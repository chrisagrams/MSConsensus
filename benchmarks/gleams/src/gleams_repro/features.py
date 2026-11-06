from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as ss
from gleams import config as gcfg
from gleams.feature import encoder as _enc
from gleams.feature import spectrum as _gs
from gleams.feature.spectrum import get_num_bins

from gleams_repro.preprocess import _entry_to_msms, _iter_mgf_xz, iter_preprocessed, preprocess_kwargs

REF_SPECTRA_SEED = 20260422  # fix so train/val/test encode use the same 500 refs
BATCH_SIZE = 512             # joblib job granularity for --workers > 1

logger = logging.getLogger(__name__)


def _bin_size() -> float:
    return getattr(gcfg, "bin_size", getattr(gcfg, "averagine_peak_separation_da"))


def build_encoder(features_cfg: dict) -> _enc.MultipleEncoder:
    """Concatenated 3010-dim encoder (precursor 61 + fragment 2449 + reference 500).

    All encoder parameters are sourced from ``gleams.config`` — using our YAML
    values here would risk drifting away from the encoding of the shipped
    reference spectra. The YAML block is retained as W&B-logged documentation.
    Seeds numpy so ReferenceSpectraEncoder's internal shuffle picks the same
    500 reference spectra every call.
    """
    np.random.seed(REF_SPECTRA_SEED)
    precursor = _enc.PrecursorEncoder(
        num_bits_mz=getattr(gcfg, "num_bits_precursor_mz", 27),
        mz_min=getattr(gcfg, "precursor_mz_min", 50.5),
        mz_max=getattr(gcfg, "precursor_mz_max", 2500.0),
        num_bits_mass=getattr(gcfg, "num_bits_precursor_mass", 27),
        mass_min=getattr(gcfg, "precursor_mass_min", 400.0),
        mass_max=getattr(gcfg, "precursor_mass_max", 6000.0),
        charge_max=getattr(gcfg, "charge_max", 7),
    )
    fragment = _enc.FragmentEncoder(
        min_mz=gcfg.fragment_mz_min,
        max_mz=gcfg.fragment_mz_max,
        bin_size=_bin_size(),
    )
    reference = _enc.ReferenceSpectraEncoder(
        filename=gcfg.ref_spectra_filename,
        preprocessing=preprocess_kwargs(features_cfg),
        fragment_mz_tol=getattr(gcfg, "fragment_mz_tol", 0.05),
        num_ref_spectra=gcfg.num_ref_spectra,
    )
    return _enc.MultipleEncoder([precursor, fragment, reference])


def feature_split() -> tuple[int, int]:
    """Index boundaries separating precursor / fragment / reference segments."""
    num_precursor = (
        getattr(gcfg, "num_bits_precursor_mz", 27)
        + getattr(gcfg, "num_bits_precursor_mass", 27)
        + getattr(gcfg, "charge_max", 7)
    )
    num_fragment = get_num_bins(gcfg.fragment_mz_min, gcfg.fragment_mz_max, _bin_size())
    return num_precursor, num_precursor + num_fragment


_WORKER_ENCODER: _enc.MultipleEncoder | None = None
_WORKER_KW: dict | None = None


def _worker_encode_batch(
    batch: list[tuple[int, dict]],
    charges_set: frozenset[int],
    features_cfg: dict,
    source_filename: str,
) -> list[tuple[dict, ss.csr_matrix]]:
    """joblib worker: reconstruct MsmsSpectrum → preprocess → encode.

    Lazily builds the encoder once per worker process and caches it at module
    scope. Input entries are raw pyteomics MGF dicts (pickleable); output is a
    list of (metadata_row, csr_feature_row) tuples.
    """
    global _WORKER_ENCODER, _WORKER_KW
    if _WORKER_ENCODER is None:
        _WORKER_ENCODER = build_encoder(features_cfg)
        _WORKER_KW = preprocess_kwargs(features_cfg)

    out: list[tuple[dict, ss.csr_matrix]] = []
    for i, entry in batch:
        spec = _entry_to_msms(entry, identifier=str(i))
        if spec is None or spec.precursor_charge not in charges_set:
            continue
        spec = _gs.preprocess(spec, **_WORKER_KW)
        if not getattr(spec, "is_valid", True):
            continue
        row = _WORKER_ENCODER.encode(spec)
        out.append(
            (
                {
                    "dataset": "massivekb",
                    "filename": source_filename,
                    "scan": i,
                    "sequence": spec._seq,
                    "charge": int(spec.precursor_charge),
                    "mz": float(spec.precursor_mz),
                },
                row,
            )
        )
    return out


def _iter_batches(mgf_xz_path: str | Path, batch_size: int = BATCH_SIZE):
    """Yield lists of (global_scan_index, pyteomics entry dict) of up to ``batch_size``."""
    batch: list[tuple[int, dict]] = []
    for i, entry in enumerate(_iter_mgf_xz(mgf_xz_path)):
        batch.append((i, entry))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_split(
    mgf_xz_path: str | Path,
    out_dir: str | Path,
    split: str,
    features_cfg: dict,
    charges: tuple[int, ...] = (2, 3, 4, 5),
    limit: int | None = None,
    workers: int = 1,
) -> None:
    """Stream MGF → preprocess → encode → write ``{split}_features.npz`` + ``{split}_metadata.parquet``.

    Metadata columns mirror what ``gleams.metadata.metadata`` expects:
    ``dataset, filename, scan, sequence, charge, mz``. Row order matches the
    feature matrix so pair-generation indices are consistent.

    ``workers`` ≥ 2 dispatches batches of ~512 spectra to a joblib pool using
    the ``loky`` backend; results are gathered in submission order to preserve
    feature↔metadata alignment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ss.csr_matrix] = []
    meta: list[dict] = []

    if workers <= 1:
        encoder = build_encoder(features_cfg)
        kw = preprocess_kwargs(features_cfg)
        for spec in iter_preprocessed(mgf_xz_path, charges, kw):
            if limit is not None and len(rows) >= limit:
                break
            rows.append(encoder.encode(spec))
            meta.append(
                {
                    "dataset": "massivekb",
                    "filename": Path(mgf_xz_path).name,
                    "scan": int(spec.identifier),
                    "sequence": spec._seq,
                    "charge": int(spec.precursor_charge),
                    "mz": float(spec.precursor_mz),
                }
            )
    else:
        charges_set = frozenset(charges)
        filename = Path(mgf_xz_path).name
        logger.info("encoding %s with %d workers (batch size %d)", filename, workers, BATCH_SIZE)
        # return_as='generator' lets us break out early when ``limit`` is hit
        # without waiting for every dispatched batch to finish.
        parallel = joblib.Parallel(
            n_jobs=workers,
            backend="loky",
            return_as="generator",
            pre_dispatch="2*n_jobs",
        )
        jobs = (
            joblib.delayed(_worker_encode_batch)(batch, charges_set, features_cfg, filename)
            for batch in _iter_batches(mgf_xz_path)
        )
        done = False
        for batch_result in parallel(jobs):
            for m, r in batch_result:
                meta.append(m)
                rows.append(r)
                if limit is not None and len(rows) >= limit:
                    done = True
                    break
            if done:
                break
            if len(rows) % (BATCH_SIZE * workers * 4) < BATCH_SIZE:
                logger.info("encoded %d spectra so far", len(rows))

    if not rows:
        raise RuntimeError(f"no spectra survived preprocessing for split={split}")

    features = ss.vstack(rows, format="csr")
    meta_df = pd.DataFrame(meta)

    feat_path = out_dir / f"{split}_features.npz"
    meta_path = out_dir / f"{split}_metadata.parquet"
    ss.save_npz(str(feat_path), features)
    meta_df.to_parquet(str(meta_path), index=False)
    n_rows, n_feats = features.shape
    logger.info("wrote %s (%d rows, %d features) and %s", feat_path, n_rows, n_feats, meta_path)
