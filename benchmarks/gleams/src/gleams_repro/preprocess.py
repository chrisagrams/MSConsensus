from __future__ import annotations

import logging
import lzma
from pathlib import Path
from typing import Iterator

import numpy as np
from pyteomics import mgf
from spectrum_utils.spectrum import MsmsSpectrum

from gleams.feature import spectrum as _gs

logger = logging.getLogger(__name__)


def preprocess_kwargs(features_cfg: dict) -> dict:
    """Translate our YAML ``features`` block into gleams.feature.spectrum.preprocess kwargs."""
    return dict(
        mz_min=features_cfg["fragment_mz_min"],
        mz_max=features_cfg["fragment_mz_max"],
        min_peaks=features_cfg["min_peaks"],
        min_mz_range=features_cfg["min_mz_range"],
        remove_precursor_tolerance=2.0,
        min_intensity=0.01,
        max_peaks_used=features_cfg["max_peaks"],
        scaling="root",
    )


def _iter_mgf_xz(path: str | Path) -> Iterator[dict]:
    """Iterate pyteomics MGF entries from .mgf or .mgf.xz (compression by extension)."""
    path_str = str(path)
    opener = lzma.open if path_str.endswith(".xz") else open
    with opener(path_str, "rt") as fh:
        yield from mgf.read(fh, use_index=False)


def _entry_to_msms(entry: dict, identifier: str) -> MsmsSpectrum | None:
    params = entry.get("params", {})
    seq = params.get("seq") or params.get("sequence")
    if seq is None:
        return None
    try:
        precursor_mz = float(params["pepmass"][0])
        charge_raw = params["charge"][0]
        charge = int(getattr(charge_raw, "real", charge_raw))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if charge <= 0:
        return None
    mz = np.asarray(entry["m/z array"], dtype=np.float64)
    intensity = np.asarray(entry["intensity array"], dtype=np.float32)
    if mz.size == 0:
        return None
    spec = MsmsSpectrum(
        identifier=identifier,
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mz,
        intensity=intensity,
    )
    # Stash the annotation as a sidecar; spectrum_utils 0.3.5's peptide setter
    # tries to parse each character as an amino acid and rejects the casanovo
    # mod syntax ("+57.021M..."), which we only need for string equality.
    spec._seq = str(seq)
    # gleams.feature.spectrum.preprocess reads this flag before doing anything.
    spec.is_processed = False
    return spec


def iter_preprocessed(
    mgf_xz_path: str | Path,
    charges: tuple[int, ...],
    preprocess_kw: dict,
) -> Iterator[MsmsSpectrum]:
    """Yield GLEAMS-preprocessed MsmsSpectrum objects from an xz-compressed MGF.

    Skips spectra missing an annotation, outside the configured charge range, or
    filtered out by ``gleams.feature.spectrum.preprocess``.
    """
    charges_set = set(charges)
    kept = dropped_meta = dropped_invalid = 0
    for i, entry in enumerate(_iter_mgf_xz(mgf_xz_path)):
        spec = _entry_to_msms(entry, identifier=str(i))
        if spec is None or spec.precursor_charge not in charges_set:
            dropped_meta += 1
            continue
        spec = _gs.preprocess(spec, **preprocess_kw)
        if not getattr(spec, "is_valid", True):
            dropped_invalid += 1
            continue
        kept += 1
        if (kept + dropped_meta + dropped_invalid) % 100_000 == 0:
            logger.info(
                "scanned %d (kept=%d, dropped_meta=%d, dropped_invalid=%d)",
                kept + dropped_meta + dropped_invalid, kept, dropped_meta, dropped_invalid,
            )
        yield spec
    logger.info(
        "MGF %s exhausted: kept=%d, dropped_meta=%d, dropped_invalid=%d",
        mgf_xz_path, kept, dropped_meta, dropped_invalid,
    )
