"""Stream a (possibly xz-compressed) MGF, annotate b/y ions using kusterlab/prosit's
upstream chemistry primitives, and write Prosit parquet shards.

Output schema matches DLOmix's ``FragmentIonIntensityDataset`` expectation:

    modified_sequence:        str    (UNIMOD-bracket; C is implicitly carbamidomethyl
                                       per kusterlab's mass table, emitted as C[UNIMOD:4]
                                       for DLOmix-side parser consistency)
    precursor_charge:         int64  1..6
    precursor_charge_onehot:  list[int]  length 6
    aligned_collision_energy: float64
    collision_energy_aligned_normed: float64  (= aligned / 100)
    intensities_raw:          list[float64]   length 174 (max-norm; -1 for impossible slots)
    masses_raw:               list[float64]   length 174 (theoretical m/z; 0 for impossible)
    score:                    float64         (passed-through SCORE)
    appearance_count:         int64           (passed-through)

The 174-dim layout, alphabet, and mass tables are kusterlab/prosit's
(``prosit_repro.upstream``) — vendored verbatim under Apache-2.0. The only
script-local logic is:

  - the MGF reader (kusterlab assumed MaxQuant-pre-matched input, so they
    never had a "raw spectrum -> 174-dim" function);
  - top-1 by SCORE per (SEQ, charge) deduplication;
  - filling the 5D ``[N, MAX_ION, ION_TYPES, NLOSSES, CHARGES]`` intensity
    tensor by mirroring upstream's ``tensorize.get_mz_applied`` mass-tensor
    layout (then running their sanitize chain unchanged);
  - injecting C->C[UNIMOD:4] / M(ox)->M[UNIMOD:35] notation in the emitted
    ``modified_sequence`` column.
"""
from __future__ import annotations

import argparse
import logging
import lzma
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyteomics import mgf

from prosit_repro.upstream import annotate as up_annotate
from prosit_repro.upstream import constants as up_const
from prosit_repro.upstream import match as up_match
from prosit_repro.upstream import sanitize as up_sanitize
from prosit_repro.upstream import tensorize as up_tensorize

logger = logging.getLogger("mgf_to_prosit_parquet")

LEN_FRAG = up_const.MAX_ION * len(up_const.ION_TYPES) * 1 * up_const.MAX_FRAG_CHARGE  # 174


# The 4G consensus dump emits modifications as mass-delta brackets in SEQ=
# (C[57.0215], M[15.9949]) instead of kusterlab's parens form. kusterlab's
# peptide_parser only understands `(...)`-style mods, and AMINO_ACID["C"] is
# already +CAM, so we strip the C tag entirely and rewrite oxidation to M(ox).
_MASS_TAG_TO_KUSTER = {
    "C[57.0215]": "C",
    "M[15.9949]": "M(ox)",
}


def _normalize_seq_tokens(seq: str) -> str:
    for tag, replacement in _MASS_TAG_TO_KUSTER.items():
        seq = seq.replace(tag, replacement)
    return seq


def _to_unimod(seq: str) -> str:
    """Translate kusterlab's parens-style mod tokens to UNIMOD bracket form, and
    inject C[UNIMOD:4] for the carbamidomethyl-by-default C. The DLOmix parser
    is most-tested against UNIMOD-bracket input."""
    out = seq.replace("M(ox)", "M[UNIMOD:35]")
    # Implicit CAM on bare C; skip already-modified C[UNIMOD:4].
    res: list[str] = []
    i = 0
    while i < len(out):
        ch = out[i]
        if ch == "C" and not (i + 1 < len(out) and out[i + 1] == "["):
            res.append("C[UNIMOD:4]")
        else:
            res.append(ch)
        i += 1
    return "".join(res)


def _annotate_one(seq: str, charge: int, mz_obs: np.ndarray, intens_obs: np.ndarray,
                  mass_analyzer: str, ppm_override: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Use upstream chemistry to fill a 5D intensity tensor and emit (intens174, mz174).

    This mirrors ``upstream.tensorize.get_mz_applied``'s logic: build the per-charge
    annotation, but instead of writing theoretical m/z, write the matched observed
    intensity. Then run upstream's sanitize chain unchanged.
    """
    fw, bw = up_match.get_forward_backward(seq)
    L = len(fw)
    # 5D tensor identical in layout to upstream's masses tensor:
    #   [MAX_ION, ION_TYPES, NLOSSES, CHARGES]
    intens = np.zeros([up_const.MAX_ION, len(up_const.ION_TYPES),
                       len(up_const.NLOSSES), len(up_const.CHARGES)],
                      dtype=np.float64)
    mz_th  = np.zeros_like(intens)

    # Sort observed peaks once (upstream's binarysearch requires sorted input).
    order = np.argsort(mz_obs)
    mz_sorted = mz_obs[order]
    int_sorted = intens_obs[order]

    cap_charge = min(charge, up_const.MAX_FRAG_CHARGE)
    for z in range(cap_charge):
        zpp = z + 1
        ann = up_annotate.get_annotation(fw, bw, zpp, "yb")
        for ion_str, mz_t in ann.items():
            it, ion_idx, nloss = up_tensorize.parse_ion(ion_str)
            if ion_idx >= up_const.MAX_ION:
                continue
            mz_th[ion_idx, it, nloss, z] = mz_t
            if len(mz_sorted) == 0:
                continue
            if ppm_override is not None:
                tol = mz_t * ppm_override * 1e-6
                lo = np.searchsorted(mz_sorted, mz_t - tol, side="left")
                hi = np.searchsorted(mz_sorted, mz_t + tol, side="right")
                if lo < hi:
                    intens[ion_idx, it, nloss, z] = float(int_sorted[lo:hi].max())
            else:
                idx = up_match.binarysearch(mz_sorted, mz_t, mass_analyzer)
                if idx is not None:
                    # binarysearch returns the first hit; take the local max within tolerance
                    intens[ion_idx, it, nloss, z] = float(int_sorted[idx])

    # Add the batch dim and run upstream's sanitize chain.
    intens5 = intens[np.newaxis, ...]                              # (1, 29, 2, 3, 6)
    mz5     = mz_th[np.newaxis, ...]
    intens5 = up_sanitize.cap(intens5, nlosses=1, z=up_const.MAX_FRAG_CHARGE)
    mz5     = up_sanitize.cap(mz5,     nlosses=1, z=up_const.MAX_FRAG_CHARGE)
    intens5 = up_sanitize.mask_outofrange(intens5, np.array([L]))
    mz5     = up_sanitize.mask_outofrange(mz5,     np.array([L]), mask=0.0)
    intens5 = up_sanitize.mask_outofcharge(intens5, np.array([charge]))
    mz5     = up_sanitize.mask_outofcharge(mz5,     np.array([charge]), mask=0.0)
    intens174 = up_sanitize.reshape_flat(intens5)[0]
    mz174     = up_sanitize.reshape_flat(mz5)[0]

    # Max-normalize valid (non-masked, non-zero) intensities to [0, 1].
    valid_mask = intens174 > 0
    if valid_mask.any():
        peak = float(intens174[valid_mask].max())
        if peak > 0:
            intens174 = np.where(intens174 > 0, intens174 / peak, intens174)
    return intens174, mz174


def _onehot_charge(c: int, n: int = 6) -> list[int]:
    v = [0] * n
    if 1 <= c <= n:
        v[c - 1] = 1
    return v


def _open_mgf_stream(path: Path):
    if str(path).endswith(".xz"):
        return lzma.open(path, "rt")
    return open(path, "rt")


def _coerce_charge(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    try:
        return int(getattr(raw, "real", raw))
    except (TypeError, ValueError):
        return None


def _flush_shard(rows: list[dict], out_dir: Path, shard_idx: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"part-{shard_idx:05d}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))
    logger.info("wrote %s (%d rows)", path, len(df))


def annotate(src: Path, out_dir: Path, *, ce: float, mass_analyzer: str,
             ppm_override: float | None, min_peaks: int, max_seq_len: int,
             charges: tuple[int, ...], shard_rows: int,
             dedup_mode: str = "none", top_k: int = 1) -> None:
    """Stream the input MGF, optionally dedupe, annotate, write parquet shards.

    ``mass_analyzer`` selects the upstream tolerance preset (FTMS=25 ppm,
    ITMS=0.35 Da, TripleTOF=50 ppm). Pass ``ppm_override`` to force a custom
    ppm tolerance.

    ``dedup_mode``:
      - ``"none"``: keep every annotated PSM. Mirrors the Prosit paper's
        training-data schema (~16x PSMs per peptide). Stream-writes shards
        as they fill, so memory stays bounded regardless of corpus size.
      - ``"top_k"``: keep the top-``top_k`` PSMs per (seq, charge) ordered by
        SCORE. ``top_k=1`` reproduces the original (over-aggressive) dedup.
        Memory grows with the unique-pair count.
    """
    if dedup_mode not in ("none", "top_k"):
        raise ValueError(f"dedup_mode must be 'none' or 'top_k', got {dedup_mode!r}")
    logger.info(
        "annotating %s -> %s (mass_analyzer=%s, ppm_override=%s, dedup=%s, k=%d)",
        src, out_dir, mass_analyzer, ppm_override, dedup_mode, top_k,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Streaming buffers (used by both modes for the final write path)
    n_seen = 0
    n_kept = 0
    t0 = time.time()
    streaming_rows: list[dict] = []
    shard_idx = 0
    survivors: dict[tuple[str, int], list[tuple[float, int, dict]]] = {}
    counter = 0  # tiebreaker so heapq doesn't try to compare dict instances

    def _build_row(seq: str, charge: int, mz_obs: np.ndarray,
                   intens_obs: np.ndarray, score: float, appear: int) -> dict | None:
        try:
            intens174, mz174 = _annotate_one(seq, charge, mz_obs, intens_obs,
                                             mass_analyzer, ppm_override)
        except (KeyError, ValueError) as e:
            logger.debug("skipping %s: %s", seq, e)
            return None
        return {
            "modified_sequence": _to_unimod(seq),
            "precursor_charge": int(charge),
            "precursor_charge_onehot": _onehot_charge(charge),
            "aligned_collision_energy": float(ce),
            "collision_energy_aligned_normed": float(ce) / 100.0,
            "intensities_raw": intens174.tolist(),
            "masses_raw": mz174.tolist(),
            "score": score,
            "appearance_count": appear,
        }

    with _open_mgf_stream(src) as fh:
        for entry in mgf.read(fh, convert_arrays=2, read_charges=False, use_index=False):
            n_seen += 1
            # Throughput log fires unconditionally every 200k SCANNED, not 200k kept.
            if n_seen % 200_000 == 0:
                logger.info("scanned=%d kept=%d elapsed=%.0fs",
                            n_seen, n_kept, time.time() - t0)

            params = entry.get("params") or {}
            seq = params.get("seq") or params.get("sequence")
            charge = _coerce_charge(params.get("charge"))
            if seq is None or charge is None:
                continue
            seq = _normalize_seq_tokens(seq)
            # If unrecognised bracket mods survived translation, skip — kusterlab's
            # peptide_parser would crash on them. Length check uses the normalised
            # form so e.g. C[57.0215]VESEIR (16 chars) compares against the bare
            # 8-residue peptide it represents.
            if "[" in seq:
                continue
            if charge not in charges:
                continue
            if len(seq) > max_seq_len:
                continue
            mz_arr_raw = entry.get("m/z array")
            mz_obs = np.asarray(mz_arr_raw if mz_arr_raw is not None else [], dtype=np.float64)
            if len(mz_obs) < min_peaks:
                continue
            try:
                score = float(params.get("score", "nan"))
            except (TypeError, ValueError):
                score = float("nan")
            try:
                appear = int(params.get("appearance_count", 0))
            except (TypeError, ValueError):
                appear = 0

            if dedup_mode == "top_k":
                key = (seq, charge)
                bucket = survivors.get(key)
                # Skip cheaply if bucket is full and our score doesn't beat its smallest.
                if bucket is not None and len(bucket) >= top_k and score <= bucket[0][0]:
                    continue
                int_arr_raw = entry.get("intensity array")
                intens_obs = np.asarray(int_arr_raw if int_arr_raw is not None else [], dtype=np.float64)
                row = _build_row(seq, charge, mz_obs, intens_obs, score, appear)
                if row is None:
                    continue
                if bucket is None:
                    survivors[key] = [(score, counter, row)]
                else:
                    if len(bucket) < top_k:
                        import heapq
                        heapq.heappush(bucket, (score, counter, row))
                    else:
                        import heapq
                        heapq.heappushpop(bucket, (score, counter, row))
                counter += 1
                n_kept = sum(len(v) for v in survivors.values())
            else:  # dedup_mode == "none": stream-write every annotated row
                int_arr_raw = entry.get("intensity array")
                intens_obs = np.asarray(int_arr_raw if int_arr_raw is not None else [], dtype=np.float64)
                row = _build_row(seq, charge, mz_obs, intens_obs, score, appear)
                if row is None:
                    continue
                streaming_rows.append(row)
                n_kept += 1
                if len(streaming_rows) >= shard_rows:
                    _flush_shard(streaming_rows, out_dir, shard_idx)
                    streaming_rows = []
                    shard_idx += 1

    logger.info("scan done: scanned=%d kept=%d wallclock=%.0fs",
                n_seen, n_kept, time.time() - t0)

    if dedup_mode == "top_k":
        # Drain the survivors dict in sharded chunks
        rows: list[dict] = []
        for bucket in survivors.values():
            for _, _, row in bucket:
                rows.append(row)
                if len(rows) >= shard_rows:
                    _flush_shard(rows, out_dir, shard_idx)
                    rows = []
                    shard_idx += 1
        _flush_shard(rows, out_dir, shard_idx)
    else:
        # Flush any remaining streaming rows
        _flush_shard(streaming_rows, out_dir, shard_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--ce", type=float, default=35.0)
    p.add_argument("--mass-analyzer", default="FTMS",
                   choices=list(up_const.TOLERANCE.keys()),
                   help="Tolerance preset from upstream constants.TOLERANCE.")
    p.add_argument("--ppm-override", type=float, default=None,
                   help="If set, use this ppm tolerance instead of the mass_analyzer preset.")
    p.add_argument("--min-peaks", type=int, default=5)
    p.add_argument("--max-seq-len", type=int, default=30)
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--shard-rows", type=int, default=250_000)
    p.add_argument("--dedup-mode", choices=["none", "top_k"], default="none",
                   help="'none' = keep every PSM (paper schema). 'top_k' = keep top-K by SCORE per (seq, charge).")
    p.add_argument("--top-k", type=int, default=1,
                   help="K for --dedup-mode top_k. Ignored otherwise.")
    args = p.parse_args()
    annotate(
        src=args.src, out_dir=args.out_dir,
        ce=args.ce, mass_analyzer=args.mass_analyzer,
        ppm_override=args.ppm_override,
        min_peaks=args.min_peaks, max_seq_len=args.max_seq_len,
        charges=tuple(args.charges), shard_rows=args.shard_rows,
        dedup_mode=args.dedup_mode, top_k=args.top_k,
    )
