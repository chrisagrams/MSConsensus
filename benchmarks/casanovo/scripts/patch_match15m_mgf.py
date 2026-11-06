"""Inject theoretical PEPMASS into the new_match_15M consensus MGFs.

The 25 source MGFs at ``data/concensus_spec_30M/new_match_15M/``
are missing ``PEPMASS=`` lines (only TITLE / CHARGE / SEQ / peaks). Casanovo's
depthcharge MGF parser populates ``precursor_mz`` from PEPMASS; without it the
column is NaN and training crashes.

We compute the monoisotopic precursor m/z from SEQ + CHARGE using the v5
residue alphabet (handles ``C[57.0215]`` and ``M[15.9949]`` as residue keys
before later normalize_lance_to_v5 rewrites them to named mods). For consensus
spectra a theoretical precursor is fine — the underlying spectra are already
ensemble-averaged.

Streamed, constant-memory, idempotent (skips files whose output already
exists). Fan out 25-way: one worker per source MGF.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pyteomics import mass

logger = logging.getLogger("casanovo_repro.patch_match15m")


# Build residue masses from pyteomics' NIST-derived monoisotopic table. The
# only two bracket forms in this dataset are C[57.0215] (carbamidomethyl,
# Unimod #4) and M[15.9949] (oxidation, Unimod #35); the bracket numbers in
# the source data are rounded — we add canonical Unimod mod masses to the
# bare residue rather than trusting the rounded literal.
_RESIDUE_MASS: dict[str, float] = dict(mass.std_aa_mass)
_RESIDUE_MASS["C[57.0215]"] = mass.std_aa_mass["C"] + mass.calculate_mass(formula="C2H3NO")  # +CAM
_RESIDUE_MASS["M[15.9949]"] = mass.std_aa_mass["M"] + mass.calculate_mass(formula="O")        # +Ox

_H2O = mass.calculate_mass(formula="H2O")
_PROTON = mass.nist_mass["H+"][0][0]

_TOKEN_RE = re.compile(r"[A-Z](?:\[[^\]]+\])?")


def _peptide_mass(seq: str) -> float:
    """Sum residue masses + water. Raises KeyError for unseen residues."""
    total = _H2O
    pos = 0
    n = len(seq)
    while pos < n:
        m = _TOKEN_RE.match(seq, pos)
        if not m:
            raise ValueError(f"unparseable residue at pos {pos} of {seq!r}")
        tok = m.group(0)
        total += _RESIDUE_MASS[tok]
        pos = m.end()
    return total


def _parse_charge(charge_line: str) -> int:
    # CHARGE=2+, CHARGE=2, CHARGE=2- (latter not expected; treat as int)
    val = charge_line.split("=", 1)[1].strip()
    val = val.rstrip("+-")
    return int(val)


def patch_one(src: Path, dst: Path) -> tuple[Path, int, float]:
    """Stream src MGF, inject PEPMASS per block, write to dst.

    Returns (dst, n_blocks, elapsed_s).
    """
    if dst.exists():
        logger.info("skip (exists): %s", dst)
        return dst, 0, 0.0

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    n_blocks = 0
    t0 = time.time()
    with open(src, "rt") as fin, open(tmp, "wt") as fout:
        # Per-block header buffer, populated up to the first peak line. We
        # need SEQ + CHARGE before we can emit PEPMASS, so the headers are
        # emitted lazily once we hit the first peak (or END IONS in the
        # degenerate empty case).
        headers: list[str] = []
        seq: str | None = None
        charge: int | None = None
        flushed = False  # emitted headers for the current block?

        def _flush_headers(out):
            nonlocal flushed
            if flushed:
                return
            assert seq is not None and charge is not None, "missing SEQ/CHARGE"
            mz = (_peptide_mass(seq) + charge * _PROTON) / charge
            # Match split_train_val._format_entry ordering: TITLE, PEPMASS,
            # CHARGE, then SEQ. Other header lines (none in this dataset) get
            # appended after CHARGE in source order.
            wrote_pepmass = False
            wrote_charge = False
            for h in headers:
                if h.startswith("CHARGE=") and not wrote_pepmass:
                    out.write(f"PEPMASS={mz:.6f}\n")
                    wrote_pepmass = True
                out.write(h)
                if h.startswith("CHARGE="):
                    wrote_charge = True
            if not wrote_pepmass:
                # Defensive: no CHARGE= line ever seen — emit PEPMASS at end.
                out.write(f"PEPMASS={mz:.6f}\n")
            assert wrote_charge, "block had no CHARGE= line"
            flushed = True

        for raw in fin:
            line = raw  # keep newline as-is
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                headers = [line]
                seq = None
                charge = None
                flushed = False
                continue
            if stripped == "END IONS":
                if not flushed:
                    _flush_headers(fout)
                fout.write(line)
                n_blocks += 1
                if n_blocks % 100_000 == 0:
                    logger.info("%s: %d blocks (%.1fs)",
                                src.name, n_blocks, time.time() - t0)
                continue

            if not flushed:
                # Still in header region. Capture SEQ / CHARGE for mass calc.
                if stripped.startswith("SEQ="):
                    seq = stripped.split("=", 1)[1].strip()
                    headers.append(line)
                    continue
                if stripped.startswith("CHARGE="):
                    charge = _parse_charge(stripped)
                    headers.append(line)
                    continue
                if stripped.startswith("PEPMASS="):
                    # Source has none, but be defensive: don't double-emit.
                    headers.append(line)
                    continue
                if stripped and not stripped[0].isdigit():
                    # Other header (TITLE, RTINSECONDS, SCANS, ...).
                    headers.append(line)
                    continue
                # First peak line → flush headers, then write the peak.
                _flush_headers(fout)
                fout.write(line)
            else:
                fout.write(line)

    tmp.replace(dst)
    dt = time.time() - t0
    logger.info("done %s: %d blocks (%.1fs)", dst.name, n_blocks, dt)
    return dst, n_blocks, dt


def main(src_dir: Path, dst_dir: Path, max_workers: int) -> int:
    sources = sorted(src_dir.glob("consensus_*.mgf"))
    if not sources:
        logger.error("no consensus_*.mgf in %s", src_dir)
        return 1
    logger.info("patching %d MGFs from %s -> %s (workers=%d)",
                len(sources), src_dir, dst_dir, max_workers)

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for src in sources:
            dst = dst_dir / src.name
            futures[pool.submit(patch_one, src, dst)] = src
        for f in as_completed(futures):
            src = futures[f]
            try:
                f.result()
            except Exception:
                logger.exception("FAILED: %s", src)
                return 2
    logger.info("all done")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path,
                   default=Path("data/concensus_spec_30M/new_match_15M"))
    p.add_argument("--dst-dir", type=Path,
                   default=Path("casanovo/data/new_match_15M/mgf_patched"))
    p.add_argument("--workers", type=int, default=25)
    args = p.parse_args()
    sys.exit(main(args.src_dir, args.dst_dir, args.workers))
