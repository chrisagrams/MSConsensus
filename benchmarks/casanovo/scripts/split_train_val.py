"""Deterministic spectrum-level train/val split for the consensus 30M MGF.

Adapted from MSBench/prosit/scripts/split_train_val.py. The consensus corpus
ships only train + test, so we carve a held-out validation set from train.
Stream the source MGF once; for each ``BEGIN IONS`` block draw a uniform
[0, 1) sample from a seeded ``random.Random`` and route to ``--out-train`` or
``--out-val`` (probability ``--val-fraction``, default 0.05).

Differences from the prosit version:

  - Output is plain MGF (not xz-compressed). Casanovo/depthcharge mmaps the
    MGF when building its Lance index, so leaving it uncompressed lets the
    preprocess job avoid an in-process lzma decode per spectrum.
  - No top-1-by-SCORE deduplication: casanovo trains on all PSMs as-is (the
    paper trains on "30 million PSMs", not 30M unique peptides).
"""
from __future__ import annotations

import argparse
import logging
import lzma
import random
import time
from pathlib import Path

from pyteomics import mgf

logger = logging.getLogger("casanovo_repro.split_train_val")


def _format_entry(entry: dict) -> str:
    lines = ["BEGIN IONS"]
    params = entry.get("params") or {}

    if (title := params.get("title")) is not None:
        lines.append(f"TITLE={title}")

    if (pm := params.get("pepmass")) is not None:
        if isinstance(pm, (list, tuple)):
            mz = pm[0]
            intensity = pm[1] if len(pm) > 1 and pm[1] is not None else None
        else:
            mz, intensity = pm, None
        lines.append(f"PEPMASS={mz}" if intensity is None else f"PEPMASS={mz} {intensity}")

    if (ch := params.get("charge")) is not None:
        if isinstance(ch, (list, tuple)):
            ch = ch[0]
        try:
            ch_int = int(getattr(ch, "real", ch))
            lines.append(f"CHARGE={ch_int}+")
        except (TypeError, ValueError):
            lines.append(f"CHARGE={ch}")

    for key in ("rtinseconds", "scans", "score", "appearance_count"):
        v = params.get(key)
        if v is not None:
            lines.append(f"{key.upper()}={v}")

    seq = params.get("seq") or params.get("sequence")
    if seq is not None:
        lines.append(f"SEQ={seq}")

    mz_arr = entry.get("m/z array")
    int_arr = entry.get("intensity array")
    if mz_arr is not None and int_arr is not None:
        for m, it in zip(mz_arr, int_arr):
            lines.append(f"{m:.5f} {it:.4f}")

    lines.append("END IONS")
    return "\n".join(lines) + "\n\n"


def _open_src(path: Path):
    s = str(path)
    return lzma.open(s, "rt") if s.endswith(".xz") else open(s, "rt")


def main(src: Path, out_train: Path, out_val: Path,
         val_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    t0 = time.time()
    with _open_src(src) as fh, \
         open(out_train, "wt") as ftrain, \
         open(out_val, "wt") as fval:
        for entry in mgf.read(fh, convert_arrays=2, read_charges=False):
            text = _format_entry(entry)
            if rng.random() < val_fraction:
                fval.write(text)
                n_val += 1
            else:
                ftrain.write(text)
                n_train += 1
            total = n_train + n_val
            if total and total % 200_000 == 0:
                logger.info(
                    "scanned %d (train=%d, val=%d, elapsed=%.0fs)",
                    total, n_train, n_val, time.time() - t0,
                )

    logger.info(
        "done: train=%d val=%d total=%d val_fraction_actual=%.4f wallclock=%.0fs",
        n_train, n_val, n_train + n_val,
        n_val / max(1, n_train + n_val), time.time() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path,
                   help="Source MGF (plain or .xz). Read once, streaming.")
    p.add_argument("--out-train", required=True, type=Path)
    p.add_argument("--out-val", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260429)
    args = p.parse_args()
    main(args.src, args.out_train, args.out_val, args.val_fraction, args.seed)
