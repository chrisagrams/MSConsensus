"""Deterministically split an xz-compressed MGF file into train_main + val.

Streams the source MGF once, assigning each spectrum to either the val output
(probability ``--val-fraction``, default 0.05) or train_main, with reproducible
draws from a seeded ``random.Random``. Used to carve a held-out validation set
from a corpus that ships with only train/test splits, so the existing pipeline
(which expects train_mgf/val_mgf/test_mgf) can run unchanged.
"""
from __future__ import annotations

import argparse
import logging
import lzma
import random
import time
from pathlib import Path

from gleams_repro.preprocess import _iter_mgf_xz

logger = logging.getLogger("split_train_val")


def _format_entry(entry: dict) -> str:
    """Re-serialize a pyteomics MGF entry dict back to MGF text.

    Preserves only the fields downstream consumers (``_entry_to_msms`` and the
    gleams encoder) actually read: TITLE, PEPMASS, CHARGE, SCANS, RTINSECONDS,
    SEQ, plus the m/z and intensity arrays. Any other params are dropped.
    """
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

    for key in ("rtinseconds", "scans"):
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


def main(src: Path, out_train: Path, out_val: Path, val_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    t0 = time.time()
    # preset=1 trades worse compression for ~3-4× faster encode; these are
    # intermediate files that get re-decoded once during feature encoding.
    with (
        lzma.open(str(out_train), "wt", preset=1) as ftrain,
        lzma.open(str(out_val), "wt", preset=1) as fval,
    ):
        for i, entry in enumerate(_iter_mgf_xz(src)):
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
        "done: train=%d, val=%d, total=%d, val_fraction_actual=%.4f, wallclock=%.0fs",
        n_train, n_val, n_train + n_val,
        n_val / max(1, n_train + n_val),
        time.time() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out-train", required=True, type=Path)
    p.add_argument("--out-val", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()
    main(args.src, args.out_train, args.out_val, args.val_fraction, args.seed)
