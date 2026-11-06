"""Deterministically split an xz-compressed MGF into train_main + val.

Adapted from MSBench/gleams/scripts/split_train_val.py. Streams the source MGF
once, assigns each spectrum to either val (probability ``--val-fraction``,
default 0.05) or train_main, with reproducible draws from a seeded
``random.Random``. Used to carve a held-out validation set from a corpus that
ships with only train/test splits, so the annotation pipeline can run with
explicit train/val/test inputs.

Note: this carves at the *spectrum* level. The downstream annotation pass
deduplicates to top-1 by SCORE per (sequence, charge) within each split, so a
peptide that appears in many input rows can end up in only one of {train,val}
once dedup runs — but if multiple rows for the same peptide land on different
sides of the split, dedup happens *after* the split, which means the same
(seq, charge) can appear in both train and val with different SCORE.
For Prosit's purposes that's fine: we evaluate val on the model's prediction
of the full 174-dim canonical spectrum, and a duplicate peptide is benign.
"""
from __future__ import annotations

import argparse
import logging
import lzma
import random
import time
from pathlib import Path

from pyteomics import mgf

logger = logging.getLogger("split_train_val")


def _format_entry(entry: dict) -> str:
    """Re-serialize a pyteomics MGF entry dict back to MGF text, preserving the
    fields downstream annotation reads (TITLE/PEPMASS/CHARGE/RTINSECONDS/SCANS/
    SEQ/SCORE/APPEARANCE_COUNT) plus the m/z and intensity arrays."""
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


def _open_text_writer(path: Path):
    """Open ``path`` for text writing. Use lzma if the suffix is ``.xz``,
    otherwise plain text. Plain text is dramatically faster (no compression
    cost) at the price of disk; intermediate split files don't need to be
    compressed because they're consumed once by the annotator."""
    if str(path).endswith(".xz"):
        return lzma.open(str(path), "wt", preset=1)
    return open(str(path), "wt", buffering=1 << 20)


def main(src: Path, out_train: Path, out_val: Path,
         val_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    t0 = time.time()
    src_open = lzma.open(str(src), "rt") if str(src).endswith(".xz") else open(str(src), "rt")
    with src_open as fh, _open_text_writer(out_train) as ftrain, _open_text_writer(out_val) as fval:
        for entry in mgf.read(fh, convert_arrays=2, read_charges=False, use_index=False):
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
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out-train", required=True, type=Path)
    p.add_argument("--out-val", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()
    main(args.src, args.out_train, args.out_val, args.val_fraction, args.seed)
