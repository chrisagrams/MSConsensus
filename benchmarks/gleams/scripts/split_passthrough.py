"""Block-level pass-through MGF splitter.

Reads MGF text from stdin, deterministically routes each ``BEGIN IONS`` /
``END IONS`` block to either the train_main or val output by seeded RNG, and
writes blocks verbatim — no spectrum parsing, no re-formatting.

Designed to be fed by ``xzcat -T 0 src.xz | python split_passthrough.py ...``
so that xz decompression parallelizes across cores while this single Python
process handles only the cheap text-block dispatch.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

logger = logging.getLogger("split_passthrough")


def main(out_train: Path, out_val: Path, val_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    t0 = time.time()

    # 8 MB buffers favor sequential-write throughput.
    with (
        open(str(out_train), "w", buffering=8 * 1024 * 1024) as ftrain,
        open(str(out_val), "w", buffering=8 * 1024 * 1024) as fval,
    ):
        buffer: list[str] = []
        in_block = False
        for line in sys.stdin:
            if line.startswith("BEGIN IONS"):
                in_block = True
                buffer = [line]
            elif line.startswith("END IONS"):
                buffer.append(line)
                buffer.append("\n")
                text = "".join(buffer)
                if rng.random() < val_fraction:
                    fval.write(text)
                    n_val += 1
                else:
                    ftrain.write(text)
                    n_train += 1
                buffer.clear()
                in_block = False
                total = n_train + n_val
                if total and total % 500_000 == 0:
                    logger.info(
                        "scanned %d (train=%d, val=%d, elapsed=%.0fs)",
                        total, n_train, n_val, time.time() - t0,
                    )
            elif in_block:
                buffer.append(line)

    logger.info(
        "done: train=%d, val=%d, total=%d, val_fraction_actual=%.4f, wallclock=%.0fs",
        n_train, n_val, n_train + n_val,
        n_val / max(1, n_train + n_val),
        time.time() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-train", required=True, type=Path)
    p.add_argument("--out-val", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260426)
    args = p.parse_args()
    main(args.out_train, args.out_val, args.val_fraction, args.seed)
