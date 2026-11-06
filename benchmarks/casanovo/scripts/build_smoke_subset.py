#!/usr/bin/env python3
"""Stratified peptide-level subset builder for casanovo smoke tests.

Reads ``casanovo/data/lance_v5/<dataset>/{train,valid}.lance`` and writes
``casanovo/data/lance_v5_smoke/<dataset>/{train,valid}.lance`` containing K
random unique peptides, with up to M random spectra drawn per peptide.

Stratifying on unique peptide (rather than uniform row sampling) is the
right move because the two datasets have very different replication factors:
~111x for concensus_spec_30M and ~19x for massive_kb_v1_shared. A uniform
50k-row sample of consensus would cover only ~450 unique peptides, which
barely exercises the model's output vocabulary. Stratified sampling lets us
target peptide-diversity coverage independently of row count.

Per-dataset (K, M) below are picked so each split is ~50k rows (train) /
~5k rows (valid) with ~4x more peptide diversity than uniform row sampling.

One-shot. Idempotent: skips any destination that already exists.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import lance
import pyarrow as pa


def _stratified_indices(src: Path,
                        n_peptides: int,
                        cap_per_peptide: int,
                        seed: int) -> tuple[list[int], int, int]:
    """Scan ``src``'s seq column, bucket row indices by seq, pick
    ``n_peptides`` distinct seqs uniformly at random, and from each pick up
    to ``cap_per_peptide`` row indices uniformly at random.

    Returns (sorted_picked_indices, total_unique_seqs, n_chosen_seqs).
    """
    ds = lance.dataset(str(src))
    seq_to_idxs: dict[str, list[int]] = defaultdict(list)
    pos = 0
    t0 = time.time()
    for batch in ds.scanner(columns=["seq"], batch_size=1_000_000).to_batches():
        for s in batch.column("seq").to_pylist():
            if s is not None:
                seq_to_idxs[s].append(pos)
            pos += 1
    print(f"  scanned {pos:,} rows -> {len(seq_to_idxs):,} unique seqs "
          f"({time.time() - t0:.1f}s)", flush=True)

    rng = random.Random(seed)
    seqs = list(seq_to_idxs)
    n_chosen = min(n_peptides, len(seqs))
    chosen = rng.sample(seqs, n_chosen)

    picked: list[int] = []
    for seq in chosen:
        idxs = seq_to_idxs[seq]
        if len(idxs) <= cap_per_peptide:
            picked.extend(idxs)
        else:
            picked.extend(rng.sample(idxs, cap_per_peptide))
    picked.sort()  # lance.take is faster on sorted indices
    return picked, len(seqs), n_chosen


def build_subset(src: Path,
                 dst: Path,
                 n_peptides: int,
                 cap_per_peptide: int,
                 seed: int) -> None:
    if dst.exists():
        print(f"  skip (exists): {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  src: {src}", flush=True)
    print(f"  dst: {dst}", flush=True)
    print(f"  K (peptides) = {n_peptides:,}, M (cap) = {cap_per_peptide}, "
          f"seed = {seed}", flush=True)

    indices, n_total_seqs, n_chosen_seqs = _stratified_indices(
        src, n_peptides, cap_per_peptide, seed,
    )
    print(f"  picked {len(indices):,} rows across {n_chosen_seqs:,} "
          f"of {n_total_seqs:,} unique seqs", flush=True)

    src_ds = lance.dataset(str(src))
    t0 = time.time()
    table = src_ds.take(indices)
    print(f"  materialized take() in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    lance.write_dataset(table, str(dst), schema=src_ds.schema, mode="create")
    print(f"  wrote {table.num_rows:,} rows -> {dst} "
          f"({time.time() - t0:.1f}s)", flush=True)


# (src_dataset, dst_dataset, split, K, M, seed).
#   src_dataset: which lance_v5/<name>/ to read from
#   dst_dataset: which lance_v5_smoke/<name>/ to write to (often == src)
#   K: number of unique peptides to draw
#   M: max spectra per peptide
# Seeds differ between train and valid so the RNG draws are independent.
# Sampling is from disjoint source lances (train vs valid splits already
# separated upstream), so no leakage risk.
#
# `concensus_spec_30M_diverse` is a peer of `concensus_spec_30M` that mirrors
# massive_kb_v1_shared's stratification profile (~50k unique peptides at ~3
# spectra/peptide for train), so the consensus smoke can be exercised at the
# same diversity-vs-replication regime as the massive_kb smoke.
_TARGETS: list[tuple[str, str, str, int, int, int]] = [
    ("concensus_spec_30M",   "concensus_spec_30M",            "train", 10_000, 25, 7),
    ("concensus_spec_30M",   "concensus_spec_30M",            "valid",  2_000, 10, 8),
    ("massive_kb_v1_shared", "massive_kb_v1_shared",          "train", 50_000,  5, 7),
    ("massive_kb_v1_shared", "massive_kb_v1_shared",          "valid",  2_000, 10, 8),
    ("concensus_spec_30M",   "concensus_spec_30M_diverse",    "train", 50_000,  3, 7),
    ("concensus_spec_30M",   "concensus_spec_30M_diverse",    "valid",  2_000,  5, 8),
    # 2x variants — same stratification ratios as above, double the K (peptide
    # count) so the model sees twice as much diversity per epoch. Schedules in
    # the matching smoke_*_2x.yaml configs are sized to the new step counts.
    ("concensus_spec_30M",   "concensus_spec_30M_2x",         "train", 20_000, 25, 7),
    ("concensus_spec_30M",   "concensus_spec_30M_2x",         "valid",  4_000, 10, 8),
    ("concensus_spec_30M",   "concensus_spec_30M_diverse_2x", "train", 100_000, 3, 7),
    ("concensus_spec_30M",   "concensus_spec_30M_diverse_2x", "valid",  4_000,  5, 8),
    ("massive_kb_v1_shared", "massive_kb_v1_shared_2x",       "train", 100_000, 5, 7),
    ("massive_kb_v1_shared", "massive_kb_v1_shared_2x",       "valid",  4_000, 10, 8),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Casanovo data root (parent of lance_v5/ and lance_v5_smoke/)",
    )
    args = ap.parse_args()

    src_root = args.data_root / "lance_v5"
    dst_root = args.data_root / "lance_v5_smoke"

    for src_dataset, dst_dataset, split, n_peptides, cap, seed in _TARGETS:
        print(f"=== {dst_dataset}/{split} (from {src_dataset}) ===", flush=True)
        src = src_root / src_dataset / f"{split}.lance"
        dst = dst_root / dst_dataset / f"{split}.lance"
        if not src.exists():
            print(f"  skip (no source): {src}", flush=True)
            continue
        build_subset(src, dst, n_peptides, cap, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
