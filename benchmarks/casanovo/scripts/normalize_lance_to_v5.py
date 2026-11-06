#!/usr/bin/env python3
"""One-shot: produce v5-tokenized copies of each lance dataset.

Reads from ``casanovo/data/lance/<dataset>/<split>.lance`` and writes to
``casanovo/data/lance_v5/<dataset>/<split>.lance`` with the ``seq`` column
rewritten into casanovo 5.x notation (``C[Carbamidomethyl]``, ``M[Oxidation]``,
``[Acetyl]-`` …). All other columns pass through unchanged.

Two source formats are normalized:
  * concensus_spec_30M  : ``C[57.0215]``  /  ``M[15.9949]``
  * massive_kb_v1_shared: ``C+57.021`` / ``M+15.995`` / ``N+0.984`` /
                          ``Q+0.984`` / leading ``+42.011`` / ``+43.006`` /
                          ``-17.027`` / ``+43.006-17.027``

Originals are not touched. ``mode='create'`` makes target paths refuse to
overwrite — delete the v5 dir manually to re-run.
"""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lance
import pyarrow as pa


_RULES_CONSENSUS = [
    (re.compile(r"C\[57\.0215\]"), "C[Carbamidomethyl]"),
    (re.compile(r"M\[15\.9949\]"), "M[Oxidation]"),
]

# Order matters: the combo N-term prefix must be matched before the single-
# mass prefixes, otherwise "+43.006-17.027..." gets rewritten to
# "[Carbamyl]--17.027...".
_RULES_MASSIVEKB = [
    (re.compile(r"^\+43\.006-17\.027"), "[+25.980265]-"),
    (re.compile(r"^\+42\.011"),         "[Acetyl]-"),
    (re.compile(r"^\+43\.006"),         "[Carbamyl]-"),
    (re.compile(r"^-17\.027"),          "[Ammonia-loss]-"),
    (re.compile(r"C\+57\.021"),         "C[Carbamidomethyl]"),
    (re.compile(r"M\+15\.995"),         "M[Oxidation]"),
    (re.compile(r"N\+0\.984"),          "N[Deamidated]"),
    (re.compile(r"Q\+0\.984"),          "Q[Deamidated]"),
]


def _normalize(seq: str | None, rules) -> str | None:
    if seq is None:
        return None
    for pat, repl in rules:
        seq = pat.sub(repl, seq)
    return seq


def _rewritten_batches(src_ds: lance.LanceDataset, rules, batch_size: int):
    """Yield record batches from src_ds with the seq column rewritten."""
    seq_idx = src_ds.schema.get_field_index("seq")
    seq_field = src_ds.schema.field("seq")
    for batch in src_ds.scanner(batch_size=batch_size).to_batches():
        new_seq = pa.array(
            [_normalize(s, rules) for s in batch.column("seq").to_pylist()],
            type=seq_field.type,
        )
        yield batch.set_column(seq_idx, seq_field, new_seq)


def rewrite(src: Path, dst: Path, rules, batch_size: int = 100_000) -> None:
    if dst.exists():
        print(f"  skip (exists): {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_ds = lance.dataset(str(src))
    n_rows = src_ds.count_rows()
    print(f"  src rows: {n_rows:,}", flush=True)

    reader = pa.RecordBatchReader.from_batches(
        src_ds.schema,
        _rewritten_batches(src_ds, rules, batch_size),
    )
    t0 = time.time()
    lance.write_dataset(reader, str(dst), schema=src_ds.schema, mode="create")
    dt = time.time() - t0

    out_ds = lance.dataset(str(dst))
    out_rows = out_ds.count_rows()
    if out_rows != n_rows:
        raise RuntimeError(
            f"row mismatch after rewrite: src={n_rows:,} dst={out_rows:,}"
        )
    print(f"  wrote {out_rows:,} rows -> {dst} ({dt:.1f}s)", flush=True)


def _rewrite_target(args):
    """ProcessPoolExecutor-friendly wrapper around rewrite()."""
    src, dst, rules, label = args
    print(f"=== {label} ===", flush=True)
    rewrite(src, dst, rules)
    return label


def main() -> int:
    data_root = Path(__file__).resolve().parents[1] / "data"

    # Original two datasets share a single (lance/, lance_v5/) layout; the
    # new_match_15M datasets are self-contained per-split under their own
    # subtree. Each entry is (src_lance, dst_lance, rules, human_label).
    legacy_root = data_root
    targets: list[tuple[Path, Path, list, str]] = []
    for split in ("train", "valid"):
        targets.append((
            legacy_root / "lance" / "concensus_spec_30M" / f"{split}.lance",
            legacy_root / "lance_v5" / "concensus_spec_30M" / f"{split}.lance",
            _RULES_CONSENSUS,
            f"concensus_spec_30M/{split}",
        ))
    for split in ("train", "valid"):
        targets.append((
            legacy_root / "lance" / "massive_kb_v1_shared" / f"{split}.lance",
            legacy_root / "lance_v5" / "massive_kb_v1_shared" / f"{split}.lance",
            _RULES_MASSIVEKB,
            f"massive_kb_v1_shared/{split}",
        ))

    new15_root = data_root / "new_match_15M"
    for variant in ("file", "pepdisjoint"):
        for split in ("train", "valid", "test"):
            targets.append((
                new15_root / f"lance_{variant}" / f"{split}.lance",
                new15_root / f"lance_v5_{variant}" / f"{split}.lance",
                _RULES_CONSENSUS,
                f"new_match_15M/{variant}/{split}",
            ))

    # Only run targets whose source lance actually exists; this lets the
    # script be re-run safely as different lance builds finish.
    runnable = [t for t in targets if t[0].exists()]
    if not runnable:
        print("no source lance datasets found — nothing to do", flush=True)
        return 0

    max_workers = min(len(runnable), 6)
    print(f"running {len(runnable)} rewrite(s) with {max_workers} worker(s)",
          flush=True)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for t in runnable:
            futures[pool.submit(_rewrite_target, t)] = t[3]
        for f in as_completed(futures):
            label = futures[f]
            try:
                f.result()
            except Exception:
                import traceback; traceback.print_exc()
                print(f"FAILED: {label}", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
