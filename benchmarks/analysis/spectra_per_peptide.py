#!/usr/bin/env python3
"""Distribution of spectra-per-peptide for the casanovo lance datasets.

Scans each lance dataset's `seq` column, counts how many spectra each
unique peptide has, and reports:

  - Summary stats: mean, median, percentiles (p1/25/50/75/90/95/99/99.9), min, max
  - Log-spaced histogram (text-rendered + tsv for external plotting)

Default targets are the rewritten v5 lance dirs (train + valid combined):

    casanovo/data/lance_v5/concensus_spec_30M/{train,valid}.lance
    casanovo/data/lance_v5/massive_kb_v1_shared/{train,valid}.lance

Usage:
    python analysis/spectra_per_peptide.py
    python analysis/spectra_per_peptide.py --root /alt/path/to/data/lance_v5
    python analysis/spectra_per_peptide.py --tsv-out /tmp/distribution.tsv
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import lance


# Log-spaced bins covering 1 -> ~10k+ spectra/peptide.
_BINS: list[tuple[int, int, str]] = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 5, "4-5"),
    (6, 10, "6-10"),
    (11, 25, "11-25"),
    (26, 50, "26-50"),
    (51, 100, "51-100"),
    (101, 250, "101-250"),
    (251, 500, "251-500"),
    (501, 1_000, "501-1000"),
    (1_001, 5_000, "1001-5000"),
    (5_001, 10**9, "5001+"),
]


def scan_dataset(paths: list[Path]) -> Counter[str]:
    """Return Counter[seq_str] -> spectrum count, aggregated across paths."""
    counts: Counter[str] = Counter()
    for p in paths:
        ds = lance.dataset(str(p))
        for batch in ds.scanner(columns=["seq"], batch_size=1_000_000).to_batches():
            for s in batch.column("seq").to_pylist():
                if s is not None:
                    counts[s] += 1
    return counts


def percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bucketize(values: list[int]) -> list[tuple[str, int, int]]:
    """Return list of (label, count_of_peptides_in_bin, total_spectra_in_bin)."""
    out: list[tuple[str, int, int]] = []
    for lo, hi, label in _BINS:
        bin_vals = [v for v in values if lo <= v <= hi]
        out.append((label, len(bin_vals), sum(bin_vals)))
    return out


def render_histogram(buckets: list[tuple[str, int, int]],
                     total_peptides: int,
                     bar_width: int = 60) -> str:
    """Text-rendered histogram by peptide count per bin."""
    if total_peptides == 0:
        return "(no peptides)"
    max_count = max(b[1] for b in buckets) or 1
    lines = []
    lines.append(f"  {'bin':>10}  {'peptides':>10}  {'%':>6}  {'spectra':>12}  bar")
    lines.append(f"  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*bar_width}")
    for label, n_pep, n_spec in buckets:
        pct = 100.0 * n_pep / total_peptides
        bar = "#" * int(round(bar_width * n_pep / max_count))
        lines.append(f"  {label:>10}  {n_pep:>10,}  {pct:>5.1f}%  {n_spec:>12,}  {bar}")
    return "\n".join(lines)


def summarize(name: str, counts: Counter[str], tsv_rows: list[dict]) -> None:
    values = sorted(counts.values())
    total_pep = len(values)
    total_spec = sum(values)
    print(f"\n=== {name} ===")
    print(f"  total spectra:  {total_spec:>14,}")
    print(f"  unique peptides:{total_pep:>14,}")
    print(f"  spectra/peptide:")
    print(f"    mean   {statistics.mean(values):>10.2f}")
    print(f"    min    {values[0]:>10}")
    for q in (1, 25, 50, 75, 90, 95, 99, 99.9):
        print(f"    p{q:<5} {percentile(values, q):>10.0f}")
    print(f"    max    {values[-1]:>10}")
    print()
    buckets = bucketize(values)
    print(render_histogram(buckets, total_pep))

    for label, n_pep, n_spec in buckets:
        tsv_rows.append({
            "dataset": name,
            "bin": label,
            "n_peptides": n_pep,
            "pct_peptides": round(100.0 * n_pep / total_pep, 4) if total_pep else 0.0,
            "n_spectra": n_spec,
            "pct_spectra": round(100.0 * n_spec / total_spec, 4) if total_spec else 0.0,
        })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path,
        default=Path("casanovo/data/lance_v5"),
        help="Parent dir containing <dataset>/{train,valid}.lance subdirs",
    )
    ap.add_argument(
        "--datasets", nargs="*",
        default=["concensus_spec_30M", "massive_kb_v1_shared"],
        help="Subdirectory names under --root to scan",
    )
    ap.add_argument(
        "--splits", nargs="*", default=["train", "valid"],
        help="Splits to combine per dataset",
    )
    ap.add_argument(
        "--tsv-out", type=Path,
        default=Path("analysis/spectra_per_peptide.tsv"),
        help="Where to write the bin counts as TSV (for external plotting)",
    )
    args = ap.parse_args()

    tsv_rows: list[dict] = []
    for ds in args.datasets:
        paths = [args.root / ds / f"{s}.lance" for s in args.splits]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"skip {ds}: missing {missing}", file=sys.stderr)
            continue
        print(f"scanning {ds} ({len(paths)} files)...", file=sys.stderr, flush=True)
        counts = scan_dataset(paths)
        summarize(ds, counts, tsv_rows)

    args.tsv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.tsv_out, "w") as fh:
        cols = ["dataset", "bin", "n_peptides", "pct_peptides", "n_spectra", "pct_spectra"]
        fh.write("\t".join(cols) + "\n")
        for row in tsv_rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"\nwrote bin counts to {args.tsv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
