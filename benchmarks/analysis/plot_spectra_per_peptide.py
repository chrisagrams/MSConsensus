#!/usr/bin/env python3
"""Plot spectra-per-peptide distribution for the casanovo lance datasets.

Re-scans the lance files (so we have full per-peptide counts, not just bins),
then renders two panels:
  1. Log-y histogram of spectra/peptide (overlaid both datasets)
  2. Complementary CDF on log-log axes (the shape comparison — flat vs
     heavy tail)

Default targets and CLI flags mirror spectra_per_peptide.py.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import lance
import numpy as np
import matplotlib.pyplot as plt


def scan_dataset(paths: list[Path]) -> np.ndarray:
    counts: Counter[str] = Counter()
    for p in paths:
        ds = lance.dataset(str(p))
        for batch in ds.scanner(columns=["seq"], batch_size=1_000_000).to_batches():
            for s in batch.column("seq").to_pylist():
                if s is not None:
                    counts[s] += 1
    return np.array(sorted(counts.values()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path,
        default=Path("casanovo/data/lance_v5"),
    )
    ap.add_argument(
        "--datasets", nargs="*",
        default=["concensus_spec_30M", "massive_kb_v1_shared"],
    )
    ap.add_argument("--splits", nargs="*", default=["train", "valid"])
    ap.add_argument(
        "--out", type=Path,
        default=Path("analysis/spectra_per_peptide.png"),
    )
    args = ap.parse_args()

    series: dict[str, np.ndarray] = {}
    for ds in args.datasets:
        paths = [args.root / ds / f"{s}.lance" for s in args.splits]
        if not all(p.exists() for p in paths):
            print(f"skip {ds}: missing {[p for p in paths if not p.exists()]}",
                  file=sys.stderr)
            continue
        print(f"scanning {ds}...", file=sys.stderr, flush=True)
        series[ds] = scan_dataset(paths)
        print(f"  {ds}: {len(series[ds]):,} peptides, {series[ds].sum():,} spectra",
              file=sys.stderr, flush=True)

    fig, (ax_hist, ax_ccdf) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"concensus_spec_30M": "#1f77b4", "massive_kb_v1_shared": "#d62728"}
    log_bins = np.logspace(0, np.log10(max(v.max() for v in series.values()) * 1.05),
                           50)

    for name, vals in series.items():
        c = colors.get(name, None)
        n_pep = len(vals)
        n_spec = vals.sum()
        mean = vals.mean()
        median = np.median(vals)
        # --- histogram (log-x bins, log-y) ---
        ax_hist.hist(
            vals, bins=log_bins, alpha=0.55, color=c, edgecolor=c,
            label=f"{name}\n  n_peptides={n_pep:,}, n_spectra={n_spec:,}\n"
                  f"  mean={mean:.1f}, median={median:.0f}",
        )
        # --- complementary CDF: P(X >= x) on log-log ---
        sorted_vals = vals  # already sorted ascending
        ccdf_y = 1.0 - np.arange(len(sorted_vals)) / len(sorted_vals)
        ax_ccdf.plot(sorted_vals, ccdf_y, color=c, linewidth=2, label=name)

    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("spectra per peptide")
    ax_hist.set_ylabel("number of peptides (log)")
    ax_hist.set_title("Histogram (log-x, log-y)")
    ax_hist.grid(True, which="both", alpha=0.3)
    ax_hist.legend(loc="upper right", fontsize=8)

    ax_ccdf.set_xscale("log")
    ax_ccdf.set_yscale("log")
    ax_ccdf.set_xlabel("spectra per peptide  (x)")
    ax_ccdf.set_ylabel("P(spectra/peptide >= x)")
    ax_ccdf.set_title("Survival function — heavy-tail shape comparison")
    ax_ccdf.grid(True, which="both", alpha=0.3)
    ax_ccdf.legend(loc="lower left")

    fig.suptitle("Spectra-per-peptide distribution: full lance_v5 corpora "
                 "(train + valid combined)", fontsize=12)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
