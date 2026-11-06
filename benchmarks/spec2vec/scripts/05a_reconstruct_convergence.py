"""Post-hoc convergence reconstruction from saved checkpoints.

Loads every per-epoch checkpoint under ``models/<name>/checkpoints/``, scores
it against a stratified Tanimoto-balanced holdout, and writes
``reports/<name>/convergence_posthoc.csv`` with columns::

    epoch, wall_sec, pearson, top_bin_mean, n_high_tanimoto, n_pairs

``wall_sec`` is reconstructed from checkpoint file mtimes (seconds since the
first checkpoint).
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.convergence import build_holdout_pairs, score_convergence  # noqa: E402


EPOCH_RE = re.compile(r"_epoch(\d+)\.model$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ckpt_dir = REPO_ROOT / "models" / args.name / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"No checkpoints at {ckpt_dir}")

    ckpts = []
    for p in sorted(ckpt_dir.glob(f"{args.name}_epoch*.model")):
        m = EPOCH_RE.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    if not ckpts:
        raise SystemExit(f"No {args.name}_epoch*.model files found")
    logging.info("Found %d checkpoints", len(ckpts))

    processed = REPO_ROOT / "data" / "processed"
    with open(processed / f"{args.name}_spectra.pkl", "rb") as f:
        spectra = pickle.load(f)
    with open(processed / f"{args.name}_documents.pkl", "rb") as f:
        documents = pickle.load(f)

    logging.info("Building stratified holdout (n_pairs=%d, seed=%d)", args.n_pairs, args.seed)
    holdout = build_holdout_pairs(
        spectra, n_pairs=args.n_pairs, seed=args.seed, stratified=True,
    )
    t = holdout["tanimoto"]
    logging.info(
        "Stratified Tanimoto distribution: "
        "[0,.2)=%d [.2,.4)=%d [.4,.6)=%d [.6,.8)=%d [.8,1]=%d",
        int(((t >= 0.0) & (t < 0.2)).sum()),
        int(((t >= 0.2) & (t < 0.4)).sum()),
        int(((t >= 0.4) & (t < 0.6)).sum()),
        int(((t >= 0.6) & (t < 0.8)).sum()),
        int(((t >= 0.8) & (t <= 1.0)).sum()),
    )

    first_mtime = ckpts[0][1].stat().st_mtime
    rows = []
    for epoch, path in ckpts:
        model = Word2Vec.load(str(path))
        metrics = score_convergence(model, documents, holdout)
        wall_sec = path.stat().st_mtime - first_mtime
        rows.append({
            "epoch": epoch,
            "wall_sec": float(wall_sec),
            "pearson": metrics["pearson"],
            "top_bin_mean": metrics["top_bin_mean"],
            "n_high_tanimoto": metrics["n_high_tanimoto"],
            "n_pairs": metrics["n_pairs"],
        })
        logging.info(
            "epoch=%d wall_sec=%.0f pearson=%.3f top_bin_mean=%.3f n_high=%d",
            epoch, wall_sec, metrics["pearson"], metrics["top_bin_mean"],
            metrics["n_high_tanimoto"],
        )

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "reports" / args.name / "convergence_posthoc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("Wrote %s", out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
