"""Produce all figures for a trained Spec2Vec run.

Reads:
  reports/<name>/fig3_bin_means.csv          (from 03_eval_fig3.py)
  reports/<name>/fig4_threshold_sweep.csv    (from 04_eval_fig4.py)
  models/<name>/history.jsonl                (if the run used the updated
                                              callback) OR wandb API fallback.

Writes:
  reports/<name>/fig3.png
  reports/<name>/fig4.png
  reports/<name>/convergence.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_history(name: str, wandb_run: str | None) -> pd.DataFrame | None:
    """Convergence data lookup, in order of fidelity:
      1. reports/<name>/convergence_posthoc.csv (stratified, from 05a_...py)
      2. models/<name>/history.jsonl            (written live by the callback)
      3. W&B API                                (requires --wandb-run)
    """
    posthoc = REPO_ROOT / "reports" / name / "convergence_posthoc.csv"
    if posthoc.exists():
        df = pd.read_csv(posthoc)
        # Normalize column names to match the history.jsonl schema used below.
        df = df.rename(columns={"pearson": "eval/pearson", "top_bin_mean": "eval/top_bin_mean"})
        return df

    hist_path = REPO_ROOT / "models" / name / "history.jsonl"
    if hist_path.exists():
        rows = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
        return pd.DataFrame(rows)

    if wandb_run is None:
        logging.warning("No convergence source found; convergence plot skipped.")
        return None

    import wandb
    api = wandb.Api()
    run = api.run(wandb_run)
    return run.history(pandas=True)


def plot_fig3(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    centers = (df["tanimoto_bin_low"] + df["tanimoto_bin_high"]) / 2
    for method, color in [
        ("spec2vec", "C0"),
        ("cosine", "C1"),
        ("modified_cosine", "C2"),
    ]:
        sub = df[df["method"] == method].sort_values("tanimoto_bin_low")
        if sub.empty:
            continue
        x = (sub["tanimoto_bin_low"] + sub["tanimoto_bin_high"]) / 2
        ax.plot(x, sub["mean"], marker="o", label=method, color=color)
        ax.fill_between(
            x,
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Tanimoto similarity (Daylight-2048)")
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Figure 3 — mean score per Tanimoto bin (UniqueInchikey)")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig4(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, color in [
        ("spec2vec", "C0"),
        ("cosine", "C1"),
        ("modified_cosine", "C2"),
    ]:
        sub = df[df["method"] == method].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["retrieval"], sub["accuracy"], marker="o", label=method, color=color)
    ax.set_xlabel("Retrieval (fraction of queries predicted)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Figure 4 — library matching, scenario 1 (AllPositive)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(hist: pd.DataFrame, out_path: Path) -> None:
    """Plot convergence metrics — rows per metric (loss/pearson), cols epoch vs wallclock."""
    has_loss = "loss_delta" in hist.columns and hist["loss_delta"].notna().any()
    has_pearson = "eval/pearson" in hist.columns and hist["eval/pearson"].notna().any()
    has_top = "eval/top_bin_mean" in hist.columns and hist["eval/top_bin_mean"].notna().any()
    has_wall = "wall_sec" in hist.columns and hist["wall_sec"].notna().any()

    rows = int(has_loss) + int(has_pearson)
    if rows == 0:
        logging.warning("No convergence data to plot.")
        return
    cols = 2 if has_wall else 1

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    r = 0

    if has_loss:
        axes[r, 0].plot(hist["epoch"], hist["loss_delta"], marker="o", color="C3")
        axes[r, 0].set_xlabel("Epoch")
        axes[r, 0].set_ylabel("Training loss (gensim delta)")
        axes[r, 0].set_title("Training loss vs epoch")
        axes[r, 0].grid(alpha=0.3)
        if has_wall:
            axes[r, 1].plot(hist["wall_sec"] / 60, hist["loss_delta"], marker="o", color="C3")
            axes[r, 1].set_xlabel("Wallclock (min)")
            axes[r, 1].set_ylabel("Training loss (gensim delta)")
            axes[r, 1].set_title("Training loss vs wallclock")
            axes[r, 1].grid(alpha=0.3)
        r += 1

    if has_pearson:
        eh = hist.dropna(subset=["eval/pearson"])
        axes[r, 0].plot(eh["epoch"], eh["eval/pearson"], marker="o", color="C0", label="Pearson")
        if has_top:
            axes[r, 0].plot(
                eh["epoch"], eh["eval/top_bin_mean"], marker="s", color="C2",
                label="Mean score, Tanimoto>=0.7",
            )
        axes[r, 0].set_xlabel("Epoch")
        axes[r, 0].set_ylabel("Convergence metric")
        axes[r, 0].set_title("Downstream convergence vs epoch")
        axes[r, 0].legend()
        axes[r, 0].grid(alpha=0.3)

        if has_wall:
            axes[r, 1].plot(eh["wall_sec"] / 60, eh["eval/pearson"], marker="o", color="C0", label="Pearson")
            if has_top:
                axes[r, 1].plot(
                    eh["wall_sec"] / 60, eh["eval/top_bin_mean"], marker="s", color="C2",
                    label="Mean score, Tanimoto>=0.7",
                )
            axes[r, 1].set_xlabel("Wallclock (min)")
            axes[r, 1].set_ylabel("Convergence metric")
            axes[r, 1].set_title("Downstream convergence vs wallclock")
            axes[r, 1].legend()
            axes[r, 1].grid(alpha=0.3)

    fig.suptitle("Convergence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--wandb-run",
        default=None,
        help="Fallback W&B run path (<entity>/<project>/<run_id>) if history.jsonl is missing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reports = REPO_ROOT / "reports" / args.name
    reports.mkdir(parents=True, exist_ok=True)

    fig3_csv = reports / "fig3_bin_means.csv"
    if fig3_csv.exists():
        plot_fig3(pd.read_csv(fig3_csv), reports / "fig3.png")
        logging.info("Wrote %s", reports / "fig3.png")

    fig4_csv = reports / "fig4_threshold_sweep.csv"
    if fig4_csv.exists():
        plot_fig4(pd.read_csv(fig4_csv), reports / "fig4.png")
        logging.info("Wrote %s", reports / "fig4.png")

    hist = load_history(args.name, args.wandb_run)
    if hist is not None and len(hist) > 0:
        plot_convergence(hist, reports / "convergence.png")
        logging.info("Wrote %s", reports / "convergence.png")


if __name__ == "__main__":
    main()
