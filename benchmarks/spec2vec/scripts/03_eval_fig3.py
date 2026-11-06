"""Reproduce Figure 3: mean similarity score vs Tanimoto bin (UniqueInchikey).

Runs all-vs-all Spec2Vec, Cosine, and Modified Cosine on the
uniqueinchikey subset and saves a parquet of bin means + counts.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import Word2Vec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.baselines import cosine_matrix, modified_cosine_matrix  # noqa: E402
from spec2vec_repro.eval import (  # noqa: E402
    bin_scores_by_tanimoto,
    compute_fingerprints,
    spec2vec_matrix,
    tanimoto_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip Cosine/ModifiedCosine (slow) — Spec2Vec + Tanimoto only.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override model path; defaults to models/<name>/<name>_final.model",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["name"]

    processed = REPO_ROOT / "data" / "processed"
    with open(processed / f"{name}_spectra.pkl", "rb") as f:
        spectra = pickle.load(f)
    with open(processed / f"{name}_documents.pkl", "rb") as f:
        documents = pickle.load(f)

    model_path = args.model or REPO_ROOT / "models" / name / f"{name}_final.model"
    logging.info("Loading model %s", model_path)
    model = Word2Vec.load(str(model_path))

    logging.info("Computing daylight-%d fingerprints", cfg["eval"]["fingerprint_nbits"])
    fps = compute_fingerprints(spectra, nbits=cfg["eval"]["fingerprint_nbits"])
    n_valid_fp = sum(fp is not None for fp in fps)
    logging.info("Fingerprints: %d / %d valid", n_valid_fp, len(fps))

    logging.info("Computing Tanimoto matrix")
    T = tanimoto_matrix(fps)

    logging.info("Computing Spec2Vec matrix")
    S = spec2vec_matrix(model, documents)

    results_dir = REPO_ROOT / "reports" / name
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "matrices.npz",
        tanimoto=T.astype(np.float32),
        spec2vec=S.astype(np.float32),
    )

    rows = []
    spec_bins = bin_scores_by_tanimoto(T, S)
    for i, edge in enumerate(spec_bins["bin_edges"][:-1]):
        rows.append({
            "method": "spec2vec",
            "tanimoto_bin_low": float(edge),
            "tanimoto_bin_high": float(spec_bins["bin_edges"][i + 1]),
            "mean": float(spec_bins["bin_means"][i]),
            "std": float(spec_bins["bin_stds"][i]),
            "count": int(spec_bins["bin_counts"][i]),
        })

    if not args.skip_baselines:
        logging.info("Computing Cosine (tol=0.005, min_match=6) ...")
        C = cosine_matrix(spectra)
        cos_bins = bin_scores_by_tanimoto(T, C)
        for i, edge in enumerate(cos_bins["bin_edges"][:-1]):
            rows.append({
                "method": "cosine",
                "tanimoto_bin_low": float(edge),
                "tanimoto_bin_high": float(cos_bins["bin_edges"][i + 1]),
                "mean": float(cos_bins["bin_means"][i]),
                "std": float(cos_bins["bin_stds"][i]),
                "count": int(cos_bins["bin_counts"][i]),
            })

        logging.info("Computing Modified Cosine (tol=0.005, min_match=10) ...")
        MC = modified_cosine_matrix(spectra)
        mc_bins = bin_scores_by_tanimoto(T, MC)
        for i, edge in enumerate(mc_bins["bin_edges"][:-1]):
            rows.append({
                "method": "modified_cosine",
                "tanimoto_bin_low": float(edge),
                "tanimoto_bin_high": float(mc_bins["bin_edges"][i + 1]),
                "mean": float(mc_bins["bin_means"][i]),
                "std": float(mc_bins["bin_stds"][i]),
                "count": int(mc_bins["bin_counts"][i]),
            })
        np.savez(
            results_dir / "baseline_matrices.npz",
            cosine=C.astype(np.float32),
            modified_cosine=MC.astype(np.float32),
        )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "fig3_bin_means.csv", index=False)
    logging.info("Wrote %s", results_dir / "fig3_bin_means.csv")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
