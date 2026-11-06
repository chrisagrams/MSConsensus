"""Compute the UniqueInchikey Cosine + Modified Cosine baseline matrices.

Reuses the existing Tanimoto matrix from ``reports/uniqueinchikey/matrices.npz``
and appends baseline bin means to ``fig3_bin_means.csv``.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.baselines import cosine_matrix, modified_cosine_matrix  # noqa: E402
from spec2vec_repro.eval import bin_scores_by_tanimoto  # noqa: E402


def _bin_rows(method_name: str, tanimoto: np.ndarray, scores: np.ndarray) -> list[dict]:
    b = bin_scores_by_tanimoto(tanimoto, scores)
    rows = []
    for i, edge in enumerate(b["bin_edges"][:-1]):
        rows.append({
            "method": method_name,
            "tanimoto_bin_low": float(edge),
            "tanimoto_bin_high": float(b["bin_edges"][i + 1]),
            "mean": float(b["bin_means"][i]),
            "std": float(b["bin_stds"][i]),
            "count": int(b["bin_counts"][i]),
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reports = REPO_ROOT / "reports" / "uniqueinchikey"
    with open(REPO_ROOT / "data" / "processed" / "uniqueinchikey_spectra.pkl", "rb") as f:
        spectra = pickle.load(f)

    mats = np.load(reports / "matrices.npz")
    T = mats["tanimoto"]
    logging.info("Loaded Tanimoto matrix %s", T.shape)

    logging.info("Cosine (tol=0.005, min_match=6), 48 workers, chunk_size=100")
    C = cosine_matrix(spectra, n_workers=48, chunk_size=100)

    logging.info("ModifiedCosineGreedy (tol=0.005, min_match=10), 48 workers, chunk_size=100")
    MC = modified_cosine_matrix(spectra, n_workers=48, chunk_size=100)

    np.savez(
        reports / "baseline_matrices.npz",
        cosine=C.astype(np.float32),
        modified_cosine=MC.astype(np.float32),
    )
    logging.info("Wrote %s", reports / "baseline_matrices.npz")

    rows = _bin_rows("cosine", T, C) + _bin_rows("modified_cosine", T, MC)
    existing = pd.read_csv(reports / "fig3_bin_means.csv")
    # Drop any stale baseline rows so this script is idempotent.
    existing = existing[existing["method"] == "spec2vec"]
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(reports / "fig3_bin_means.csv", index=False)
    logging.info("Wrote %s", reports / "fig3_bin_means.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
