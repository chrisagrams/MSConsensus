"""Reproduce Figure 4: library-matching scenario 1 (AllPositive).

For each of three scoring methods (Spec2Vec, Cosine, Modified Cosine),
builds the 1000-query / ~94k-library split, scores, applies precursor-mz
prefilter (±1 Da), and sweeps score thresholds.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosineGreedy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.eval import embed_documents  # noqa: E402
from spec2vec_repro.library_matching import (  # noqa: E402
    build_scenario1_split,
    threshold_sweep,
)


def _matchms_query_scores(
    queries: list,
    library: list,
    sim,
    score_key: str,
    matches_key: str,
    min_match: int,
) -> np.ndarray:
    scores_obj = calculate_scores(
        references=library, queries=queries, similarity_function=sim
    )
    arr = scores_obj.to_array()
    score = arr[score_key].astype(np.float32)
    matches = arr[matches_key]
    score[matches < min_match] = 0.0
    return score  # shape (n_queries, n_library)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["spec2vec", "cosine", "modified_cosine"],
        choices=["spec2vec", "cosine", "modified_cosine"],
    )
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["name"]

    processed = REPO_ROOT / "data" / "processed"
    with open(processed / f"{name}_spectra.pkl", "rb") as f:
        spectra_raw = pickle.load(f)
    with open(processed / f"{name}_documents.pkl", "rb") as f:
        documents_raw = pickle.load(f)

    # Drop spectra missing a usable precursor_mz — Modified Cosine asserts it.
    keep = [
        i for i, s in enumerate(spectra_raw)
        if s.get("precursor_mz") is not None and float(s.get("precursor_mz", 0)) > 0
    ]
    spectra = [spectra_raw[i] for i in keep]
    documents = [documents_raw[i] for i in keep]
    logging.info("Kept %d / %d spectra with precursor_mz > 0", len(spectra), len(spectra_raw))

    q_idx, l_idx = build_scenario1_split(spectra, seed=cfg.get("seed", 42))
    q_spec = [spectra[i] for i in q_idx]
    l_spec = [spectra[i] for i in l_idx]
    q_doc = [documents[i] for i in q_idx]
    l_doc = [documents[i] for i in l_idx]

    results_dir = REPO_ROOT / "reports" / name
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if "spec2vec" in args.methods:
        model_path = args.model or REPO_ROOT / "models" / name / f"{name}_final.model"
        logging.info("Loading model %s", model_path)
        model = Word2Vec.load(str(model_path))
        logging.info("Embedding %d queries + %d library docs", len(q_doc), len(l_doc))
        q_vecs = embed_documents(model, q_doc)
        l_vecs = embed_documents(model, l_doc)
        scores = (q_vecs @ l_vecs.T).astype(np.float32)
        rows = threshold_sweep(scores, q_spec, l_spec)
        for r in rows:
            r["method"] = "spec2vec"
        all_rows.extend(rows)

    if "cosine" in args.methods:
        logging.info("Computing Cosine for %d queries × %d library", len(q_spec), len(l_spec))
        scores = _matchms_query_scores(
            q_spec, l_spec, CosineGreedy(tolerance=0.005),
            "CosineGreedy_score", "CosineGreedy_matches", min_match=6,
        )
        rows = threshold_sweep(scores, q_spec, l_spec)
        for r in rows:
            r["method"] = "cosine"
        all_rows.extend(rows)

    if "modified_cosine" in args.methods:
        logging.info("Computing ModifiedCosine for %d queries × %d library", len(q_spec), len(l_spec))
        scores = _matchms_query_scores(
            q_spec, l_spec, ModifiedCosineGreedy(tolerance=0.005),
            "ModifiedCosineGreedy_score", "ModifiedCosineGreedy_matches", min_match=10,
        )
        rows = threshold_sweep(scores, q_spec, l_spec)
        for r in rows:
            r["method"] = "modified_cosine"
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "fig4_threshold_sweep.csv", index=False)
    logging.info("Wrote %s", results_dir / "fig4_threshold_sweep.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
