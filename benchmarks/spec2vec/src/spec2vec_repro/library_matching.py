"""Figure 4 reproduction: library matching, scenario 1.

Per ``iomega-8-library-matching-scenario1.ipynb``:
  - Restrict to inchikey14s that occur >=2 times in the filtered dataset.
  - With ``np.random.seed(42)`` draw 1000 inchikey14s without replacement.
  - For each, pick one spectrum (uniformly at random) as the *query*.
  - The remaining spectra form the *library* (each query's inchikey14 still
    has >=1 spectrum in the library by construction).
  - Score each query against the library with a precursor-mz prefilter
    (|Δ m/z| <= ``precursor_mz_tol``, default 1.0 Da).
  - At each score threshold, report top-1 accuracy, retrieval, recall.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from matchms import Spectrum

from spec2vec_repro.split import get_inchikey14

logger = logging.getLogger(__name__)


def build_scenario1_split(
    spectra: list[Spectrum],
    n_queries: int = 1000,
    min_occurrences: int = 2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return (query_indices, library_indices) into ``spectra``."""
    rng = np.random.default_rng(seed)

    by_ik14: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(spectra):
        ik = get_inchikey14(s)
        if ik is not None:
            by_ik14[ik].append(idx)

    eligible = sorted(ik for ik, ids in by_ik14.items() if len(ids) >= min_occurrences)
    if len(eligible) < n_queries:
        raise ValueError(
            f"Only {len(eligible)} eligible inchikey14s (need {n_queries})."
        )

    chosen = rng.choice(np.array(eligible), size=n_queries, replace=False)
    query_indices: list[int] = []
    for ik in chosen:
        query_indices.append(int(rng.choice(by_ik14[str(ik)])))

    query_set = set(query_indices)
    library_indices = [i for i in range(len(spectra)) if i not in query_set]

    logger.info(
        "Scenario 1: %d queries drawn from %d eligible inchikey14s (library=%d)",
        n_queries, len(eligible), len(library_indices),
    )
    return query_indices, library_indices


def _precursor_mz_array(spectra: list[Spectrum]) -> np.ndarray:
    """Return precursor m/z as float32; NaN for spectra missing it."""
    out = np.full(len(spectra), np.nan, dtype=np.float32)
    for i, s in enumerate(spectra):
        v = s.get("precursor_mz")
        if v is not None:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _ik14_array(spectra: list[Spectrum]) -> np.ndarray:
    return np.array([get_inchikey14(s) or "" for s in spectra])


def threshold_sweep(
    query_scores: np.ndarray,
    query_spectra: list[Spectrum],
    library_spectra: list[Spectrum],
    precursor_mz_tol: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Per-threshold top-1 retrieval metrics.

    Parameters
    ----------
    query_scores : ndarray of shape (n_queries, n_library)
        Similarity scores (higher is better).
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 0.96, 0.05)

    q_pmz = _precursor_mz_array(query_spectra)
    l_pmz = _precursor_mz_array(library_spectra)
    q_ik = _ik14_array(query_spectra)
    l_ik = _ik14_array(library_spectra)

    diff = np.abs(q_pmz[:, None] - l_pmz[None, :])
    filtered = np.where(diff <= precursor_mz_tol, query_scores, -np.inf)

    top_idx = filtered.argmax(axis=1)
    n_q = len(query_spectra)
    top_scores = filtered[np.arange(n_q), top_idx]
    top_ik = l_ik[top_idx]

    has_candidate = np.isfinite(top_scores)

    rows: list[dict] = []
    for thr in thresholds:
        predicted = has_candidate & (top_scores >= thr)
        correct = predicted & (top_ik == q_ik)
        n_pred = int(predicted.sum())
        n_correct = int(correct.sum())
        rows.append({
            "threshold": float(thr),
            "n_predicted": n_pred,
            "n_correct": n_correct,
            "accuracy": (n_correct / n_pred) if n_pred else 0.0,
            "retrieval": n_pred / n_q,
            "recall": n_correct / n_q,
        })
    return rows
