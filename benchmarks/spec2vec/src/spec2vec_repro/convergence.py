"""Periodic downstream-task eval for tracking training convergence.

Build a held-out sample of (doc_i, doc_j) pairs with precomputed Tanimoto
similarities (Daylight-2048 fingerprints). At each eval step, compute the
corresponding Spec2Vec cosine scores using the in-progress model and report
Pearson(spec2vec, tanimoto) plus mean spec2vec score in the high-Tanimoto bin.
"""
from __future__ import annotations

import logging

import numpy as np
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from matchms.similarity import FingerprintSimilarity
from scipy.stats import pearsonr
from spec2vec.vector_operations import calc_vector

logger = logging.getLogger(__name__)


def build_holdout_pairs(
    spectra: list[Spectrum],
    n_pairs: int = 5000,
    fingerprint_nbits: int = 2048,
    high_tanimoto_threshold: float = 0.7,
    seed: int = 42,
    stratified: bool = True,
    bin_edges: np.ndarray | None = None,
    pool_multiplier: int = 40,
) -> dict:
    """Sample doc-pairs with precomputed Daylight-Tanimoto ground truth.

    With ``stratified=True`` (default), oversamples a pool of candidate pairs
    and picks ~evenly across Tanimoto bins so high-Tanimoto pairs are not
    drowned out by the low-Tanimoto mass. Only spectra with a valid
    SMILES-derived fingerprint are eligible.
    """
    rng = np.random.default_rng(seed)

    fp_spectra: list[Spectrum] = []
    fp_doc_idx: list[int] = []
    for i, s in enumerate(spectra):
        s2 = add_fingerprint(s, fingerprint_type="daylight", nbits=fingerprint_nbits)
        if s2 is None:
            continue
        fp = s2.get("fingerprint")
        if fp is None:
            continue
        fp_spectra.append(s2)
        fp_doc_idx.append(i)

    logger.info(
        "Fingerprinted %d / %d spectra (daylight-%d)",
        len(fp_spectra), len(spectra), fingerprint_nbits,
    )

    n_fp = len(fp_spectra)
    if n_fp < 2:
        raise ValueError("Not enough fingerprinted spectra for convergence eval.")
    sim = FingerprintSimilarity(similarity_measure="jaccard")

    if not stratified:
        pairs = rng.integers(0, n_fp, size=(n_pairs * 2, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]][:n_pairs]
        tanimotos = np.array(
            [sim.pair(fp_spectra[i], fp_spectra[j]) for i, j in pairs],
            dtype=np.float64,
        )
    else:
        if bin_edges is None:
            bin_edges = np.arange(0.0, 1.01, 0.1)
        n_bins = len(bin_edges) - 1
        target_per_bin = max(1, n_pairs // n_bins)
        pool_size = max(n_pairs * pool_multiplier, 200_000)
        pool = rng.integers(0, n_fp, size=(pool_size * 2, 2))
        pool = pool[pool[:, 0] != pool[:, 1]][:pool_size]
        pool_tanimoto = np.array(
            [sim.pair(fp_spectra[i], fp_spectra[j]) for i, j in pool],
            dtype=np.float64,
        )
        keep_pairs: list[np.ndarray] = []
        keep_tanimoto: list[float] = []
        for b in range(n_bins):
            lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
            mask = (pool_tanimoto >= lo) & (pool_tanimoto < hi)
            idx = np.where(mask)[0]
            take = min(target_per_bin, len(idx))
            if take == 0:
                continue
            chosen = rng.choice(idx, size=take, replace=False)
            for k in chosen:
                keep_pairs.append(pool[k])
                keep_tanimoto.append(float(pool_tanimoto[k]))
        pairs = np.array(keep_pairs, dtype=np.int64)
        tanimotos = np.array(keep_tanimoto, dtype=np.float64)
        logger.info(
            "Stratified holdout: %d pairs across %d bins (pool=%d)",
            len(pairs), n_bins, pool_size,
        )

    doc_pairs = np.array(
        [[fp_doc_idx[i], fp_doc_idx[j]] for i, j in pairs], dtype=np.int64
    )

    return {
        "doc_pairs": doc_pairs,
        "tanimoto": tanimotos,
        "high_tanimoto_threshold": high_tanimoto_threshold,
    }


def score_convergence(
    model,
    documents: list,
    holdout: dict,
    intensity_weighting_power: float = 0.5,
    allowed_missing_percentage: float = 100.0,
) -> dict:
    doc_pairs = holdout["doc_pairs"]
    tanimotos = holdout["tanimoto"]
    threshold = holdout["high_tanimoto_threshold"]

    unique_idx = np.unique(doc_pairs)
    embeddings: dict[int, np.ndarray] = {}
    for idx in unique_idx:
        embeddings[int(idx)] = calc_vector(
            model,
            documents[int(idx)],
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
        )

    scores = np.empty(len(doc_pairs), dtype=np.float64)
    for k, (i, j) in enumerate(doc_pairs):
        v1 = embeddings[int(i)]
        v2 = embeddings[int(j)]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        scores[k] = float(v1 @ v2 / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

    valid = ~np.isnan(scores) & ~np.isnan(tanimotos)
    s = scores[valid]
    t = tanimotos[valid]

    pearson = float(pearsonr(s, t).statistic) if len(s) > 1 else float("nan")
    high = t >= threshold
    top_mean = float(s[high].mean()) if high.any() else float("nan")

    return {
        "pearson": pearson,
        "top_bin_mean": top_mean,
        "n_pairs": int(valid.sum()),
        "n_high_tanimoto": int(high.sum()),
    }
