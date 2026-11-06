"""All-vs-all Spec2Vec scoring and Tanimoto ground truth (Figure 3).

Mirrors ``iomega-4-fingerprint-based-similarities.ipynb`` (Daylight-2048
fingerprints, Jaccard/Tanimoto) and ``iomega-6-compute-spec2vec-similarities.ipynb``
(weighted-sum document embeddings, L2-normalized, cosine) from
``iomega/spec2vec_gnps_data_analysis``.
"""
from __future__ import annotations

import logging

import numpy as np
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from spec2vec.vector_operations import calc_vector

logger = logging.getLogger(__name__)


def compute_fingerprints(
    spectra: list[Spectrum], nbits: int = 2048
) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for s in spectra:
        s2 = add_fingerprint(s, fingerprint_type="daylight", nbits=nbits)
        fp = None if s2 is None else s2.get("fingerprint")
        out.append(None if fp is None else np.asarray(fp, dtype=np.int32))
    return out


def tanimoto_matrix(fps: list[np.ndarray | None]) -> np.ndarray:
    """N x N Jaccard/Tanimoto matrix; rows/cols of missing fps are NaN."""
    n = len(fps)
    valid = np.array([fp is not None for fp in fps])
    if not valid.any():
        raise ValueError("No valid fingerprints.")
    nbits = int(next(fp for fp in fps if fp is not None).shape[0])

    mat = np.zeros((n, nbits), dtype=np.float32)
    for i, fp in enumerate(fps):
        if fp is not None:
            mat[i] = fp
    counts = mat.sum(axis=1)
    inter = mat @ mat.T  # float32 BLAS GEMM — orders of magnitude faster than int32
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = (inter / union).astype(np.float32)
    jacc[~valid, :] = np.nan
    jacc[:, ~valid] = np.nan
    return jacc


def embed_documents(
    model,
    documents: list,
    intensity_weighting_power: float = 0.5,
    allowed_missing_percentage: float = 5.0,
) -> np.ndarray:
    """Compute L2-normalized Spec2Vec vectors for every document.

    Documents that exceed ``allowed_missing_percentage`` (unseen words) will
    be set to zeros; their downstream scores are thus 0.
    """
    n_dim = model.wv.vector_size
    vecs = np.zeros((len(documents), n_dim), dtype=np.float32)
    n_skipped = 0
    for i, doc in enumerate(documents):
        try:
            v = calc_vector(
                model,
                doc,
                intensity_weighting_power=intensity_weighting_power,
                allowed_missing_percentage=allowed_missing_percentage,
            )
        except AssertionError:
            n_skipped += 1
            continue
        vecs[i] = v
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    if n_skipped:
        logger.info("embed_documents: skipped %d documents over missing-word limit", n_skipped)
    return vecs


def spec2vec_matrix(
    model,
    documents: list,
    intensity_weighting_power: float = 0.5,
    allowed_missing_percentage: float = 5.0,
) -> np.ndarray:
    vecs = embed_documents(
        model,
        documents,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
    )
    return (vecs @ vecs.T).astype(np.float32)


def bin_scores_by_tanimoto(
    tanimotos: np.ndarray,
    scores: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """Mean score per Tanimoto bin on the upper triangle (excluding diagonal)."""
    if bin_edges is None:
        bin_edges = np.arange(0.0, 1.01, 0.1)
    mask = np.triu(np.ones_like(tanimotos, dtype=bool), k=1)
    t = tanimotos[mask]
    s = scores[mask]
    valid = ~np.isnan(t) & ~np.isnan(s)
    t, s = t[valid], s[valid]
    idx = np.digitize(t, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    means = np.full(n_bins, np.nan, dtype=np.float64)
    stds = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        counts[b] = m.sum()
        if counts[b] > 0:
            means[b] = float(s[m].mean())
            stds[b] = float(s[m].std())
    return {
        "bin_edges": bin_edges,
        "bin_means": means,
        "bin_stds": stds,
        "bin_counts": counts,
    }
