"""Classical spectral similarity baselines: Cosine and Modified Cosine.

Parameters match ``iomega-3-classical-spectra-similarities.ipynb``:
  Cosine:                 tolerance=0.005, min_match=6
  Modified Cosine Greedy: tolerance=0.005, min_match=10

``matchms.calculate_scores`` is serial. For the 12 846 × 12 846 UniqueInchikey
matrix it takes ~100 min per similarity function. These helpers parallelize
by chunking the query axis across a multiprocessing Pool.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosineGreedy

logger = logging.getLogger(__name__)


def _score_chunk(
    query_idx_range: tuple[int, int],
    references: list,
    queries: list,
    sim_name: str,
    tolerance: float,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Compute score + match arrays for queries[query_idx_range[0]:query_idx_range[1]] vs references."""
    start, stop = query_idx_range
    if sim_name == "cosine":
        sim = CosineGreedy(tolerance=tolerance)
        score_key, match_key = "CosineGreedy_score", "CosineGreedy_matches"
    elif sim_name == "modified_cosine":
        sim = ModifiedCosineGreedy(tolerance=tolerance)
        score_key, match_key = "ModifiedCosineGreedy_score", "ModifiedCosineGreedy_matches"
    else:
        raise ValueError(sim_name)

    chunk_queries = queries[start:stop]
    scores_obj = calculate_scores(references=references, queries=chunk_queries, similarity_function=sim)
    arr = scores_obj.to_array()
    return start, stop, arr[score_key].astype(np.float32), arr[match_key]


def _parallel_pairwise(
    spectra: list,
    sim_name: str,
    tolerance: float,
    min_match: int,
    n_workers: int | None,
    chunk_size: int,
) -> np.ndarray:
    n = len(spectra)
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 2)
    score_mat = np.zeros((n, n), dtype=np.float32)
    match_mat = np.zeros((n, n), dtype=np.int32)

    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    logger.info(
        "Parallel %s: n=%d spectra, %d chunks × %d queries, %d workers",
        sim_name, n, len(chunks), chunk_size, n_workers,
    )

    worker = partial(
        _score_chunk, references=spectra, queries=spectra,
        sim_name=sim_name, tolerance=tolerance,
    )
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(worker, c) for c in chunks]
        for fut in as_completed(futures):
            start, stop, score_block, match_block = fut.result()
            # calculate_scores returns (n_references, n_queries) where queries is chunk.
            # Transpose so rows correspond to queries (= original spectra range).
            score_mat[start:stop, :] = score_block.T
            match_mat[start:stop, :] = match_block.T
            done += 1
            if done % 5 == 0 or done == len(chunks):
                logger.info("%s: %d / %d chunks done", sim_name, done, len(chunks))

    score_mat[match_mat < min_match] = 0.0
    # Symmetrize (take mean of (i,j) and (j,i); they should be equal for symmetric similarities,
    # but matchms rounds internally so enforce symmetry)
    score_mat = 0.5 * (score_mat + score_mat.T)
    return score_mat


def cosine_matrix(
    spectra: list,
    tolerance: float = 0.005,
    min_match: int = 6,
    n_workers: int | None = None,
    chunk_size: int = 100,
) -> np.ndarray:
    return _parallel_pairwise(spectra, "cosine", tolerance, min_match, n_workers, chunk_size)


def modified_cosine_matrix(
    spectra: list,
    tolerance: float = 0.005,
    min_match: int = 10,
    n_workers: int | None = None,
    chunk_size: int = 100,
) -> np.ndarray:
    return _parallel_pairwise(spectra, "modified_cosine", tolerance, min_match, n_workers, chunk_size)


def _score_asym_chunk(
    query_idx_range: tuple[int, int],
    library: list,
    queries: list,
    sim_name: str,
    tolerance: float,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    start, stop = query_idx_range
    if sim_name == "cosine":
        sim = CosineGreedy(tolerance=tolerance)
        score_key, match_key = "CosineGreedy_score", "CosineGreedy_matches"
    elif sim_name == "modified_cosine":
        sim = ModifiedCosineGreedy(tolerance=tolerance)
        score_key, match_key = "ModifiedCosineGreedy_score", "ModifiedCosineGreedy_matches"
    else:
        raise ValueError(sim_name)
    chunk_q = queries[start:stop]
    scores_obj = calculate_scores(references=library, queries=chunk_q, similarity_function=sim)
    arr = scores_obj.to_array()
    return start, stop, arr[score_key].astype(np.float32), arr[match_key]


def query_library_matrix(
    queries: list,
    library: list,
    sim_name: str,
    tolerance: float,
    min_match: int,
    n_workers: int | None = None,
    chunk_size: int = 20,
) -> np.ndarray:
    """Score (n_queries, n_library) similarity matrix with matchms in parallel."""
    n_q = len(queries)
    n_l = len(library)
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 2)
    score_mat = np.zeros((n_q, n_l), dtype=np.float32)
    match_mat = np.zeros((n_q, n_l), dtype=np.int32)

    chunks = [(i, min(i + chunk_size, n_q)) for i in range(0, n_q, chunk_size)]
    logger.info(
        "Parallel %s query×library: %d queries × %d library, %d chunks × %d, %d workers",
        sim_name, n_q, n_l, len(chunks), chunk_size, n_workers,
    )

    worker = partial(
        _score_asym_chunk, library=library, queries=queries,
        sim_name=sim_name, tolerance=tolerance,
    )
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(worker, c) for c in chunks]
        for fut in as_completed(futures):
            start, stop, score_block, match_block = fut.result()
            # score_block shape: (n_library, chunk_size) — transpose to (chunk_size, n_library)
            score_mat[start:stop, :] = score_block.T
            match_mat[start:stop, :] = match_block.T
            done += 1
            if done % 5 == 0 or done == len(chunks):
                logger.info("%s q×lib: %d / %d chunks done", sim_name, done, len(chunks))

    score_mat[match_mat < min_match] = 0.0
    return score_mat
