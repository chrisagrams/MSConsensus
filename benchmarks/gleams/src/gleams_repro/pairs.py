from __future__ import annotations

import logging
from pathlib import Path

from gleams.metadata import metadata as _gm

logger = logging.getLogger(__name__)


def build_pairs(
    out_dir: str | Path,
    split: str,
    charges: tuple[int, ...] = (2, 3, 4, 5),
    pair_mz_tolerance_ppm: float = 10.0,
    negative_pair_fragment_tolerance: float = 0.01,
    negative_pair_matching_fragments_threshold: float = 0.25,
) -> None:
    """Generate positive and negative pair index files for a split.

    Writes ``{split}_metadata.parquet_pairs_pos_{charge}.npy`` and
    ``..._pairs_neg_{charge}.npy`` beside the metadata parquet — the file-name
    layout upstream ``PairSequence`` expects. Each file is an ``(N, 2)`` uint32
    array of row indices into the feature matrix.
    """
    meta_path = Path(out_dir) / f"{split}_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    # Upstream iterates np.arange(charges[0], charges[1] + 1) — pass as (min, max).
    charge_bounds = (min(charges), max(charges))

    logger.info("generating positive pairs for %s (charges=%s)", meta_path, charge_bounds)
    _gm.generate_pairs_positive(str(meta_path), charges=charge_bounds)

    logger.info("generating negative pairs for %s (charges=%s)", meta_path, charge_bounds)
    _gm.generate_pairs_negative(
        str(meta_path),
        charges=charge_bounds,
        mz_tolerance=pair_mz_tolerance_ppm,
        fragment_tolerance=negative_pair_fragment_tolerance,
        matching_fragments_threshold=negative_pair_matching_fragments_threshold,
    )
