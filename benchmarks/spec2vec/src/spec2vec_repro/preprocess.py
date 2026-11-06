"""Preprocessing for spec2vec reproduction.

Mirrors `iomega-5b-train-spec2vec-model-all-gnps.ipynb` from
`iomega/spec2vec_gnps_data_analysis`: six-step matchms pipeline followed by
SpectrumDocument conversion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Iterable

from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    reduce_to_number_of_peaks,
    require_minimum_number_of_peaks,
    select_by_mz,
    select_by_relative_intensity,
)
from matchms.importing import load_from_json

from spec2vec_repro.document import SpectrumDocumentWithLosses

logger = logging.getLogger(__name__)

CLEANED_JSON_NAME = "gnps_positive_ionmode_cleaned_by_matchms_and_lookups.json"


def load_cleaned_gnps(zip_path: str | Path, extract_dir: str | Path) -> list[Spectrum]:
    """Load the Zenodo 3978118 cleaned GNPS positive-mode JSON.

    Extracts the JSON to ``extract_dir`` on first call, then loads via matchms.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    json_path = extract_dir / CLEANED_JSON_NAME
    if not json_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s -> %s", zip_path, json_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(CLEANED_JSON_NAME, extract_dir)
    logger.info("Loading spectra from %s", json_path)
    spectra = list(load_from_json(str(json_path)))
    logger.info("Loaded %d raw spectra", len(spectra))
    return spectra


def post_process(
    spectrum: Spectrum,
    *,
    mz_from: float,
    mz_to: float,
    min_peaks: int,
    relative_intensity_from: float,
    reduce_n_required: int,
    reduce_ratio_desired: float,
) -> Spectrum | None:
    """Peak filter pipeline from iomega-5b (losses handled at document time)."""
    s = normalize_intensities(spectrum)
    if s is None:
        return None
    s = select_by_mz(s, mz_from=mz_from, mz_to=mz_to)
    s = require_minimum_number_of_peaks(s, n_required=min_peaks)
    if s is None:
        return None
    s_low_removed = select_by_relative_intensity(s, intensity_from=relative_intensity_from)
    if s_low_removed is not None and len(s_low_removed.peaks) >= min_peaks:
        s = s_low_removed
    s = reduce_to_number_of_peaks(
        s,
        n_required=reduce_n_required,
        n_max=1000,
        ratio_desired=reduce_ratio_desired,
    )
    return s


def apply_pipeline(spectra: Iterable[Spectrum], params: dict) -> list[Spectrum]:
    processed: list[Spectrum] = []
    n_in = 0
    for s in spectra:
        n_in += 1
        out = post_process(s, **params)
        if out is not None:
            processed.append(out)
    logger.info("Kept %d / %d spectra after filtering", len(processed), n_in)
    return processed


def to_documents(
    spectra: list[Spectrum],
    *,
    n_decimals: int,
    loss_mz_from: float,
    loss_mz_to: float,
) -> list[SpectrumDocumentWithLosses]:
    return [
        SpectrumDocumentWithLosses(
            s,
            n_decimals=n_decimals,
            loss_mz_from=loss_mz_from,
            loss_mz_to=loss_mz_to,
        )
        for s in spectra
    ]


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def get_or_build_filtered(
    zip_path: str | Path,
    extract_dir: str | Path,
    cache_dir: str | Path,
    params: dict,
) -> list[Spectrum]:
    """Filter-step cache — both configs share these params, so we only do it once."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"filtered_{params_hash(params)}.pkl"
    if cache_path.exists():
        logger.info("Loading cached filtered spectra from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    raw = load_cleaned_gnps(zip_path=zip_path, extract_dir=extract_dir)
    filtered = apply_pipeline(raw, params)
    with open(cache_path, "wb") as f:
        pickle.dump(filtered, f)
    logger.info("Cached filtered spectra to %s", cache_path)
    return filtered
