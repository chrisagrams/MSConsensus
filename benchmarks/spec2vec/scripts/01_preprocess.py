"""Preprocess GNPS cleaned spectra: filter + subset + SpectrumDocument.

Produces under ``data/processed/``:
  <name>_spectra.pkl    — filtered matchms Spectrum objects for the subset
  <name>_documents.pkl  — list of SpectrumDocument for training
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.preprocess import get_or_build_filtered, to_documents  # noqa: E402
from spec2vec_repro.split import select_all_positive, select_unique_inchikey  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["name"]
    pp = cfg["preprocess"]
    filter_params = {
        "mz_from": pp["mz_from"],
        "mz_to": pp["mz_to"],
        "min_peaks": pp["min_peaks"],
        "relative_intensity_from": pp["relative_intensity_from"],
        "reduce_n_required": pp["reduce_n_required"],
        "reduce_ratio_desired": pp["reduce_ratio_desired"],
    }

    out_dir = REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = get_or_build_filtered(
        zip_path=cfg["dataset"]["source"],
        extract_dir=REPO_ROOT / "data" / "raw",
        cache_dir=out_dir,
        params=filter_params,
    )

    subset = cfg["dataset"]["subset"]
    if subset == "all_positive":
        spectra = select_all_positive(filtered)
    elif subset == "unique_inchikey14":
        spectra = select_unique_inchikey(filtered)
    else:
        raise ValueError(f"Unknown subset: {subset}")

    docs = to_documents(
        spectra,
        n_decimals=pp["n_decimals"],
        loss_mz_from=pp["loss_mz_from"],
        loss_mz_to=pp["loss_mz_to"],
    )

    spectra_path = out_dir / f"{name}_spectra.pkl"
    docs_path = out_dir / f"{name}_documents.pkl"
    with open(spectra_path, "wb") as f:
        pickle.dump(spectra, f)
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

    logging.info("Saved %d spectra -> %s", len(spectra), spectra_path)
    logging.info("Saved %d documents -> %s", len(docs), docs_path)


if __name__ == "__main__":
    main()
