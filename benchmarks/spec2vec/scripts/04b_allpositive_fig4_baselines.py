"""AllPositive Fig 4 Cosine + Modified Cosine baselines (parallel).

Runs the same 1000-query × ~94k-library scenario-1 split built by
``04_eval_fig4.py`` and appends cosine + modified_cosine rows to
``reports/allpositive/fig4_threshold_sweep.csv``.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.baselines import query_library_matrix  # noqa: E402
from spec2vec_repro.library_matching import (  # noqa: E402
    build_scenario1_split,
    threshold_sweep,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "allpositive.yaml").read_text())
    processed = REPO_ROOT / "data" / "processed"
    with open(processed / "allpositive_spectra.pkl", "rb") as f:
        spectra_raw = pickle.load(f)

    keep = [
        i for i, s in enumerate(spectra_raw)
        if s.get("precursor_mz") is not None and float(s.get("precursor_mz", 0)) > 0
    ]
    spectra = [spectra_raw[i] for i in keep]
    logging.info("Kept %d / %d spectra with precursor_mz > 0", len(spectra), len(spectra_raw))

    q_idx, l_idx = build_scenario1_split(spectra, seed=cfg.get("seed", 42))
    q_spec = [spectra[i] for i in q_idx]
    l_spec = [spectra[i] for i in l_idx]

    new_rows: list[dict] = []
    for sim_name, min_match in [("cosine", 6), ("modified_cosine", 10)]:
        logging.info("Running %s ...", sim_name)
        scores = query_library_matrix(
            q_spec, l_spec, sim_name=sim_name, tolerance=0.005,
            min_match=min_match, n_workers=32, chunk_size=20,
        )
        rows = threshold_sweep(scores, q_spec, l_spec)
        for r in rows:
            r["method"] = sim_name
        new_rows.extend(rows)

    reports = REPO_ROOT / "reports" / "allpositive"
    csv = reports / "fig4_threshold_sweep.csv"
    existing = pd.read_csv(csv)
    existing = existing[existing["method"] == "spec2vec"]
    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(csv, index=False)
    logging.info("Wrote %s", csv)


if __name__ == "__main__":
    main()
