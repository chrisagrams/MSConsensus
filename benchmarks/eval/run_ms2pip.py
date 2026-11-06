"""Run MS2PIP HCD2021 against a Prosit-format parquet via the Koina HTTP API.

MS2PIP only predicts singly-charged b/y ions (58 dims), while our observed
intensities are in Prosit's 174-dim layout (29 positions × {y,b} × {1+,2+,3+}).
We slice the observed array to the 58-dim 1+-only subspace before comparing.

Layout of the observed 174-dim Prosit array (kusterlab tensor shape
``[MAX_ION=29, ION_TYPES=2, NLOSSES_used=1, CHARGES=3]`` flattened C-order),
with ``ION_TYPES = ['y', 'b']`` (y first, NOT b):

  flat_idx = position*6 + ion_type*3 + charge
    where ion_type ∈ {0:y, 1:b}, charge ∈ {0:1+, 1:2+, 2:3+}

So observed[3::6]  = 29 entries of b@1+ for positions 0..28
   observed[0::6]  = 29 entries of y@1+
   concatenated [b1+, y1+] = 58-dim aligned with MS2PIP's
     [b1@1+, ..., b29@1+, y1@1+, ..., y29@1+] output.

Peptides with unsupported tokens (N-term mods, non-standard residues that
MS2PIP rejects) are dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KOINA_URL = "https://koina.wilhelmlab.org/v2/models/{model}/infer"
HOLDOUT = Path("data/holdout_pxd053296")
EPS = 1e-7


def koina_predict(model: str, peptides: list[str], charges: list[int],
                  retries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Call Koina with one batch and return (intensities, mz) of shape (N, 58).
    -1 values mean 'ion not predictable' (e.g., position past peptide length)."""
    n = len(peptides)
    body = {
        "id": "0",
        "inputs": [
            {"name": "peptide_sequences", "shape": [n, 1],
             "datatype": "BYTES", "data": peptides},
            {"name": "precursor_charges", "shape": [n, 1],
             "datatype": "INT32", "data": [[c] for c in charges]},
        ],
    }
    req = urllib.request.Request(
        KOINA_URL.format(model=model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            break
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Koina failed after {retries} retries: {last_err}")
    by_name = {o["name"]: o for o in resp["outputs"]}
    intens = np.asarray(by_name["intensities"]["data"], dtype=np.float32).reshape(n, 58)
    mz = np.asarray(by_name["mz"]["data"], dtype=np.float32).reshape(n, 58)
    return intens, mz


def slice_observed(obs174: np.ndarray) -> np.ndarray:
    """Reduce a (N, 174) Prosit-layout observed array to the (N, 58) MS2PIP
    layout: [b@1+ at p=0..28, y@1+ at p=0..28]. Recall ION_TYPES=['y','b']
    in kusterlab, so y is at offset 0 and b at offset 3 within each
    6-element position block."""
    y1 = obs174[:, 0::6]   # (N, 29) — y@1+ across positions
    b1 = obs174[:, 3::6]   # (N, 29) — b@1+ across positions
    return np.concatenate([b1, y1], axis=1)  # MS2PIP order: b first, then y


def spectral_angle(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Same per-row formula as prosit_repro.convergence._spectral_angle but on
    numpy. Mask y_true == -1 slots before normalization."""
    pred = ((y_true + 1) * y_pred) / (y_true + 1 + EPS)
    true = ((y_true + 1) * y_true) / (y_true + 1 + EPS)
    pn = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + EPS)
    tn = true / (np.linalg.norm(true, axis=-1, keepdims=True) + EPS)
    cos = np.clip((pn * tn).sum(axis=-1), -1 + EPS, 1 - EPS)
    return 2 * np.arccos(cos) / np.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path,
                    help="Prosit-format parquet with modified_sequence, "
                         "precursor_charge, intensities_raw cols.")
    ap.add_argument("--label", required=True,
                    help="Output dir name under HOLDOUT/eval/ms2pip/")
    ap.add_argument("--model", default="ms2pip_HCD2021",
                    help="Koina model name. HCD2021 is the standard HCD variant.")
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="Koina max batch is 1000.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_root = HOLDOUT / "eval/ms2pip" / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.parquet}")
    shards = sorted(args.parquet.glob("*.parquet")) if args.parquet.is_dir() else [args.parquet]
    df = pd.concat([pq.read_table(str(s)).to_pandas() for s in shards], ignore_index=True)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    n = len(df)
    print(f"  {n} rows")

    # Slice observed to the 58-dim 1+-only subspace.
    obs174 = np.asarray(df["intensities_raw"].tolist(), dtype=np.float32)
    obs58 = slice_observed(obs174)

    peps = df["modified_sequence"].tolist()
    charges = df["precursor_charge"].astype(int).tolist()

    # Predict in batches.
    pred58 = np.full((n, 58), -1.0, dtype=np.float32)
    n_failed = 0
    t0 = time.time()
    for s in range(0, n, args.batch_size):
        e = min(s + args.batch_size, n)
        try:
            intens, _ = koina_predict(args.model, peps[s:e], charges[s:e])
            pred58[s:e] = intens
        except Exception as exc:
            n_failed += (e - s)
            print(f"  batch [{s},{e}) failed: {exc}")
        if (s // args.batch_size) % 20 == 0:
            rate = (s + 1) / max(time.time() - t0, 1e-3)
            print(f"  progress: {s}/{n}  rate={rate:.0f} specs/sec")

    print(f"\nfinished {n - n_failed}/{n} predictions in {time.time() - t0:.0f}s")

    # Compute spectral angle on rows where MS2PIP returned a valid prediction
    # AND the observed has at least one non-mask value.
    valid = (pred58 >= 0).any(axis=1) & (obs58 > 0).any(axis=1)
    sa = np.full(n, np.nan, dtype=np.float32)
    if valid.any():
        sa[valid] = spectral_angle(obs58[valid], pred58[valid])

    # Aggregate
    metrics: dict = {
        "n_rows_total": int(n),
        "n_rows_evaluated": int(valid.sum()),
        "n_rows_failed_or_unscored": int((~valid).sum()),
    }
    if valid.any():
        valid_sa = sa[valid]
        metrics["median_spectral_angle"] = float(np.median(valid_sa))
        metrics["mean_spectral_angle"] = float(valid_sa.mean())
        metrics["median_spectral_angle_similarity"] = float(np.median(1 - valid_sa))
        for c in [1, 2, 3, 4, 5, 6]:
            mask = valid & (df["precursor_charge"].to_numpy() == c)
            if mask.any():
                metrics[f"median_spectral_angle_charge_{c}"] = float(np.median(sa[mask]))
                metrics[f"n_charge_{c}"] = int(mask.sum())
    print(f"metrics: {metrics}")

    csv_path = out_root / "metrics.csv"
    cols = ["timestamp", "label", "model", "parquet", *sorted(metrics.keys())]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join([
            datetime.utcnow().isoformat(), args.label, args.model,
            str(args.parquet),
            *[str(metrics[k]) for k in sorted(metrics.keys())],
        ]) + "\n")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
