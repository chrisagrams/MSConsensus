"""Run a GLEAMS checkpoint against a pre-encoded dataset and compute top-1
peptide-NN accuracy.

Mirrors the convergence eval used during training (gleams_repro/convergence.py
DownstreamEval) but as a standalone CLI so we can score arbitrary
(checkpoint, encoded-dataset) pairs from the eval harness.

Pipeline:
  1. Load the .hdf5 checkpoint via gleams' Embedder.
  2. Load <split>_features.npz + <split>_metadata.parquet.
  3. Sample <num_query> queries + <num_library> library spectra (deterministic
     by seed); embed each with the single-arm encoder.
  4. For each query, find the nearest library embedding by Euclidean distance.
  5. Top-1 = matching peptide (after I→L folding).
  6. Write metrics.csv with top-1 accuracy and per-charge breakdown.

Usage:
  python eval/run_gleams.py \\
      --checkpoint gleams/data/raw/gleams_82c0124b.hdf5 \\
      --features-dir <eval_root>/gleams_data \\
      --split pxd053296 \\
      --label published_pxd053296
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# gleams package needs TF; force CPU for deterministic embedding (small data,
# inference is fast enough on CPU and avoids GPU contention with other evals).
# Override with --device cuda if desired.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as ss  # noqa: E402

# Reuse gleams_repro modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gleams/src"))
from gleams_repro.features import feature_split  # noqa: E402

HOLDOUT = Path("data/holdout_pxd053296")


def _load_embedder(checkpoint: Path):
    """Build the GLEAMS Embedder and load weights from the .hdf5 checkpoint."""
    from gleams.nn import embedder as _gem  # type: ignore[import-not-found]
    from gleams import config as gcfg  # type: ignore[import-not-found]
    i1, i2 = feature_split()
    em = _gem.Embedder(
        num_precursor_features=i1,
        num_fragment_features=i2 - i1,
        num_ref_spectra_features=gcfg.num_ref_spectra,
        lr=2e-4,
        filename=str(checkpoint),
    )
    em.load()
    return em


def _embed(em, feats: ss.csr_matrix, batch_size: int = 1024) -> np.ndarray:
    """Run the single-arm encoder on a sparse feature matrix."""
    i1, i2 = feature_split()
    x_prec = feats[:, :i1].toarray()
    x_frag = feats[:, i1:i2].toarray()
    x_ref = feats[:, i2:].toarray()
    single_arm = em.siamese_model.get_layer("embedder")
    return single_arm.predict([x_prec, x_frag, x_ref], batch_size=batch_size, verbose=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="GLEAMS .hdf5 checkpoint.")
    ap.add_argument("--features-dir", required=True, type=Path,
                    help="Directory containing <split>_features.npz + "
                         "<split>_metadata.parquet.")
    ap.add_argument("--split", required=True,
                    help="Split-name prefix used at encode time.")
    ap.add_argument("--label", required=True,
                    help="Output dir name under HOLDOUT/eval/gleams/")
    ap.add_argument("--num-query", type=int, default=10000)
    ap.add_argument("--num-library", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    out_root = HOLDOUT / "eval/gleams" / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    feat_path = args.features_dir / f"{args.split}_features.npz"
    meta_path = args.features_dir / f"{args.split}_metadata.parquet"
    print(f"loading {feat_path.name} + {meta_path.name}")
    feats = ss.load_npz(str(feat_path))
    meta = pd.read_parquet(str(meta_path))
    print(f"  {feats.shape[0]} encoded spectra, {feats.shape[1]} features")

    want = args.num_query + args.num_library
    if want > feats.shape[0]:
        raise ValueError(
            f"need {want} rows for eval, only {feats.shape[0]} available "
            f"({args.num_query} query + {args.num_library} library)"
        )

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(feats.shape[0])
    q_idx, lib_idx = idx[: args.num_query], idx[args.num_query : want]

    q_feats = feats[q_idx]
    lib_feats = feats[lib_idx]
    q_seq = meta["sequence"].iloc[q_idx].str.replace("I", "L").to_numpy()
    lib_seq = meta["sequence"].iloc[lib_idx].str.replace("I", "L").to_numpy()
    q_charge = meta["charge"].iloc[q_idx].to_numpy()

    print(f"loading checkpoint {args.checkpoint}")
    em = _load_embedder(args.checkpoint)

    print(f"embedding {len(q_idx)} queries + {len(lib_idx)} library")
    q_emb = _embed(em, q_feats, args.batch_size).astype(np.float32)
    lib_emb = _embed(em, lib_feats, args.batch_size).astype(np.float32)

    # Compute top-K accuracy for several K, plus mean reciprocal rank.
    # We argsort each chunk's distances ascending so we get both an exact
    # ranking (for MRR) and the top-K prefix (for the accuracies) from the
    # same pass. Cost: O(L log L) per query — fine at L = 100k.
    TOP_KS = [1, 5, 10, 50]
    print(f"computing top-K NN (K={TOP_KS}) + MRR")
    lib_sq = (lib_emb * lib_emb).sum(axis=1)
    hits: dict[int, np.ndarray] = {k: np.zeros(len(q_emb), dtype=bool) for k in TOP_KS}
    rr = np.zeros(len(q_emb), dtype=np.float32)
    chunk = 512
    for s in range(0, len(q_emb), chunk):
        q = q_emb[s : s + chunk]
        q_sq = (q * q).sum(axis=1, keepdims=True)
        d2 = q_sq + lib_sq[None, :] - 2.0 * q @ lib_emb.T   # (chunk, L)
        order = np.argsort(d2, axis=1)                       # ascending distance
        matches = lib_seq[order] == q_seq[s : s + chunk, None]  # (chunk, L) bool
        for k in TOP_KS:
            hits[k][s : s + chunk] = matches[:, :k].any(axis=1)
        first_match = matches.argmax(axis=1)                 # 0 if all False
        has_match = matches.any(axis=1)
        rr[s : s + chunk] = np.where(has_match, 1.0 / (first_match + 1), 0.0)

    overall: dict[str, float] = {}
    for k in TOP_KS:
        overall[f"top{k}_acc"] = float(hits[k].mean())
    overall["mrr"] = float(rr.mean())

    per_charge: dict[str, float | int] = {}
    for z in [2, 3, 4, 5]:
        mask = q_charge == z
        if not mask.any():
            continue
        per_charge[f"n_charge_{z}"] = int(mask.sum())
        for k in TOP_KS:
            per_charge[f"top{k}_charge_{z}"] = float(hits[k][mask].mean())
        per_charge[f"mrr_charge_{z}"] = float(rr[mask].mean())

    print(f"overall: {overall}")
    print(f"per-charge: {per_charge}")

    csv = out_root / "metrics.csv"
    cols = ["timestamp", "label", "checkpoint", "split", "n_query", "n_library",
            *sorted(overall.keys()), *sorted(per_charge.keys())]
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join([
            datetime.utcnow().isoformat(), args.label, str(args.checkpoint),
            args.split, str(len(q_idx)), str(len(lib_idx)),
            *[f"{overall[k]}" for k in sorted(overall.keys())],
            *[f"{per_charge[k]}" for k in sorted(per_charge.keys())],
        ]) + "\n")
    print(f"wrote {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
