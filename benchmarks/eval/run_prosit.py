"""Run a Prosit retrain checkpoint on a Prosit-format parquet test set.

Loads a state_dict from a ``best.pt`` produced by ``prosit-repro train``,
rebuilds the model with the training alphabet (re-derived from the original
train parquet), and runs ``prosit_repro.convergence.evaluate`` on the eval
parquet to compute median spectral-angle similarity (overall + per charge).

Why we re-derive the alphabet from the train parquet: the embedding layer's
size is determined by the training-time alphabet, which isn't persisted
alongside the .pt checkpoint. Different retrains have different alphabets
(paper/4g: 23 tokens, consensus_30m: 21 tokens), so passing the wrong train
parquet here will trigger a state_dict size mismatch.

Usage:
  python eval/run_prosit.py \\
      --checkpoint prosit/models/paper/best.pt \\
      --train-parquet prosit/data/processed/prosit_paper/train.parquet \\
      --eval-parquet  prosit/data/processed/prosit_paper/holdout.parquet \\
      --label paper_paper-holdout
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Activate the PyTorch backend before importing dlomix.
os.environ.setdefault("DLOMIX_BACKEND", "torch")

import torch  # noqa: E402

# Reuse the existing prosit_repro modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prosit/src"))
from prosit_repro.convergence import evaluate  # noqa: E402
from prosit_repro.data import _resolve_parquet  # noqa: E402

from dlomix.data import FragmentIonIntensityDataset  # noqa: E402
from dlomix.data.dataset_utils import EncodingScheme  # noqa: E402
from dlomix.models import PrositIntensityPredictor  # noqa: E402

HOLDOUT = Path("data/holdout_pxd053296")

# Match the model architecture pinned in configs/prosit_paper.yaml — every
# retrain config uses these same architectural hyperparameters.
_MODEL_ARGS = dict(
    seq_length=30,
    with_termini=False,
    use_meta_data=True,
    use_prosit_ptm_features=False,
    embedding_output_dim=16,
    recurrent_layers_sizes=(256, 512),
    regressor_layer_size=512,
    dropout_rate=0.2,
    latent_dropout_rate=0.1,
    input_keys={"SEQUENCE_KEY": "modified_sequence"},
    meta_data_keys={
        "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
        "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
    },
)
_MODEL_FEATURES = ["collision_energy_aligned_normed", "precursor_charge_onehot"]
_LABEL_COLUMN = "intensities_raw"


_REQUIRED_COLUMNS = [
    "modified_sequence",
    "precursor_charge",
    "precursor_charge_onehot",
    "aligned_collision_energy",
    "collision_energy_aligned_normed",
    "intensities_raw",
    "masses_raw",
]


def _conformed_eval_parquet(eval_parquet: Path) -> Path:
    """Project the eval parquet to the train-parquet column set so HF Datasets
    accepts a combined train+test dataset (it refuses if schemas differ).

    Some eval parquets carry extra columns (``score``, ``appearance_count``)
    from the kusterlab annotate pipeline that the prosit paper train parquet
    doesn't have. We rewrite a slimmed copy under /tmp and use that.

    The slim copy is keyed by source path so it's stable across reruns.
    """
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    digest = hashlib.sha1(str(eval_parquet.resolve()).encode()).hexdigest()[:12]
    slim_dir = Path("/tmp") / f"prosit_eval_slim_{digest}"
    if slim_dir.exists() and any(slim_dir.glob("*.parquet")):
        return slim_dir

    slim_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(eval_parquet.glob("*.parquet")) if eval_parquet.is_dir() else [eval_parquet]
    for i, src in enumerate(shards):
        tbl = pq.read_table(str(src))
        keep = [c for c in _REQUIRED_COLUMNS if c in tbl.schema.names]
        if not all(c in tbl.schema.names for c in _REQUIRED_COLUMNS):
            missing = [c for c in _REQUIRED_COLUMNS if c not in tbl.schema.names]
            raise RuntimeError(f"{src} missing required cols: {missing}")
        slim = tbl.select(keep)
        # Force the canonical column order to match the train parquet so HF
        # Datasets schema-cast doesn't trip on column ordering.
        slim = slim.select(_REQUIRED_COLUMNS)
        pq.write_table(slim, str(slim_dir / f"part-{i:05d}.parquet"))
    return slim_dir


def _build_dataset(train_parquet: Path, eval_parquet: Path,
                   batch_size: int) -> FragmentIonIntensityDataset:
    """Build a combined train+test dataset. Alphabet is learned from the union
    of train+eval (so eval-only tokens get proper indices instead of being
    folded to UNK), and the eval split is exposed via ``tensor_test_data`` so
    we can iterate it for inference."""
    eval_slim = _conformed_eval_parquet(eval_parquet)
    return FragmentIonIntensityDataset(
        data_source=_resolve_parquet(str(train_parquet)),
        test_data_source=_resolve_parquet(str(eval_slim)),
        data_format="parquet",
        sequence_column="modified_sequence",
        label_column=_LABEL_COLUMN,
        max_seq_len=30,
        dataset_type="pt",
        batch_size=batch_size,
        shuffle=False,
        model_features=_MODEL_FEATURES,
        with_termini=False,
        alphabet=None,
        encoding_scheme=EncodingScheme.NAIVE_MODS,
        num_proc=-1,
        torch_dataloader_kwargs=dict(pin_memory=True),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to a best.pt produced by prosit-repro train.")
    ap.add_argument("--train-parquet", required=True, type=Path,
                    help="Train parquet the checkpoint was trained on. Used to "
                         "re-derive the alphabet matching the checkpoint's "
                         "embedding layer.")
    ap.add_argument("--eval-parquet", required=True, type=Path,
                    help="Parquet with the test spectra to evaluate against.")
    ap.add_argument("--label", required=True,
                    help="Output dir name under HOLDOUT/eval/prosit/")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    out_root = HOLDOUT / "eval/prosit" / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    print(f"building dataset (train={args.train_parquet.name} eval={args.eval_parquet.name})")
    ds = _build_dataset(args.train_parquet, args.eval_parquet, args.batch_size)
    alphabet = ds.extended_alphabet
    print(f"alphabet size={len(alphabet)}  (must match checkpoint's embedding rows)")

    print("building model + loading state_dict")
    model = PrositIntensityPredictor(alphabet=alphabet, **_MODEL_ARGS).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    eval_loader = ds.tensor_test_data
    print(f"running eval over {len(eval_loader)} batches")
    metrics = evaluate(
        model, eval_loader, device,
        label_column=_LABEL_COLUMN,
        per_charge_charges=[1, 2, 3, 4, 5, 6],
    )
    print(f"metrics: {metrics}")

    csv = out_root / "metrics.csv"
    cols = ["timestamp", "label", "checkpoint", "train_parquet", "eval_parquet"]
    cols += sorted(metrics.keys())
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join([
            datetime.utcnow().isoformat(),
            args.label,
            str(args.checkpoint),
            str(args.train_parquet),
            str(args.eval_parquet),
            *[f"{metrics[k]}" for k in sorted(metrics.keys())],
        ]) + "\n")
    print(f"wrote {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
