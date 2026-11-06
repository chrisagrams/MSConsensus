"""Train a Spec2Vec model with W&B + convergence logging.

Reads preprocessed pickles from ``data/processed/<name>_{spectra,documents}.pkl``
and writes the final model + per-eval checkpoints to ``models/<name>/``.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spec2vec_repro.convergence import build_holdout_pairs  # noqa: E402
from spec2vec_repro.train import build_word2vec, train_model  # noqa: E402
from spec2vec_repro.wandb_callback import SpecTrainCallback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from the config (useful for smoke tests).",
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["name"]
    if args.epochs is not None:
        cfg["model"]["epochs"] = args.epochs

    processed = REPO_ROOT / "data" / "processed"
    with open(processed / f"{name}_spectra.pkl", "rb") as f:
        spectra = pickle.load(f)
    with open(processed / f"{name}_documents.pkl", "rb") as f:
        documents = pickle.load(f)
    logging.info("Loaded %d spectra / %d documents", len(spectra), len(documents))

    holdout = build_holdout_pairs(
        spectra,
        n_pairs=cfg["eval"]["holdout_pairs"],
        fingerprint_nbits=cfg["eval"]["fingerprint_nbits"],
        seed=cfg.get("seed", 42),
    )

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
        mode=args.wandb_mode,
    )

    model_dir = REPO_ROOT / "models" / name
    callback = SpecTrainCallback(
        documents=documents,
        holdout=holdout,
        checkpoint_dir=model_dir / "checkpoints",
        eval_every_n_epochs=cfg["eval"]["every_n_epochs"],
        run_name=name,
    )

    model = build_word2vec(cfg)
    model = train_model(
        model,
        documents,
        epochs=cfg["model"]["epochs"],
        callbacks=[callback],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / f"{name}_final.model"
    model.save(str(final_path))
    logging.info("Saved final model to %s", final_path)

    run.summary["final_model_path"] = str(final_path)
    run.summary["vocab_size"] = len(model.wv)
    run.finish()


if __name__ == "__main__":
    main()
