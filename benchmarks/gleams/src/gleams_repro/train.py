from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
import wandb
import yaml
from gleams.nn import data_generator as _dg

from gleams_repro.convergence import DownstreamEval
from gleams_repro.features import feature_split as _feature_split
from gleams_repro.model import build_embedder
from gleams_repro.wandb_callback import WandbConvergence

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _pair_files(processed_dir: Path, split: str, kind: str, charges: tuple[int, ...]) -> list[str]:
    # Upstream writes {basename}_pairs_{kind}_{charge}.npy (replacing .parquet).
    base = processed_dir / f"{split}_metadata"
    return [f"{base}_pairs_{kind}_{c}.npy" for c in charges]


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    proc = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["logging"]["checkpoint_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(cfg["training"]["seed"])

    wandb.init(
        project=cfg["logging"]["wandb_project"],
        entity=cfg["logging"].get("wandb_entity"),
        name=cfg["run_name"],
        config=cfg,
        dir=cfg["logging"]["log_dir"],
    )

    charges = tuple(cfg["data"]["charges"])
    fsplit = _feature_split()

    train_seq = _dg.PairSequence(
        filename_feat=str(proc / "train_features.npz"),
        filenames_pairs_pos=_pair_files(proc, "train", "pos", charges),
        filenames_pairs_neg=_pair_files(proc, "train", "neg", charges),
        batch_size=cfg["training"]["batch_size"],
        feature_split=fsplit,
        max_num_pairs=None,
        shuffle=True,
    )
    val_seq = _dg.PairSequence(
        filename_feat=str(proc / "val_features.npz"),
        filenames_pairs_pos=_pair_files(proc, "val", "pos", charges),
        filenames_pairs_neg=_pair_files(proc, "val", "neg", charges),
        batch_size=cfg["training"]["batch_size"],
        feature_split=fsplit,
        max_num_pairs=cfg["training"]["max_num_pairs_val"],
        shuffle=False,
    )

    # Embedder.build() creates its own MirroredStrategy scope and compiles
    # with RectifiedAdam + contrastive_loss — exactly the paper setup.
    embedder = build_embedder(
        lr=cfg["optim"]["learning_rate"],
        filename=str(model_dir / f"{cfg['run_name']}.hdf5"),
    )
    embedder.build()

    callbacks: list[tf.keras.callbacks.Callback] = [
        WandbConvergence(log_every_n_steps=cfg["logging"]["log_every_n_steps"]),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / (cfg["run_name"] + "_epoch{epoch:03d}.hdf5")),
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(Path(cfg["logging"]["log_dir"]) / f"{cfg['run_name']}.csv"),
            append=True,
        ),
    ]
    if cfg["convergence"]["enabled"]:
        callbacks.append(
            DownstreamEval(
                embedder=embedder,
                processed_dir=proc,
                split="val",
                num_query=cfg["convergence"]["num_query"],
                num_library=cfg["convergence"]["num_library"],
                eval_every_n_epochs=cfg["convergence"]["eval_every_n_epochs"],
                feature_split=fsplit,
                batch_size=cfg["training"]["batch_size"],
                seed=cfg["convergence"]["seed"],
            )
        )

    embedder.siamese_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg["training"]["num_epochs"],
        steps_per_epoch=cfg["training"]["steps_per_epoch"],
        callbacks=callbacks,
    )

    embedder.siamese_model.save(str(model_dir / f"{cfg['run_name']}_final.hdf5"))
    wandb.finish()
