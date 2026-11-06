from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from gleams_repro import features as _features
from gleams_repro import pairs as _pairs
from gleams_repro import train as _train


def _load(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def cli(log_level: str) -> None:
    """GLEAMS reproduction pipeline."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@cli.command("encode")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--split", required=True, type=click.Choice(["train", "val", "test"]))
@click.option("--limit", type=int, default=None, help="Cap the number of encoded spectra (smoke tests).")
@click.option("--workers", type=int, default=1, help="Parallel worker processes (1 = serial).")
def encode_cmd(config_path: str, split: str, limit: int | None, workers: int) -> None:
    cfg = _load(config_path)
    mgf_key = {"train": "train_mgf", "val": "val_mgf", "test": "test_mgf"}[split]
    _features.encode_split(
        mgf_xz_path=cfg["data"][mgf_key],
        out_dir=cfg["data"]["processed_dir"],
        split=split,
        features_cfg=cfg["features"],
        charges=tuple(cfg["data"]["charges"]),
        limit=limit,
        workers=workers,
    )


@cli.command("pairs")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--split", required=True, type=click.Choice(["train", "val", "test"]))
def pairs_cmd(config_path: str, split: str) -> None:
    cfg = _load(config_path)
    _pairs.build_pairs(
        out_dir=cfg["data"]["processed_dir"],
        split=split,
        charges=tuple(cfg["data"]["charges"]),
        pair_mz_tolerance_ppm=cfg["pairs"]["pair_mz_tolerance_ppm"],
        negative_pair_fragment_tolerance=cfg["pairs"]["negative_pair_fragment_tolerance"],
        negative_pair_matching_fragments_threshold=cfg["pairs"]["negative_pair_matching_fragments_threshold"],
    )


@cli.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def train_cmd(config_path: str) -> None:
    _train.train(config_path)


@cli.command("pipeline")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def pipeline_cmd(config_path: str) -> None:
    """Run encode(train+val) → pairs(train+val) → train end-to-end."""
    cfg = _load(config_path)
    proc = Path(cfg["data"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    charges = tuple(cfg["data"]["charges"])
    workers = int(cfg.get("hardware", {}).get("encode_workers", 1))
    for split, mgf_key in [("train", "train_mgf"), ("val", "val_mgf")]:
        if not (proc / f"{split}_features.npz").exists():
            _features.encode_split(
                mgf_xz_path=cfg["data"][mgf_key],
                out_dir=proc,
                split=split,
                features_cfg=cfg["features"],
                charges=charges,
                workers=workers,
            )
        if not (proc / f"{split}_metadata_pairs_pos_{min(charges)}.npy").exists():
            _pairs.build_pairs(
                out_dir=proc,
                split=split,
                charges=charges,
                pair_mz_tolerance_ppm=cfg["pairs"]["pair_mz_tolerance_ppm"],
                negative_pair_fragment_tolerance=cfg["pairs"]["negative_pair_fragment_tolerance"],
                negative_pair_matching_fragments_threshold=cfg["pairs"]["negative_pair_matching_fragments_threshold"],
            )
    _train.train(config_path)


if __name__ == "__main__":
    cli()
