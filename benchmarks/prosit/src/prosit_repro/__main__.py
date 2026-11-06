"""prosit-repro CLI."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

# Make the sibling scripts/ dir importable so preprocess subcommands can
# reuse split_train_val.main and mgf_to_prosit_parquet.annotate without
# duplicating logic. scripts/ isn't a package; we just put it on sys.path.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from prosit_repro import train as _train


def _load(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def cli(log_level: str) -> None:
    """Prosit reproduction pipeline (DLOmix PyTorch backend)."""
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


@cli.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def train_cmd(config_path: str) -> None:
    """Train Prosit on the parquet train/val/test set referenced in --config."""
    _train.train(config_path)


@cli.command("preprocess-paper")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def preprocess_paper_cmd(config_path: str) -> None:
    """Convert the figshare HDF5s into parquet shards for the paper run."""
    cfg = _load(config_path)
    raw = Path("prosit/data/raw/prosit_paper")
    out = Path(cfg["data"]["train_parquet"]).parent
    from importlib import import_module
    mod = import_module("scripts.hdf5_to_parquet")  # only works when scripts/ is on PYTHONPATH
    mod.main(raw / "traintest_hcd.hdf5", raw / "holdout_hcd.hdf5", out,
             val_fraction=0.10, seed=20260428, shard_rows=500_000)


@cli.command("preprocess-concensus")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def preprocess_concensus_cmd(config_path: str) -> None:
    """Annotate the consensus MGFs into Prosit-format parquet shards.

    Steps: split train.xz -> train_main.xz + val.xz; annotate each of
    {train_main, val, test} into its parquet directory.
    """
    cfg = _load(config_path)
    pre = cfg["preprocess"]
    # Derive the working dir from the config so different consensus corpora
    # (e.g. 30m vs 4g) don't collide on split intermediates / sentinels.
    work = Path(cfg["data"]["train_parquet"]).parent
    work.mkdir(parents=True, exist_ok=True)

    train_xz = Path(pre["raw_train_xz"])
    test_xz = Path(pre["raw_test_xz"])
    # Nest the split intermediates under the per-corpus work dir so that running
    # 4g and 30m preprocesses (or re-running with a different val_seed) doesn't
    # accidentally reuse a stale _DONE sentinel from a different corpus.
    split_dir = work / "concensus_split"
    # Plain MGF (no .xz) for the intermediate split outputs: encoding to xz
    # is ~10× slower than just writing text, and these files are read once.
    train_main = split_dir / "concensus.mgf_train_main.mgf"
    val_xz = split_dir / "concensus.mgf_val.mgf"

    # Use a sentinel file so we don't mistake half-written xz stubs (left
    # behind by an interrupted prior run) for a completed split.
    split_sentinel = split_dir / "_DONE"
    if not split_sentinel.exists():
        # Wipe any partial outputs from a prior crash before re-splitting.
        for p in (train_main, val_xz):
            if p.exists():
                p.unlink()
        from split_train_val import main as split_main  # type: ignore[import-not-found]
        split_main(train_xz, train_main, val_xz,
                   val_fraction=float(pre.get("val_fraction", 0.05)),
                   seed=int(pre.get("val_seed", 20260426)))
        split_sentinel.touch()

    from mgf_to_prosit_parquet import annotate  # type: ignore[import-not-found]
    common = dict(
        ce=float(pre.get("collision_energy", 35.0)),
        mass_analyzer=str(pre.get("mass_analyzer", "FTMS")),
        ppm_override=pre.get("ppm_override"),
        min_peaks=int(pre.get("min_peaks", 5)),
        max_seq_len=int(pre.get("max_seq_len", 30)),
        charges=tuple(pre.get("charges", [1, 2, 3, 4, 5, 6])),
        shard_rows=int(pre.get("shard_rows", 250_000)),
        dedup_mode=str(pre.get("dedup_mode", "none")),
        top_k=int(pre.get("top_k", 1)),
    )
    annotate(train_main, Path(cfg["data"]["train_parquet"]), **common)
    annotate(val_xz,    Path(cfg["data"]["val_parquet"]),   **common)
    annotate(test_xz,   Path(cfg["data"]["test_parquet"]),  **common)


if __name__ == "__main__":
    cli()
