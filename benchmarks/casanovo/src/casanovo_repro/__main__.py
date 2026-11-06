"""casanovo-repro CLI."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def cli(log_level: str) -> None:
    """Casanovo paper-faithful retrain pipeline (MassIVE-KB and consensus 30M)."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


@cli.command("prep")
@click.option("--config", "overlay_path", required=True,
              type=click.Path(exists=True),
              help="Per-dataset overlay YAML (massive_kb.yaml / consensus.yaml).")
@click.option("--paper-config", "paper_path", default=None,
              type=click.Path(exists=True),
              help="Override casanovo_paper.yaml (default: shipped with package).")
def prep_cmd(overlay_path: str, paper_path: str | None) -> None:
    """Dataset-level prep that must run before lance builds. For consensus,
    splits the full-train MGF into train_main + val and aliases test. For
    massive_kb, just verifies inputs exist. Idempotent."""
    from casanovo_repro.data import load_merged
    from casanovo_repro import preprocess

    _, repro_cfg = load_merged(overlay_path, paper_path)
    preprocess.prep(repro_cfg)


@cli.command("preprocess")
@click.option("--config", "overlay_path", required=True,
              type=click.Path(exists=True),
              help="Per-dataset overlay YAML (massive_kb.yaml / consensus.yaml).")
@click.option("--paper-config", "paper_path", default=None,
              type=click.Path(exists=True),
              help="Override casanovo_paper.yaml (default: shipped with package).")
@click.option("--split",
              type=click.Choice(["train", "valid", "test", "all"]),
              default="all", show_default=True,
              help="Which split to lance-build. `all` does train+valid (test is "
                   "inference-only and skipped). Use one split per process when "
                   "fanning out across CPUs in the PBS preprocess job.")
def preprocess_cmd(overlay_path: str, paper_path: str | None, split: str) -> None:
    """Build casanovo's depthcharge Lance index for one or all splits. Run
    `prep` first if the dataset needs MGF prep (consensus). depthcharge is
    single-threaded internally; run one invocation per (dataset, split) pair
    concurrently from the PBS script for max throughput."""
    from casanovo_repro.data import load_merged
    from casanovo_repro import preprocess

    casanovo_cfg, repro_cfg = load_merged(overlay_path, paper_path)
    preprocess.run(casanovo_cfg, repro_cfg, split)


@cli.command("train")
@click.option("--config", "overlay_path", required=True,
              type=click.Path(exists=True),
              help="Per-dataset overlay YAML (massive_kb.yaml / consensus.yaml).")
@click.option("--paper-config", "paper_path", default=None,
              type=click.Path(exists=True),
              help="Override casanovo_paper.yaml (default: shipped with package).")
def train_cmd(overlay_path: str, paper_path: str | None) -> None:
    """Train one dataset's casanovo model from scratch with paper-faithful
    hyperparameters and W&B logging."""
    from casanovo_repro.data import load_merged
    from casanovo_repro import train as _train

    casanovo_cfg, repro_cfg = load_merged(overlay_path, paper_path)
    _train.run(casanovo_cfg, repro_cfg)


if __name__ == "__main__":
    cli()
