"""Config loading + path resolution shared by preprocess and train.

casanovo.config.Config validates that every key from its bundled config.yaml
is present in the user-supplied YAML. Our paper.yaml is intentionally just
an overlay (paper-specified knobs only), so we layer:

    upstream casanovo/config.yaml   (full default set, ground truth)
        + casanovo/configs/casanovo_paper.yaml   (paper deltas)
            + casanovo/configs/<dataset>.yaml    (paths, seed, tokenizer)

before handing the result to casanovo.Config.
"""
from __future__ import annotations

import copy
import importlib.resources as ir
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("casanovo_repro.data")

_REPRO_KEY = "_repro"

# __file__:                                 .../casanovo/src/casanovo_repro/data.py
# parents[0]: .../casanovo/src/casanovo_repro
# parents[1]: .../casanovo/src
# parents[2]: .../casanovo                  <- package root with configs/, scripts/
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_CONFIG = PACKAGE_ROOT / "configs" / "casanovo_paper.yaml"


def _upstream_default_config() -> dict[str, Any]:
    """Read the config.yaml bundled inside the installed `casanovo` package."""
    with ir.files("casanovo").joinpath("config.yaml").open("r") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_merged(overlay_path: str | Path,
                paper_config_path: str | Path | None = None,
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(casanovo_cfg, repro_cfg)``.

    ``casanovo_cfg`` is the dict that gets handed to ``casanovo.config.Config``
    after the ``_repro`` block is stripped. ``repro_cfg`` carries the wrapper-
    only fields (paths, output dir, W&B metadata).
    """
    base = _upstream_default_config()

    paper = paper_config_path or DEFAULT_PAPER_CONFIG
    with open(paper) as fh:
        paper_overlay = yaml.safe_load(fh) or {}
    base = _deep_merge(base, paper_overlay)

    with open(overlay_path) as fh:
        dataset_overlay = yaml.safe_load(fh) or {}
    merged = _deep_merge(base, dataset_overlay)

    repro = merged.pop(_REPRO_KEY, {}) or {}
    if not repro:
        raise ValueError(
            f"overlay {overlay_path} is missing the `_repro:` block "
            "(dataset paths, output_dir, wandb metadata)"
        )
    return merged, repro


def write_casanovo_config(casanovo_cfg: dict[str, Any],
                          out_path: Path | None = None) -> Path:
    """Materialize the merged casanovo config to a YAML on disk.

    casanovo.config.Config takes a file path, not a dict, so we write a temp
    file. If ``out_path`` is None, a tempfile in $TMPDIR is created (callers
    should keep the returned path alive for the duration of the run).
    """
    if out_path is None:
        fd, name = tempfile.mkstemp(prefix="casanovo_repro_", suffix=".yaml")
        out_path = Path(name)
        import os
        os.close(fd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(casanovo_cfg, fh, sort_keys=False)
    logger.info("wrote merged casanovo config: %s", out_path)
    return out_path


def lance_paths(lance_dir: str | Path) -> dict[str, Path]:
    """Conventional Lance index paths under a single `lance_dir`."""
    base = Path(lance_dir)
    return {
        "train": base / "train.lance",
        "valid": base / "valid.lance",
        "test":  base / "test.lance",
    }


def resolve_input_paths(repro_cfg: dict[str, Any],
                        lance_dir: str | Path,
                        prefer_lance: bool) -> dict[str, list[str]]:
    """For each split, return ``[<lance_dir>/<split>.lance]`` if it exists and
    ``prefer_lance`` is True, else fall back to the source MGF in repro_cfg.

    casanovo's _get_input_paths short-circuits on .lance suffixes, so passing
    the prebuilt lance dir avoids any MGF re-parsing on the GPU node.
    """
    lance = lance_paths(lance_dir)
    out: dict[str, list[str]] = {}
    pairs = [
        ("train", "train_mgf", lance["train"]),
        ("valid", "val_mgf",   lance["valid"]),
        ("test",  "test_mgf",  lance["test"]),
    ]
    for split, mgf_key, lance_path in pairs:
        if prefer_lance and lance_path.exists():
            out[split] = [str(lance_path)]
        else:
            mgf = repro_cfg.get(mgf_key)
            if not mgf:
                continue
            # Per-split MGF paths may be a single string or a list of strings
            # (the new_match_15M datasets fan out across many shards).
            if isinstance(mgf, list):
                out[split] = [str(p) for p in mgf]
            else:
                out[split] = [str(mgf)]
    return out
