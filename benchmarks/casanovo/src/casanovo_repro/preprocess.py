"""Preprocess: trigger casanovo's depthcharge-backed Lance build.

We don't reimplement MGF parsing; we instantiate the same DeNovoDataModule
that ModelRunner.train would build, then call .setup() once per split. That
materializes ``<lance_dir>/{train,valid,test}.lance`` on Eagle so the GPU job
can mmap them off node-local scratch with zero CPU-side data prep.

The source MGFs are assumed to already be decompressed on Eagle. For the
consensus dataset we additionally split the full train MGF into train_main +
val (5%, fixed seed) the first time this runs; subsequent runs skip if the
outputs are already present.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("casanovo_repro.preprocess")


def _ensure_mgf_alias(src: Path, dst: Path) -> None:
    """Casanovo's input-path filter only accepts files ending in .mgf/.lance,
    but the consensus test corpus on Eagle is named `concensus.mgf_test` (no
    `.mgf` suffix). Create a stable symlink so casanovo accepts the path
    without us having to rename or copy 13 GB of data."""
    if dst.exists() or dst.is_symlink():
        logger.info("skip alias (already present): %s -> %s", dst, src)
        return
    if not src.exists():
        raise FileNotFoundError(f"source MGF missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    logger.info("symlinked %s -> %s", dst, src)


def _split_consensus(repro_cfg: dict[str, Any]) -> None:
    """Split the pre-decompressed consensus full-train MGF into train_main+val."""
    train_main = Path(repro_cfg["train_mgf"])
    val = Path(repro_cfg["val_mgf"])
    if train_main.exists() and val.exists():
        logger.info("skip split (already present): %s, %s", train_main, val)
        return

    full_train = Path(repro_cfg["source_full_train"])
    if not full_train.exists():
        raise FileNotFoundError(f"source full-train MGF missing: {full_train}")

    from scripts.split_train_val import main as split_main  # type: ignore

    split_main(
        src=full_train,
        out_train=train_main,
        out_val=val,
        val_fraction=0.05,
        seed=20260429,
    )


def _consensus_prep(repro_cfg: dict[str, Any]) -> None:
    """Split full-train into train_main+val, alias the test file."""
    _split_consensus(repro_cfg)
    _ensure_mgf_alias(
        src=Path(repro_cfg["source_test"]),
        dst=Path(repro_cfg["test_mgf"]),
    )


_SPLIT_TO_MGF_KEY = {"train": "train_mgf", "valid": "val_mgf", "test": "test_mgf"}
_SPLIT_TO_LANCE = {"train": "train.lance", "valid": "valid.lance", "test": "test.lance"}


def _coerce_mgf_list(val: Any) -> list[Path]:
    """Allow per-split MGF paths to be either a single path string or a list
    of path strings (the new_match_15M datasets fan out across many shards)."""
    if val is None:
        return []
    if isinstance(val, (str, Path)):
        return [Path(val)]
    if isinstance(val, list):
        return [Path(p) for p in val]
    raise TypeError(f"expected str or list[str] for MGF paths, got {type(val).__name__}")


def _build_one_split(casanovo_cfg: dict[str, Any],
                     repro_cfg: dict[str, Any],
                     split: str) -> None:
    """Lance-build exactly one split for one dataset.

    Idempotent: if ``<lance_dir>/<split>.lance/`` already exists and is non-
    empty, returns immediately. depthcharge is single-threaded inside, so the
    intended use is to fan this out one-per-process across (dataset × split)
    pairs from the orchestrator.
    """
    if split not in _SPLIT_TO_MGF_KEY:
        raise ValueError(f"unknown split {split!r}; expected one of "
                         f"{tuple(_SPLIT_TO_MGF_KEY)}")

    dataset = repro_cfg.get("dataset", "?")
    tag = f"[{dataset}/{split}]"

    srcs = _coerce_mgf_list(repro_cfg.get(_SPLIT_TO_MGF_KEY[split]))
    if not srcs:
        logger.info("%s skip: no source MGF configured", tag)
        return
    missing = [p for p in srcs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{tag} {len(missing)} source MGF(s) missing, first: {missing[0]}"
        )

    # ``lance_dir`` in the merged casanovo cfg is the *training-time* path —
    # i.e. where the v5-normalized lance lives. The depthcharge preprocess
    # writes the *raw* lance (with the on-MGF residue strings like
    # ``C[57.0215]``); normalize_lance_to_v5.py rewrites raw -> v5 afterwards.
    # ``_repro.build_lance_dir``, if set, redirects the preprocess write so
    # raw and v5 outputs don't share a path. Falls back to lance_dir when
    # absent (the legacy single-step convention used pre-15M).
    build_dir = Path(repro_cfg.get("build_lance_dir") or casanovo_cfg["lance_dir"])
    casanovo_cfg = dict(casanovo_cfg)
    casanovo_cfg["lance_dir"] = str(build_dir)
    target = build_dir / _SPLIT_TO_LANCE[split]
    if target.exists() and any(target.iterdir()):
        logger.info("%s skip: lance already populated at %s", tag, target)
        return

    # Force single-device, CPU-friendly setup. Doing this on the input dict
    # is local to this process; the orchestrator passes the unmutated cfg.
    casanovo_cfg["accelerator"] = "cpu"
    casanovo_cfg["devices"] = 1

    from casanovo.config import Config

    from .data import write_casanovo_config

    out_dir = Path(repro_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_casanovo_config(
        casanovo_cfg, out_dir / f"config_preprocess_{split}.yaml",
    )
    config = Config(str(cfg_path))

    from casanovo.denovo.model_runner import ModelRunner

    with ModelRunner(
        config=config,
        model_filename=None,
        output_dir=out_dir,
        output_rootname=f"{dataset}_{split}",
        overwrite_ckpt_check=False,
    ) as runner:
        runner.initialize_trainer(train=False)
        runner.initialize_tokenizer()
        # Pass paths only for the requested split. DeNovoDataModule.setup()
        # (called with stage=None) iterates {train, valid, test} but skips
        # any whose paths are None, so we get exactly one Lance build per
        # invocation -> one process -> trivial fan-out across splits.
        kw = {"train_paths": None, "valid_paths": None, "test_paths": None}
        kw[f"{split}_paths"] = [str(p) for p in srcs]
        runner.initialize_data_module(**kw)
        if len(srcs) == 1:
            logger.info("%s building Lance from %s -> %s", tag, srcs[0], target)
        else:
            logger.info("%s building Lance from %d MGFs -> %s", tag, len(srcs), target)
        runner.loaders.setup()
        logger.info("%s Lance build done", tag)


def _verify_listed_mgfs(repro_cfg: dict[str, Any]) -> None:
    """For datasets whose MGFs are produced by an out-of-band step (the
    new_match_15M file/pepdisjoint variants), prep() just checks they all
    exist. The patch + reshard scripts are run by hand before this."""
    for k in ("train_mgf", "val_mgf", "test_mgf"):
        for p in _coerce_mgf_list(repro_cfg.get(k)):
            if not p.exists():
                raise FileNotFoundError(f"missing {k} entry = {p}")


def prep(repro_cfg: dict[str, Any]) -> None:
    """Dataset-level prep that has to run before any lance build.

    For consensus: split the full-train MGF into train_main + val and alias
    the test MGF. For massive_kb / consensus_15m_*: just verify the expected
    MGFs exist (the 15M datasets are pre-staged by the patch + reshard
    scripts). Idempotent. Single-threaded by design — fan-out happens at
    lance level.
    """
    dataset = repro_cfg.get("dataset")
    logger.info("prep: dataset=%s", dataset)
    if dataset == "consensus":
        _consensus_prep(repro_cfg)
    elif dataset == "massive_kb":
        for k in ("train_mgf", "val_mgf", "test_mgf"):
            p = Path(repro_cfg[k])
            if not p.exists():
                raise FileNotFoundError(f"missing {k} = {p}")
    elif dataset in {"consensus_15m_file", "consensus_15m_pepdisjoint"}:
        _verify_listed_mgfs(repro_cfg)
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def run(casanovo_cfg: dict[str, Any],
        repro_cfg: dict[str, Any],
        split: str) -> None:
    """Lance-build one or all splits. Assumes prep() has already run for
    consensus (i.e. train_main.mgf, val.mgf, test.mgf all exist on Eagle)."""
    dataset = repro_cfg.get("dataset")
    logger.info("preprocess: dataset=%s split=%s", dataset, split)

    if split == "all":
        # Train + valid only. Test is inference-time, not training-time, so
        # we don't lance-build it as part of the preprocess job. Add it back
        # via --split test on demand.
        for s in ("train", "valid"):
            _build_one_split(casanovo_cfg, repro_cfg, s)
    else:
        _build_one_split(casanovo_cfg, repro_cfg, split)
