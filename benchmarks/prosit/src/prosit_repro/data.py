"""Thin wrapper around DLOmix's FragmentIonIntensityDataset for our YAML configs.

Centralizes the dataset construction so train.py / convergence.py share the
exact same loader configuration (alphabet, padding, meta features).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Activate the PyTorch backend before importing dlomix.
os.environ.setdefault("DLOMIX_BACKEND", "torch")

from dlomix.data import FragmentIonIntensityDataset  # noqa: E402
from dlomix.data.dataset_utils import EncodingScheme  # noqa: E402


def _resolve_parquet(path: Optional[str]) -> Optional[str | list[str]]:
    """HF datasets needs a file path or a list. Expand a directory of shards."""
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        shards = sorted(str(s) for s in p.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no .parquet shards in {p}")
        return shards
    return str(p)


def _build_dataloader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build torch DataLoader kwargs from the hardware section.

    Default behavior is conservative (no workers) because Prosit's small forward
    pass means data IO can saturate the GPU; bump num_workers when you see GPU
    util well below 100%.
    """
    hw = cfg.get("hardware", {}) or {}
    nw = int(hw.get("num_workers", 0))
    out: dict[str, Any] = {}
    if nw > 0:
        out["num_workers"] = nw
        # persistent_workers avoids re-spawning between epochs; prefetch_factor
        # determines how many batches each worker buffers ahead.
        out["persistent_workers"] = True
        out["prefetch_factor"] = int(hw.get("prefetch_factor", 4))
    if bool(hw.get("pin_memory", True)):
        out["pin_memory"] = True
    return out


def build_dataset(cfg: dict[str, Any]) -> FragmentIonIntensityDataset:
    """Construct the DLOmix dataset from a parsed YAML config."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    kwargs: dict[str, Any] = dict(
        data_source=_resolve_parquet(data_cfg["train_parquet"]),
        val_data_source=_resolve_parquet(data_cfg.get("val_parquet")),
        test_data_source=_resolve_parquet(data_cfg.get("test_parquet")),
        data_format="parquet",
        sequence_column=data_cfg.get("sequence_column", "modified_sequence"),
        label_column=data_cfg.get("label_column", "intensities_raw"),
        max_seq_len=cfg["model"].get("seq_length", 30),
        dataset_type="pt",
        batch_size=train_cfg.get("batch_size", 1024),
        shuffle=True,
        model_features=list(data_cfg.get("model_features", [])),
        with_termini=cfg["model"].get("with_termini", False),
        alphabet=None,  # learn from data
        # naive-mods adds (residue, ptm) combos as distinct vocab tokens — required
        # to preserve C[UNIMOD:4] / M[UNIMOD:35] etc. UNMOD strips them silently.
        encoding_scheme=EncodingScheme.NAIVE_MODS,
        num_proc=train_cfg.get("dataset_num_proc", -1),
        torch_dataloader_kwargs=_build_dataloader_kwargs(cfg),
    )
    if data_cfg.get("val_ratio") is not None:
        kwargs["val_ratio"] = float(data_cfg["val_ratio"])

    return FragmentIonIntensityDataset(**kwargs)
