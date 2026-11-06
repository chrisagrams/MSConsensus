"""Per-epoch evaluation: median spectral angle on val/test, broken down by charge."""
from __future__ import annotations

import os
from typing import Any, Iterable

os.environ.setdefault("DLOMIX_BACKEND", "torch")

import numpy as np
import torch
import torch.nn.functional as F


def _spectral_angle(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-row spectral angle (0=identical, 1=orthogonal); masks y_true == -1 slots."""
    epsilon = 1e-7
    pred = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)
    pn = F.normalize(pred, p=2, dim=-1)
    tn = F.normalize(true, p=2, dim=-1)
    cos = (pn * tn).sum(dim=-1).clamp(-1.0 + epsilon, 1.0 - epsilon)
    return 2 * torch.arccos(cos) / np.pi


def _charge_from_onehot(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    oh = batch.get("precursor_charge_onehot")
    if oh is None:
        return None
    return torch.argmax(oh, dim=-1) + 1


@torch.inference_mode()
def evaluate(model, loader: Iterable[dict[str, torch.Tensor]], device: torch.device,
             label_column: str, per_charge_charges: list[int] | None) -> dict[str, float]:
    """Compute median spectral angle (overall + per charge) over a dataloader."""
    model.eval()
    sas: list[np.ndarray] = []
    charges: list[np.ndarray] = []

    for batch in loader:
        batch_dev = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        y_true = batch_dev[label_column]
        y_pred = model(batch_dev)
        sa = _spectral_angle(y_true, y_pred).detach().cpu().numpy()
        sas.append(sa)
        ch = _charge_from_onehot(batch_dev)
        charges.append(ch.detach().cpu().numpy() if ch is not None else np.zeros_like(sa))

    sa_all = np.concatenate(sas) if sas else np.array([])
    ch_all = np.concatenate(charges) if charges else np.array([])

    out: dict[str, float] = {}
    if sa_all.size:
        out["median_spectral_angle"] = float(np.median(sa_all))
        out["mean_spectral_angle"] = float(np.mean(sa_all))
        # Spectral angle similarity is conventionally reported as 1 - SA:
        out["median_spectral_angle_similarity"] = float(np.median(1.0 - sa_all))

    if per_charge_charges:
        for c in per_charge_charges:
            mask = ch_all == c
            if mask.any():
                out[f"median_spectral_angle_charge_{c}"] = float(np.median(sa_all[mask]))
                out[f"n_charge_{c}"] = int(mask.sum())
    return out
