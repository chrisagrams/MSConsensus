"""PyTorch training loop for Prosit (DLOmix backend).

- Adam(lr=cfg.optim.learning_rate), grad clip, ReduceLROnPlateau on val loss.
- Loss: dlomix.losses.intensity_torch.masked_spectral_distance (Prosit's loss).
- Per-epoch validation + per-charge spectral-angle convergence eval (W&B).
- Early stopping on val_loss with configurable patience.
- Checkpoints best (val_loss) + last per-epoch.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("DLOMIX_BACKEND", "torch")

import torch
import yaml
from dlomix.losses.intensity_torch import masked_spectral_distance

from .data import build_dataset
from .model import build_model
from .convergence import evaluate
from .wandb_callback import WandbLogger

logger = logging.getLogger("prosit_repro.train")


def _load(config_path: str) -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _device(cfg: dict[str, Any]) -> torch.device:
    pref = (cfg.get("hardware", {}) or {}).get("device", "auto")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config_path: str) -> None:
    cfg = _load(config_path)
    train_cfg = cfg["training"]
    optim_cfg = cfg["optim"]
    log_cfg = cfg.get("logging", {}) or {}
    conv_cfg = cfg.get("convergence", {}) or {}

    _seed_all(int(train_cfg.get("seed", 42)))
    device = _device(cfg)
    logger.info("device=%s", device)

    Path(log_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(log_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("building dataset")
    ds = build_dataset(cfg)
    logger.info("alphabet size: %d", len(ds.extended_alphabet))

    logger.info("building model")
    model = build_model(cfg, ds.extended_alphabet).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_cfg["learning_rate"]))
    sch_cfg = optim_cfg.get("scheduler", {}) or {}
    if sch_cfg.get("name") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 5)),
            min_lr=float(sch_cfg.get("min_lr", 1e-7)),
        )
    else:
        scheduler = None

    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0)) or None
    label_col = cfg["data"].get("label_column", "intensities_raw")
    n_epochs = int(train_cfg["num_epochs"])
    es_patience = int(train_cfg.get("early_stop_patience", 0))

    wb = WandbLogger(cfg)
    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    best_val = float("inf")
    epochs_since_improve = 0
    global_step = 0
    t0 = time.time()

    try:
        for epoch in range(n_epochs):
            model.train()
            ep_t0 = time.time()
            ep_loss_sum = 0.0
            ep_n = 0
            for batch in ds.tensor_train_data:
                batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(batch)
                loss = masked_spectral_distance(batch[label_col], y_pred)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                                   error_if_nonfinite=False)
                optimizer.step()
                global_step += 1
                ep_loss_sum += float(loss.item())
                ep_n += 1
                wb.log_step(loss.item(), global_step)

            train_loss = ep_loss_sum / max(1, ep_n)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.inference_mode():
                for batch in ds.tensor_val_data:
                    batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
                    y_pred = model(batch)
                    val_loss_sum += float(masked_spectral_distance(batch[label_col], y_pred).item())
                    val_n += 1
            val_loss = val_loss_sum / max(1, val_n)

            payload: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_wallclock_s": time.time() - ep_t0,
            }

            if conv_cfg.get("enabled") and (epoch % int(conv_cfg.get("eval_every_n_epochs", 1)) == 0):
                eval_metrics = evaluate(
                    model, ds.tensor_val_data, device,
                    label_column=label_col,
                    per_charge_charges=(conv_cfg.get("charges") if conv_cfg.get("per_charge") else None),
                )
                payload.update({f"val_{k}": v for k, v in eval_metrics.items()})

            wb.log_epoch(epoch, payload)
            logger.info("epoch %d train_loss=%.4f val_loss=%.4f wallclock=%.0fs",
                        epoch, train_loss, val_loss, time.time() - t0)

            # Schedule + early stop
            if scheduler is not None:
                scheduler.step(val_loss)
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_since_improve = 0
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                logger.info("new best val_loss=%.4f -> best.pt", best_val)
            else:
                epochs_since_improve += 1
                if es_patience and epochs_since_improve >= es_patience:
                    logger.info("early stop: no val_loss improvement in %d epochs", es_patience)
                    break

        torch.save(model.state_dict(), ckpt_dir / "final.pt")
        logger.info("training done. best val_loss=%.4f, total wallclock=%.0fs",
                    best_val, time.time() - t0)

        # Held-out test eval (if test set provided)
        if cfg["data"].get("test_parquet"):
            logger.info("evaluating held-out test set")
            test_metrics = evaluate(
                model, ds.tensor_test_data, device,
                label_column=label_col,
                per_charge_charges=(conv_cfg.get("charges") if conv_cfg.get("per_charge") else None),
            )
            logger.info("test metrics: %s", test_metrics)
            wb.log_epoch(epoch=99999, payload={f"test_{k}": v for k, v in test_metrics.items()})

    finally:
        wb.finish()
