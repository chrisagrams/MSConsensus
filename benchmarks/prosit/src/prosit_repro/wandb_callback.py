"""Lightweight W&B logger for the PyTorch training loop.

Pairs with train.py and convergence.py: train.py calls log_step / log_epoch
on the loop; convergence.py calls log_epoch with eval payloads. wandb_mode is
honored from the config so smoke runs stay offline.
"""
from __future__ import annotations

import os
import time
from typing import Any

import wandb


class WandbLogger:
    def __init__(self, cfg: dict[str, Any]) -> None:
        log_cfg = cfg.get("logging", {}) or {}
        self.mode = log_cfg.get("wandb_mode", "online")
        self.log_every_n_steps = int(log_cfg.get("log_every_n_steps", 50))
        self._t0 = time.time()
        self._step = 0
        self._enabled = self.mode != "disabled"

        if self._enabled:
            os.environ.setdefault("WANDB_MODE", self.mode)
            self._run = wandb.init(
                project=log_cfg.get("wandb_project", "msbench-prosit"),
                entity=log_cfg.get("wandb_entity"),
                name=cfg.get("run_name"),
                config=cfg,
                mode=self.mode,
                dir=log_cfg.get("log_dir"),
            )
        else:
            self._run = None

    def log_step(self, loss: float, step: int) -> None:
        self._step = step
        if not self._enabled:
            return
        if step % self.log_every_n_steps != 0:
            return
        wandb.log({
            "train/loss": float(loss),
            "train/step": int(step),
            "train/wallclock_s": time.time() - self._t0,
        }, step=step)

    def log_epoch(self, epoch: int, payload: dict[str, float]) -> None:
        if not self._enabled:
            return
        out = {f"epoch/{k}": float(v) for k, v in payload.items()}
        out["epoch"] = epoch
        out["epoch/wallclock_s"] = time.time() - self._t0
        wandb.log(out, step=self._step)

    def finish(self) -> None:
        if self._enabled and self._run is not None:
            wandb.finish()
