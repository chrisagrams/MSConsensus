from __future__ import annotations

import time

import tensorflow as tf
import wandb


class WandbConvergence(tf.keras.callbacks.Callback):
    """Logs train loss per step, val loss per charge per epoch, and wallclock.

    Pairs with ``convergence.DownstreamEval`` which logs top-1 peptide accuracy.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._t0: float | None = None

    def on_train_begin(self, logs=None) -> None:
        self._t0 = time.time()

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        if logs is None or batch % self.log_every_n_steps != 0:
            return
        wandb.log(
            {
                "train/loss": float(logs.get("loss", float("nan"))),
                "train/step": int(self.model.optimizer.iterations.numpy()),
                "train/wallclock_s": time.time() - (self._t0 or time.time()),
            }
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if logs is None:
            return
        payload = {f"epoch/{k}": float(v) for k, v in logs.items()}
        payload["epoch"] = epoch
        payload["epoch/wallclock_s"] = time.time() - (self._t0 or time.time())
        wandb.log(payload)
