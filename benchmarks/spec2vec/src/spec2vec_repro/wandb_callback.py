"""gensim training callback: per-epoch loss, wallclock, W&B logging, periodic eval."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import wandb
from gensim.models.callbacks import CallbackAny2Vec

from spec2vec_repro.convergence import score_convergence

logger = logging.getLogger(__name__)


class SpecTrainCallback(CallbackAny2Vec):
    def __init__(
        self,
        documents: list,
        holdout: dict,
        checkpoint_dir: str | Path,
        eval_every_n_epochs: int = 1,
        run_name: str = "run",
    ):
        self.documents = documents
        self.holdout = holdout
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.checkpoint_dir.parent / "history.jsonl"
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))
        self.run_name = run_name

        self.epoch = 0
        self.loss_prev = 0.0
        self.train_start: float | None = None
        self.epoch_start: float | None = None

    def on_train_begin(self, model):
        self.train_start = time.monotonic()
        self.epoch_start = self.train_start

    def on_epoch_begin(self, model):
        self.epoch_start = time.monotonic()

    def on_epoch_end(self, model):
        now = time.monotonic()
        self.epoch += 1
        epoch_sec = now - self.epoch_start
        wall_sec = now - self.train_start

        loss_cum = float(model.get_latest_training_loss())
        loss_delta = loss_cum - self.loss_prev
        self.loss_prev = loss_cum

        log = {
            "epoch": self.epoch,
            "loss_cumulative": loss_cum,
            "loss_delta": loss_delta,
            "epoch_sec": epoch_sec,
            "wall_sec": wall_sec,
        }

        pearson_str = "skip"
        if self.epoch % self.eval_every_n_epochs == 0 or self.epoch == 1:
            eval_start = time.monotonic()
            metrics = score_convergence(model, self.documents, self.holdout)
            log["eval/pearson"] = metrics["pearson"]
            log["eval/top_bin_mean"] = metrics["top_bin_mean"]
            log["eval/sec"] = time.monotonic() - eval_start
            pearson_str = f"{metrics['pearson']:.3f}"

            ckpt = self.checkpoint_dir / f"{self.run_name}_epoch{self.epoch:03d}.model"
            model.save(str(ckpt))

        wandb.log(log, step=self.epoch)
        with self.history_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        logger.info(
            "epoch=%d loss_delta=%.1f epoch_sec=%.1f wall_sec=%.1f pearson=%s",
            self.epoch, loss_delta, epoch_sec, wall_sec, pearson_str,
        )
