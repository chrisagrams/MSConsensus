"""Training entry point: wraps casanovo.denovo.ModelRunner with W&B logging
and Lightning checkpoint resume.

Casanovo's ``ModelRunner`` doesn't expose a way to:

  1. Add an extra Lightning logger (we want W&B alongside the built-in CSV).
  2. Save a periodic ``last.ckpt`` (upstream only saves end-of-epoch + best-
     by-val, neither of which makes mid-run resume work for a 1-epoch
     training schedule).
  3. Pass ``ckpt_path`` into ``Trainer.fit`` so a re-launch resumes from
     ``last.ckpt`` instead of starting over.

We solve all three with a thin ``ModelRunner`` subclass that:
  - appends a ``ModelCheckpoint(save_last=True, every_n_train_steps=N)``
    callback to ``self.callbacks`` in ``__init__`` (before
    ``initialize_trainer`` copies callbacks into the Trainer);
  - attaches a ``WandbLogger`` to ``self.trainer._loggers`` after
    ``initialize_trainer`` runs;
  - replicates ``ModelRunner.train``'s body but passes ``ckpt_path`` to
    ``trainer.fit`` so a re-launch with an existing ``last.ckpt`` continues
    from the last optimizer state, not from scratch.

Autoresume is opt-in via ``_repro.autoresume: true`` in the per-dataset
overlay. When enabled, ``<output_dir>/last.ckpt`` (if present) is loaded and
``trainer.fit`` continues from it; PBS just re-submits on walltime, no flags
or manual cleanup needed. When disabled (the default), every launch starts
from scratch even if a stale ``last.ckpt`` is on disk — the right behavior
for smoke runs and one-off experiments.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("casanovo_repro.train")

# How often to save last.ckpt during training (in optimizer steps). 10k steps
# at our ~400 steps/min estimate is ~25 min between checkpoints, which bounds
# the work lost on a wall-hit. Cheap on disk: ~600 MB per ckpt, atomically
# overwritten in place.
_RESUME_CKPT_EVERY_N_STEPS = 10_000

# DDP-averaged train_CELoss is emitted by _TrainStepLossLogger at the
# `log_every_n_steps` cadence read from the casanovo config — same knob
# Lightning already uses for its CSV/progress-bar logging. Smoke runs
# (log_every_n_steps=25) get a dense curve in minutes; paper runs
# (log_every_n_steps=50) get a dense curve over the full Polaris job.
# Casanovo's built-in train_loss logger fires only on_epoch=True, so without
# this callback W&B would only see one train_loss per epoch.


def _build_wandb_logger(repro_cfg: dict[str, Any], output_dir: Path):
    from lightning.pytorch.loggers import WandbLogger
    wb = repro_cfg.get("wandb") or {}
    return WandbLogger(
        project=wb.get("project", "msbench-casanovo"),
        name=wb.get("run_name"),
        save_dir=str(output_dir),
        log_model=False,        # checkpoints are large; keep them on Eagle, not W&B
        # `reinit=True` lets a re-launched run reattach to the same W&B run id
        # (id is set by the PBS script via WANDB_RUN_ID) so resume looks like
        # one continuous run in the W&B UI.
        reinit=True,
    )


def _patched_runner_cls():
    """Build a ModelRunner subclass that injects:
      - a periodic save_last ModelCheckpoint
      - a WandbLogger
      - ckpt_path resumption in train()

    Done lazily so that ``import casanovo_repro.train`` doesn't import
    casanovo unless training is actually invoked.
    """
    from casanovo.denovo.model_runner import ModelRunner  # noqa: WPS433
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    import lightning.pytorch as pl
    import torch.distributed as dist

    def _ddp_avg(loss):
        """All-reduce a scalar loss tensor across ranks and return its mean."""
        loss = loss.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
        return loss

    class _TrainStepLossLogger(Callback):
        """Log the DDP-averaged training loss every N steps to all attached
        loggers. Bypasses Spec2Pep's on_step=False/on_epoch=True choice
        without modifying its training_step. All ranks participate in the
        all-reduce; only rank 0 calls log_metrics."""
        def __init__(self, every_n_steps: int):
            self.every_n_steps = every_n_steps

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step == 0 or step % self.every_n_steps != 0:
                return
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = _ddp_avg(loss)
            if trainer.global_rank != 0:
                return
            for lg in (trainer.loggers or []):
                lg.log_metrics(
                    {"train_loss_step": float(loss)},
                    step=step,
                )

    class _StepLogger(Callback):
        """Replace casanovo's broken `_log_history` console output.

        Casanovo logs `Step | Train loss | Valid loss` from
        ``Spec2Pep.on_validation_epoch_end`` -> ``_log_history``, but
        ``train_CELoss`` is registered with ``on_step=False, on_epoch=True``
        and only lands in ``trainer.callback_metrics`` at end-of-train-epoch.
        Mid-epoch (every val_check_interval), the train field defaults to
        ``np.nan`` -> the printout is misleading.

        This callback maintains a DDP-averaged running mean of train_CELoss
        across each val window and, at ``on_validation_end`` (after the
        validation metric is fresh in callback_metrics), emits a clean
        ``step | train | valid | lr`` line on rank 0. The casanovo
        text logger is silenced separately by raising ``model.n_log`` to a
        sentinel post-init.
        """
        def __init__(self):
            self._train_sum = 0.0
            self._train_n = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = _ddp_avg(loss)
            if trainer.global_rank != 0:
                return
            self._train_sum += float(loss)
            self._train_n += 1

        def on_validation_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            cm = trainer.callback_metrics
            valid_t = cm.get("valid_CELoss")
            valid = float(valid_t) if valid_t is not None else float("nan")
            train = (self._train_sum / self._train_n) if self._train_n else float("nan")
            try:
                lr = trainer.optimizers[0].param_groups[0]["lr"]
            except (IndexError, KeyError, AttributeError):
                lr = float("nan")
            logger.info(
                "step=%d  train=%.4f  valid=%.4f  lr=%.2e  (avg over %d train batches since last val)",
                trainer.global_step, train, valid, lr, self._train_n,
            )
            self._train_sum = 0.0
            self._train_n = 0

    class _ReproModelRunner(ModelRunner):
        wandb_logger = None   # populated by run() before .train()
        resume_ckpt = None    # populated by run() before .train()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Casanovo's built-in checkpoints (best + end-of-epoch) don't
            # produce a last.ckpt usable for mid-epoch resume. Add one. The
            # callback writes <output_dir>/last.ckpt and overwrites it in
            # place every N training steps.
            self.callbacks.append(
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    save_top_k=0,
                    save_last=True,
                    every_n_train_steps=_RESUME_CKPT_EVERY_N_STEPS,
                    save_on_train_epoch_end=False,
                    enable_version_counter=False,
                )
            )
            # Per-epoch checkpoint preservation (Option B). Casanovo's built-in
            # end-of-epoch ModelCheckpoint already writes "{rootname}.{epoch}-
            # {step}.ckpt", but it defaults to save_top_k=1, which deletes
            # earlier epochs. Bumping it to -1 keeps every epoch's weights so
            # we can compare epoch-1 vs epoch-2 after training. We mutate the
            # existing callback in place instead of adding a second one,
            # because Lightning forbids two ModelCheckpoint callbacks with the
            # same state_key (and a fresh save_on_train_epoch_end=True
            # callback would collide with this one).
            for cb in self.callbacks:
                if (
                    isinstance(cb, ModelCheckpoint)
                    and cb.monitor is None
                    and cb._save_on_train_epoch_end is True
                    and cb.save_top_k != 0  # don't touch our save_last cb
                ):
                    cb.save_top_k = -1
                    break
            self.callbacks.append(
                _TrainStepLossLogger(
                    every_n_steps=self.config.log_every_n_steps,
                )
            )
            self.callbacks.append(_StepLogger())

        def initialize_model(self, train: bool, db_search: bool = False) -> None:
            super().initialize_model(train=train, db_search=db_search)
            # Suppress casanovo's _log_history text logger. It prints
            # `Step | Train loss | Valid loss` from on_validation_epoch_end
            # but train_CELoss is only populated at end-of-train-epoch, so
            # mid-epoch lines have train=nan. _StepLogger above replaces it
            # with a correct rank-0 line at every val_check. n_log is only
            # used at model.py:1286 in `step % n_log == 0`; a value larger
            # than any realistic step count silences the print without other
            # side effects.
            self.model.n_log = 10**12

        def initialize_trainer(self, train: bool) -> None:
            super().initialize_trainer(train)
            if train and self.wandb_logger is not None:
                existing = list(getattr(self.trainer, "_loggers", []) or [])
                existing.append(self.wandb_logger)
                self.trainer._loggers = existing
                logger.info("attached WandbLogger to Trainer (n_loggers=%d)",
                            len(existing))

        def train(self, train_peak_path, valid_peak_path) -> None:
            """Mirror upstream ModelRunner.train but pass ckpt_path to fit()."""
            self.initialize_trainer(train=True)
            self.initialize_tokenizer()
            self.initialize_model(train=True)
            train_paths = self._get_input_paths(train_peak_path, True, "train")
            valid_paths = self._get_input_paths(valid_peak_path, True, "valid")
            self.initialize_data_module(train_paths, valid_paths)
            self.loaders.setup()

            ckpt_path = self.resume_ckpt
            if ckpt_path:
                logger.info("resuming training from %s", ckpt_path)
            else:
                logger.info("no last.ckpt found; training from scratch")

            self.trainer.fit(
                self.model,
                self.loaders.train_dataloader(),
                self.loaders.val_dataloader(),
                ckpt_path=ckpt_path,
            )

    return _ReproModelRunner


def _detect_resume_ckpt(output_dir: Path) -> str | None:
    """Return the path to last.ckpt if it exists, else None."""
    last = output_dir / "last.ckpt"
    return str(last) if last.exists() else None


def run(casanovo_cfg: dict[str, Any], repro_cfg: dict[str, Any]) -> None:
    """Execute one training run. Auto-resumes from <output_dir>/last.ckpt
    if present."""
    from casanovo.config import Config

    from .data import (
        lance_paths,
        resolve_input_paths,
        write_casanovo_config,
    )

    output_dir = Path(repro_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize the merged casanovo config to disk so casanovo.config.Config
    # can ingest it. Place it next to the run's checkpoints for provenance.
    cfg_path = write_casanovo_config(casanovo_cfg, output_dir / "config_used.yaml")

    config = Config(str(cfg_path))

    lance_dir = casanovo_cfg.get("lance_dir")
    if not lance_dir:
        raise ValueError(
            "lance_dir must be set; preprocess job builds the Lance index "
            "before training can start."
        )
    lance = lance_paths(lance_dir)
    if not lance["train"].exists() or not lance["valid"].exists():
        raise FileNotFoundError(
            f"missing prebuilt Lance index under {lance_dir}; "
            "run `casanovo-repro preprocess --config <overlay>` first."
        )

    paths = resolve_input_paths(repro_cfg, lance_dir, prefer_lance=True)
    train_paths = paths.get("train")
    valid_paths = paths.get("valid")
    if not train_paths or not valid_paths:
        raise FileNotFoundError("could not resolve train/valid Lance paths")

    runner_cls = _patched_runner_cls()
    wandb_logger = _build_wandb_logger(repro_cfg, output_dir)
    # Autoresume is opt-in: only consult <output_dir>/last.ckpt when the
    # per-dataset overlay sets ``_repro.autoresume: true`` (production runs
    # that need walltime recovery). Smoke runs and one-off experiments leave
    # this unset, so they always start from scratch even if a stale
    # last.ckpt is on disk from an earlier session.
    autoresume = bool(repro_cfg.get("autoresume", False))
    if autoresume:
        resume_ckpt = _detect_resume_ckpt(output_dir)
    else:
        resume_ckpt = None
        stale = output_dir / "last.ckpt"
        if stale.exists():
            logger.info(
                "autoresume disabled in config; ignoring existing %s", stale,
            )

    with runner_cls(
        config=config,
        model_filename=None,            # train from scratch (or resume via ckpt_path)
        output_dir=output_dir,
        output_rootname=repro_cfg.get("dataset"),
        overwrite_ckpt_check=False,     # tolerate re-runs into the same dir
    ) as runner:
        runner.wandb_logger = wandb_logger
        runner.resume_ckpt = resume_ckpt
        logger.info(
            "starting train: dataset=%s train=%s valid=%s output=%s resume=%s",
            repro_cfg.get("dataset"), train_paths, valid_paths,
            output_dir, resume_ckpt,
        )
        runner.train(train_paths, valid_paths)

    # WandbLogger needs an explicit finish under DDP to flush.
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.finish()
    except Exception:  # noqa: BLE001
        logger.exception("wandb.finish() failed (non-fatal)")
