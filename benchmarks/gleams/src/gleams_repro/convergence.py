from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss
import tensorflow as tf
import wandb


class DownstreamEval(tf.keras.callbacks.Callback):
    """Per-epoch held-out top-1 peptide accuracy via Euclidean NN search.

    Samples ``num_query`` queries and ``num_library`` library spectra from the
    processed ``split``, embeds each with the single-arm encoder, and for each
    query records whether the nearest library embedding shares the query peptide
    (after I/L equivalencing). Logs ``eval/top1_acc`` to W&B.
    """

    def __init__(
        self,
        embedder,
        processed_dir: str | Path,
        split: str,
        num_query: int,
        num_library: int,
        eval_every_n_epochs: int,
        feature_split: tuple[int, int],
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.num_query = num_query
        self.num_library = num_library
        self.eval_every_n_epochs = eval_every_n_epochs
        self.feature_split = feature_split
        self.batch_size = batch_size
        self.seed = seed
        self._prepared = False

    def _prepare(self) -> None:
        feats = ss.load_npz(str(self.processed_dir / f"{self.split}_features.npz"))
        meta = pd.read_parquet(str(self.processed_dir / f"{self.split}_metadata.parquet"))
        n = feats.shape[0]
        want = self.num_query + self.num_library
        if want > n:
            raise ValueError(f"need {want} rows for convergence eval, only {n} available in {self.split}")
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        q_idx, lib_idx = idx[: self.num_query], idx[self.num_query : want]
        self._q_feats = feats[q_idx]
        self._lib_feats = feats[lib_idx]
        self._q_seq = meta["sequence"].iloc[q_idx].str.replace("I", "L").to_numpy()
        self._lib_seq = meta["sequence"].iloc[lib_idx].str.replace("I", "L").to_numpy()
        self._prepared = True

    def _embed(self, feats: ss.csr_matrix) -> np.ndarray:
        i1, i2 = self.feature_split
        x_prec = feats[:, :i1].toarray()
        x_frag = feats[:, i1:i2].toarray()
        x_ref = feats[:, i2:].toarray()
        single_arm = self.embedder.siamese_model.get_layer("embedder")
        return single_arm.predict(
            [x_prec, x_frag, x_ref], batch_size=self.batch_size, verbose=0
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if (epoch + 1) % self.eval_every_n_epochs != 0:
            return
        if not self._prepared:
            self._prepare()

        q_emb = self._embed(self._q_feats).astype(np.float32)
        lib_emb = self._embed(self._lib_feats).astype(np.float32)
        # ||q - l||^2 = ||q||^2 + ||l||^2 - 2 q·l — compute in chunks to cap memory.
        lib_sq = (lib_emb * lib_emb).sum(axis=1)
        correct = 0
        chunk = 512
        for s in range(0, len(q_emb), chunk):
            q = q_emb[s : s + chunk]
            q_sq = (q * q).sum(axis=1, keepdims=True)
            d2 = q_sq + lib_sq[None, :] - 2.0 * q @ lib_emb.T
            nn = np.argmin(d2, axis=1)
            correct += int((self._lib_seq[nn] == self._q_seq[s : s + chunk]).sum())
        acc = correct / len(q_emb)
        wandb.log({"eval/top1_acc": float(acc), "epoch": int(epoch)})
