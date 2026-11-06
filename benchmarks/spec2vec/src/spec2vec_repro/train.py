"""Spec2Vec Word2Vec training — paper hyperparameters via gensim."""
from __future__ import annotations

import logging
from pathlib import Path

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def build_word2vec(cfg: dict) -> Word2Vec:
    """Build an untrained Word2Vec with paper defaults.

    gensim decays alpha linearly from ``alpha`` to ``min_alpha`` across all
    training epochs; we compute ``min_alpha`` from ``alpha_decay`` to match
    spec2vec's ``set_spec2vec_defaults``.
    """
    m = cfg["model"]
    min_alpha = m["alpha"] - m["alpha_decay"] * m["epochs"]
    return Word2Vec(
        vector_size=m["vector_size"],
        window=m["window"],
        min_count=m["min_count"],
        sg=m["sg"],
        negative=m["negative"],
        workers=m["workers"],
        alpha=m["alpha"],
        min_alpha=min_alpha,
    )


def train_model(
    model: Word2Vec,
    documents: list,
    epochs: int,
    callbacks: list | None = None,
) -> Word2Vec:
    model.build_vocab(documents)
    model.train(
        corpus_iterable=documents,
        total_examples=len(documents),
        epochs=epochs,
        compute_loss=True,
        callbacks=callbacks or [],
    )
    return model
