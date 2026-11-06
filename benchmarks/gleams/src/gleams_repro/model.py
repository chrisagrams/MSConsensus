from __future__ import annotations

from gleams.nn import embedder as _embedder


def build_embedder(
    lr: float,
    filename: str,
    num_precursor_features: int = 61,
    num_fragment_features: int = 2449,
    num_ref_spectra_features: int = 500,
) -> _embedder.Embedder:
    """Instantiate the upstream GLEAMS ``Embedder``. Call ``.build()`` to
    populate ``siamese_model`` (under MirroredStrategy, compiled with
    RectifiedAdam + contrastive_loss).
    """
    return _embedder.Embedder(
        num_precursor_features=num_precursor_features,
        num_fragment_features=num_fragment_features,
        num_ref_spectra_features=num_ref_spectra_features,
        lr=lr,
        filename=filename,
    )


def single_arm_model(embedder: _embedder.Embedder):
    """Return the single-spectrum encoder (shared weights of the Siamese)."""
    return embedder.siamese_model.get_layer("embedder")


contrastive_loss = _embedder.contrastive_loss
