"""Build the PyTorch PrositIntensityPredictor from our YAML config.

Unlike DLOmix's reference run script, we wire ``meta_data_keys`` so the model
sees collision energy and precursor charge — required for paper fidelity.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("DLOMIX_BACKEND", "torch")

from dlomix.models import PrositIntensityPredictor  # noqa: E402


def build_model(cfg: dict[str, Any], extended_alphabet: dict[str, int]) -> PrositIntensityPredictor:
    m = cfg["model"]
    meta_keys: dict[str, str] = {}
    if m.get("use_meta_data", True):
        # The keys here are DLOmix's internal labels; the values are the
        # column/feature names the dataloader emits in each batch dict.
        feature_names = set(cfg["data"].get("model_features", []))
        if "collision_energy_aligned_normed" in feature_names:
            meta_keys["COLLISION_ENERGY_KEY"] = "collision_energy_aligned_normed"
        if "precursor_charge_onehot" in feature_names:
            meta_keys["PRECURSOR_CHARGE_KEY"] = "precursor_charge_onehot"
        if not meta_keys:
            raise ValueError(
                "use_meta_data=True but no recognized meta features in "
                "data.model_features. Expected collision_energy_aligned_normed "
                "and/or precursor_charge_onehot."
            )

    return PrositIntensityPredictor(
        seq_length=m.get("seq_length", 30),
        with_termini=m.get("with_termini", False),
        use_meta_data=bool(meta_keys),
        use_prosit_ptm_features=m.get("use_prosit_ptm_features", False),
        embedding_output_dim=m.get("embedding_output_dim", 16),
        recurrent_layers_sizes=tuple(m.get("recurrent_layers_sizes", [256, 512])),
        regressor_layer_size=m.get("regressor_layer_size", 512),
        dropout_rate=m.get("dropout_rate", 0.2),
        latent_dropout_rate=m.get("latent_dropout_rate", 0.1),
        input_keys={"SEQUENCE_KEY": cfg["data"].get("sequence_column", "modified_sequence")},
        meta_data_keys=meta_keys or None,
        alphabet=extended_alphabet,
    )
