"""SpectrumDocument with loss tokens restored.

spec2vec >= 0.9 removed loss support from ``SpectrumDocument``; the paper
(Huber et al., 2021) trained on peaks *and* neutral losses. This subclass
re-adds ``loss@<mz>`` words computed as ``parent_mass - peak_mz``, filtered
to the paper's default range (5.0 – 200.0 Da).
"""
from __future__ import annotations

import numpy as np
from matchms import Spectrum
from spec2vec import SpectrumDocument


class SpectrumDocumentWithLosses(SpectrumDocument):
    def __init__(
        self,
        spectrum: Spectrum,
        n_decimals: int = 2,
        loss_mz_from: float = 5.0,
        loss_mz_to: float = 200.0,
    ):
        self.loss_mz_from = loss_mz_from
        self.loss_mz_to = loss_mz_to
        super().__init__(spectrum, n_decimals=n_decimals)

    def _loss_arrays(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        pm = self._obj.get("parent_mass")
        if pm is None:
            return None, None
        losses = float(pm) - self._obj.peaks.mz
        mask = (losses >= self.loss_mz_from) & (losses <= self.loss_mz_to)
        return losses[mask], mask

    def _make_words(self):
        n = self.n_decimals
        peak_words = [f"peak@{mz:.{n}f}" for mz in self._obj.peaks.mz]
        loss_mzs, _ = self._loss_arrays()
        loss_words = (
            [] if loss_mzs is None else [f"loss@{mz:.{n}f}" for mz in loss_mzs]
        )
        self.words = peak_words + loss_words
        return self

    def _add_weights(self):
        assert self._obj.peaks.intensities.max() <= 1, "peak intensities not normalized"
        peak_int = self._obj.peaks.intensities.tolist()
        _, mask = self._loss_arrays()
        loss_int = (
            [] if mask is None else self._obj.peaks.intensities[mask].tolist()
        )
        self.weights = peak_int + loss_int
        return self
