"""Subset selection: AllPositive vs UniqueInchikey."""
from __future__ import annotations

import logging
from collections import defaultdict

from matchms import Spectrum

logger = logging.getLogger(__name__)


def get_inchikey14(spectrum: Spectrum) -> str | None:
    inchikey = spectrum.get("inchikey")
    if not isinstance(inchikey, str):
        return None
    head = inchikey.split("-")[0]
    return head if len(head) == 14 else None


def select_unique_inchikey(spectra: list[Spectrum]) -> list[Spectrum]:
    """One spectrum per unique planar InChIKey.

    Within each group, picks the spectrum with the most peaks; ties broken
    deterministically by spectrum_id.
    """
    groups: dict[str, list[Spectrum]] = defaultdict(list)
    dropped = 0
    for s in spectra:
        ik = get_inchikey14(s)
        if ik is None:
            dropped += 1
            continue
        groups[ik].append(s)

    reps: list[Spectrum] = []
    for group in groups.values():
        group.sort(key=lambda s: (-len(s.peaks), s.get("spectrum_id") or ""))
        reps.append(group[0])

    logger.info(
        "UniqueInchikey: %d groups from %d spectra (dropped %d w/o valid inchikey)",
        len(reps),
        len(spectra),
        dropped,
    )
    return reps


def select_all_positive(spectra: list[Spectrum]) -> list[Spectrum]:
    return list(spectra)
