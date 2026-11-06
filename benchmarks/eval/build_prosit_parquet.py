"""Annotate spectra.mgf into Prosit-format parquet (174-dim b/y intensity tensor).

Reuses the existing kusterlab-vendored chemistry pipeline at
``prosit/scripts/mgf_to_prosit_parquet.annotate``. Inputs:
  - SEQ= field in MSFragger notation (e.g. C[57.0215], M[15.9949])
  - CHARGE= field
  - peak m/z + intensity arrays
  - aligned_collision_energy = 28 (Q Exactive Plus NCE per mzML metadata)
  - mass analyzer = FTMS (Q Exactive Plus is FTMS, drives 25 ppm tolerance)

PSMs with mods the prosit pipeline can't normalize (N-term acetyl, pyro-Q,
pyro-E) are silently dropped — those mods are out of distribution for the
prosit base model anyway, so it's the right behavior for fair evaluation.

Output: eval/spectra.parquet/ as DLOmix-compatible shards.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the vendored prosit annotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prosit/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prosit/src"))
from mgf_to_prosit_parquet import annotate  # type: ignore[import-not-found]

DEST = Path("data/holdout_pxd053296")
SRC = DEST / "eval/spectra.mgf"
OUT = DEST / "eval/spectra.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    annotate(
        src=SRC,
        out_dir=OUT,
        ce=28.0,                  # NCE confirmed from mzML
        mass_analyzer="FTMS",     # Q Exactive Plus
        ppm_override=None,        # use FTMS preset (25 ppm)
        min_peaks=5,
        max_seq_len=30,
        charges=(1, 2, 3, 4, 5, 6),
        shard_rows=250_000,
        dedup_mode="none",        # already deduped at PSM-table stage
        top_k=1,
    )
    shards = sorted(OUT.glob("*.parquet"))
    total_size = sum(p.stat().st_size for p in shards)
    print(f"\nwrote {len(shards)} shards, total {total_size / 1e6:.1f} MB to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
