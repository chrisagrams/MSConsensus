"""Annotate the consensus 4M test MGF into Prosit-format parquet shards.

Reuses ``prosit/scripts/mgf_to_prosit_parquet.annotate`` (kusterlab-vendored
chemistry; see project_msbench_prosit memory). Consensus SEQ is already in
MSFragger mass-delta form (e.g. ``C[57.0215]``), which the annotate function
handles natively via ``_normalize_seq_tokens``.

Output: <HOLDOUT>/eval/prosit_data/consensus_4m_test.parquet/ as DLOmix-
compatible shards, ready to be passed to eval/run_prosit.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the vendored prosit annotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prosit/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prosit/src"))
from mgf_to_prosit_parquet import annotate  # type: ignore[import-not-found]

SRC = Path("data/concensus_spec_30M/4G_dataset/concensus_4m.mgf_test.xz")
OUT = Path(
    "data/holdout_pxd053296/eval/prosit_data/"
    "consensus_4m_test.parquet"
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    annotate(
        src=SRC,
        out_dir=OUT,
        ce=35.0,                # NCE confirmed for consensus (project memory)
        mass_analyzer="FTMS",   # all consensus sources are Orbitrap FTMS
        ppm_override=None,      # use FTMS preset (25 ppm)
        min_peaks=5,
        max_seq_len=30,
        charges=(1, 2, 3, 4, 5, 6),
        shard_rows=250_000,
        dedup_mode="none",      # the consensus already deduped at submission time
        top_k=1,
    )
    shards = sorted(OUT.glob("*.parquet"))
    total = sum(p.stat().st_size for p in shards)
    print(f"\nwrote {len(shards)} shards, total {total / 1e6:.1f} MB to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
