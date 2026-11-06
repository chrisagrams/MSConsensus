"""Encode an MGF (or .xz MGF) into GLEAMS feature/metadata files.

Thin wrapper around ``gleams_repro.features.encode_split`` so we can point at
arbitrary MGFs without authoring a new YAML config per dataset.

Output:
  <out_dir>/<split>_features.npz
  <out_dir>/<split>_metadata.parquet

Usage:
  python eval/encode_for_gleams.py --mgf <path> --out <dir> --split <name> --workers 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse gleams_repro's encoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gleams/src"))
from gleams_repro import features as _features  # noqa: E402

# Same encoder params as configs/gleams_paper.yaml.features (W&B-logged only;
# real values come from gleams.config), echoed here for explicit provenance.
_FEATURES_CFG = {
    "fragment_mz_min": 50.5,
    "fragment_mz_max": 2500.0,
    "fragment_bin_size": 1.0005079,
    "min_peaks": 10,
    "min_mz_range": 250.0,
    "max_peaks": 150,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgf", required=True, type=Path,
                    help="Source MGF (plain or .xz) with SEQ= and CHARGE= per spectrum.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory; receives <split>_features.npz + _metadata.parquet")
    ap.add_argument("--split", required=True,
                    help="Split name used as the output filename prefix.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap encoded spectra (for smoke testing). Default: encode all.")
    ap.add_argument("--charges", type=int, nargs="+", default=[2, 3, 4, 5],
                    help="Precursor charges to retain (paper scope: 2-5).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"encoding {args.mgf} -> {args.out}/{args.split}_features.npz "
          f"(workers={args.workers}, charges={args.charges})")
    _features.encode_split(
        mgf_xz_path=str(args.mgf),
        out_dir=str(args.out),
        split=args.split,
        features_cfg=_FEATURES_CFG,
        charges=tuple(args.charges),
        limit=args.limit,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
