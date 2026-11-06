"""Translate the MassIVE-KB casanovo training-set MGF SEQ tokens into the
canonical ProForma named-bracket form expected at runtime by
``PeptideTokenizer`` (default casanovo config).

MassIVE-KB SEQ uses concatenated mass-delta notation (``C+57.021``,
``M+15.995``, N-term ``+42.011``). Depthcharge ships
``Peptide.massivekb_to_proforma`` which maps each token to its named-bracket
equivalent via ``MSKB_TO_UNIMOD``. We just stream the MGF, rewrite ``SEQ=``
lines via that helper, and pass everything else through unchanged.

Used to prep the MassIVE-KB test split (~996k spectra) for casanovo eval
alongside the held-out PXD053296 set, so we can compare in-distribution vs
out-of-distribution performance for each checkpoint.

Usage:
  python eval/translate_massivekb_mgf.py \\
      --src data/.../massivekb_82c0124b_test.mgf \\
      --dst /tmp/spectra_massivekb_translated.mgf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from depthcharge.primitives import Peptide


def translate_seq(seq: str) -> str:
    """Apply depthcharge's MSKB→ProForma translation. We strip the trailing
    ``/<charge>`` component that ``massivekb_to_proforma`` would otherwise
    append — casanovo's MGF reader gets the charge from the CHARGE= header,
    so embedding it in the seq string is redundant and confuses tokenizers
    further downstream."""
    proforma = Peptide.massivekb_to_proforma(seq)
    if "/" in proforma:
        proforma = proforma.split("/", 1)[0]
    return proforma


def translate_mgf(src_mgf: Path, dst_mgf: Path) -> tuple[int, int]:
    n_kept = 0
    n_dropped = 0
    with src_mgf.open("r") as f_in, dst_mgf.open("w") as f_out:
        for line in f_in:
            if line.startswith("SEQ="):
                seq = line[len("SEQ="):].strip()
                try:
                    new = translate_seq(seq)
                except Exception:
                    n_dropped += 1
                    f_out.write(f"SEQ=__SKIP__\n")  # poison marker, filtered below
                    continue
                f_out.write(f"SEQ={new}\n")
            else:
                f_out.write(line)
            if line.startswith("BEGIN IONS"):
                n_kept += 1
    return n_kept, n_dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source MassIVE-KB MGF (uncompressed).")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output translated MGF.")
    args = ap.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_kept, n_dropped = translate_mgf(args.src, args.dst)
    print(f"entries scanned={n_kept}  seq-translation-failures={n_dropped}")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
