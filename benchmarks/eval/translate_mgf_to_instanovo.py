"""Adapt an MGF for InstaNovo's loader.

Two issues observed feeding consensus_4m's MGF directly to InstaNovo:

  1. SEQ uses MSFragger-style mass-delta brackets (``C[57.0215]``,
     ``M[15.9949]``). InstaNovo's vocab is UNIMOD-named (``C[UNIMOD:4]``,
     ``M[UNIMOD:35]``) and treats anything else as an unsupported residue,
     dropping those rows in eval mode.

  2. TITLE strings carry confounding ``=`` substrings, e.g.
     ``TITLE=consensus:PEPK:charge=2:n_spectra=50`` — InstaNovo's parser
     appears to extract the last ``=N`` as a charge candidate, picking up
     ``n_spectra=50`` and flagging ~25% of rows as ``charge > 10``. They're
     also non-unique across replicate consensus rows, which trips dedup.

This script streams the source MGF, rewrites TITLE to a sequential unique
``<label>:N`` and SEQ via a small mass-delta → UNIMOD map, and emits a
clean copy. Peptides with mods we can't map (e.g., ``n[42.0106]`` N-term
acetyl, ``Q[-17.0265]`` pyro-Q, ``E[-18.0106]``) are dropped to avoid
unsupported-residue warnings; consensus_4m has none of these in our sample.

Usage:
  python eval/translate_mgf_to_instanovo.py --src <mgf> --dst <mgf> --label consensus_4m
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# MSFragger mass-delta → UNIMOD on residue. Same mapping as
# prosit/scripts/mgf_to_prosit_parquet.py:_to_unimod, restricted to the
# residue mods that show up in consensus_4m.
_RESIDUE_MAP = {
    "[57.0215]": "[UNIMOD:4]",     # C carbamidomethyl
    "[15.9949]": "[UNIMOD:35]",    # M oxidation
    "[0.9840]":  "[UNIMOD:7]",     # N/Q deamidation
}

_RESIDUE_RE = re.compile(r"([A-Z])(\[[-\d.]+\])")
_NTERM_RE = re.compile(r"^n(\[[-\d.]+\])")
_PYRO_PREFIX = ("Q[-17.0265]", "E[-18.0106]")


def translate_seq(seq: str) -> str | None:
    """MSFragger → UNIMOD per-residue translation. Returns None if any mod
    we can't represent appears (drop the row)."""
    if _NTERM_RE.match(seq):
        return None
    if any(seq.startswith(p) for p in _PYRO_PREFIX):
        return None

    def _replace(m: re.Match) -> str:
        residue, delta = m.group(1), m.group(2)
        token = _RESIDUE_MAP.get(delta)
        if token is None:
            return f"__UNK_{delta}__"
        return residue + token

    out = _RESIDUE_RE.sub(_replace, seq)
    if "__UNK_" in out:
        return None
    return out


def translate_mgf(src: Path, dst: Path, label: str) -> tuple[int, int]:
    n_kept = 0
    n_dropped = 0
    idx = 0
    with src.open("r") as f_in, dst.open("w") as f_out:
        block: list[str] = []
        keep = True
        for line in f_in:
            if line.startswith("BEGIN IONS"):
                block = [line]
                keep = True
            elif line.startswith("TITLE="):
                # Rewrite to clean unique title
                block.append(f"TITLE={label}:{idx}\n")
                idx += 1
            elif line.startswith("SEQ="):
                seq = line[len("SEQ="):].strip()
                new = translate_seq(seq)
                if new is None:
                    keep = False
                else:
                    block.append(f"SEQ={new}\n")
            elif line.startswith("END IONS"):
                block.append(line)
                if keep:
                    f_out.writelines(block)
                    n_kept += 1
                else:
                    n_dropped += 1
                block = []
            else:
                block.append(line)
    return n_kept, n_dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--label", required=True,
                    help="Used as the new TITLE prefix.")
    args = ap.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_kept, n_dropped = translate_mgf(args.src, args.dst, args.label)
    print(f"kept={n_kept}  dropped={n_dropped}")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
