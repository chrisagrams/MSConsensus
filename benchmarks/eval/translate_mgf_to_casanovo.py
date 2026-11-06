"""Translate the canonical eval/spectra.mgf SEQ tokens into a casanovo-
compatible alphabet.

The canonical MGF carries SEQ in MSFragger notation (e.g. ``C[57.0215]``,
``M[15.9949]``, ``n[42.0106]MK...``). Casanovo's tokenizers expect either:

  * style=named  → ProForma named-bracket form (``C[Carbamidomethyl]``,
                   ``[Acetyl]-VPK``). Used by the default ``PeptideTokenizer``.
                   This is the format the published v5 checkpoint's
                   vocabulary stores internally and the format our retrained
                   ckpts (config sets ``massivekb_tokenizer: false``) read.

  * style=mskb   → MassIVE-KB mass-delta form (``C+57.021``, ``+42.011VPK``).
                   Used by ``MskbPeptideTokenizer``. The runtime translates
                   it back to named-bracket via ``massivekb_to_proforma`` in
                   depthcharge — but only if that tokenizer is actually
                   instantiated, which requires ``massivekb_tokenizer: true``
                   in the config. Default casanovo config has it false, so
                   the named style works for every checkpoint we care about.

PSMs whose mods can't be represented (currently: pyro-Glu = ``E[-18.0106]``,
~420 entries / 0.1%) are dropped.

Usage:
  python eval/translate_mgf_to_casanovo.py \\
      --src data/.../eval/spectra.mgf \\
      --dst data/.../eval/spectra_casanovo.mgf \\
      --style named
"""
from __future__ import annotations

import argparse
import lzma
import re
import sys
from pathlib import Path

# Two output styles, selected at runtime by --style:
#
#   "named" → casanovo PeptideTokenizer (used by published v5 + our retrains).
#              Produces C[Carbamidomethyl] etc. (valid ProForma).
#
#   "mskb"  → MskbPeptideTokenizer (only used when a config explicitly opts in
#              with ``massivekb_tokenizer: true``). Produces C+57.021 etc.
#
# The MSFragger emits residue mods as Letter[delta], N-term mods as
# n[delta]<rest>, and pyro-Q as Q[-17.0265]<rest> (mod sits on the residue,
# but both casanovo styles treat it as an N-term token preceding the residue).

_NAMED_RESIDUE = {
    "[57.0215]":  "[Carbamidomethyl]",
    "[15.9949]":  "[Oxidation]",
    "[0.9840]":   "[Deamidated]",
}
_NAMED_NTERM = {
    "[42.0106]":  "[Acetyl]-",
    "[43.0058]":  "[Carbamyl]-",
    "[-17.0265]": "[Ammonia-loss]-",
    "[25.9803]":  "[+25.980265]-",
}

# MSKB strings: NO brackets, mass delta concatenated to residue (or peptide
# start). Source of truth: depthcharge.primitives.peptides.MSKB_TO_UNIMOD.
_MSKB_RESIDUE = {
    "[57.0215]":  "+57.021",
    "[15.9949]":  "+15.995",
    "[0.9840]":   "+0.984",
}
_MSKB_NTERM = {
    "[42.0106]":  "+42.011",
    "[43.0058]":  "+43.006",
    "[-17.0265]": "-17.027",
    "[25.9803]":  "+43.006-17.027",
}

_RESIDUE_RE = re.compile(r"([A-Z])(\[[-\d.]+\])")
_NTERM_RE = re.compile(r"^n(\[[-\d.]+\])")          # MSFragger N-term: n[42.0106]MK...
_CTERM_RE = re.compile(r"c(\[[-\d.]+\])$")          # defensive (none in our PSMs)
# Pyro-Q on first residue: MSFragger 'Q[-17.0265]' → casanovo's N-term form.
# Pyro-E (E[-18.0106]) has no canonical token in either casanovo style: drop.
_NTERM_RES_PYRO = ("Q[-17.0265]",)


def translate_seq(seq: str, style: str) -> str | None:
    """Convert one MSFragger-format peptide string. Returns None if the
    peptide carries a mod we can't represent in ``style``."""
    if style == "named":
        rmap, nmap = _NAMED_RESIDUE, _NAMED_NTERM
        pyro_q_prefix = "[Ammonia-loss]-Q"
    elif style == "mskb":
        rmap, nmap = _MSKB_RESIDUE, _MSKB_NTERM
        pyro_q_prefix = "-17.027Q"
    else:
        raise ValueError(f"unknown style {style!r}")

    out = seq

    # 1) MSFragger N-term mod: 'n[<delta>]<rest>' → maps in nmap.
    m = _NTERM_RE.match(out)
    if m:
        delta = m.group(1)
        token = nmap.get(delta)
        if token is None:
            return None
        out = token + out[m.end():]

    # 2) Pyro-Q: 'Q[-17.0265]<rest>' (after stripping any other N-term)
    for src in _NTERM_RES_PYRO:
        if out.startswith(src):
            out = pyro_q_prefix + out[len(src):]
            break

    # 3) Per-residue mods anywhere in the sequence
    def _replace(m: re.Match) -> str:
        residue, delta = m.group(1), m.group(2)
        token = rmap.get(delta)
        if token is None:
            return f"__UNKNOWN_{delta}__"
        return residue + token

    out = _RESIDUE_RE.sub(_replace, out)

    if _CTERM_RE.search(out):
        return None
    if "__UNKNOWN_" in out:
        return None
    if style == "mskb" and "[" in out:
        return None
    return out


def _open_text(path: Path):
    """Open a plain or .xz MGF for streaming text reads."""
    if str(path).endswith(".xz"):
        return lzma.open(path, "rt")
    return path.open("r")


def translate_mgf(src_mgf: Path, dst_mgf: Path, style: str) -> tuple[int, int]:
    """Stream src_mgf (plain or .xz), rewrite SEQ= lines via translate_seq, drop
    entries whose peptide can't be represented. Returns (n_kept, n_dropped)."""
    n_kept = 0
    n_dropped = 0
    with _open_text(src_mgf) as f_in, dst_mgf.open("w") as f_out:
        block: list[str] = []
        keep = True
        for line in f_in:
            if line.startswith("BEGIN IONS"):
                block = [line]
                keep = True
            elif line.startswith("SEQ="):
                seq = line[len("SEQ="):].strip()
                new = translate_seq(seq, style)
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
    ap.add_argument("--src", required=True, type=Path,
                    help="Canonical eval/spectra.mgf (MSFragger-format SEQ).")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output MGF with translated SEQ.")
    ap.add_argument("--style", choices=["named", "mskb"], required=True,
                    help="Casanovo SEQ alphabet to translate to. 'named' "
                         "covers the published v5 ckpt + our retrains "
                         "(default config). 'mskb' is for opt-in MassIVE-KB "
                         "tokenizer configs.")
    args = ap.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_kept, n_dropped = translate_mgf(args.src, args.dst, args.style)
    pct = n_dropped / max(n_kept + n_dropped, 1)
    print(f"kept={n_kept}  dropped={n_dropped}  ({pct:.1%})")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
