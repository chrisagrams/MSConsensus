"""Build the canonical PSM table from the 30 search-result zips.

Reads each zip's ``psm_fido.tsv`` (Percolator-validated 1% FDR PSMs from MSFragger),
parses MSFragger-style peptide notation (mass-delta brackets) into a structured
table, and writes a single parquet for the per-model eval scripts to consume.

Schema:
  source_mzml         basename of source mzML (e.g. "200505-NanoV-20_0728-04-AWO-F02")
  scan                int, MS2 scan number from PSMId
  charge              int
  rank                int (Percolator rank, 1 = top hit; we filter to 1 only)
  peptide_msfragger   peptide with MSFragger mass-delta notation (C[57.0215] etc.)
  peptide_unmod       plain amino-acid sequence (no brackets)
  pep_len             int
  q_value             float (≤ 0.01 = 1% PSM FDR)
  score               float (Percolator score, higher is better)
  pep                 float (Percolator posterior error probability)
  proteins            str (semicolon-joined accessions)
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEST = Path("data/holdout_pxd053296")
ZIP_DIR = DEST / "search/results"
OUT = DEST / "eval/psms.parquet"

# PSMId format: input.<scan_lo>.<scan_hi>.<charge>_<rank>
_PSMID_RE = re.compile(r"^input\.(\d+)\.\d+\.(\d+)_(\d+)$")
# peptide format: <prev>.<peptide_with_mods><charge>.<next>
# Peptide body matches [A-Z\[\]\-\.0-9] but the leading/trailing single residue
# is what we strip. Examples:
#   R.AEGSDVANAVLDGADC[57.0215]IMLSGETAK3.G  ->  AEGSDVANAVLDGADC[57.0215]IMLSGETAK
#   K.[42.0106]VIPAADLSEQISTAGTEASGTGNMK2.F  ->  [42.0106]VIPAADLSEQISTAGTEASGTGNMK
_PEPTIDE_RE = re.compile(r"^([A-Z\-])\.(.+?)(\d+)\.([A-Z\-])$")


def parse_peptide(raw: str) -> tuple[str, int] | None:
    """Strip flanking residues and trailing charge. Returns (peptide_with_mods, charge)
    or None if the format is unexpected (we drop the row)."""
    m = _PEPTIDE_RE.match(raw)
    if not m:
        return None
    return m.group(2), int(m.group(3))


def parse_psmid(psmid: str) -> tuple[int, int, int] | None:
    m = _PSMID_RE.match(psmid)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def strip_mods(pep_with_mods: str) -> str:
    """Remove [...] mass-delta tags to get the plain AA sequence."""
    return re.sub(r"\[[^]]+\]", "", pep_with_mods)


def main() -> int:
    rows = []
    skipped = {"psmid": 0, "peptide": 0, "charge_mismatch": 0, "rank_gt_1": 0}
    zips = sorted(ZIP_DIR.glob("*.zip"))
    print(f"reading {len(zips)} zips from {ZIP_DIR}")
    for zp in zips:
        source = zp.stem  # basename without .zip
        with zipfile.ZipFile(zp) as zf:
            with zf.open("psm_fido.tsv") as fh:
                df = pd.read_csv(fh, sep="\t", dtype=str)
        for _, r in df.iterrows():
            psmid = r["PSMId"]
            pid_parsed = parse_psmid(psmid)
            if pid_parsed is None:
                skipped["psmid"] += 1
                continue
            scan, charge_id, rank = pid_parsed
            if rank != 1:
                skipped["rank_gt_1"] += 1
                continue
            pep_parsed = parse_peptide(r["peptide"])
            if pep_parsed is None:
                skipped["peptide"] += 1
                continue
            pep_mods, charge_pep = pep_parsed
            if charge_pep != charge_id:
                # If the trailing-digit-as-charge convention disagrees with PSMId,
                # trust PSMId (it's authoritative) and log.
                skipped["charge_mismatch"] += 1
            rows.append({
                "source_mzml": source,
                "scan": scan,
                "charge": charge_id,
                "rank": rank,
                "peptide_msfragger": pep_mods,
                "peptide_unmod": strip_mods(pep_mods),
                "pep_len": len(strip_mods(pep_mods)),
                "q_value": float(r["q-value"]),
                "score": float(r["score"]),
                "pep": float(r["posterior_error_prob"]),
                "proteins": r["proteinIds"],
            })

    print(f"  parsed rows={len(rows)}  skipped={skipped}")
    df = pd.DataFrame(rows)

    # Filter to confident PSMs at our acceptance ranges
    before = len(df)
    df = df[df["q_value"] <= 0.01].copy()
    after_q = len(df)
    df = df[df["charge"].between(2, 5)].copy()
    after_z = len(df)
    df = df[df["pep_len"].between(7, 30)].copy()
    after_l = len(df)
    print(f"filter: rows={before} -> q≤0.01: {after_q} -> charge 2-5: {after_z} -> length 7-30: {after_l}")

    # Within each (source, scan) keep highest-score row (defensive — should already be unique
    # since we filtered to rank=1)
    df = df.sort_values("score", ascending=False).drop_duplicates(["source_mzml", "scan"])
    print(f"after dedup on (source, scan): {len(df)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}  cols={list(df.columns)}")

    # Quick stats
    print("\n=== charge distribution ===")
    print(df["charge"].value_counts().sort_index())
    print("\n=== mods distribution (top 10 modified residues) ===")
    mods = df["peptide_msfragger"].str.findall(r"[A-Z]\[[^]]+\]|\[[^]]+\]").explode()
    print(mods.value_counts().head(10))
    print("\n=== unique peptides ===", df["peptide_msfragger"].nunique())
    print("=== unique source mzMLs ===", df["source_mzml"].nunique())

    return 0


if __name__ == "__main__":
    sys.exit(main())
