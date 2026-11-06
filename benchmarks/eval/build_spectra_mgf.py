"""Build the canonical annotated MGF from psms.parquet + the 30 mzMLs.

Reads psms.parquet (one row per confident PSM at 1% FDR), then for each
(source_mzml, scan) extracts the MS2 spectrum from the corresponding indexed
mzML and emits an MGF entry with MSFragger-format peptide annotation in SEQ=.

Output schema (per MGF entry):
  TITLE=<source>:<scan>            # uniquely identifies the spectrum
  RTINSECONDS=<float>              # retention time
  PEPMASS=<float>                  # observed precursor m/z
  CHARGE=<int>+                    # precursor charge
  SEQ=<peptide_msfragger>          # MSFragger notation, e.g. C[57.0215]
  SCANS=<scan>                     # native MS2 scan number
  SOURCE=<source>                  # source mzML basename (no path)

Downstream models translate SEQ= per their own alphabet (casanovo 5.x named
brackets; prosit UNIMOD; gleams uses unmodified strings for retrieval).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import pandas as pd
from pyteomics import mzml

DEST = Path("data/holdout_pxd053296")
PSMS = DEST / "eval/psms.parquet"
MZML_DIR = DEST / "mzml"
OUT = DEST / "eval/spectra.mgf"


def _scan_id(spec: dict) -> int | None:
    """Extract the native MS2 scan number from a pyteomics mzML spectrum dict.
    Thermo writes 'scan=N' in the spectrum's id; pyteomics doesn't normalize
    that into an int field, so we parse it ourselves."""
    sid = spec.get("id", "")
    for token in sid.split():
        if token.startswith("scan="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _ms2_iter(mzml_path: Path):
    """Yield (scan_int, spec_dict) for MS level 2 spectra only."""
    with mzml.read(str(mzml_path)) as reader:
        for spec in reader:
            if spec.get("ms level") != 2:
                continue
            scan = _scan_id(spec)
            if scan is None:
                continue
            yield scan, spec


def _format_charge(z: int) -> str:
    return f"{z}+"


def _precursor_mz(spec: dict) -> float | None:
    """Pull the selected ion m/z from the precursor list."""
    plist = spec.get("precursorList", {}).get("precursor", [])
    if not plist:
        return None
    sel = plist[0].get("selectedIonList", {}).get("selectedIon", [])
    if not sel:
        return None
    return float(sel[0].get("selected ion m/z"))


def _retention_seconds(spec: dict) -> float | None:
    scan_list = spec.get("scanList", {}).get("scan", [])
    if not scan_list:
        return None
    rt = scan_list[0].get("scan start time")
    if rt is None:
        return None
    # pyteomics returns a UnitFloat; treat unit as either minute or second.
    rt_val = float(rt)
    unit = getattr(rt, "unit_info", "") or ""
    if "minute" in unit.lower():
        rt_val *= 60.0
    return rt_val


def _process_one(args: tuple[str, str, str]) -> tuple[str, int, int]:
    """Worker: extract all wanted spectra from one mzML into its own per-source
    MGF chunk file. Returns (source, n_written, n_missing).

    The PSM subset is passed as a parquet path (not a DataFrame) so each worker
    reads its own slice — pickling a multi-MB DataFrame across the Pool boundary
    is slower than re-reading.
    """
    src, mzml_path, chunk_path = args
    sub = pd.read_parquet(chunk_path)
    wanted = sub.set_index("scan").to_dict(orient="index")
    n_written = 0
    out_path = Path(chunk_path).with_suffix(".mgf")
    with out_path.open("w") as out:
        for scan, spec in _ms2_iter(Path(mzml_path)):
            row = wanted.get(scan)
            if row is None:
                continue
            mz = _precursor_mz(spec)
            mzs = spec.get("m/z array", [])
            ints = spec.get("intensity array", [])
            if mz is None or len(mzs) == 0:
                continue
            rt = _retention_seconds(spec) or 0.0
            charge = int(row["charge"])
            seq = row["peptide_msfragger"]
            out.write("BEGIN IONS\n")
            out.write(f"TITLE={src}:{scan}\n")
            out.write(f"RTINSECONDS={rt:.4f}\n")
            out.write(f"PEPMASS={mz:.6f}\n")
            out.write(f"CHARGE={_format_charge(charge)}\n")
            out.write(f"SEQ={seq}\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SOURCE={src}\n")
            for m, i in zip(mzs, ints):
                out.write(f"{m:.6f} {i:.4f}\n")
            out.write("END IONS\n")
            n_written += 1
            if n_written >= len(wanted):
                break
    n_missing = len(wanted) - n_written
    return src, n_written, n_missing


def main() -> int:
    df = pd.read_parquet(PSMS)
    print(f"loaded {len(df)} PSMs across {df['source_mzml'].nunique()} mzMLs")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Spill per-source PSM subsets to /tmp parquets so each worker reads its own
    # slice independently — avoids pickling a 395k-row DataFrame across the pool.
    tmpdir = Path(tempfile.mkdtemp(prefix="mgf_chunks_"))
    args = []
    skipped = 0
    for src, group in df.groupby("source_mzml"):
        mzml_path = MZML_DIR / f"{src}.mzML"
        if not mzml_path.exists():
            print(f"  SKIP {src}: mzML missing at {mzml_path}")
            skipped += len(group)
            continue
        chunk = tmpdir / f"{src}.parquet"
        group.to_parquet(chunk, index=False)
        args.append((src, str(mzml_path), str(chunk)))

    n_workers = min(len(args), mp.cpu_count())
    print(f"dispatching {len(args)} mzMLs to {n_workers} workers")

    n_written_total = 0
    n_missing_total = skipped
    chunk_mgfs = []
    with mp.Pool(n_workers) as pool:
        for src, n_written, n_missing in pool.imap_unordered(_process_one, args):
            chunk_mgfs.append(tmpdir / f"{src}.mgf")
            n_written_total += n_written
            n_missing_total += n_missing
            print(f"  {src}: wrote={n_written} missing={n_missing}")

    print(f"\nconcatenating {len(chunk_mgfs)} chunk MGFs -> {OUT}")
    with OUT.open("wb") as out:
        for cm in sorted(chunk_mgfs):
            with cm.open("rb") as src_fh:
                while True:
                    buf = src_fh.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)

    # Tidy up tmp parquet+chunk files
    for p in tmpdir.glob("*"):
        p.unlink()
    tmpdir.rmdir()

    print(f"\nwrote {n_written_total} entries to {OUT}; missing {n_missing_total}")
    print(f"size: {OUT.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
