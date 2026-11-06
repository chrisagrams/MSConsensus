#!/usr/bin/env python3
"""Compare unique peptide / precursor-key diversity between two datasets.

Default targets concensus_spec_30M vs massive_kb_v1_shared under data/. Two
read modes:

  --mode mgf    Stream MGF text files, skipping peak lines.
  --mode lance  Project `seq` + `precursor_charge` columns from a Lance
                dataset (verified casanovo schema). Requires the casanovo
                venv with `lance` installed; lance mode also caps Tokio
                thread count via TOKIO_WORKER_THREADS=2 to play nicely
                on shared login nodes.

The two modes are not strictly comparable: the Lance dump only ships
train+valid (no test split), while the MGF set has train+val+test. Use
matching --mgfs-* / --lance-* flags if you need apples-to-apples.

Reports total/unique/intersection/union/Jaccard on:

    - precursor keys    key = (peptide_with_mods, charge)
                        (i.e. unique annotated precursor identities; not
                        PSMs in the strict peptide-spectrum-match sense —
                        that count equals total_spectra)
    - modified peptides key = peptide_with_mods
    - peptides (bare)   key = peptide_with_mods stripped of non-letters

Usage:
    python compare_diversity.py                       # mgf mode, data/ defaults
    python compare_diversity.py --mode lance          # Lance fast path
    python compare_diversity.py --sample 100000       # smoke test
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

DATA_ROOT = Path("data/")

DEFAULT_DATASETS: dict[str, list[Path]] = {
    "concensus_spec_30M": [
        DATA_ROOT / "concensus_spec_30M" / "concensus.mgf_train_main.mgf",
        DATA_ROOT / "concensus_spec_30M" / "concensus.mgf_val.mgf",
        DATA_ROOT / "concensus_spec_30M" / "concensus.mgf_test.mgf",
    ],
    "massive_kb_v1_shared": [
        DATA_ROOT / "massive_kb_v1_shared" / "massivekb_82c0124b_train.mgf",
        DATA_ROOT / "massive_kb_v1_shared" / "massivekb_82c0124b_val.mgf",
        DATA_ROOT / "massive_kb_v1_shared" / "massivekb_82c0124b_test.mgf",
    ],
}

DEFAULT_LANCE_DATASETS: dict[str, list[Path]] = {
    "concensus_spec_30M": [
        DATA_ROOT / "concensus_spec_30M" / "casanovo_lance" / "train.lance",
        DATA_ROOT / "concensus_spec_30M" / "casanovo_lance" / "valid.lance",
    ],
    "massive_kb_v1_shared": [
        DATA_ROOT / "massive_kb_v1_shared" / "casanovo_lance" / "train.lance",
        DATA_ROOT / "massive_kb_v1_shared" / "casanovo_lance" / "valid.lance",
    ],
}

LANCE_SEQ_COL = "seq"
LANCE_CHARGE_COL = "precursor_charge"

# Strip everything except A-Z to get the bare peptide. This collapses any
# modification annotation style (PEPT[+57]IDE, PEPT(ox)IDE, lowercase mods,
# etc.) to the underlying sequence. Modified-peptide counts use the raw
# SEQ string instead.
_BARE_PEP_RE = re.compile(rb"[^A-Z]")


def parse_mgf(
    path: Path, sample: int | None = None
) -> tuple[str, int, set[bytes], set[bytes], set[bytes]]:
    """Stream one MGF file. Returns (path, n_spectra, precursor_keys, mod_peps, peps).

    Reads in binary, skips peak lines (anything starting with a digit), and
    only inspects header lines. SEQ= and CHARGE= are extracted per spectrum;
    on END IONS we emit keys.
    """
    precursor_keys: set[bytes] = set()
    mod_peps: set[bytes] = set()
    peps: set[bytes] = set()
    n = 0

    seq: bytes | None = None
    charge: bytes | None = None

    # 8 MB read buffer — Lustre likes large blocks.
    with open(path, "rb", buffering=8 * 1024 * 1024) as fh:
        for line in fh:
            # Cheap reject for peak lines: they start with a digit.
            c0 = line[:1]
            if c0 >= b"0" and c0 <= b"9":
                continue
            if line.startswith(b"SEQ="):
                seq = line[4:].rstrip()
            elif line.startswith(b"CHARGE="):
                # CHARGE=2+  -> b"2"
                charge = line[7:].rstrip().rstrip(b"+-")
            elif line.startswith(b"END IONS"):
                if seq is not None:
                    mod_peps.add(seq)
                    peps.add(_BARE_PEP_RE.sub(b"", seq))
                    if charge is not None:
                        precursor_keys.add(seq + b"/" + charge)
                    else:
                        precursor_keys.add(seq + b"/?")
                    n += 1
                    if sample is not None and n >= sample:
                        break
                seq = None
                charge = None
            # BEGIN IONS / TITLE / PEPMASS / RTINSECONDS / SCANS / SCORE /
            # APPEARANCE_COUNT lines are ignored.

    return str(path), n, precursor_keys, mod_peps, peps


def _worker(args):
    path, sample = args
    return parse_mgf(path, sample=sample)


def parse_lance(
    path: Path,
    seq_col: str = LANCE_SEQ_COL,
    charge_col: str = LANCE_CHARGE_COL,
    sample: int | None = None,
    batch_size: int = 1_000_000,
) -> tuple[str, int, set[bytes], set[bytes], set[bytes]]:
    """Project peptide+charge columns from a Lance dataset.

    Far faster than MGF parsing — peaks stay on disk. Reads in row-group
    batches; updates sets per batch.
    """
    import lance  # imported lazily so MGF mode doesn't require it

    precursor_keys: set[bytes] = set()
    mod_peps: set[bytes] = set()
    peps: set[bytes] = set()
    n = 0

    ds = lance.dataset(str(path))
    scanner = ds.scanner(columns=[seq_col, charge_col], batch_size=batch_size)
    for batch in scanner.to_batches():
        seqs = batch.column(seq_col).to_pylist()
        charges = batch.column(charge_col).to_pylist()
        for seq, ch in zip(seqs, charges):
            if seq is None:
                continue
            sb = seq.encode() if isinstance(seq, str) else bytes(seq)
            mod_peps.add(sb)
            peps.add(_BARE_PEP_RE.sub(b"", sb))
            cb = str(ch).encode() if ch is not None else b"?"
            precursor_keys.add(sb + b"/" + cb)
            n += 1
            if sample is not None and n >= sample:
                return str(path), n, precursor_keys, mod_peps, peps

    return str(path), n, precursor_keys, mod_peps, peps


def _lance_worker(args):
    path, sample = args
    return parse_lance(path, sample=sample)


def aggregate(
    paths: list[Path], sample: int | None, workers: int, mode: str
) -> tuple[int, set[bytes], set[bytes], set[bytes]]:
    """Run parse_mgf or parse_lance across paths; merge sets."""
    precursor_keys: set[bytes] = set()
    mod_peps: set[bytes] = set()
    peps: set[bytes] = set()
    total = 0

    parse = parse_mgf if mode == "mgf" else parse_lance
    worker = _worker if mode == "mgf" else _lance_worker

    if workers <= 1 or len(paths) == 1:
        for p in paths:
            _, n, a, b, c = parse(p, sample=sample)
            total += n
            precursor_keys |= a
            mod_peps |= b
            peps |= c
            print(f"  [{p.name}] {n:,} spectra, "
                  f"{len(a):,} cum precursor_keys, {len(b):,} cum mod-peps, "
                  f"{len(c):,} cum peps", file=sys.stderr)
    else:
        with mp.Pool(min(workers, len(paths))) as pool:
            for path_str, n, a, b, c in pool.imap_unordered(
                worker, [(p, sample) for p in paths]
            ):
                total += n
                precursor_keys |= a
                mod_peps |= b
                peps |= c
                print(f"  [{Path(path_str).name}] {n:,} spectra "
                      f"(file-local: {len(a):,} precursor_keys, {len(b):,} mod-peps, "
                      f"{len(c):,} peps)", file=sys.stderr)
    return total, precursor_keys, mod_peps, peps


def fmt(n: int) -> str:
    return f"{n:,}"


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["mgf", "lance"], default="mgf")
    ap.add_argument("--dataset-a-name", default="concensus_spec_30M")
    ap.add_argument("--dataset-b-name", default="massive_kb_v1_shared")
    ap.add_argument("--mgfs-a", nargs="*", type=Path,
                    help="MGF paths for dataset A (default: data/ paths)")
    ap.add_argument("--mgfs-b", nargs="*", type=Path,
                    help="MGF paths for dataset B (default: data/ paths)")
    ap.add_argument("--lance-a", nargs="*", type=Path,
                    help="Lance dataset paths for A (default: data/ paths)")
    ap.add_argument("--lance-b", nargs="*", type=Path,
                    help="Lance dataset paths for B (default: data/ paths)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Per-file spectrum cap for smoke testing")
    ap.add_argument("--workers", type=int, default=4,
                    help="Process pool size across all paths")
    ap.add_argument("--out", type=Path, default=Path("diversity_table.tsv"))
    args = ap.parse_args()

    if args.mode == "mgf":
        a_paths = args.mgfs_a or DEFAULT_DATASETS[args.dataset_a_name]
        b_paths = args.mgfs_b or DEFAULT_DATASETS[args.dataset_b_name]
    else:
        a_paths = args.lance_a or DEFAULT_LANCE_DATASETS[args.dataset_a_name]
        b_paths = args.lance_b or DEFAULT_LANCE_DATASETS[args.dataset_b_name]

    for p in (*a_paths, *b_paths):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    print(f"[A] {args.dataset_a_name}: {len(a_paths)} files",
          file=sys.stderr)
    for p in a_paths:
        print(f"    {p}", file=sys.stderr)
    print(f"[B] {args.dataset_b_name}: {len(b_paths)} files",
          file=sys.stderr)
    for p in b_paths:
        print(f"    {p}", file=sys.stderr)
    if args.sample:
        print(f"SAMPLE MODE: first {args.sample:,} spectra per file",
              file=sys.stderr)

    t0 = time.time()
    print(f"\n[A] parsing... (mode={args.mode})", file=sys.stderr)
    a_total, a_pk, a_mp, a_p = aggregate(
        a_paths, args.sample, args.workers, args.mode
    )
    print(f"\n[B] parsing... (mode={args.mode})", file=sys.stderr)
    b_total, b_pk, b_mp, b_p = aggregate(
        b_paths, args.sample, args.workers, args.mode
    )
    dt = time.time() - t0
    print(f"\nparsed in {dt:.1f}s", file=sys.stderr)

    rows = [
        ("total_spectra", a_total, b_total, None, None, None),
        ("unique_precursor_keys",
         len(a_pk), len(b_pk),
         len(a_pk & b_pk), len(a_pk | b_pk), jaccard(a_pk, b_pk)),
        ("unique_modified_peptides",
         len(a_mp), len(b_mp),
         len(a_mp & b_mp), len(a_mp | b_mp), jaccard(a_mp, b_mp)),
        ("unique_peptides_bare",
         len(a_p), len(b_p),
         len(a_p & b_p), len(a_p | b_p), jaccard(a_p, b_p)),
    ]

    header = ["metric", args.dataset_a_name, args.dataset_b_name,
              "intersection", "union", "jaccard"]

    with open(args.out, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for metric, a, b, i, u, j in rows:
            fh.write("\t".join([
                metric, fmt(a), fmt(b),
                "" if i is None else fmt(i),
                "" if u is None else fmt(u),
                "" if j is None else f"{j:.4f}",
            ]) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)

    # Markdown table to stdout for quick visual.
    widths = [max(len(str(r[k])) for r in [header] + [
        (m, fmt(a), fmt(b),
         "" if i is None else fmt(i),
         "" if u is None else fmt(u),
         "" if j is None else f"{j:.4f}")
        for m, a, b, i, u, j in rows
    ]) for k in range(len(header))]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(line(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for metric, a, b, i, u, j in rows:
        print(line([
            metric, fmt(a), fmt(b),
            "" if i is None else fmt(i),
            "" if u is None else fmt(u),
            "" if j is None else f"{j:.4f}",
        ]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
