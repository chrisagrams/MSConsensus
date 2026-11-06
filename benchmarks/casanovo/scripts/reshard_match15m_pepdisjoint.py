"""Peptide-disjoint reshard of the patched new_match_15M MGFs.

Reads the 25 PEPMASS-injected MGFs in
``casanovo/data/new_match_15M/mgf_patched/`` and routes each spectrum to
{train, valid, test} by ``sha256(SEQ) mod 100``:

  buckets 0..3   -> test   (~4%)
  buckets 4..7   -> valid  (~4%)
  buckets 8..99  -> train  (~92%)

Hashing on the full ``SEQ`` string keeps every charge state of a given
modified peptide together in the same split — this is the standard
peptide-disjoint convention. (Same bare peptide with different mods is
treated as different peptides; if you want bare-peptide disjointness, hash
``re.sub(r'\\[[^\\]]+\\]', '', seq)`` instead.)

Each worker reads one source MGF and writes to its own per-shard outputs:
``mgf_pepdisjoint/{train,valid,test}/consensus_NN.mgf``. No cross-worker
contention because each worker owns shard NN exclusively. depthcharge
accepts a list of MGFs as input, so we don't concatenate.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("casanovo_repro.reshard_match15m")


def _bucket(seq: str) -> int:
    """sha256(seq) mod 100. Stable across processes / Python versions
    (unlike Python's built-in hash())."""
    return int.from_bytes(hashlib.sha256(seq.encode()).digest()[:8], "big") % 100


def _route(bucket: int) -> str:
    if bucket < 4:
        return "test"
    if bucket < 8:
        return "valid"
    return "train"


def reshard_one(src: Path, dst_root: Path, shard_name: str) -> dict:
    """Stream src MGF, route blocks by sha256(SEQ)%100, write to per-split
    shard files. Idempotent: if all 3 shard outputs exist non-empty, skip."""
    out_paths = {sp: dst_root / sp / shard_name for sp in ("train", "valid", "test")}
    if all(p.exists() and p.stat().st_size > 0 for p in out_paths.values()):
        logger.info("skip (exists): %s", shard_name)
        return {"shard": shard_name, "skipped": True,
                "train": 0, "valid": 0, "test": 0}

    for p in out_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths = {sp: p.with_suffix(p.suffix + ".tmp") for sp, p in out_paths.items()}

    counts = {"train": 0, "valid": 0, "test": 0}
    t0 = time.time()
    with open(src, "rt") as fin, \
         open(tmp_paths["train"], "wt") as ftr, \
         open(tmp_paths["valid"], "wt") as fva, \
         open(tmp_paths["test"], "wt") as fte:
        out_for_split = {"train": ftr, "valid": fva, "test": fte}
        block_lines: list[str] = []
        seq: str | None = None
        in_block = False

        for line in fin:
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                block_lines = [line]
                seq = None
                in_block = True
                continue
            if not in_block:
                # Allow blank lines between blocks
                continue
            block_lines.append(line)
            if stripped.startswith("SEQ="):
                seq = stripped.split("=", 1)[1].strip()
            if stripped == "END IONS":
                if seq is None:
                    raise ValueError(
                        f"{shard_name}: block at line {sum(map(len, block_lines))} "
                        "had no SEQ= line"
                    )
                split = _route(_bucket(seq))
                fh = out_for_split[split]
                fh.writelines(block_lines)
                counts[split] += 1
                in_block = False
                block_lines = []
                seq = None
                total = counts["train"] + counts["valid"] + counts["test"]
                if total % 100_000 == 0:
                    logger.info("%s: %d blocks (%.1fs)", shard_name, total, time.time() - t0)

    for sp, tmp in tmp_paths.items():
        tmp.replace(out_paths[sp])

    dt = time.time() - t0
    logger.info("done %s: train=%d valid=%d test=%d (%.1fs)",
                shard_name, counts["train"], counts["valid"], counts["test"], dt)
    counts.update({"shard": shard_name, "skipped": False})
    return counts


def main(src_dir: Path, dst_dir: Path, max_workers: int) -> int:
    sources = sorted(src_dir.glob("consensus_*.mgf"))
    if not sources:
        logger.error("no consensus_*.mgf in %s", src_dir)
        return 1
    logger.info("resharding %d MGFs from %s -> %s/{train,valid,test}/ (workers=%d)",
                len(sources), src_dir, dst_dir, max_workers)

    futures = {}
    totals = {"train": 0, "valid": 0, "test": 0}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for src in sources:
            futures[pool.submit(reshard_one, src, dst_dir, src.name)] = src
        for f in as_completed(futures):
            src = futures[f]
            try:
                r = f.result()
                if not r["skipped"]:
                    for k in ("train", "valid", "test"):
                        totals[k] += r[k]
            except Exception:
                logger.exception("FAILED: %s", src)
                return 2

    grand = sum(totals.values())
    if grand:
        logger.info(
            "totals: train=%d (%.2f%%) valid=%d (%.2f%%) test=%d (%.2f%%) all=%d",
            totals["train"], 100 * totals["train"] / grand,
            totals["valid"], 100 * totals["valid"] / grand,
            totals["test"], 100 * totals["test"] / grand,
            grand,
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path,
                   default=Path("casanovo/data/new_match_15M/mgf_patched"))
    p.add_argument("--dst-dir", type=Path,
                   default=Path("casanovo/data/new_match_15M/mgf_pepdisjoint"))
    p.add_argument("--workers", type=int, default=25)
    args = p.parse_args()
    sys.exit(main(args.src_dir, args.dst_dir, args.workers))
