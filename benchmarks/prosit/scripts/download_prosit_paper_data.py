"""Fetch the original Prosit fragmentation training/holdout HDF5s from figshare.

Pulls the two files published alongside Gessulat et al. 2019 (figshare article
6860261) and verifies their MD5 checksums:

    traintest_hcd.hdf5  (5.23 GB)  — train + test, 174-dim Prosit format
    holdout_hcd.hdf5    (263 MB)   — held-out set used in the paper figures

Idempotent: re-running with files already present and matching MD5 is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("download_prosit_paper_data")

FILES = [
    {
        "name": "holdout_hcd.hdf5",
        "url": "https://ndownloader.figshare.com/files/12506534",
        "md5": "ea960a3e531db1d812d95603b338d9b1",
        "size": 262_781_348,
    },
    {
        "name": "traintest_hcd.hdf5",
        "url": "https://ndownloader.figshare.com/files/12785291",
        "md5": "ff5d05cee72a7095587267701e815960",
        "size": 5_232_963_475,
    },
]


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    logger.info("downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
        total = int(r.headers.get("Content-Length", "0")) or None
        n = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            n += len(buf)
            if total and n // (50 << 20) != (n - len(buf)) // (50 << 20):
                logger.info("  %.1f%% (%d MB)", 100 * n / total, n >> 20)
    tmp.rename(dest)
    logger.info("done in %.0fs", time.time() - t0)


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in FILES:
        dest = out_dir / spec["name"]
        if dest.exists() and dest.stat().st_size == spec["size"]:
            digest = _md5(dest)
            if digest == spec["md5"]:
                logger.info("%s already present and md5-clean, skipping", dest)
                continue
            logger.warning("%s md5 mismatch (%s vs expected %s); re-downloading",
                           dest, digest, spec["md5"])
            dest.unlink()
        _download(spec["url"], dest)
        digest = _md5(dest)
        if digest != spec["md5"]:
            logger.error("md5 mismatch on %s after download: got %s expected %s",
                         dest, digest, spec["md5"])
            return 1
        logger.info("verified %s md5 ok", dest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("prosit/data/raw/prosit_paper"),
    )
    args = p.parse_args()
    sys.exit(main(args.out_dir))
