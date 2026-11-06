"""Convert the Prosit-paper HDF5 files into DLOmix-style parquet shards.

The figshare HDF5s (``traintest_hcd.hdf5`` / ``holdout_hcd.hdf5``) are already
in the original Prosit numerical format. DLOmix's ``FragmentIonIntensityDataset``
wants parquet. This script just unpacks: it inverts the integer sequence
encoding using the *upstream* alphabet (``prosit_repro.upstream.constants.ALPHABET``)
and writes the columns DLOmix expects, no chemistry of our own.

The only translation is parens-style upstream mod tokens (``M(ox)``) -> UNIMOD
bracket (``M[UNIMOD:35]``), and explicit ``C[UNIMOD:4]`` injection (since the
upstream alphabet treats every C as carbamidomethyl by mass-table convention,
but DLOmix's parser benefits from the explicit token). Both translations are
lossless: the integer ``2`` in ``sequence_integer`` always means CAM-C; ``21``
always means oxidized M.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prosit_repro.upstream.constants import ALPHABET as UPSTREAM_ALPHABET

logger = logging.getLogger("hdf5_to_parquet")

# Translate upstream tokens to UNIMOD bracket form for DLOmix-side compat.
_TRANSLATE = {
    "C": "C[UNIMOD:4]",      # upstream alphabet treats C as carbamidomethyl by default
    "M(ox)": "M[UNIMOD:35]",
}
INV_ALPHABET: dict[int, str] = {
    idx: _TRANSLATE.get(tok, tok) for tok, idx in UPSTREAM_ALPHABET.items()
}


def _decode_sequence(int_row: np.ndarray) -> str:
    out: list[str] = []
    for x in int_row:
        x = int(x)
        if x == 0:
            break
        tok = INV_ALPHABET.get(x)
        if tok is None:
            raise ValueError(f"unknown alphabet index {x} (upstream alphabet has {len(UPSTREAM_ALPHABET)} entries)")
        out.append(tok)
    return "".join(out)


def _try_decode_sequences(seq_dataset: h5py.Dataset) -> np.ndarray:
    arr = seq_dataset[...]
    if arr.dtype.kind in ("S", "O", "U"):
        return np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
    if arr.ndim == 2:
        return np.array([_decode_sequence(row) for row in arr], dtype=object)
    raise ValueError(f"unexpected sequence dataset shape/dtype: {arr.shape} {arr.dtype}")


def _list_keys(h: h5py.File) -> list[str]:
    keys: list[str] = []
    h.visit(lambda name: keys.append(name))
    return keys


def _required(h: h5py.File, *candidates: str) -> str:
    keys = set(_list_keys(h))
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"none of {candidates} in HDF5; have: {sorted(keys)[:30]}")


def _peptide_split(peps: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(peps)
    val_set = set(rng.choice(unique, size=int(round(len(unique) * val_fraction)),
                             replace=False))
    return np.array([p in val_set for p in peps], dtype=bool)


def _write_shards(df: pd.DataFrame, out_path: Path, shard_rows: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_rows is None or len(df) <= shard_rows:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out_path))
        logger.info("wrote %s (%d rows)", out_path, len(df))
        return
    out_path.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, len(df), shard_rows)):
        chunk = df.iloc[start:start + shard_rows]
        shard = out_path / f"part-{i:05d}.parquet"
        pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False), str(shard))
        logger.info("wrote %s (%d rows)", shard, len(chunk))


def _build_df(h: h5py.File) -> pd.DataFrame:
    seq_key = _required(h, "sequence_integer", "modified_sequence", "sequence")
    int_key = _required(h, "intensities_raw")
    mz_key  = _required(h, "masses_raw")
    ce_key  = _required(h, "collision_energy_aligned_normed", "collision_energy")
    ch_key  = _required(h, "precursor_charge_onehot", "precursor_charge")

    n = h[int_key].shape[0]
    logger.info("decoding %d rows (seq=%s int=%s mz=%s ce=%s ch=%s)",
                n, seq_key, int_key, mz_key, ce_key, ch_key)

    modified_sequence = _try_decode_sequences(h[seq_key])
    intensities_raw = h[int_key][...].astype(np.float64)
    masses_raw      = h[mz_key][...].astype(np.float64)
    ce_normed       = h[ce_key][...].astype(np.float64).reshape(-1)
    ch_onehot_raw   = h[ch_key][...]
    if ch_onehot_raw.ndim == 1:
        ch_onehot = np.zeros((len(ch_onehot_raw), 6), dtype=np.int64)
        for i, c in enumerate(ch_onehot_raw):
            ci = int(c)
            if 1 <= ci <= 6:
                ch_onehot[i, ci - 1] = 1
        precursor_charge = ch_onehot_raw.astype(np.int64)
    else:
        ch_onehot = ch_onehot_raw.astype(np.int64)
        precursor_charge = (ch_onehot.argmax(axis=1) + 1).astype(np.int64)

    aligned_ce = ce_normed * 100.0

    return pd.DataFrame({
        "modified_sequence": modified_sequence,
        "precursor_charge": precursor_charge,
        "precursor_charge_onehot": list(ch_onehot),
        "aligned_collision_energy": aligned_ce,
        "collision_energy_aligned_normed": ce_normed,
        "intensities_raw": list(intensities_raw),
        "masses_raw": list(masses_raw),
    })


def main(traintest: Path, holdout: Path, out_dir: Path,
         val_fraction: float, seed: int, shard_rows: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("converting %s", traintest)
    with h5py.File(traintest, "r") as h:
        df = _build_df(h)
    logger.info("traintest: %d rows, %d unique peptides",
                len(df), df["modified_sequence"].nunique())
    val_mask = _peptide_split(df["modified_sequence"].to_numpy(),
                              val_fraction=val_fraction, seed=seed)
    train_df = df.loc[~val_mask].reset_index(drop=True)
    val_df   = df.loc[val_mask].reset_index(drop=True)
    logger.info("split train=%d val=%d (val_fraction=%.3f, seed=%d)",
                len(train_df), len(val_df), val_fraction, seed)

    _write_shards(train_df, out_dir / "train.parquet", shard_rows)
    _write_shards(val_df, out_dir / "val.parquet", shard_rows)

    logger.info("converting %s", holdout)
    with h5py.File(holdout, "r") as h:
        ho_df = _build_df(h)
    _write_shards(ho_df, out_dir / "holdout.parquet", shard_rows)
    logger.info("done. outputs in %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traintest", type=Path,
                   default=Path("prosit/data/raw/prosit_paper/traintest_hcd.hdf5"))
    p.add_argument("--holdout", type=Path,
                   default=Path("prosit/data/raw/prosit_paper/holdout_hcd.hdf5"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("prosit/data/processed/prosit_paper"))
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--shard-rows", type=int, default=500_000)
    args = p.parse_args()
    main(args.traintest, args.holdout, args.out_dir,
         args.val_fraction, args.seed, args.shard_rows)
