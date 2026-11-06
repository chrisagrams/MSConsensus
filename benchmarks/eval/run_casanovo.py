"""Run Casanovo evaluation on a prebuilt Lance dataset.

Inputs:
  --lance     Path to a casanovo-compatible Lance dataset (built by
              eval/build_casanovo_lance.py from a translated, named-bracket
              MGF — see eval/translate_mgf_to_casanovo.py).
  --model     Path to a .ckpt or 'published' to let casanovo auto-download
              the official MassIVE-KB v5 weights.
  --label     Output dir name under HOLDOUT/eval/casanovo/<label>/

Invokes ``casanovo sequence --evaluate`` and parses the casanovo log for
amino-acid + peptide precision/recall, writing one row to metrics.csv.

Why Lance and not MGF: casanovo's MGF→Lance conversion is the slow and
fragile part of the pipeline. By pre-building Lance once with
build_casanovo_lance.py we (a) isolate parsing crashes from the eval, and
(b) reuse the same Lance across every checkpoint we evaluate.
"""
from __future__ import annotations

import argparse
import importlib.resources as ir
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

HOLDOUT = Path("data/holdout_pxd053296")
# Path to the casanovo CLI; override with $CASANOVO_BIN (e.g. casanovo/.venv/bin/casanovo).
CASANOVO_BIN = Path(os.environ.get("CASANOVO_BIN", shutil.which("casanovo") or "casanovo"))

# Inference perf overrides on top of casanovo's bundled config defaults.
# Empty by default. Things we'd want to override but can't:
#   - precision: bf16-mixed crashes beam_search_decode (fp32 scatter buffer
#     can't accept bf16 source). The casanovo paper config uses bf16-mixed
#     for training but inference is bugged.
#   - n_workers: casanovo.utils.n_workers() hard-codes 0 to avoid a
#     multiprocessing deadlock (see casanovo/utils.py FIXME), and Config
#     rejects the key as "Unrecognized config option".
_PERF_OVERRIDES: dict = {}

# Tokenizer style toggle. The published v5 ckpt was saved with
# MskbPeptideTokenizer; if the runtime uses PeptideTokenizer (default config
# `massivekb_tokenizer: false`), the model emits MSKB-style strings while
# ground-truth SEQ is parsed as named-bracket → 0% peptide precision even
# though AA-level matches. Setting massivekb_tokenizer=true aligns both
# sides on MSKB and makes peptide_precision actually meaningful.
_STYLE_OVERRIDES = {
    "mskb":  {"massivekb_tokenizer": True},
    "named": {"massivekb_tokenizer": False},
}


def _write_eval_config(style: str) -> Path:
    """Materialize a casanovo config: bundled defaults + perf + tokenizer
    overrides for the chosen ``style``. Returns the path to a temp YAML
    usable as ``casanovo sequence -c``."""
    with ir.files("casanovo").joinpath("config.yaml").open("r") as fh:
        merged = yaml.safe_load(fh) or {}
    merged.update(_PERF_OVERRIDES)
    merged.update(_STYLE_OVERRIDES[style])
    tmp = Path(tempfile.mkstemp(suffix="_casanovo_eval.yaml")[1])
    tmp.write_text(yaml.safe_dump(merged))
    return tmp


def run_casanovo(lance_path: Path, model: str | None, output_dir: Path,
                 output_root: str, config_path: Path) -> int:
    cmd = [
        str(CASANOVO_BIN), "sequence", "--evaluate",
        "-d", str(output_dir),
        "-o", output_root,
        "-c", str(config_path),
        "-f",
    ]
    if model is not None:
        cmd += ["-m", model]
    cmd.append(str(lance_path))
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


_AA_PRECISION_RE = re.compile(r"amino acid precision[: ]+([\d.]+)", re.IGNORECASE)
_AA_RECALL_RE = re.compile(r"amino acid recall[: ]+([\d.]+)", re.IGNORECASE)
_PEP_PRECISION_RE = re.compile(r"peptide precision[: ]+([\d.]+)", re.IGNORECASE)


def parse_metrics(log_file: Path) -> dict[str, float]:
    if not log_file.exists():
        return {}
    text = log_file.read_text()
    out: dict[str, float] = {}
    if (m := _AA_PRECISION_RE.search(text)):
        out["aa_precision"] = float(m.group(1))
    if (m := _AA_RECALL_RE.search(text)):
        out["aa_recall"] = float(m.group(1))
    if (m := _PEP_PRECISION_RE.search(text)):
        out["peptide_precision"] = float(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", required=True, type=Path,
                    help="Path to a casanovo Lance dataset.")
    ap.add_argument("--model", default=None,
                    help="Path or URL to .ckpt. Use 'published' to let "
                         "casanovo auto-download the official MassIVE-KB v5.")
    ap.add_argument("--label", required=True,
                    help="Output dir name under HOLDOUT/eval/casanovo/")
    ap.add_argument("--style", choices=["mskb", "named"], required=True,
                    help="SEQ format the Lance was built with. Sets "
                         "massivekb_tokenizer in the runtime config so the "
                         "tokenizer matches both ground-truth SEQ and the "
                         "model's output decoder.")
    args = ap.parse_args()

    out_root = HOLDOUT / "eval/casanovo" / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    config_path = _write_eval_config(args.style)
    overrides = {**_PERF_OVERRIDES, **_STYLE_OVERRIDES[args.style]}
    print(f"using eval config (overrides: {overrides}) at {config_path}")

    model = None if args.model in (None, "published") else args.model
    rc = run_casanovo(args.lance, model, out_root, args.label, config_path)
    print(f"casanovo exit code: {rc}")

    log_file = out_root / f"{args.label}.log"
    metrics = parse_metrics(log_file)

    csv = out_root / "metrics.csv"
    with csv.open("w") as f:
        cols = ["timestamp", "label", "model", "lance",
                "aa_precision", "aa_recall", "peptide_precision"]
        f.write(",".join(cols) + "\n")
        f.write(",".join([
            datetime.utcnow().isoformat(),
            args.label,
            args.model or "published",
            str(args.lance),
            f"{metrics.get('aa_precision', '')}",
            f"{metrics.get('aa_recall', '')}",
            f"{metrics.get('peptide_precision', '')}",
        ]) + "\n")
    print(f"metrics: {metrics}")
    print(f"wrote {csv}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
