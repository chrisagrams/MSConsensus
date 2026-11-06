"""Pre-build the casanovo Lance dataset from the translated hold-out MGF.

Mirrors the flow in casanovo_repro/preprocess.py but for the test split:
instantiate the same ModelRunner that ``casanovo sequence`` would build,
hand it the MGF as the test peak path, and call loaders.setup() once.
That triggers depthcharge's MGF→Arrow→Lance write, isolated from the
inference path.

If lance.write_dataset crashes, this harness lets us iterate on the bug
without spinning up GPU inference each time. Once it succeeds, the resulting
``test.lance/`` directory can be passed to all three casanovo checkpoints
(published + retrains) — Lance write happens once, not three times.

Usage:
  python eval/build_casanovo_lance.py --mgf /tmp/spectra_casanovo_translated.mgf \
                                       --out  data/.../eval/casanovo/lance
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgf", required=True, type=Path,
                    help="Translated MGF (with style-correct SEQ tokens).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Lance output directory (will hold test.lance/).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # casanovo.Config validates that every key from its bundled config.yaml is
    # present, so we start from upstream defaults and only override what we
    # need (CPU, lance_dir, no shuffling).
    import importlib.resources as ir
    import copy
    from casanovo.config import Config

    with ir.files("casanovo").joinpath("config.yaml").open("r") as fh:
        merged = yaml.safe_load(fh) or {}
    overrides = {
        "accelerator": "cpu",
        "devices": 1,
        "lance_dir": str(args.out),
        "shuffle": False,
        "num_sanity_val_steps": 0,
    }
    merged.update(overrides)
    tmp_cfg = Path(tempfile.mkstemp(suffix="_casanovo.yaml")[1])
    tmp_cfg.write_text(yaml.safe_dump(merged))
    config = Config(str(tmp_cfg))

    from casanovo.denovo.model_runner import ModelRunner

    out_root = args.out / "_runner_workdir"
    out_root.mkdir(parents=True, exist_ok=True)

    with ModelRunner(
        config=config,
        model_filename=None,
        output_dir=out_root,
        output_rootname="prebuild",
        overwrite_ckpt_check=False,
    ) as runner:
        runner.initialize_trainer(train=False)
        runner.initialize_tokenizer()
        runner.initialize_data_module(
            train_paths=None,
            valid_paths=None,
            test_paths=[str(args.mgf)],
        )
        print(f"setting up loaders → triggers MGF→Lance write into {args.out}/test.lance")
        runner.loaders.setup(stage="test", annotated=True)
        print("Lance build succeeded.")

    print(f"  output: {args.out}/test.lance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
