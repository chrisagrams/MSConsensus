# MSConsensus benchmarks

Code for the baseline experiments in the MSConsensus paper. Each baseline is
retrained on MSConsensus and on its original reference dataset, then
evaluated on a shared held-out set (PXD053296).

| Directory | Contents |
|---|---|
| `casanovo/` | Casanovo (de novo sequencing) retraining: `casanovo-repro` CLI and per-dataset configs |
| `prosit/` | Prosit (fragment intensity prediction) retraining: `prosit-repro` CLI and configs |
| `gleams/` | GLEAMS (spectrum embedding) retraining: `gleams-repro` CLI and configs |
| `spec2vec/` | Spec2Vec reproduction (numbered scripts `01_` to `05_`) |
| `eval/` | Held-out evaluation: builds model-specific inputs, then `run_{casanovo,prosit,gleams,ms2pip}.py` |
| `analysis/` | Dataset analyses: peptide diversity, vocabulary audit, spectra per peptide |

Each of `casanovo/`, `prosit/`, `gleams/`, and `spec2vec/` is a separate
Python project with its own dependencies (GLEAMS requires Python 3.10).

## Paths

All paths are relative. Run commands from this `benchmarks/` directory.

- `data/`: input datasets. Download MSConsensus from Harvard Dataverse
  ([doi:10.7910/DVN/NUCT4N](https://doi.org/10.7910/DVN/NUCT4N)) and place
  it here, e.g. `data/concensus_spec_30M/`, `data/massive_kb_v1_shared/`,
  `data/holdout_pxd053296/`.
- `runs/`: training checkpoints for the full-scale Casanovo runs.
- `<model>/data`, `<model>/models`, `<model>/logs`: per-baseline
  intermediate files.

These directories are ignored by git. To keep data somewhere else, symlink
`data/` to it.

## Example

```bash
cd benchmarks
uv sync --project casanovo
uv run --project casanovo casanovo-repro prep --config casanovo/configs/consensus.yaml
uv run --project casanovo casanovo-repro train --config casanovo/configs/consensus.yaml

# Evaluate on the held-out set
CASANOVO_BIN=casanovo/.venv/bin/casanovo \
  uv run --project casanovo python eval/run_casanovo.py --help
```

Experiments are logged to Weights & Biases (run `wandb login` first, or set
`WANDB_MODE=offline`).
