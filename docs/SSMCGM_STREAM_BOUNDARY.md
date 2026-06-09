# SSMCGM-Stream Boundary

This document defines what belongs to the AI-READI SSMCGM-Stream layer in this fork and what should remain local/generated. It is meant to keep the GitHub repository clean when pushing code, report source, and reproducibility assets.

## What SSMCGM-Stream Is

SSMCGM-Stream is the stateful AI-READI CGM forecasting path added in this fork. It streams each clean `participant_id + segment_id` chronologically, initializes recurrent state from static context, carries state only inside the same segment, and decodes multi-horizon glucose forecasts from anchor states.

The current production stream mode is `stateful_stream` with Mamba scan on CUDA. The debug/comparison mode is `windowed_debug`, which uses fixed 24-hour contexts with T1DEXI-style batch size.

## What It Is Not

SSMCGM-Stream is not the original upstream SSM-CGM code path. The upstream code, benchmarking notebooks, counterfactual examples, interpretability notebooks, and earlier Experiment C work remain separate.

SSMCGM-Stream is also not a raw-results repository. Model checkpoints, parquet predictions, participant-level generated outputs, and local experiment folders are produced under `outputs/` and should not be pushed.

## Path Ownership

| Path | Role | Commit to GitHub? |
| --- | --- | --- |
| `ssmcgm/` | Reusable stream package: data loading, model components, training, evaluation, causal/proxy helpers | Yes |
| `configs/aireadi_stream_full.yaml` | Main AI-READI stream config | Yes |
| `configs/ablations/` | Ablation configs for stream runs | Yes |
| `scripts/train_stream_aireadi.py` | Stream training entrypoint | Yes |
| `scripts/evaluate_stream_aireadi.py` | Stream checkpoint evaluation entrypoint | Yes |
| `scripts/evaluate_stream_diagnostics.py` | Diagnostic and persistence-baseline evaluation entrypoint | Yes |
| `scripts/run_aireadi_stream_suite.sh` | Convenience run wrapper | Yes |
| `scripts/run_stream_diagnostics.sh` | Convenience diagnostic wrapper | Yes |
| `scripts/launch_ablation_probes.sh` | Ablation probe wrapper | Yes |
| `scripts/report/` | Report inventory, synchronization, table/figure generation, and validation | Yes |
| `report/` | LaTeX manuscript source plus generated `.tex` tables and `.png` figures | Yes, excluding build logs/PDF and raw metric CSV/JSON snapshots |
| `outputs/` | Local training/evaluation outputs, checkpoints, predictions, raw metrics | No |
| `aireadi_ssmcgm_report_overleaf.zip` | Local upload artifact for Overleaf | No |

## Canonical Commands

Train the current CUDA stateful stream model:

```bash
python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --device cuda \
  --train-mode stateful_stream \
  --batch-size 8
```

Run the fixed-window debug mode:

```bash
python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --smoke \
  --device cuda \
  --train-mode windowed_debug \
  --batch-size 32
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --ckpt outputs/aireadi_stream_full/checkpoints/best_model_checkpoint.pt \
  --device cuda
```

Regenerate report material from local outputs:

```bash
python scripts/report/sync_stream_results_to_report.py --outputs-root outputs --report-dir report
python scripts/report/update_report_from_results.py --outputs-root outputs --report-dir report
python scripts/report/validate_report_outputs.py --outputs-root outputs --report-dir report
```

Compile the report locally:

```bash
cd report
make
```

## Git Hygiene

Before pushing, inspect staged files and make sure none of these are included:

- `outputs/`
- `*.pt`, `*.pth`, `*.ckpt`
- `*.parquet`, `*.feather`
- generated raw metric CSV/JSON/YAML snapshots under `report/tables/generated/`
- `report/main.pdf` and LaTeX build intermediates
- `aireadi_ssmcgm_report_overleaf.zip`

The repository should contain the code and manuscript source needed to reproduce and explain the run, while large/private/generated experiment artifacts remain local.
