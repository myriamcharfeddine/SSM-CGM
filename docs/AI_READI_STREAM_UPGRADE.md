# AI-READI Stream Upgrade

## Added Files

- `ssmcgm/data/aireadi.py`: AI-READI parquet loader, schema inference, schema mapping, participant-heldout split reuse, train-only scaling/encoding, scenario values and masks, and participant+segment stream construction.
- `ssmcgm/models/aireadi_stream.py`: AI-READI stream model using the T1DEXI SSMCGM-Stream primitives: static h0, static FiLM, grouped fusion, MES stream stack, masked scenario decoder, base+scenario-effect decomposition, and residual-current forecasting.
- `ssmcgm/training/aireadi_stream_trainer.py`: stateful truncated-BPTT trainer that walks participant segments in chunks and never materializes rolling windows.
- `ssmcgm/evaluation/aireadi_streaming.py`: forecast/factual/proxy scenario evaluation, personalization sweep, bias diagnostics, subgroup metrics, clinical safety metrics, memory, and latency reporting.
- `configs/aireadi_stream_full.yaml`: default AI-READI-Stream-Full config.
- `scripts/train_stream_aireadi.py`: training entrypoint.
- `scripts/evaluate_stream_aireadi.py`: evaluation entrypoint.
- `scripts/benchmark_stream_aireadi.py`: hardware benchmark entrypoint.
- `scripts/run_aireadi_stream_suite.sh`: train/evaluate/benchmark wrapper.

The requested T1DEXI reference stream modules were also ported under `ssmcgm/stream`, `ssmcgm/models/ssmcgm_stream.py`, `ssmcgm/data/streaming.py`, `ssmcgm/evaluation/{streaming,target_transform,metrics}.py`, `ssmcgm/training/stream_trainer.py`, and the causal helper files listed in the request.

## Stream Model vs Old Window Model

The old path builds overlapping 24h/48h windows and trains a window model. The new AI-READI path streams each `participant_id + segment_id` chronologically. The recurrent state is initialized once from static covariates, advanced through observed dynamic rows, detached at chunk boundaries for truncated BPTT, and decoded at forecast anchors.

Future scenario variables do not update the stream state. They enter only the horizon decoder with explicit masks, so `predmeal_flag=0` with mask `0` means unknown future meal, not known no meal.

The default target is residual-current:

```text
future glucose = current observed glucose + predicted residual
```

## Smoke Training

```bash
python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --smoke \
  --device cpu
```

CUDA smoke, when a CUDA build of PyTorch is installed:

```bash
python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --smoke \
  --device cuda
```

## Full Training

```bash
python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --device cuda
```

Outputs are written under `outputs/aireadi_stream_full/`:

- `checkpoints/`
- `predictions/`
- `metrics/`
- `figures/`
- `hardware/`
- `config_resolved.yaml`
- `schema_mapping.json`

## Evaluation

```bash
python scripts/evaluate_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --ckpt outputs/aireadi_stream_full/checkpoints/best_model_checkpoint.pt \
  --device cuda
```

This evaluates one checkpoint across:

- `forecast_only`
- `factual_future`
- `meal_proxy`
- `activity_proxy`
- `sleep_rest_proxy`

It writes one complete prediction file at `outputs/aireadi_stream_full/predictions/predictions.parquet`, plus overall, horizon, scenario, personalization sweep, bias diagnostic, subgroup, clinical safety, memory, and latency outputs.

## Valid AI-READI Assumptions

- Existing `final_multimodal_dataset_*.parquet`, `participant_static_features.parquet`, `cohort.csv`, and Experiment C split artifacts are reused.
- Streams are sorted by `participant_id`, `segment_id`, and per-segment `time_idx`.
- Segment boundaries are derived from 5-minute timestamp continuity after observed CGM filtering, so the model does not stream across long gaps.
- Static covariates initialize `h0` and FiLM-modulate the dynamic stream.
- Scalers and categorical encoders are fit on train participants only and reused for validation/test participants.
- Scenario support is limited to AI-READI-supported proxies: meal proxy, activity-like wearable proxy, and sleep/rest proxy.

## Invalid AI-READI Assumptions

- AI-READI does not contain insulin timing, dose, or IOB action histories.
- `med_insulin` is static medication-profile metadata, not an editable insulin action.
- Insulin causal ranking losses are therefore disabled and logged as disabled.
- Meal proxy scenarios are not proven causal meal interventions because AI-READI has no meal annotations.
- Activity and sleep/rest scenarios are wearable-derived proxy scenarios, not proven causal actions.
