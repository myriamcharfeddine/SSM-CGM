# CGM Project Progress: Static Features and Personalization

This page summarizes the current extension work on SSM-CGM for continuous glucose forecasting. The public repository contains code and experiment configuration only; raw clinical, wearable, CGM, and participant-level result files are intentionally excluded.

## Research Direction

The project extends the original SSM-CGM forecasting pipeline toward a more clinically realistic setting:

- compare dynamic-only forecasting against models that include static clinical and demographic covariates;
- evaluate participant-disjoint generalization, where validation and test participants are unseen during global training;
- add per-participant adaptation for unseen participants using a frozen global backbone and a small trainable prediction head;
- analyze whether personalization gains differ by clinical subgroup, including HbA1c, BMI, medication indicators, study group, and site.

## Experiment Tracks

| Track | Purpose | Main files | Status |
| --- | --- | --- | --- |
| Experiment A | Dynamic CGM and wearable inputs baseline | `Benchmarking/Day1/mamba288_local.py`, `Benchmarking/Day1/mamba288_exp_A.yaml` | Implemented |
| Experiment B | Dynamic + static clinical features with the original train/test setup | `Benchmarking/Day1/mamba288_static_local.py`, `Benchmarking/Day1/mamba288_exp_B.yaml` | Implemented |
| Experiment C | Participant-disjoint dynamic + static training/evaluation | `Benchmarking/Day1/mamba288_static_participant_split_local.py` | Active cloud rerun |
| Personalization C1 | Freeze global Experiment C backbone and fine-tune prediction head per unseen participant | `Benchmarking/Day1/personalize_exp_C_head.py` | Implemented |
| Subgroup analysis | Quantify where personalization improves MAE/RMSE/TIR | `Benchmarking/Day1/subgroup_analysis.py` | Implemented |
| Adaptation sweep | Compare adaptation-window and fine-tuning settings | `Benchmarking/Day1/sweep_adaptation.py` | Implemented |

## Current Experiment C Configuration

The active cloud rerun is configured for a longer and more stable training window:

- 24-hour encoder context: 288 five-minute bins;
- 1-hour prediction horizon: 12 five-minute bins;
- participant-disjoint train/validation/test split;
- static categoricals: clinical site and study group;
- participant ID retained as the time-series group ID but removed from static categorical embeddings by default;
- rolling-window validation/evaluation by default, with an option for last-window-only ablation;
- maximum 20 epochs, minimum 10 epochs, and early stopping patience of 5 validation checks;
- final full rolling-window evaluation is separated from training to keep the cloud training job tractable.

## Privacy and Reproducibility Notes

The repository should not include raw data, protected health information, model checkpoints, Lightning logs, or generated prediction tables. Local paths in scripts point to the project workspace used for the experiments and can be adapted to another environment by changing the data root or command-line arguments.

## Next Steps

- Monitor the active Experiment C rerun and compare validation curves against the previous 3-epoch early-stopped run.
- Run full rolling-window evaluation from the best Experiment C checkpoint.
- Run C1 personalization on validation and test participants.
- Use subgroup analysis to identify which clinical groups benefit most from participant-specific head adaptation.
- Summarize Experiment A/B/C and C1 metrics in a publication-ready table.
