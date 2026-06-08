# Myriam Charfeddine's SSM-CGM Upgrades

This document separates the original SSM-CGM repository from the extensions added in this fork. The added work turns the original forecasting codebase into a clinically enriched, participant-disjoint, and personalization-aware CGM forecasting pipeline.

## Original Base

The upstream SSM-CGM project already provided:

- the Mamba/TFT forecasting model implementation;
- baseline benchmarking scripts for CGM forecasting;
- interpretability and counterfactual analysis notebooks;
- meal-detection experiments on CGMacros;
- model evaluation utilities and benchmark notebooks.

Those components remain in the repository as the base model and reference experiments.

## Added Upgrade Layer

The added work is organized as a named upgrade layer on top of the original codebase:

1. Enriched multimodal preprocessing
   - Builds CGM/wearable/clinical/static multimodal tables.
   - Adds demographics, clinical measurements, medication indicators, and derived static covariates.
   - Applies cohort selection, duration filters, gap-aware segmentation, completeness checks, and stratum validation.

2. Experiment C participant-disjoint evaluation
   - Creates train/validation/test splits by participant rather than by window.
   - Separates global-training eligibility from held-out participant evaluation eligibility.
   - Labels context, adaptation, and evaluation periods for unseen participants.

3. Dynamic + static forecasting upgrades
   - Adds Experiment A/B/C local scripts for dynamic-only, dynamic + static, and participant-disjoint global models.
   - Keeps `participant_id` as the grouping key while excluding it from predictive static categoricals by default.
   - Adds controls for context length, horizon, validation windows, batch size, worker count, learning rate, dropout, weight decay, and checkpoint policy.

4. Cloud-scale Experiment C tuning
   - Adds Batch/W&B launch scripts for private long-running Experiment C jobs.
   - Adds memory-survival controls for large rolling-window validation, including shared-memory sizing, reduced validation workers, scalar-only validation diagnostics, and optional final-evaluation skipping.
   - Adds a W&B notebook for comparing tuning runs and selecting checkpoints.

5. Personalization and subgroup analysis
   - Adds a C1 personalization protocol that freezes the global model and fine-tunes the prediction head per held-out participant.
   - Adds adaptation-window sweeps.
   - Adds subgroup and stratum summaries for clinical interpretation of personalization gains.

## Main Added Files

| Upgrade area | Files |
| --- | --- |
| Enriched multimodal data | `Preprocessing/create_multimodal_with_clinical.py`, `Preprocessing/cohort_selection.py` |
| Experiment C split and model-ready data | `Preprocessing/create_experiment_c_split.py`, `Preprocessing/prepare_ssmcgm_data_enriched.py`, `Preprocessing/prepare_ssmcgm_data_experiment_c.py` |
| Experiment A/B/C training | `Benchmarking/Day1/mamba288_local.py`, `Benchmarking/Day1/mamba288_static_local.py`, `Benchmarking/Day1/mamba288_static_participant_split_local.py` |
| Personalization and subgroup analysis | `Benchmarking/Day1/personalize_exp_C_head.py`, `Benchmarking/Day1/sweep_adaptation.py`, `Benchmarking/Day1/subgroup_analysis.py` |
| Cloud tuning and monitoring | `scripts/submit_exp_C_tuning_batch.sh`, `scripts/run_exp_C_tuning_cloud.sh`, `scripts/launch_exp_C_small_sweep.py`, `notebooks/exp_C_tuning_results.ipynb` |

## Public Repository Boundary

The repository publishes code, experiment design, and reproducible configuration. It intentionally excludes raw AI-READI/CGMacros data, protected clinical exports, generated participant-level predictions, model checkpoints, Lightning logs, and private result directories.

Cloud scripts are parameterized through environment variables. Private GCP project IDs, service accounts, container images, buckets, W&B credentials, and result paths should be supplied at runtime rather than committed as defaults.
