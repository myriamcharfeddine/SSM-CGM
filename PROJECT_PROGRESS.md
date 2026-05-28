# My CGM Project Extensions

This page documents the work I added on top of the original SSM-CGM repository. The goal is to move from a dynamic-sensor forecasting benchmark toward a clinically enriched, participant-disjoint, and personalization-aware CGM forecasting pipeline.

## What I Added Compared With The Original SSM-CGM Work

The original repository centered on the SSM-CGM model, benchmarking, interpretability, counterfactual notebooks, and meal-detection experiments. My current contribution extends that base in four directions:

1. **Enriched multimodal clinical dataset construction**
   - Built a new multimodal data pipeline that merges CGM, wearable streams, participant metadata, clinical measurements, demographics, medication indicators, and derived static features.
   - Added cohort selection logic with duration floors, gap-aware segmentation, completeness checks, forecasting-window generation, and stratum validation.
   - Added static feature exports used by dynamic + static forecasting models.

2. **Experiment C: realistic participant-disjoint evaluation**
   - Designed a participant-level split where validation and test participants are unseen during global training.
   - Separated training eligibility from personalization/evaluation eligibility: fragmented participants can still contribute training windows, while val/test participants must satisfy stricter contiguous coverage rules.
   - Labeled context, adaptation, and evaluation phases for unseen participants.

3. **Dynamic + static Mamba/TFT forecasting strategy**
   - Added dynamic-only, dynamic + static, and participant-disjoint static-feature training scripts.
   - Removed participant ID from predictive static categorical inputs by default to avoid out-of-vocabulary participant embedding collapse.
   - Kept participant ID as the time-series group ID for normalization and sequence construction.
   - Added ablation flags to restore participant-ID embeddings or last-window-only validation when needed.

4. **Personalization and subgroup analysis**
   - Added a C1 personalization protocol: freeze the global Experiment C backbone and fine-tune only the output head or broader prediction head for each unseen participant.
   - Added adaptation sweeps over participant-specific windows and fine-tuning settings.
   - Added subgroup analysis to quantify who benefits from personalization by HbA1c, BMI, medication indicators, study group, and site.

## Added Code Map

| Area | Files | Purpose |
| --- | --- | --- |
| Enriched data pipeline | `Preprocessing/create_multimodal_with_clinical.py` | Build enriched multimodal dataset from exported modality files and clinical/static information. |
| Cohort selection | `Preprocessing/cohort_selection.py` | Apply duration, gap, completeness, sensitivity, and stratum checks; produce cohort, segments, and forecast windows. |
| Experiment C split | `Preprocessing/create_experiment_c_split.py` | Create participant-disjoint train/val/test split with context/adaptation/evaluation labels. |
| Model-ready data | `Preprocessing/prepare_ssmcgm_data_enriched.py`, `Preprocessing/prepare_ssmcgm_data_experiment_c.py` | Convert enriched multimodal/cohort artifacts into model-ready time-series tables and static feature lists. |
| Dynamic baseline | `Benchmarking/Day1/mamba288_local.py`, `Benchmarking/Day1/mamba288_exp_A.yaml` | Experiment A dynamic-only 24 h context model. |
| Static features | `Benchmarking/Day1/mamba288_static_local.py`, `Benchmarking/Day1/mamba288_exp_B.yaml` | Experiment B dynamic + static feature model. |
| Participant-disjoint model | `Benchmarking/Day1/mamba288_static_participant_split_local.py` | Experiment C global model with unseen participant validation/test splits. |
| Personalization | `Benchmarking/Day1/personalize_exp_C_head.py` | Per-participant head adaptation from a global Experiment C checkpoint. |
| Subgroup analysis | `Benchmarking/Day1/subgroup_analysis.py` | Analyze personalization improvements by clinical subgroup. |
| Adaptation sweep | `Benchmarking/Day1/sweep_adaptation.py` | Sweep adaptation windows and fine-tuning settings. |
| Cloud execution | `cloud/scripts/`, `cloud/configs/` | Google Batch scripts and configs for Experiment A/B/C and personalization runs. |

## Current Experiment C Strategy

The active strategy is built around a more deployment-like setting:

- Train a global model on participants seen during training only.
- Validate and test on held-out participants.
- Use dynamic CGM/wearable inputs plus static clinical covariates.
- Exclude participant ID from predictive static embeddings by default.
- Use rolling-window evaluation where possible rather than one last window per participant.
- Run cloud training with up to 20 epochs, at least 10 epochs, and early stopping patience of 5 validation checks.
- Run final rolling-window evaluation and personalization separately from the main training job.

## Privacy and Data Policy

This repository is intended to publish code, experiment design, and reproducible configuration. It intentionally excludes:

- raw CGM and wearable data;
- clinical CSV/parquet/feather exports;
- participant-level generated predictions;
- model checkpoints;
- Lightning logs and cloud result folders.

Local paths in scripts reflect the working environment used for development. They can be changed through script arguments or path constants when reproducing the workflow elsewhere.

## Next Steps

- Monitor the current Experiment C rerun and compare the validation curve against the earlier early-stopped run.
- Run full rolling-window validation/test evaluation from the best checkpoint.
- Run C1 personalization on validation and test participants.
- Summarize Experiment A/B/C/C1 metrics in a compact table.
- Use subgroup analysis to identify clinical groups where personalization helps most.
