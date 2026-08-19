# SSM-CGM Stream for AIREADI  
**Interpretable State-Space Model for Continuous Glucose Forecasting and Counterfactual Analysis**

> **Isaac, S., Collin, Y., & Patel, C.J. (2025).**  
> *SSM-CGM: Interpretable State-Space Forecasting Model of Continuous Glucose Monitoring for Personalized Diabetes Management.*  
> Accepted at the *NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)*.  
> Preprint available on [arXiv:2510.04386](https://arxiv.org/abs/2510.04386)

---

## Overview

**SSM-CGM** is a neural **state-space model** for **interpretable glucose forecasting** and **counterfactual analysis** using continuous glucose monitoring (CGM) and wearable sensor data.

It integrates CGM with physiological signals such as **heart rate, respiration, stress, etc.** to:
- Improve short-term glucose forecasting over transformer-based baselines  
- Provide interpretable variable and temporal importance  
- Simulate counterfactual forecasts (e.g., “what if heart rate increases?”)

---

## Key Features

- **Mamba-based state-space core** efficient long-context forecasting  
- **Variable Selection Networks (VSNs)** feature-level interpretability  
- **Hidden Attention maps** identify influential time windows  
- **Counterfactual simulation** using sequential g-formula framework  
- **Benchmarking** against Temporal Fusion Transformer (TFT)

---


## Upgrade Layer

This fork presents Myriam Charfeddine's upgrades beyond the original SSM-CGM codebase. The original repository provides the base SSM-CGM model, benchmarking, interpretability, counterfactual, and meal-detection code. The added layer extends it with:

- enriched multimodal clinical preprocessing from CGM, wearable, demographics, medication, and clinical measurement sources;
- Experiment C participant-disjoint train/validation/test construction;
- dynamic-only, dynamic + static, and participant-disjoint Mamba/TFT training scripts;
- cloud-scale Experiment C tuning, W&B run tracking, and memory-survival controls;
- per-participant C1 personalization and subgroup/stratum analysis.

See [`MYRIAM_UPGRADES.md`](MYRIAM_UPGRADES.md) for the original-vs-added separation and [`PROJECT_PROGRESS.md`](PROJECT_PROGRESS.md) for the current strategy and code map.


## AI-READI SSMCGM-Stream Layer

This fork also contains a separate **AI-READI SSMCGM-Stream** layer. It is the new stateful streaming CGM forecasting path, not the original upstream SSM-CGM experiment code and not the older window-only benchmark path.

The stream layer lives in:
- `ssmcgm/`: reusable stream model, data, training, evaluation, causal/proxy, and reporting helpers
- `configs/aireadi_stream_full.yaml` and `configs/ablations/`: AI-READI stream configs
- `scripts/train_stream_aireadi.py`, `scripts/evaluate_stream_aireadi.py`, `scripts/evaluate_stream_diagnostics.py`: train/evaluate/diagnose entrypoints
- `scripts/report/`: report synchronization, table/figure generation, and validation utilities
- `report/`: LaTeX manuscript source and generated `.tex`/figure artifacts for Overleaf/GitHub

Generated training outputs stay local under `outputs/` and are not pushed: checkpoints, parquet predictions, raw metric CSV/JSON snapshots, and the Overleaf ZIP. See [`docs/SSMCGM_STREAM_BOUNDARY.md`](docs/SSMCGM_STREAM_BOUNDARY.md) for the full boundary.

## Repository Structure

```
SSM-CGM/
├── Preprocessing/           # Myriam's enriched data, cohort, and Experiment C split pipeline
├── Benchmarking/            # Forecasting experiments, Experiment A/B/C, personalization, metrics
├── scripts/                 # Myriam's Experiment C cloud tuning and Batch launch scripts
├── notebooks/               # Myriam's Experiment C tuning analysis notebooks
├── Counterfactuals/         # Counterfactual simulations & plausibility checks
├── Interpretability/        # Variable and temporal attribution analyses
├── MealDetection/           # CNN-BiLSTM meal detection model (CGMacros)
├── Miscellaneous/           # Embedding visualizations & error analyses
├── MYRIAM_UPGRADES.md       # Original base vs Myriam upgrade layer
├── PROJECT_PROGRESS.md      # Summary of Myriam's current CGM project extensions
├── SSM_CGM.py               # Core model implementation
├── LICENSE
└── README.md
```

---

## Quick Start


**Environment setup**
```bash
conda env create -f environment.yml
conda activate ssmcgm
```

**Run example**
```bash
python SSM_CGM.py
```

Or explore the Jupyter notebooks and python scripts under:
- `Benchmarking/` for forecasting  
- `Counterfactuals/Notebook/` for counterfactual simulations  
- `Interpretability/` for model attributions  

---

## Dataset Summary

- **AI-READI:** 741 participants with 8–10 days of CGM and wearable data (5-min intervals)  
- **CGMacros:** 45 participants with CGM and annotated meals (used for meal detection training)

<!-- ---

## Notes

- Counterfactual forecasts are *associational*, not causal.  
- AI-READI lacks meal and medication annotations (meals are inferred).  

--- -->

## Citation

If you use this work, please cite:

```bibtex
@article{isaac2025ssmcgm,
  title={SSM-CGM: Interpretable State-Space Forecasting Model of Continuous Glucose Monitoring for Personalized Diabetes Management},
  author={Isaac, Shakson and Collin, Yentl and Patel, Chirag J.},
  journal={arXiv preprint arXiv:2510.04386 [cs.LG]},
  year={2025},
  note={Accepted at the NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)}
}
```
