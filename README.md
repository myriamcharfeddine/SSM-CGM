# SSM-CGM  
**Interpretable State-Space Model for Continuous Glucose Forecasting and Counterfactual Analysis**

> **Isaac, S., Collin, Y., & Patel, C.J. (2025).**  
> *SSM-CGM: Interpretable State-Space Forecasting Model of Continuous Glucose Monitoring for Personalized Diabetes Management.*  
> Accepted at the *NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)*.  
> Preprint available on [arXiv](https://arxiv.org/).

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

## Repository Structure

```
SSM-CGM/
├── Benchmarking/            # Forecasting experiments & metrics
├── Counterfactuals/         # Counterfactual simulations & plausibility checks
├── Interpretability/        # Variable and temporal attribution analyses
├── MealDetection/           # CNN-BiLSTM meal detection model (CGMacros)
├── Miscellaneous/           # Embedding visualizations & error analyses
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
  journal={arXiv preprint},
  year={2025},
  note={Accepted at the NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)}
}
```

---

## 👥 Authors

- **Shakson Isaac**, Harvard Medical School  
- **Yentl Collin**, Harvard Medical School / ENPC  
- **Chirag J. Patel**, Harvard Medical School  

---

## 📜 License
This project is distributed under a **custom research license**.  
Usage and redistribution are subject to the terms described in `LICENSE`.