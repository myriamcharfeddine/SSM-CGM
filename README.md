# Streaming multimodal glucose forecasting across the glycemic spectrum

Master's thesis code and results, Myriam Charfeddine, EPFL, completed at Harvard DBMI.
Supervised by Chirag Patel and Shakson Isaac (Harvard DBMI). Committee: Maria Brbic (EPFL).

## Model

All forecasting in this repository runs on **SSM-CGM-Stream**, a Mamba-2-based
streaming state-space architecture. This ports SSM-CGM-Stream to the AI-READI
multimodal cohort and builds every downstream analysis in the thesis on top of it:
forecasting and personalization, event-aware scenario reasoning, environmental
exposure analysis, and hidden-state interpretability and phenotyping.

> Isaac, S., Collin, Y., & Patel, C.J. (2025). *SSM-CGM: Interpretable State-Space
> Forecasting Model of Continuous Glucose Monitoring for Personalized Diabetes
> Management.* NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H).
> [arXiv:2510.04386](https://arxiv.org/abs/2510.04386)

```bibtex
@article{isaac2025ssmcgm,
  title={SSM-CGM: Interpretable State-Space Forecasting Model of Continuous Glucose Monitoring for Personalized Diabetes Management},
  author={Isaac, Shakson and Collin, Yentl and Patel, Chirag J.},
  journal={arXiv preprint arXiv:2510.04386 [cs.LG]},
  year={2025},
  note={Accepted at the NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)}
}
```

## Repository structure
SSM-CGM/
├── ssmcgm/ # SSM-CGM-Stream: model, data, training, evaluation code
├── configs/ # AI-READI stream configs and ablations
├── scripts/ # Training, evaluation, and analysis scripts (flat, see mapping below)
├── Preprocessing/ # AI-READI cohort selection and multimodal dataset construction
├── experiments_outputs/ # Final figures, tables, and checkpoint reported in the thesis
│ ├── model/ # best_model_checkpoint.pt and resolved eval configs
│ ├── tables/ #  LaTeX and JSON result tables
│ └── experiments_scripts_figures/ # Per-chapter figures, see mapping below
├── Benchmarking/ # Earlier-phase window-based forecasting experiments (pre-streaming)
├── Counterfactuals/ # Earlier-phase counterfactual simulation and plausibility checks
├── MealDetection/ # Base SSM-CGM meal detector, not the meal analysis in this thesis
├── Miscellaneous/ # Embedding visualizations and error analyses
├── SSM_CGM.py # Base SSM-CGM entry point (Isaac et al. 2025)
├── environment.yml
├── LICENSE
└── README.md

`Benchmarking/`, `Counterfactuals/`, `MealDetection/`, and `Miscellaneous/` hold
earlier, window-based experiments run before the pivot to the streaming
architecture. They are not the pipeline behind the results in the thesis and are
kept for project history.

## Thesis chapter to repository mapping

| Chapter | Content | Figures and tables |
|---|---|---|
| 3 | Data, model, and evaluation protocol | `experiments_outputs/experiments_scripts_figures/Preprocessing_cohort_selection/` |
| 4 | Forecasting performance and personalization | `experiments_outputs/experiments_scripts_figures/Forecasting_personalization/` |
| 5 | Event-aware scenario reasoning, exercise | `experiments_outputs/experiments_scripts_figures/Exercise_detector_model/` |
| 5 | Event-aware scenario reasoning, meal | `experiments_outputs/experiments_scripts_figures/Meal_counterfactual/` |
| 6 | Environmental exposure analysis | `experiments_outputs/experiments_scripts_figures/Environmental_exposure/` |
| 7 | Interpretability and hidden-state phenotyping | `experiments_outputs/experiments_scripts_figures/Interpretability/` and `Clinical_hidden_state_phenotyping/` |

Each figure's generating script is named the same as the figure file, or close to
it, and lives in `scripts/`. `experiments_outputs/tables/_scripts/` holds the two
scripts (`collect_latest_results.py`, `make_report_tables.py`) that assemble the
result tables from raw run outputs.

## Reproducing the reported model

```bash
conda env create -f environment.yml
conda activate ssmcgm

python scripts/train_stream_aireadi.py \
  --config configs/aireadi_stream_full.yaml \
  --smoke --device cpu
```

The checkpoint reported in the thesis is
`experiments_outputs/model/best_model_checkpoint.pt`, the epoch-5
best-validation checkpoint from a 10-epoch training run (validation pinball loss
3.286316), evaluated on the `adapt6h_seed42` split.

## License

Distributed for academic and non-commercial research use under the terms in
`LICENSE`. The base SSM-CGM software was developed by Shakson Isaac, Yentl
Collin, and Chirag Patel at Harvard University.
