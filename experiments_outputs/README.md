# experiments_outputs

Everything needed to trace and, where possible, regenerate the figures and tables in
`MasterThesisProject_draft/main.tex`. This replaces the old `report/` folder as the
code-facing home for thesis artifacts on GitHub — it holds final images, tables, and
(where the generating code still exists) the exact script that produced each one,
instead of the 45GB of raw intermediate `outputs/` data behind them.

Total size: ~38MB.

Figures under `experiments_scripts_figures/` are organized by thesis chapter/topic,
not by an artificial "generated" bucket — the pipeline-produced figures that used to
live in one flat folder were split up based on which chapter or appendix section
actually `\includegraphics`'d them (checked directly against `chapters/*.tex` and
`appendices/appendix.tex`).

**Shared reporting-pipeline scripts live in one place only** —
`tables/_scripts/` — rather than copied into every figure folder that uses them.
`make_report_figures.py`, `collect_latest_results.py`, `update_report_from_results.py`,
and `stream_report_utils.py` are one pipeline (normally at `scripts/report/` in the
main repo) that produces figures for several chapters (Environmental_exposure,
Forecasting_personalization, Interpretability) plus every table under `tables/`. If
you're looking at a figure in one of those folders and want the code that made it,
check `tables/_scripts/`.

## model/

- `best_model_checkpoint.pt` — the final trained SSM-CGM Stream model. Despite the
  "10epoch" naming on the eval output folders, the config each eval run actually
  loaded (`ckpt:` field) points here: this is a checkpoint from
  `outputs/aireadi_stream_mamba_stateful_5epoch/`, not a separate 10-epoch training run.
- `eval_test_config_resolved.yaml` / `eval_validation_config_resolved.yaml` — the
  resolved configs for the two eval runs that produced the main-report and appendix
  metrics.
- `training_curves.png` — appendix "Training configuration and convergence" figure.

## experiments_scripts_figures/

| Folder | Generating script(s) | Status |
|---|---|---|
| `Environmental_exposure/` | `scripts/plot_environmental_exposure_figures.py` (+ report pipeline, see `tables/_scripts/`) | Mostly traced. `env_temp_sensitivity_*` (3 files) and `fig_case_high_humidity_v2.pdf`: **no generating notebook found** — confirmed deleted (not in git, no `.ipynb_checkpoints`, no trash). Raw output data still exists at `outputs/figures/env_temperature_*` and `outputs/interpretability_v2/`. See `environment_model_trained/` below for the closest recoverable source. |
| `Environmental_exposure/environment_model_trained/` | — | The environment-augmented model and its own eval run, copied from `outputs/environment_model_trained/`. Contains: the checkpoint (`checkpoint/best_model_checkpoint.pt` + `final_model_checkpoint.pt`, ~6.6MB each), its training history/summary, its own eval `config_resolved.yaml`/figures/metrics/hardware (the 175MB `predictions/` raw-array folder was left out), and the 3 root-level interpretability figures + 4 CSV attribution tables that sit directly in that output folder. Note: `config_resolved.yaml` references a checkpoint path `outputs/environment_ht/...` that no longer exists on disk — the folder was evidently renamed to `environment_model_trained` after that config was written; the checkpoint copied here is the one that's actually present. **Removed:** this folder originally also had `original_baseline_eval_test/` — its metrics/config turned out to be byte-identical to what's already in `tables/generated/` and `model/`, so it was dropped as pure duplication. |
| `Clinical_hidden_state_/` | `run_extended_cluster_metabolic_profiles.py`, `recreate_main_appendix_neighborhood_figures_v6.py`, `recreate_hidden_state_dynamics_figures.py`, `recreate_event_4a_a5_v3.py`, `recreate_event_figures_v2.py` (in `_scripts/`) | Traced at the family level. |
| `Interpretability/` | report pipeline, see `tables/_scripts/` | Traced — chapter 07 ("Representation analysis") figures and its appendix supplement (hidden-state probes, local case studies, interpretability/LOMO attribution). |
| `Preprocessing_cohort_selection/` | `Preprocessing/figures_streaming_cohort.ipynb` (in `_scripts/`) | Traced — reads AI-READI cohort tables directly, not `outputs/`. |
| `Exercise_detector_model/Exercise_scenario_model_training/` | `rebuild_figure_13_test_stream_history_ablation.py` | Partially traced. `figure_1` and `figure_8` came from `scripts/generate_final_exercise_figures.py` — **gone**, only a `.pyc` cache remains. |
| `Exercise_detector_model/exercise_detector_2/` | — | **Script gone.** Produced by `scripts/render_exercise_thesis_figure_suite.py`; only a `.pyc` cache remains. `exer_responce_bycadence.png` (in `Exercise_scenario_model_training/`) is the same family — confirmed source: `outputs/exercise_thesis_figures_2/exercise_response_by_cadence_strain.png`. |
| `Exercise_detector_model/Exercise_detector/` | — | **Notebook gone, not recoverable.** Source output data: `notebooks/outputs/exercise_episode_detection_v2/figures_v3/` and `outputs/stage2_residual_diagnostic/`. |
| `Forecasting_personalization/` | `plot_convergence_outcome.py`, `plot_forecasting_progression.py`, `plot_personalization_audit.py`, `state_reset_warmup_sweep.py`, `build_forecasting_report.py` (all in `scripts/forecasting_story/` originally) + report pipeline | Traced — chapter 04 ("Forecasting glycemic spectrum") headline figures and warm-up/state-reset audit. |
| `Meal_counterfactual/` | `plot_f1_leakage_redundancy.py` ... `plot_f6_summary_schematic.py`, `report_style.py` | Traced — found colocated with the images in the thesis draft folder itself. |

Not included: the six hand-made/external architecture diagrams
(`AIREADI_DATA_COLLECTIONS`, `SSMCGMStreamArchitecture`, `SSMCGMarchitecture`,
`StateSpaceModel`, `Streaminght`, `Trati_test_spli_starteggy`) — taken from the
internet or drawn by hand, not produced by any script.

## tables/

`generated/` and `meal_flags/` — all 65 table files, all traced to the SSM-CGM
report pipeline. `_scripts/` holds `make_report_tables.py`,
`collect_latest_results.py`, `update_report_from_results.py`,
`stream_report_utils.py`, and a copy of `report/results_manifest.csv` — the
authoritative source-to-destination trace for every table. This is also the single
canonical copy of these scripts; see the note at the top of this file.

## Known remaining duplication (left in place, not worth removing)

A handful of byte-identical file pairs exist and are harmless — either pre-existing
in the source data or a deliberate "figure lives at its chapter location AND inside
the raw output bundle" choice:

- `fig_interp_global_ht_environment_v1.png` and `fig_interp_environment_accuracy_comparison_v1.png`
  each appear both directly under `Environmental_exposure/` and inside
  `Environmental_exposure/environment_model_trained/` — same bytes, kept in both
  places on purpose (chapter-figure location vs. full output bundle).
- `fig_warmup_audit_state_reset.png` / `var_warm_up_hours.png` (Forecasting_personalization) —
  identical content under two different names, inherited from the source `outputs/` folder.
- `exercise_response_by_baseline_glucose.pdf` / `..._bin.pdf` (exercise_detector_2) —
  same, inherited duplicate naming from `outputs/exercise_thesis_figures_2/`.

## What's deliberately left out

`outputs/` is 45GB; the vast majority of that (32GB in
`static_phenotype_trajectory_stratified_v2/extended_clinical_latent_dynamics_v1/`
alone, plus the 175MB `predictions/` folders inside each eval run) is raw
intermediate data that no figure reads directly — regenerating a figure from
scratch would need it, but *having* the figure doesn't. This folder prioritizes
"what proves and reproduces the thesis's final artifacts" over "full from-scratch
pipeline reproducibility," which would need external storage (Git LFS, cloud
bucket, etc.) rather than a plain GitHub repo.
