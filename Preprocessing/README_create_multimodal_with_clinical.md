# create_multimodal_with_clinical.py

Extension of `create_multimodal.py` that adds optional clinical, demographic, and medication enrichment to the AI-READI multimodal wearable dataset.

> **Constraint**: `create_multimodal.py` is never modified. This script is a copy with enrichment added on top.

---

## What it does

**Step 1 — Modality building** (same as original script):
Loads 8 wearable modalities (CGM, heart rate, respiration, oxygen, sleep, activity, stress, calories) from GCS, resamples to 5-min bins, and merges into a single time-series parquet (one row per participant × timestamp).

**Step 2 — Clinical enrichment** (new, optional):
Left-joins static clinical features onto every time-series row. Never drops rows or participants.

| Source file | Columns added |
|---|---|
| `participants.tsv` | `participants_study_group`, `participants_clinical_site`, `participants_age`, `participants_study_visit_date` |
| `measurement.csv` | 12 baseline measurements (HbA1c, BP, BMI, cholesterol, glucose, C-peptide, etc.) |
| Demographics Excel | Sex, race (one-hot + primary), ethnicity (one-hot), marital status, ancestry |
| Medications Excel | Diabetes drug class flags (`med_metformin`, `med_insulin`, `med_glp1_or_gip_glp1`, `med_sglt2`, etc.) |

---

## Output directory

```
/home/myriamcharfeddine/CGM/Data/enriched_multimodal/
├── final_multimodal_dataset_{timestamp}.parquet       # main enriched time-series
├── final_multimodal_dataset_{timestamp}_metadata.json
├── participant_static_features.parquet                # one row per participant (all enriched cols)
├── participant_static_features.csv
├── participant_measurements_selected_long.parquet     # raw measurement records
├── participant_medications_long.parquet               # raw medication records with matched_keywords
├── clinical_measurement_unit_audit.csv               # unit validation per measurement
├── clinical_measurement_coverage.csv                 # % participants with each measurement
├── medication_class_coverage.csv                     # drug class flags summary
├── demographic_coverage.csv                          # demographic column coverage
└── participant_freetext_misc.csv                     # free-text demographic fields (not joined)
```

---

## How to run

> Always use the `ssmcgm` conda environment.

### Full run (modality building + enrichment) — ~30–60 min

```bash
conda run -n ssmcgm python /home/myriamcharfeddine/CGM/Preprocess/create_multimodal_with_clinical.py \
  --mode standard \
  --fast-groupby \
  --use-columns \
  --workers 4
```

### Enrichment only — ~3–5 min (recommended after first full run)

Skips the slow modality-building loop and re-runs only the clinical enrichment on an existing parquet.

```bash
conda run -n ssmcgm python /home/myriamcharfeddine/CGM/Preprocess/create_multimodal_with_clinical.py \
  --enrich-only \
  --input-parquet /home/myriamcharfeddine/CGM/Data/final_multimodal_dataset_20260501_171205.parquet \
  --mode standard
```

> If `--input-parquet` is omitted, the script auto-discovers the latest `final_multimodal_dataset_*.parquet` in the output directory.

### No enrichment (replicates original script exactly)

```bash
conda run -n ssmcgm python /home/myriamcharfeddine/CGM/Preprocess/create_multimodal_with_clinical.py \
  --mode minimal
```

---

## Enrichment modes

| Mode | What is added |
|---|---|
| `minimal` | Nothing — identical output to original `create_multimodal.py` |
| `standard` | Clinical baselines + demographics (race, sex, ethnicity) + diabetes medication flags |
| `full` | `standard` + optional medication flags (statins, BP drugs, steroids, thyroid, etc.) + medication name summary |

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `minimal` | Enrichment level: `minimal`, `standard`, `full` |
| `--enrich-only` | off | Skip modality building; load existing parquet |
| `--input-parquet` | auto-discover | Path to existing merged parquet (used with `--enrich-only`) |
| `--output-dir` | `…/enriched_multimodal` | Where all outputs are written |
| `--workers` | 1 | Parallel threads for participant merging (use 4 on GPU server) |
| `--fast-groupby` | off | Faster groupby implementation (recommended) |
| `--use-columns` | off | Read only needed columns from parquet (saves memory) |
| `--dry-run` | off | Run enrichment but skip writing the final parquet |

---

## Clinical measurements extracted (OMOP concept IDs)

| Column | Concept ID | Unit |
|---|---|---|
| `clinical_systolic_bp_mmhg_baseline` | 3004249 | mmHg |
| `clinical_diastolic_bp_mmhg_baseline` | 3012888 | mmHg |
| `clinical_resting_hr_bpm_baseline` | 4239408 | beats/min |
| `weight_kg_baseline` | 3025315 | kg |
| `height_cm_baseline` | 3036277 | cm |
| `bmi_baseline` | 4245997 | kg/m² |
| `waist_cm_baseline` | 4172830 | cm |
| `hip_cm_baseline` | 4111665 | cm |
| `waist_to_hip_ratio_baseline` | 44809433 | — |
| `hba1c_percent_baseline` | 3004410 | % |
| `serum_glucose_mgdl_baseline` | 3004501 | mg/dL |
| `c_peptide_ngml_baseline` | 3010084 | ng/mL |
| `serum_insulin_uuml_baseline` | 3016244 | uIU/mL |
| `hdl_cholesterol_mgdl_baseline` | 3007070 | mg/dL |
| `ldl_cholesterol_mgdl_baseline` | 3028288 | mg/dL |
| `triglycerides_mgdl_baseline` | 3022192 | mg/dL |

Each measurement also gets `_n_records`, `_baseline_date`, `_value_range`, and `_days_to_cgm_start` companion columns.

---

## Notes

- **First-value baseline**: for each measurement, only the earliest valid record per participant is used — avoids leaking future clinic visits into the time-series.
- **Unit validation**: each measurement has a list of accepted units with automatic conversions (mmol/L → mg/dL, IFCC → NGSP for HbA1c, etc.). Empty unit strings are accepted for measurements where the unit is implicit (BP, BMI, anthropometrics).
- **`demo_race_primary`**: single string per participant following the same logic as `clinical_data_exploration.ipynb` — single selection → that label; multiple → `race2` tiebreaker or `"Multi-racial (unspecified)"`.
- **Medication `matched_keywords`**: per-row audit column listing which keywords triggered each drug class flag, enabling future debugging without re-running.
