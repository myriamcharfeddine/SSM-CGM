"""
subgroup_analysis.py
====================
Subgroup analysis of Exp C personalization results.

Merges personalization_metrics_{split}.csv with:
  - participant_static_features.parquet  (HbA1c, BMI, medications, study group, site)
  - cohort.csv                           (clinical site, BMI, HbA1c, stratum)

Analyzes MAE improvement by:
  - participants_study_group
  - HbA1c quartile
  - BMI quartile
  - med_insulin (bool)
  - med_any_diabetes_drug (bool)
  - clinical site

Outputs:
  Data/results/exp_C_personalization/subgroup_analysis_{split}.csv
  Data/results/exp_C_personalization/subgroup_analysis_{split}_detail.csv

Usage:
  python subgroup_analysis.py
  python subgroup_analysis.py --split val
  python subgroup_analysis.py --metrics-dir Data/results/exp_C_personalization/sweep_adapt48h
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

_DATA_ROOT   = Path("/home/myriamcharfeddine/CGM/Data")
_STATIC_F    = _DATA_ROOT / "enriched_multimodal/participant_static_features.parquet"
_COHORT_F    = _DATA_ROOT / "enriched_multimodal/cohort.csv"
_DEFAULT_DIR = _DATA_ROOT / "results/exp_C_personalization"


def parse_args():
    p = argparse.ArgumentParser(description="Subgroup analysis of Exp C personalization")
    p.add_argument("--split",            choices=["val", "test"], default="test")
    p.add_argument("--metrics-dir",      type=Path, default=_DEFAULT_DIR,
                   help="Directory containing personalization_metrics_{split}.csv")
    p.add_argument("--out-dir",          type=Path, default=None,
                   help="Output directory (default: same as --metrics-dir)")
    p.add_argument("--static-features",  type=Path, default=_STATIC_F,
                   help="participant_static_features.parquet")
    p.add_argument("--cohort",           type=Path, default=_COHORT_F,
                   help="cohort.csv")
    return p.parse_args()


def quartile_label(series: pd.Series, col: str) -> pd.Series:
    labels = [f"Q1 (low {col})", f"Q2", f"Q3", f"Q4 (high {col})"]
    return pd.qcut(series, q=4, labels=labels, duplicates="drop")


def subgroup_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, grp in df.groupby(group_col, observed=True):
        n = len(grp)
        n_imp = int(grp["improved_mae"].sum())
        mae_imp = grp["mae_improvement"]
        t_stat, p_val = scipy_stats.ttest_1samp(mae_imp.dropna(), 0)
        rows.append({
            "subgroup":                group_col,
            "value":                   str(val),
            "n_participants":          n,
            "n_improved":              n_imp,
            "pct_improved":            round(n_imp / n * 100, 1) if n > 0 else np.nan,
            "mean_global_mae":         round(grp["global_mae"].mean(), 3),
            "mean_personalized_mae":   round(grp["personalized_mae"].mean(), 3),
            "mean_mae_improvement":    round(mae_imp.mean(), 3),
            "median_mae_improvement":  round(mae_imp.median(), 3),
            "std_mae_improvement":     round(mae_imp.std(), 3),
            "mean_global_rmse":        round(grp["global_rmse"].mean(), 3),
            "mean_personalized_rmse":  round(grp["personalized_rmse"].mean(), 3),
            "mean_global_tir":         round(grp["global_tir"].mean(), 3),
            "mean_personalized_tir":   round(grp["personalized_tir"].mean(), 3),
            "t_stat":                  round(t_stat, 3),
            "p_value":                 round(p_val, 4),
            "significant_p05":         bool(p_val < 0.05),
        })
    return pd.DataFrame(rows).sort_values("mean_mae_improvement", ascending=False)


def print_subgroup(label: str, df: pd.DataFrame) -> None:
    print(f"\n  ── {label} ──")
    for _, row in df.iterrows():
        sig = "*" if row["significant_p05"] else " "
        print(f"    {str(row['value']):<35}  n={row['n_participants']:>4}  "
              f"improved={row['pct_improved']:>5.1f}%  "
              f"ΔMAE={row['mean_mae_improvement']:>+7.3f}  p={row['p_value']:.3f}{sig}")


def main():
    args = parse_args()
    out_dir = args.out_dir or args.metrics_dir

    metrics_path = args.metrics_dir / f"personalization_metrics_{args.split}.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    metrics_df["participant_id"] = metrics_df["participant_id"].astype(str)
    print(f"[load] {len(metrics_df)} participants from {metrics_path}")

    static_df = pd.read_parquet(args.static_features)
    static_df["participant_id"] = static_df["participant_id"].astype(str)
    print(f"[load] {len(static_df)} rows from participant_static_features")

    cohort_df = pd.read_csv(args.cohort)
    cohort_df["participant_id"] = cohort_df["participant_id"].astype(str)
    print(f"[load] {len(cohort_df)} rows from cohort")

    # Merge
    df = metrics_df.merge(
        static_df[[
            "participant_id", "participants_study_group", "participants_clinical_site",
            "hba1c_percent_baseline", "bmi_baseline",
            "med_insulin", "med_any_diabetes_drug",
            "med_glp1_or_gip_glp1", "med_sglt2", "med_metformin",
        ]],
        on="participant_id", how="left"
    ).merge(
        cohort_df[["participant_id", "clinical_site", "BMI", "HbA1c", "stratum"]],
        on="participant_id", how="left"
    )

    # Fill BMI / HbA1c from cohort if missing from static features
    df["bmi_use"]   = df["bmi_baseline"].fillna(df["BMI"])
    df["hba1c_use"] = df["hba1c_percent_baseline"].fillna(df["HbA1c"])
    df["site_use"]  = df["participants_clinical_site"].fillna(df["clinical_site"])

    n_merged = df[["hba1c_use", "bmi_use"]].notna().all(axis=1).sum()
    print(f"[merge] {n_merged}/{len(df)} participants have both HbA1c and BMI")

    # Create quartile columns
    df["hba1c_quartile"] = quartile_label(df["hba1c_use"].dropna().reindex(df.index), "HbA1c")
    df["bmi_quartile"]   = quartile_label(df["bmi_use"].dropna().reindex(df.index), "BMI")

    # ── Run subgroup analyses ──────────────────────────────────────────────────
    subgroups = {
        "study_group":        ("participants_study_group", None),
        "clinical_site":      ("site_use",                None),
        "hba1c_quartile":     ("hba1c_quartile",          None),
        "bmi_quartile":       ("bmi_quartile",            None),
        "med_insulin":        ("med_insulin",              None),
        "med_any_diabetes":   ("med_any_diabetes_drug",   None),
    }

    print(f"\n{'='*62}")
    print(f"  SUBGROUP ANALYSIS — split={args.split}  n={len(df)}")
    print(f"  Global MAE: {df['global_mae'].mean():.3f}  →  "
          f"Personal MAE: {df['personalized_mae'].mean():.3f}  "
          f"(+{df['mae_improvement'].mean():.3f})")
    print(f"  Improved: {df['improved_mae'].sum()}/{len(df)} ({df['improved_mae'].mean()*100:.1f}%)")
    print(f"{'='*62}")

    all_rows = []
    for name, (col, _) in subgroups.items():
        if col not in df.columns or df[col].isna().all():
            print(f"\n  [SKIP] {name}: column {col} missing or all-NaN")
            continue
        sub_df = subgroup_stats(df, col)
        sub_df.insert(0, "subgroup_name", name)
        all_rows.append(sub_df)
        print_subgroup(name, sub_df)

    # Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    detail_path = out_dir / f"subgroup_analysis_{args.split}_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"\n[save] Detail → {detail_path}")

    # Wide summary: one row per subgroup × value, metrics side by side
    summary_path = out_dir / f"subgroup_analysis_{args.split}.csv"
    detail_df.to_csv(summary_path, index=False)
    print(f"[save] Summary → {summary_path}")

    # Also save enriched metrics with subgroup columns for further analysis
    enriched_path = out_dir / f"personalization_metrics_{args.split}_enriched.csv"
    keep_cols = list(metrics_df.columns) + [
        "participants_study_group", "site_use", "hba1c_use", "bmi_use",
        "hba1c_quartile", "bmi_quartile",
        "med_insulin", "med_any_diabetes_drug", "med_glp1_or_gip_glp1",
        "med_sglt2", "med_metformin", "stratum",
    ]
    df[[c for c in keep_cols if c in df.columns]].to_csv(enriched_path, index=False)
    print(f"[save] Enriched metrics → {enriched_path}")


if __name__ == "__main__":
    main()
