"""Phase 0 (Gate A): schema resolution, split confirmation, coverage, k-selection preview.

Static-only clinical phenotype partition project. See build prompt for full spec.
This script is preview-only: it does NOT save any participant-level cluster labels.
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
N_INIT = 25
K_RANGE = list(range(2, 7))
GAP_N_REFS = 20

DATASET = "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
SPLIT_PATH = "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv"
OUTPUT_DIR = "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/static_phenotype_trajectory/step0"

# Requested factor -> resolved parquet column
FACTOR_COLUMN_MAP = {
    "age": "participants_age",
    "bmi": "bmi_baseline",
    "hba1c_baseline": "hba1c_percent_baseline",
    "c_peptide_baseline": "c_peptide_ngml_baseline",
    "triglycerides_baseline": "triglycerides_mgdl_baseline",
    "hdl_cholesterol_baseline": "hdl_cholesterol_mgdl_baseline",
}
FACTORS = ["age", "bmi", "hba1c_baseline", "c_peptide_baseline", "tg_hdl_ratio"]


def gap_statistic(x_scaled, k_range, n_refs, seed):
    rng = np.random.default_rng(seed)
    mins = x_scaled.min(axis=0)
    maxs = x_scaled.max(axis=0)
    gaps = {}
    sk = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
        km.fit(x_scaled)
        log_wk = np.log(km.inertia_)

        ref_log_wks = []
        for _ in range(n_refs):
            ref = rng.uniform(mins, maxs, size=x_scaled.shape)
            km_ref = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
            km_ref.fit(ref)
            ref_log_wks.append(np.log(km_ref.inertia_))
        ref_log_wks = np.array(ref_log_wks)
        gap = ref_log_wks.mean() - log_wk
        sk_k = ref_log_wks.std() * np.sqrt(1 + 1.0 / n_refs)
        gaps[k] = float(gap)
        sk[k] = float(sk_k)
    return gaps, sk


def main():
    print("=" * 80)
    print("STEP 1: Resolve column names")
    print("=" * 80)
    schema = pq.read_schema(DATASET)
    all_cols = set(schema.names)
    for factor, col in FACTOR_COLUMN_MAP.items():
        present = col in all_cols
        print(f"  {factor:28s} -> {col:35s} {'OK' if present else 'MISSING'}")
        assert present, f"Column {col} not found in parquet schema"

    read_cols = ["participant_id", "participants_study_group"] + list(FACTOR_COLUMN_MAP.values())
    df = pd.read_parquet(DATASET, columns=read_cols)
    df = df.drop_duplicates(subset="participant_id").reset_index(drop=True)
    print(f"\n  Loaded {len(df)} unique participants, columns: {list(df.columns)}")

    df = df.rename(columns={v: k for k, v in FACTOR_COLUMN_MAP.items()})
    df["tg_hdl_ratio"] = df["triglycerides_baseline"] / df["hdl_cholesterol_baseline"]

    print("\n" + "=" * 80)
    print("STEP 2: Load split, confirm adapt6h_seed42, participant-held-out")
    print("=" * 80)
    assert "adapt6h_seed42" in SPLIT_PATH, "Split path does not reference adapt6h_seed42"
    assert "adapt48h" not in SPLIT_PATH, "STOP: split path references adapt48h, not adapt6h"
    split_df = pd.read_csv(SPLIT_PATH, dtype={"participant_id": str})
    print(f"  Split file: {SPLIT_PATH}")
    print(f"  Columns: {list(split_df.columns)}")
    counts = split_df["split"].value_counts()
    total = len(split_df)
    print(f"  Total participants in split file: {total}")
    for s, c in counts.items():
        print(f"    {s:12s}: {c:5d}  ({100 * c / total:.2f}%)")
    dup_check = split_df["participant_id"].duplicated().sum()
    print(f"  Duplicate participant_id rows in split file: {dup_check}")

    merged = df.merge(split_df[["participant_id", "split", "stratum"]], on="participant_id", how="inner")
    print(f"  Participants with both dataset row and split assignment: {len(merged)} / {total}")
    missing_from_dataset = total - len(merged)
    if missing_from_dataset:
        print(f"  WARNING: {missing_from_dataset} split participants not found in dataset parquet")

    print("\n" + "=" * 80)
    print("STEP 3: Coverage table (train split)")
    print("=" * 80)
    train = merged[merged["split"] == "train"].copy()
    n_train = len(train)
    print(f"  Train n = {n_train}")

    coverage_cols = FACTORS + ["triglycerides_baseline", "hdl_cholesterol_baseline"]
    coverage = {}
    for col in coverage_cols:
        non_missing = train[col].notna().sum()
        frac = non_missing / n_train
        coverage[col] = {"non_missing": int(non_missing), "fraction": float(frac)}
        print(f"  {col:28s}: {non_missing:5d} / {n_train:5d}  ({frac * 100:.2f}%)")

    complete_case_mask = train[FACTORS].notna().all(axis=1)
    n_complete = int(complete_case_mask.sum())
    print(f"\n  Complete-case count on all five factors: {n_complete} / {n_train} ({100 * n_complete / n_train:.2f}%)")

    print("\n" + "=" * 80)
    print("STEP 4: Recommended missing-data rule (NOT applied automatically)")
    print("=" * 80)
    min_factor_coverage = min(coverage[f]["fraction"] for f in FACTORS)
    if min_factor_coverage >= 0.90 and (n_complete / n_train) >= 0.85:
        recommendation = "complete-case"
        reason = (
            f"All five factors have >=90% coverage (min={min_factor_coverage * 100:.2f}%) and "
            f"complete-case retains {100 * n_complete / n_train:.2f}% of train -> complete-case on the five."
        )
    else:
        recommendation = "median-impute-with-flag"
        reason = (
            f"Coverage or complete-case retention below threshold "
            f"(min factor coverage={min_factor_coverage * 100:.2f}%, "
            f"complete-case retention={100 * n_complete / n_train:.2f}%) -> median-impute-with-flag recommended."
        )
    print(f"  RECOMMENDATION: {recommendation}")
    print(f"  REASON: {reason}")
    print("  Waiting for user confirmation before Phase 1.")

    print("\n" + "=" * 80)
    print(f"STEP 5: k-selection preview (train complete-case, n={n_complete})")
    print("=" * 80)
    train_cc = train[complete_case_mask].copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(train_cc[FACTORS].values)

    inertia = {}
    silhouette = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
        labels = km.fit_predict(x)
        inertia[k] = float(km.inertia_)
        silhouette[k] = float(silhouette_score(x, labels))
        print(f"  k={k}: inertia={inertia[k]:.2f}  silhouette={silhouette[k]:.4f}")

    print(f"\n  Computing gap statistic ({GAP_N_REFS} reference draws per k, this takes a moment)...")
    gaps, sk = gap_statistic(x, K_RANGE, GAP_N_REFS, SEED)
    for k in K_RANGE:
        print(f"  k={k}: gap={gaps[k]:.4f}  s_k={sk[k]:.4f}")

    gap_k_star = None
    for i, k in enumerate(K_RANGE[:-1]):
        k_next = K_RANGE[i + 1]
        if gaps[k] >= gaps[k_next] - sk[k_next]:
            gap_k_star = k
            break
    if gap_k_star is None:
        gap_k_star = K_RANGE[-1]
    best_silhouette_k = max(silhouette, key=silhouette.get)
    print(f"\n  Gap statistic selected k (Tibshirani rule): {gap_k_star}")
    print(f"  Best-silhouette k: {best_silhouette_k}")

    report = {
        "factor_column_map": FACTOR_COLUMN_MAP,
        "split_path": SPLIT_PATH,
        "split_counts": {str(k): int(v) for k, v in counts.items()},
        "split_total": int(total),
        "duplicate_participant_rows_in_split": int(dup_check),
        "participants_missing_from_dataset": int(missing_from_dataset),
        "train_n": int(n_train),
        "coverage": coverage,
        "complete_case_n": n_complete,
        "complete_case_fraction": n_complete / n_train,
        "missing_rule_recommendation": recommendation,
        "missing_rule_reason": reason,
        "k_preview": {
            "k_range": K_RANGE,
            "inertia": inertia,
            "silhouette": silhouette,
            "gap": gaps,
            "gap_sk": sk,
            "gap_selected_k": gap_k_star,
            "silhouette_selected_k": best_silhouette_k,
        },
    }
    with open(f"{OUTPUT_DIR}/phase0_gate_a_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Wrote {OUTPUT_DIR}/phase0_gate_a_report.json")


if __name__ == "__main__":
    main()
