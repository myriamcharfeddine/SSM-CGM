#!/usr/bin/env python3
"""Study 2 Block A/B diagnostics on the 5-minute-stride forecast cache.

Reads the parquet produced by generate_study2_forecast_cache.py and verifies:
  1. anchor_ds spacing mode == 1 for every (participant, segment) in each split.
  2. Valid t-1 one-step forecasts exist (consecutive-anchor join succeeds).
  3. One-step 80% empirical coverage (q10--q90 interval) by split.
  4. trigger_up and trigger_down rates by split (at --tau-up / --tau-down).
  5. Abstention rate broken down by abstain_reason by split.
  6. Triggered-anchor count and triggered-participant count by split,
     restricted to non-insulin participants (primary analysis cohort).

Outputs a JSON report and per-diagnostic CSV tables into --output-dir.

Usage:
    python scripts/run_study2_blocks_ab_diagnostics.py \\
        --cache-path outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet \\
        [--tau-up 1.0] [--tau-down 1.0] [--eps 1e-3] \\
        [--output-dir outputs/study2_forecast_cache_5min/diagnostics]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


_SPLITS = ("train", "validation", "test")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Study 2 Block A/B diagnostics on the 5-min forecast cache"
    )
    ap.add_argument(
        "--cache-path",
        required=True,
        help="Path to study2_forecast_cache.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostic outputs; defaults to <cache-dir>/diagnostics",
    )
    ap.add_argument(
        "--tau-up",
        type=float,
        default=1.0,
        help="Trigger threshold for upward surprise (default=1.0; diagnostic only)",
    )
    ap.add_argument(
        "--tau-down",
        type=float,
        default=1.0,
        help="Trigger threshold for downward surprise (default=1.0; diagnostic only)",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Regulariser added to interval width in standardised surprise (default=1e-3)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 1. Spacing check
# ---------------------------------------------------------------------------

def check_anchor_spacing(df: pd.DataFrame) -> dict:
    """For each (participant, segment, split), compute the mode of consecutive
    anchor_ds diffs.  Pass criterion: mode == 1 for every group.
    """
    fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()
    anchors = fo.drop_duplicates(["participant_id", "segment_id", "split", "anchor_ds"]).copy()
    anchors = anchors.sort_values(["participant_id", "segment_id", "split", "anchor_ds"])

    results_by_split: dict[str, dict] = {}
    for split_name in _SPLITS:
        sub = anchors[anchors["split"].eq(split_name)]
        if sub.empty:
            results_by_split[split_name] = {"status": "no_data"}
            continue
        group_keys = ["participant_id", "segment_id"]
        diffs = (
            sub.groupby(group_keys, sort=False)["anchor_ds"]
            .diff()
            .dropna()
        )
        if diffs.empty:
            results_by_split[split_name] = {
                "status": "single_anchor_per_stream",
                "n_groups": sub.groupby(group_keys).ngroups,
            }
            continue
        mode_val = int(diffs.mode().iloc[0])
        pct_mode = float((diffs == mode_val).mean())
        n_groups = sub.groupby(group_keys).ngroups
        # groups that violate mode-1 spacing
        group_modes = (
            sub.groupby(group_keys, sort=False)["anchor_ds"]
            .apply(lambda s: int(s.sort_values().diff().dropna().mode().iloc[0]) if len(s) > 1 else 1)
        )
        n_violating = int((group_modes != 1).sum())
        pass_check = (mode_val == 1) and (n_violating == 0)
        results_by_split[split_name] = {
            "status": "PASS" if pass_check else "FAIL",
            "spacing_mode": mode_val,
            "pct_at_mode": round(pct_mode, 4),
            "n_stream_segments": n_groups,
            "n_violating_segments": n_violating,
            "diff_distribution": diffs.value_counts().head(5).to_dict(),
        }
    return results_by_split


# ---------------------------------------------------------------------------
# 2. One-step join and coverage
# ---------------------------------------------------------------------------

def build_anchor_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each forecast anchor (anchor_ds = t), join the one-step forecast
    that was issued at (anchor_ds = t-1, horizon_step = 1).

    Returns one row per (participant, segment, anchor_ds) with columns:
        anchor_ds, split, participant_id, segment_id,
        current_glucose (= target from t-1,h=1),
        q10_one_step, q50_one_step, q90_one_step,
        one_step_issue_anchor_ds,
        valid_one_step  (bool: consecutive t-1 row found),
        interval_width, one_step_residual, z_t,
        abstain_reason, abstain (bool),
        non_insulin, participants_study_group, hba1c_percent_baseline,
        participants_clinical_site, med_insulin, med_any_diabetes_drug.
    """
    fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()

    # Fresh-12-step rows (any horizon) - one row per anchor for metadata
    anchor_meta_cols = [
        "participant_id", "segment_id", "split", "anchor_ds",
        "current_glucose", "med_insulin", "participants_study_group",
        "hba1c_percent_baseline", "participants_clinical_site",
        "med_any_diabetes_drug",
    ]
    anchor_meta_cols = [c for c in anchor_meta_cols if c in fo.columns]
    anchors = (
        fo.sort_values(["participant_id", "segment_id", "anchor_ds"])
        .drop_duplicates(["participant_id", "segment_id", "anchor_ds"])
        [anchor_meta_cols]
        .copy()
    )

    # Horizon-1 rows - these give us the one-step forecast issued at t-1
    h1 = fo[fo["horizon_step"].eq(1)][
        ["participant_id", "segment_id", "anchor_ds", "q10", "q50", "q90", "target"]
    ].copy()
    h1 = h1.rename(columns={
        "anchor_ds": "one_step_issue_anchor_ds",
        "q10": "q10_one_step",
        "q50": "q50_one_step",
        "q90": "q90_one_step",
        "target": "one_step_target",
    })
    h1["anchor_ds"] = h1["one_step_issue_anchor_ds"] + 1
    h1 = h1.drop_duplicates(["participant_id", "segment_id", "anchor_ds"])

    out = anchors.merge(
        h1[["participant_id", "segment_id", "anchor_ds",
            "one_step_issue_anchor_ds", "q10_one_step", "q50_one_step", "q90_one_step", "one_step_target"]],
        on=["participant_id", "segment_id", "anchor_ds"],
        how="left",
    )

    # Derive fields
    out["valid_one_step"] = out["one_step_issue_anchor_ds"].notna()
    out["consecutive"] = out["one_step_issue_anchor_ds"].eq(out["anchor_ds"] - 1)
    out["interval_width"] = (
        pd.to_numeric(out["q90_one_step"], errors="coerce")
        - pd.to_numeric(out["q10_one_step"], errors="coerce")
    )
    # current_glucose: prefer the column directly; fall back to one_step_target
    if "current_glucose" not in out.columns or out["current_glucose"].isna().all():
        out["current_glucose"] = pd.to_numeric(out.get("one_step_target"), errors="coerce")
    else:
        out["current_glucose"] = pd.to_numeric(out["current_glucose"], errors="coerce").fillna(
            pd.to_numeric(out.get("one_step_target"), errors="coerce")
        )
    out["one_step_residual"] = out["current_glucose"] - pd.to_numeric(out["q50_one_step"], errors="coerce")

    valid_interval = np.isfinite(out["interval_width"]) & out["interval_width"].gt(0)
    valid_residual = np.isfinite(out["one_step_residual"])
    valid_consecutive = out["consecutive"].fillna(False)

    # Abstain reason (mutually exclusive priority: non-consecutive > missing interval > missing residual)
    reasons = pd.Series("none", index=out.index)
    reasons = reasons.where(valid_residual, "missing_residual")
    reasons = reasons.where(valid_interval, "missing_interval")
    reasons = reasons.where(valid_consecutive, "non_consecutive")
    # First anchor in each stream has no prior anchor; mark as "first_anchor"
    is_first = (
        out.groupby(["participant_id", "segment_id"], sort=False)["anchor_ds"]
        .transform("min")
        .eq(out["anchor_ds"])
    )
    reasons = reasons.where(~is_first, "first_anchor")

    out["abstain_reason"] = reasons
    out["abstain"] = reasons.ne("none")

    med_insulin = pd.to_numeric(out.get("med_insulin", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0)
    out["non_insulin"] = med_insulin.eq(0)

    return out


def check_one_step_valid(anchor_table: pd.DataFrame) -> dict:
    results = {}
    for split_name in _SPLITS:
        sub = anchor_table[anchor_table["split"].eq(split_name)]
        if sub.empty:
            results[split_name] = {"status": "no_data"}
            continue
        n_total = len(sub)
        n_valid = int(sub["valid_one_step"].sum())
        n_consecutive = int(sub["consecutive"].fillna(False).sum())
        results[split_name] = {
            "n_anchors": n_total,
            "n_with_one_step": n_valid,
            "pct_with_one_step": round(n_valid / n_total, 4) if n_total else float("nan"),
            "n_consecutive": n_consecutive,
            "pct_consecutive": round(n_consecutive / n_total, 4) if n_total else float("nan"),
        }
    return results


# ---------------------------------------------------------------------------
# 3. One-step 80% empirical coverage
# ---------------------------------------------------------------------------

def check_one_step_coverage(df: pd.DataFrame) -> dict:
    """80% empirical coverage of the horizon-step=1 forecast (q10--q90)."""
    fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()
    h1 = fo[fo["horizon_step"].eq(1)].copy()

    results = {}
    for split_name in _SPLITS:
        sub = h1[h1["split"].eq(split_name)]
        if sub.empty or "q10" not in sub.columns or "q90" not in sub.columns:
            results[split_name] = {"status": "no_data"}
            continue
        lo = pd.to_numeric(sub["q10"], errors="coerce")
        hi = pd.to_numeric(sub["q90"], errors="coerce")
        y = pd.to_numeric(sub["target"], errors="coerce")
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
        n = int(valid.sum())
        covered = int(((y >= lo) & (y <= hi) & valid).sum())
        coverage = round(covered / n, 4) if n > 0 else float("nan")
        results[split_name] = {
            "n_one_step_rows": n,
            "covered": covered,
            "empirical_coverage_80pct_interval": coverage,
            "pass_80pct": coverage >= 0.80 if n > 0 else None,
        }
    return results


# ---------------------------------------------------------------------------
# 4. Trigger rates
# ---------------------------------------------------------------------------

def compute_trigger_rates(anchor_table: pd.DataFrame, *, tau_up: float,
                          tau_down: float, eps: float) -> dict:
    """Apply the rolling-3-anchor trigger rule and return rates by split."""
    at = anchor_table.copy()
    at["z_t"] = np.where(
        at["abstain"],
        np.nan,
        at["one_step_residual"] / (at["interval_width"] + eps),
    )
    at["up_obs"] = (~at["abstain"]) & at["z_t"].gt(tau_up)
    at["down_obs"] = (~at["abstain"]) & at["z_t"].lt(-tau_down)
    at["valid_obs"] = ~at["abstain"]

    group_keys = ["participant_id", "segment_id"]

    def _rolling3(series: pd.Series) -> pd.Series:
        return series.rolling(3, min_periods=3).sum()

    results = {}
    for split_name in _SPLITS:
        sub = at[at["split"].eq(split_name)].sort_values(group_keys + ["anchor_ds"]).copy()
        if sub.empty:
            results[split_name] = {"status": "no_data"}
            continue
        grp = sub.groupby(group_keys, sort=False)
        valid3 = grp["valid_obs"].transform(lambda s: _rolling3(s.astype(int)))
        up3 = grp["up_obs"].transform(lambda s: _rolling3(s.astype(int)))
        down3 = grp["down_obs"].transform(lambda s: _rolling3(s.astype(int)))
        sub["trigger_up"] = (valid3 >= 3) & (up3 >= 3)
        sub["trigger_down"] = (valid3 >= 3) & (down3 >= 3)
        n = len(sub)
        n_valid = int((valid3 >= 3).sum())
        n_up = int(sub["trigger_up"].sum())
        n_down = int(sub["trigger_down"].sum())
        results[split_name] = {
            "n_anchors": n,
            "n_evaluable_for_trigger": n_valid,
            "n_trigger_up": n_up,
            "n_trigger_down": n_down,
            "trigger_up_rate": round(n_up / n_valid, 4) if n_valid else float("nan"),
            "trigger_down_rate": round(n_down / n_valid, 4) if n_valid else float("nan"),
            "tau_up_used": tau_up,
            "tau_down_used": tau_down,
        }
    return results


# ---------------------------------------------------------------------------
# 5. Abstention rate by reason
# ---------------------------------------------------------------------------

def check_abstention_by_reason(anchor_table: pd.DataFrame) -> dict:
    results = {}
    for split_name in _SPLITS:
        sub = anchor_table[anchor_table["split"].eq(split_name)]
        if sub.empty:
            results[split_name] = {"status": "no_data"}
            continue
        n = len(sub)
        reason_counts = sub["abstain_reason"].value_counts().to_dict()
        abstain_rate = round(sub["abstain"].mean(), 4)
        results[split_name] = {
            "n_anchors": n,
            "abstain_rate": abstain_rate,
            "abstain_rate_excluding_first_anchor": round(
                sub[sub["abstain_reason"].ne("first_anchor")]["abstain"].mean(), 4
            ) if sub[sub["abstain_reason"].ne("first_anchor")].shape[0] > 0 else float("nan"),
            "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
            "reason_rates": {
                str(k): round(int(v) / n, 4)
                for k, v in reason_counts.items()
            },
        }
    return results


# ---------------------------------------------------------------------------
# 6. Triggered anchors and participants (non-insulin primary cohort)
# ---------------------------------------------------------------------------

def check_triggered_non_insulin(anchor_table: pd.DataFrame, *,
                                 tau_up: float, tau_down: float, eps: float) -> dict:
    at = anchor_table.copy()
    at["z_t"] = np.where(
        at["abstain"],
        np.nan,
        at["one_step_residual"] / (at["interval_width"] + eps),
    )
    at["up_obs"] = (~at["abstain"]) & at["z_t"].gt(tau_up)
    at["down_obs"] = (~at["abstain"]) & at["z_t"].lt(-tau_down)
    at["valid_obs"] = ~at["abstain"]

    group_keys = ["participant_id", "segment_id"]

    def _rolling3(s: pd.Series) -> pd.Series:
        return s.rolling(3, min_periods=3).sum()

    results = {}
    for split_name in _SPLITS:
        sub = at[at["split"].eq(split_name)].sort_values(group_keys + ["anchor_ds"]).copy()
        ni = sub[sub["non_insulin"].fillna(True)]
        if ni.empty:
            results[split_name] = {"status": "no_data"}
            continue
        grp = ni.groupby(group_keys, sort=False)
        valid3 = grp["valid_obs"].transform(lambda s: _rolling3(s.astype(int)))
        up3 = grp["up_obs"].transform(lambda s: _rolling3(s.astype(int)))
        down3 = grp["down_obs"].transform(lambda s: _rolling3(s.astype(int)))
        ni = ni.copy()
        ni["trigger_up"] = (valid3 >= 3) & (up3 >= 3)
        ni["trigger_down"] = (valid3 >= 3) & (down3 >= 3)
        ni["triggered"] = ni["trigger_up"] | ni["trigger_down"]
        results[split_name] = {
            "n_non_insulin_anchors": int(len(ni)),
            "n_non_insulin_participants": int(ni["participant_id"].nunique()),
            "n_triggered_anchors": int(ni["triggered"].sum()),
            "n_triggered_participants": int(ni[ni["triggered"]]["participant_id"].nunique()),
            "pct_triggered_anchors": round(ni["triggered"].mean(), 4),
            "pct_triggered_participants": round(
                ni[ni["triggered"]]["participant_id"].nunique() / ni["participant_id"].nunique(), 4
            ) if ni["participant_id"].nunique() > 0 else float("nan"),
        }
    return results


# ---------------------------------------------------------------------------
# Summary pass/fail
# ---------------------------------------------------------------------------

def overall_pass_fail(spacing: dict, one_step: dict, coverage: dict) -> dict:
    spacing_ok = all(
        v.get("status") == "PASS"
        for v in spacing.values()
        if v.get("status") != "no_data"
    )
    one_step_ok = all(
        v.get("pct_consecutive", 0) > 0.90
        for v in one_step.values()
        if "pct_consecutive" in v
    )
    coverage_ok = all(
        v.get("pass_80pct") is True
        for v in coverage.values()
        if "pass_80pct" in v
    )
    return {
        "spacing_mode_1": "PASS" if spacing_ok else "FAIL",
        "one_step_consecutive_gt90pct": "PASS" if one_step_ok else "FAIL",
        "one_step_80pct_coverage": "PASS" if coverage_ok else "FAIL",
        "all_block_a_checks": "PASS" if (spacing_ok and one_step_ok and coverage_ok) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    out_dir = Path(args.output_dir or (cache_path.parent / "diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] Loading cache: {cache_path}")
    df = pd.read_parquet(cache_path)
    print(f"[diag] {len(df):,} rows, splits: {df['split'].value_counts().to_dict() if 'split' in df.columns else 'N/A'}")

    # anchor_ds is written by generate_study2_forecast_cache.py; fall back to
    # anchor_time_idx when running against an older cache for compatibility.
    if "anchor_ds" not in df.columns:
        if "anchor_time_idx" in df.columns:
            print("[diag] WARNING: 'anchor_ds' not found; using 'anchor_time_idx' as fallback.")
            df["anchor_ds"] = df["anchor_time_idx"]
        else:
            raise KeyError("Cache must contain 'anchor_ds' or 'anchor_time_idx'.")

    # 1. Spacing
    print("[diag] 1/6  Checking anchor_ds spacing ...")
    spacing = check_anchor_spacing(df)

    # 2. One-step valid rate
    print("[diag] 2/6  Building anchor table and checking one-step join ...")
    anchor_table = build_anchor_table(df)
    anchor_table.to_parquet(out_dir / "anchor_table.parquet", index=False)
    one_step = check_one_step_valid(anchor_table)

    # 3. 80% empirical coverage
    print("[diag] 3/6  Computing one-step 80% coverage ...")
    coverage = check_one_step_coverage(df)

    # 4. Trigger rates
    print(f"[diag] 4/6  Computing trigger rates (tau_up={args.tau_up}, tau_down={args.tau_down}) ...")
    triggers = compute_trigger_rates(
        anchor_table, tau_up=args.tau_up, tau_down=args.tau_down, eps=args.eps
    )

    # 5. Abstention by reason
    print("[diag] 5/6  Abstention breakdown by reason ...")
    abstention = check_abstention_by_reason(anchor_table)

    # 6. Non-insulin triggered
    print("[diag] 6/6  Non-insulin triggered anchors and participants ...")
    triggered_ni = check_triggered_non_insulin(
        anchor_table, tau_up=args.tau_up, tau_down=args.tau_down, eps=args.eps
    )

    # Pass/fail summary
    summary = overall_pass_fail(spacing, one_step, coverage)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(cache_path),
        "tau_up": args.tau_up,
        "tau_down": args.tau_down,
        "eps": args.eps,
        "block_a_pass_fail": summary,
        "1_anchor_spacing": spacing,
        "2_one_step_valid": one_step,
        "3_one_step_80pct_coverage": coverage,
        "4_trigger_rates": triggers,
        "5_abstention_by_reason": abstention,
        "6_triggered_non_insulin": triggered_ni,
    }
    report_path = out_dir / "blocks_ab_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[diag] Report -> {report_path}")

    # Also write per-split flat CSVs for each diagnostic
    for name, table in [
        ("spacing", spacing), ("one_step", one_step), ("coverage", coverage),
        ("triggers", triggers), ("abstention", abstention), ("triggered_ni", triggered_ni),
    ]:
        rows = [{"split": k, **v} for k, v in table.items()]
        pd.DataFrame(rows).to_csv(out_dir / f"diag_{name}.csv", index=False)

    # Pretty-print key results
    print("\n" + "=" * 60)
    print("STUDY 2 BLOCK A/B DIAGNOSTICS SUMMARY")
    print("=" * 60)
    for check, result in summary.items():
        mark = "OK" if result == "PASS" else "!!"
        print(f"  [{mark}] {check}: {result}")
    print()
    for split_name in _SPLITS:
        sp = spacing.get(split_name, {})
        cv = coverage.get(split_name, {})
        tr = triggers.get(split_name, {})
        ab = abstention.get(split_name, {})
        ni = triggered_ni.get(split_name, {})
        print(f"  {split_name}:")
        print(f"    spacing mode: {sp.get('spacing_mode', 'N/A')}  "
              f"violating segments: {sp.get('n_violating_segments', 'N/A')}")
        print(f"    one-step coverage (80% interval): {cv.get('empirical_coverage_80pct_interval', 'N/A')}")
        print(f"    abstain rate: {ab.get('abstain_rate', 'N/A')}  "
              f"reasons: {ab.get('reason_counts', {})}")
        print(f"    trigger_up rate: {tr.get('trigger_up_rate', 'N/A')}  "
              f"trigger_down rate: {tr.get('trigger_down_rate', 'N/A')}")
        print(f"    non-insulin triggered anchors: {ni.get('n_triggered_anchors', 'N/A')} / "
              f"{ni.get('n_non_insulin_anchors', 'N/A')}  "
              f"triggered participants: {ni.get('n_triggered_participants', 'N/A')} / "
              f"{ni.get('n_non_insulin_participants', 'N/A')}")
        print()
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
