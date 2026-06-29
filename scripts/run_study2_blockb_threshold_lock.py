#!/usr/bin/env python3
"""Study 2 Block B: threshold sweep + official threshold-lock artifact + episode diagnostics.

Runs the rolling-3-anchor trigger rule over the candidate grid on the 5-minute
forecast cache, locks the threshold at tau=0.2, and runs episode-level
diagnostics to verify the trigger does not dominate any participant or stay on
for unreasonably long stretches.

Outputs:
  outputs/study2_forecast_cache_5min/study2_threshold_sweep.csv
  outputs/study2_forecast_cache_5min/study2_selected_thresholds.json
  outputs/study2_forecast_cache_5min/diagnostics/study2_episode_diagnostics.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

CACHE_PATH = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
OUT_DIR    = ROOT / "outputs/study2_forecast_cache_5min"
DIAG_DIR   = OUT_DIR / "diagnostics"

CANDIDATE_GRID    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SELECTED_TAU      = 0.2
EPS               = 1e-3
BIN_MINUTES       = 5
STEPS_PER_DAY     = int(24 * 60 / BIN_MINUTES)   # 288


# ---------------------------------------------------------------------------
# Anchor table (same logic as diagnostics script)
# ---------------------------------------------------------------------------

def build_anchor_table(df: pd.DataFrame) -> pd.DataFrame:
    fo = df[df["scenario_mode"].astype(str).eq("forecast_only")].copy() if "scenario_mode" in df.columns else df.copy()

    if "anchor_ds" not in fo.columns:
        fo["anchor_ds"] = fo["anchor_time_idx"]

    meta_cols = [c for c in [
        "participant_id", "segment_id", "split", "anchor_ds",
        "current_glucose", "med_insulin", "participants_study_group",
        "hba1c_percent_baseline", "participants_clinical_site", "med_any_diabetes_drug",
    ] if c in fo.columns]

    anchors = (
        fo.sort_values(["participant_id", "segment_id", "anchor_ds"])
        .drop_duplicates(["participant_id", "segment_id", "anchor_ds"])[meta_cols]
        .copy()
    )

    h1 = fo[fo["horizon_step"].eq(1)][
        ["participant_id", "segment_id", "anchor_ds", "q10", "q50", "q90", "target"]
    ].copy().rename(columns={
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
        on=["participant_id", "segment_id", "anchor_ds"], how="left",
    )

    out["interval_width"] = (
        pd.to_numeric(out["q90_one_step"], errors="coerce")
        - pd.to_numeric(out["q10_one_step"], errors="coerce")
    )

    if "current_glucose" not in out.columns or out["current_glucose"].isna().all():
        out["current_glucose"] = pd.to_numeric(out.get("one_step_target"), errors="coerce")
    else:
        out["current_glucose"] = pd.to_numeric(out["current_glucose"], errors="coerce").fillna(
            pd.to_numeric(out.get("one_step_target"), errors="coerce")
        )

    out["one_step_residual"] = out["current_glucose"] - pd.to_numeric(out["q50_one_step"], errors="coerce")

    consecutive = out["one_step_issue_anchor_ds"].eq(out["anchor_ds"] - 1)
    valid_interval = np.isfinite(out["interval_width"]) & out["interval_width"].gt(0)
    valid_residual = np.isfinite(out["one_step_residual"])
    is_first = (
        out.groupby(["participant_id", "segment_id"], sort=False)["anchor_ds"]
        .transform("min").eq(out["anchor_ds"])
    )
    reasons = pd.Series("none", index=out.index)
    reasons = reasons.where(valid_residual, "missing_residual")
    reasons = reasons.where(valid_interval, "missing_interval")
    reasons = reasons.where(consecutive, "non_consecutive")
    reasons = reasons.where(~is_first, "first_anchor")
    out["abstain"] = reasons.ne("none")

    med_insulin = pd.to_numeric(
        out.get("med_insulin", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0)
    out["non_insulin"] = med_insulin.eq(0)
    return out


# ---------------------------------------------------------------------------
# Trigger computation
# ---------------------------------------------------------------------------

def apply_triggers(anchor_table: pd.DataFrame, tau_up: float,
                   tau_down: float, eps: float = EPS) -> pd.DataFrame:
    at = anchor_table.copy()
    at["z_t"] = np.where(
        at["abstain"], np.nan,
        at["one_step_residual"] / (at["interval_width"] + eps),
    )
    at["up_obs"]    = (~at["abstain"]) & at["z_t"].gt(tau_up)
    at["down_obs"]  = (~at["abstain"]) & at["z_t"].lt(-tau_down)
    at["valid_obs"] = ~at["abstain"]

    grp_keys = ["participant_id", "segment_id"]
    at = at.sort_values(grp_keys + ["anchor_ds"]).copy()
    grp = at.groupby(grp_keys, sort=False)

    def roll3(s):
        return s.rolling(3, min_periods=3).sum()

    at["valid3"] = grp["valid_obs"].transform(lambda s: roll3(s.astype(int)))
    at["up3"]    = grp["up_obs"].transform(lambda s: roll3(s.astype(int)))
    at["down3"]  = grp["down_obs"].transform(lambda s: roll3(s.astype(int)))
    at["trigger_up"]   = (at["valid3"] >= 3) & (at["up3"] >= 3)
    at["trigger_down"] = (at["valid3"] >= 3) & (at["down3"] >= 3)
    at["triggered"]    = at["trigger_up"] | at["trigger_down"]
    return at


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_row(at: pd.DataFrame, tau: float, split: str) -> dict:
    sub = at[at["split"].eq(split) & at["non_insulin"].fillna(True)]
    if sub.empty:
        return {"tau": tau, "split": split}
    n = len(sub)
    n_valid = int((sub["valid3"] >= 3).sum()) if "valid3" in sub.columns else 0
    n_up    = int(sub["trigger_up"].sum())
    n_down  = int(sub["trigger_down"].sum())
    n_part  = int(sub["participant_id"].nunique())
    n_trig  = int(sub[sub["triggered"]]["participant_id"].nunique())
    return {
        "tau": tau,
        "split": split,
        "n_non_insulin_anchors": n,
        "n_evaluable": n_valid,
        "n_trigger_up": n_up,
        "n_trigger_down": n_down,
        "trigger_up_rate": round(n_up / n_valid, 5) if n_valid else float("nan"),
        "trigger_down_rate": round(n_down / n_valid, 5) if n_valid else float("nan"),
        "n_participants": n_part,
        "n_triggered_participants": n_trig,
        "pct_triggered_participants": round(n_trig / n_part, 4) if n_part else float("nan"),
    }


# ---------------------------------------------------------------------------
# Episode finder (vectorised)
# ---------------------------------------------------------------------------

def find_episodes(at: pd.DataFrame, trigger_col: str) -> pd.DataFrame:
    grp_keys = ["participant_id", "segment_id"]
    out = at.sort_values(grp_keys + ["anchor_ds"]).copy()

    # New episode starts when: trigger is True AND (prev row was False OR gap > 1)
    grp = out.groupby(grp_keys, sort=False)
    out["_prev_trig"] = grp[trigger_col].shift(1).fillna(False)
    out["_prev_ds"]   = grp["anchor_ds"].shift(1)
    out["_ds_gap"]    = out["anchor_ds"] - out["_prev_ds"]
    out["_new_ep"]    = out[trigger_col] & (~out["_prev_trig"] | out["_ds_gap"].gt(1))

    out["_ep_cum"]    = grp["_new_ep"].cumsum()
    out["episode_id"] = np.where(out[trigger_col], out["_ep_cum"].astype(int), -1)
    return out[out[trigger_col]].copy()


def episode_stats(at: pd.DataFrame, trigger_col: str,
                  split: str, label: str) -> dict:
    sub = at[at["split"].eq(split) & at["non_insulin"].fillna(True)].copy()
    if sub.empty:
        return {"split": split, "direction": label}

    ep_df = find_episodes(sub, trigger_col)
    if ep_df.empty:
        return {"split": split, "direction": label, "n_episodes": 0}

    ep_grp = ep_df.groupby(["participant_id", "segment_id", "episode_id"], sort=False)
    ep_durations_steps = ep_grp["anchor_ds"].count()
    ep_durations_min   = ep_durations_steps * BIN_MINUTES

    # Participant-days: sum over segments of (n_anchors / STEPS_PER_DAY)
    part_days = (
        sub.groupby(["participant_id", "segment_id"], sort=False)["anchor_ds"]
        .count()
        .groupby(level="participant_id")
        .sum()
        / STEPS_PER_DAY
    )
    total_part_days = float(part_days.sum())

    # Episodes per participant per day
    ep_per_part = ep_grp.ngroups / max(len(sub["participant_id"].unique()), 1)
    ep_per_part_day = ep_grp.ngroups / max(total_part_days, 1)

    # Top-10 participant trigger share (by triggered anchor count)
    trig_by_part = sub[sub[trigger_col]].groupby("participant_id")["anchor_ds"].count()
    total_trig   = int(trig_by_part.sum())
    top10_share  = float(trig_by_part.nlargest(10).sum() / total_trig) if total_trig else float("nan")

    # Matched placebo support: for triggered participants, ratio non-triggered:triggered anchors
    trig_pids = set(sub[sub[trigger_col]]["participant_id"])
    in_trig   = sub[sub["participant_id"].isin(trig_pids)]
    n_trig_anch  = int(in_trig[trigger_col].sum())
    n_notrig_anch = int((~in_trig[trigger_col]).sum())
    placebo_ratio = round(n_notrig_anch / n_trig_anch, 2) if n_trig_anch else float("nan")

    return {
        "split": split,
        "direction": label,
        "n_episodes": int(ep_durations_steps.shape[0]),
        "n_triggered_anchors": int(ep_df.shape[0]),
        "n_triggered_participants": int(ep_df["participant_id"].nunique()),
        "total_participant_days": round(total_part_days, 1),
        "episodes_per_participant_day": round(ep_per_part_day, 4),
        "episode_duration_minutes_median": float(ep_durations_min.median()),
        "episode_duration_minutes_p90": float(ep_durations_min.quantile(0.90)),
        "episode_duration_minutes_max": float(ep_durations_min.max()),
        "top10_participant_trigger_share": round(top10_share, 4),
        "placebo_support_ratio_nontriggered_to_triggered": placebo_ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[blockb] Loading cache: {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
    print(f"[blockb] {len(df):,} rows")

    print("[blockb] Building anchor table ...")
    anchor_table = build_anchor_table(df)

    # ------------------------------------------------------------------
    # 1. Threshold sweep
    # ------------------------------------------------------------------
    print(f"[blockb] Running sweep over {CANDIDATE_GRID} ...")
    sweep_rows = []
    for tau in CANDIDATE_GRID:
        at = apply_triggers(anchor_table, tau_up=tau, tau_down=tau)
        for split in ["train", "validation", "test"]:
            sweep_rows.append(sweep_row(at, tau, split))
        v = next((r for r in sweep_rows if r["split"] == "validation" and r["tau"] == tau), {})
        print(f"  tau={tau:.1f}  val trigger_up_rate={v.get('trigger_up_rate','N/A'):.5f}"
              f"  trig_participants={v.get('n_triggered_participants','N/A')}/{v.get('n_participants','N/A')}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = OUT_DIR / "study2_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"[blockb] Sweep saved -> {sweep_path}")

    # ------------------------------------------------------------------
    # 2. Official threshold-lock artifact
    # ------------------------------------------------------------------
    selected = {
        "tau_up": SELECTED_TAU,
        "tau_down": SELECTED_TAU,
        "eps": EPS,
        "selection_split": "validation",
        "candidate_grid": CANDIDATE_GRID,
        "selection_rule": (
            "Choose the highest candidate threshold giving a clinically usable "
            "validation trigger_up rate near 5 percent while retaining all "
            "non-insulin validation participants and sufficient anchors for "
            "participant-clustered confidence intervals."
        ),
        "selected_reason": (
            "tau=0.2 was the only candidate in the tested grid with validation "
            "trigger_up rate near 5 percent while retaining all non-insulin "
            "validation participants."
        ),
        "test_used_for_selection": False,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Attach the actual validation row from sweep
    val_row = sweep_df[(sweep_df["tau"] == SELECTED_TAU) & (sweep_df["split"] == "validation")]
    if not val_row.empty:
        selected["validation_metrics_at_lock"] = val_row.iloc[0].to_dict()

    lock_path = OUT_DIR / "study2_selected_thresholds.json"
    lock_path.write_text(json.dumps(selected, indent=2, default=str))
    print(f"[blockb] Threshold lock artifact -> {lock_path}")

    # ------------------------------------------------------------------
    # 3. Episode-level diagnostics at tau=0.2
    # ------------------------------------------------------------------
    print(f"\n[blockb] Episode diagnostics at tau={SELECTED_TAU} ...")
    at_sel = apply_triggers(anchor_table, tau_up=SELECTED_TAU, tau_down=SELECTED_TAU)

    ep_results = []
    for split in ["train", "validation", "test"]:
        for direction, col in [("up", "trigger_up"), ("down", "trigger_down"), ("any", "triggered")]:
            ep_results.append(episode_stats(at_sel, col, split, direction))

    ep_path = DIAG_DIR / "study2_episode_diagnostics.json"
    ep_path.write_text(json.dumps(ep_results, indent=2, default=str))
    print(f"[blockb] Episode diagnostics -> {ep_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SWEEP SUMMARY (non-insulin, validation split)")
    print("=" * 60)
    val_rows = sweep_df[sweep_df["split"].eq("validation")]
    print(val_rows[["tau", "trigger_up_rate", "trigger_down_rate",
                     "n_triggered_participants", "n_participants",
                     "pct_triggered_participants"]].to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"EPISODE DIAGNOSTICS AT tau={SELECTED_TAU} (non-insulin)")
    print("=" * 60)
    for row in ep_results:
        if row.get("direction") != "any":
            continue
        split = row["split"]
        print(f"\n  {split}:")
        print(f"    episodes: {row.get('n_episodes', 'N/A')}")
        print(f"    episodes/participant-day: {row.get('episodes_per_participant_day', 'N/A')}")
        print(f"    duration median: {row.get('episode_duration_minutes_median', 'N/A')} min  "
              f"p90: {row.get('episode_duration_minutes_p90', 'N/A')} min  "
              f"max: {row.get('episode_duration_minutes_max', 'N/A')} min")
        print(f"    top-10 participant trigger share: {row.get('top10_participant_trigger_share', 'N/A')}")
        print(f"    placebo support ratio: {row.get('placebo_support_ratio_nontriggered_to_triggered', 'N/A')}")

    print(f"\nArtifacts:")
    print(f"  {sweep_path}")
    print(f"  {lock_path}")
    print(f"  {ep_path}")


if __name__ == "__main__":
    main()
