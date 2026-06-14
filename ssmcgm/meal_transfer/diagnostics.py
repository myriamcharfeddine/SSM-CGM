"""Phase C — target-domain diagnostics.

Produces the distributions and event statistics the brief lists, writes a few
representative event PNGs, and compares source/target CGM feature distributions
when CGMacros data are available (they are not in this environment; the function
records that explicitly rather than silently skipping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _quantile_row(prefix: str, split: str, group: str, s: pd.Series) -> dict:
    s = s.dropna()
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if len(s) else pd.Series(dtype=float)
    return {
        "metric": prefix, "split": split, "group": group, "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "p05": float(q.get(0.05, np.nan)), "p25": float(q.get(0.25, np.nan)),
        "p50": float(q.get(0.50, np.nan)), "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
    }


def _events_per_participant_day(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Count contiguous positive runs of ``flag_col`` per participant, normalise
    by recorded days (5-min cadence => 288 rows/day)."""
    recs = []
    for pid, sub in df.groupby("participant_id", sort=False):
        f = sub[flag_col].fillna(0).to_numpy()
        starts = int(((f == 1) & (np.r_[0, f[:-1]] == 0)).sum())
        days = max(len(sub) / 288.0, 1e-6)
        recs.append({"participant_id": pid, "events": starts,
                     "days": days, "events_per_day": starts / days})
    return pd.DataFrame(recs)


def run_diagnostics(df: pd.DataFrame, output_dir: Path) -> dict:
    """``df`` carries: split, study_group, med_insulin, hour, and the columns
    cgmacros_teacher_probability/flag, student_meal_probability,
    predmeal_flag_clean. Returns {name: DataFrame}; also writes event PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    prob_cols = [c for c in ("cgmacros_teacher_probability", "student_meal_probability")
                 if c in df.columns]
    flag_cols = [c for c in ("cgmacros_teacher_flag", "predmeal_flag_clean")
                 if c in df.columns]

    # --- probability distribution by split (and overall) ---
    rows = []
    for split, sub in list(df.groupby("split")) + [("ALL", df)]:
        for pc in prob_cols:
            rows.append(_quantile_row(pc, split, "ALL", sub[pc]))
    out["prob_distribution_by_split"] = pd.DataFrame(rows)

    # --- positive flag coverage by split ---
    rows = []
    for split, sub in list(df.groupby("split")) + [("ALL", df)]:
        for fc in flag_cols:
            part_cov = sub.groupby("participant_id")[fc].max()
            rows.append({
                "split": split, "flag": fc,
                "row_pct_flagged": float(sub[fc].mean()),
                "participants": int(sub["participant_id"].nunique()),
                "participants_with_flag": int((part_cov > 0).sum()),
                "participant_pct_with_flag": float((part_cov > 0).mean()),
            })
    out["flag_coverage_by_split"] = pd.DataFrame(rows)

    # --- events per participant-day (teacher + decoded) ---
    ev_rows = []
    for fc in flag_cols:
        ev = _events_per_participant_day(df, fc)
        ev_rows.append({
            "flag": fc,
            "mean_events_per_day": float(ev["events_per_day"].mean()),
            "median_events_per_day": float(ev["events_per_day"].median()),
            "p90_events_per_day": float(ev["events_per_day"].quantile(0.9)),
            "participants": int(len(ev)),
        })
    out["events_per_participant_day"] = pd.DataFrame(ev_rows)

    # --- distributions by study group ---
    rows = []
    if "study_group" in df.columns:
        for grp, sub in df.groupby("study_group"):
            for pc in prob_cols:
                rows.append(_quantile_row(pc, "ALL", str(grp), sub[pc]))
            for fc in flag_cols:
                rows.append({"metric": fc, "split": "ALL", "group": str(grp),
                             "n": int(len(sub)), "mean": float(sub[fc].mean()),
                             "p05": np.nan, "p25": np.nan, "p50": np.nan,
                             "p75": np.nan, "p95": np.nan})
    out["distributions_by_study_group"] = pd.DataFrame(rows)

    # --- distributions by med_insulin ---
    rows = []
    if "med_insulin" in df.columns:
        for val, sub in df.groupby(df["med_insulin"].fillna(0)):
            for pc in prob_cols:
                rows.append(_quantile_row(pc, "ALL", f"med_insulin={int(val)}", sub[pc]))
            for fc in flag_cols:
                rows.append({"metric": fc, "split": "ALL",
                             "group": f"med_insulin={int(val)}", "n": int(len(sub)),
                             "mean": float(sub[fc].mean()), "p05": np.nan,
                             "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan})
    out["distributions_by_med_insulin"] = pd.DataFrame(rows)

    # --- distributions by hour of day ---
    rows = []
    if "hour" in df.columns:
        for hr, sub in df.groupby("hour"):
            entry = {"hour": int(hr), "n": int(len(sub))}
            for pc in prob_cols:
                entry[f"{pc}_mean"] = float(sub[pc].mean())
            for fc in flag_cols:
                entry[f"{fc}_rate"] = float(sub[fc].mean())
            rows.append(entry)
    out["distributions_by_hour"] = pd.DataFrame(rows)

    # --- source/target feature comparison (CGMacros not present here) ---
    tgt = df["cgm_glucose"].dropna()
    cmp_rows = [{
        "feature": "cgm_glucose", "domain": "target_aireadi",
        "n": int(len(tgt)), "mean": float(tgt.mean()), "std": float(tgt.std()),
        "p05": float(tgt.quantile(0.05)), "p50": float(tgt.quantile(0.5)),
        "p95": float(tgt.quantile(0.95)),
    }, {
        "feature": "cgm_glucose", "domain": "source_cgmacros",
        "n": 0, "mean": np.nan, "std": np.nan, "p05": np.nan, "p50": np.nan,
        "p95": np.nan,
        "note": "CGMacros raw data not present in this environment; comparison "
                "unavailable. Units verified identical (mg/dL).",
    }]
    out["source_target_feature_comparison"] = pd.DataFrame(cmp_rows)

    # --- representative event plots ---
    _plot_events(df, flag_cols, prob_cols, fig_dir)

    return out


def _plot_events(df, flag_cols, prob_cols, fig_dir: Path, n_events: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flag_col = "predmeal_flag_clean" if "predmeal_flag_clean" in flag_cols else (
        flag_cols[0] if flag_cols else None)
    if flag_col is None:
        return
    plotted = 0
    for pid, sub in df.groupby("participant_id", sort=False):
        sub = sub.reset_index(drop=True)
        f = sub[flag_col].fillna(0).to_numpy()
        starts = np.where((f == 1) & (np.r_[0, f[:-1]] == 0))[0]
        if len(starts) == 0:
            continue
        s = starts[0]
        lo, hi = max(0, s - 24), min(len(sub), s + 36)  # ~ -2h..+3h
        win = sub.iloc[lo:hi]
        t = np.arange(len(win)) * 5
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = ax1.twinx()
        ax1.plot(t, win["cgm_glucose"].to_numpy(), color="tab:blue", label="CGM")
        for pc, c in zip(prob_cols, ("tab:orange", "tab:green")):
            ax2.plot(t, win[pc].to_numpy(), "--", color=c, label=pc, alpha=0.8)
        ax2.step(t, win[flag_col].to_numpy(), where="post", color="tab:red",
                 alpha=0.6, label=flag_col)
        ax1.set_xlabel("minutes"); ax1.set_ylabel("CGM mg/dL")
        ax2.set_ylabel("prob / flag"); ax2.set_ylim(-0.05, 1.05)
        ax1.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        grp = win["study_group"].iloc[0] if "study_group" in win else ""
        ax1.set_title(f"participant {pid} ({grp}) — representative event")
        fig.tight_layout()
        fig.savefig(fig_dir / f"event_{pid}.png", dpi=90)
        plt.close(fig)
        plotted += 1
        if plotted >= n_events:
            break
