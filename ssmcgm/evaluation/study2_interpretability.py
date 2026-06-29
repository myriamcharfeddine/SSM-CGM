"""Study 2 descriptive interpretability artifacts.

The functions in this module build an anchor-level failure-mode map after the
surprise trigger threshold has already been selected. They do not select,
search, rank, or tune trigger thresholds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Study2InterpretabilityConfig:
    tau_up: float
    tau_down: float
    eps: float = 1e-3
    bin_minutes: int = 5
    require_consecutive_anchors: bool = True


FAILURE_MAP_COLUMNS = [
    "participant_id",
    "split",
    "timestamp",
    "non_insulin",
    "study_group",
    "hba1c_quartile",
    "clinical_site",
    "baseline_glucose",
    "one_step_residual",
    "standardized_surprise_z_t",
    "interval_width",
    "slope",
    "curvature",
    "trigger_up",
    "trigger_down",
    "abstain",
    "anchor_family",
    "event_enriched",
    "medication_group",
    "anchor_ds",
    "segment_id",
    "one_step_issue_anchor_ds",
]

SUMMARY_DIMENSIONS = [
    ("time_of_day", "time_of_day_bin"),
    ("baseline_glucose_bin", "baseline_glucose_bin"),
    ("slope_bin", "slope_bin"),
    ("interval_width_bin", "interval_width_bin"),
    ("hba1c_quartile", "hba1c_quartile"),
    ("study_group", "study_group"),
    ("clinical_site", "clinical_site"),
    ("medication_group", "medication_group"),
]


def build_failure_mode_map(
    forecast_rows: pd.DataFrame,
    source_rows: pd.DataFrame | None,
    event_rows: pd.DataFrame | None,
    config: Study2InterpretabilityConfig,
) -> pd.DataFrame:
    """Return one descriptive Study 2 row per forecast anchor.

    ``forecast_rows`` is the frozen-base long forecast cache. The one-step
    residual for anchor t is read from the horizon-1 forecast issued at t-1.
    The fresh 12-step forecast rows issued at t define the anchors but are not
    used to tune the trigger threshold here.
    """
    _require(forecast_rows, ["participant_id", "anchor_ds", "horizon_step", "target", "q10", "q50", "q90"], "forecast_rows")
    cfg = config
    fc = _forecast_only(forecast_rows)
    fc = fc.copy()
    fc["participant_id"] = fc["participant_id"].astype(str)
    fc["anchor_ds"] = pd.to_numeric(fc["anchor_ds"], errors="coerce")
    fc["horizon_step"] = pd.to_numeric(fc["horizon_step"], errors="coerce")
    fc = fc[np.isfinite(fc["anchor_ds"]) & np.isfinite(fc["horizon_step"])].copy()
    fc["anchor_ds"] = fc["anchor_ds"].astype(int)
    fc["horizon_step"] = fc["horizon_step"].astype(int)

    anchors = _anchor_rows(fc)
    one_step = _one_step_rows(fc)
    out = anchors.merge(
        one_step,
        left_on=["participant_id", "anchor_ds"],
        right_on=["participant_id", "one_step_target_ds"],
        how="left",
    )
    out = out.drop(columns=["one_step_target_ds"], errors="ignore")
    out = _attach_source_columns(out, source_rows)
    out = _attach_event_family(out, event_rows, fc)
    out = _derive_required_columns(out, cfg)
    out = _apply_directional_triggers(out, cfg)
    out = _add_summary_bins(out, cfg.bin_minutes)
    for col in FAILURE_MAP_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    sort_cols = [c for c in ["split", "participant_id", "segment_id", "anchor_ds"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out[FAILURE_MAP_COLUMNS + [c for c in out.columns if c not in FAILURE_MAP_COLUMNS]]


def summarize_failure_mode_map(failure_map: pd.DataFrame) -> pd.DataFrame:
    """Aggregate descriptive trigger and surprise summaries by requested axes."""
    if failure_map.empty:
        return pd.DataFrame()
    work = failure_map.copy()
    if "time_of_day_bin" not in work.columns:
        work = _add_summary_bins(work, Study2InterpretabilityConfig(1.0, 1.0).bin_minutes)

    frames = []
    cohorts = [("all", work)]
    if "non_insulin" in work.columns:
        cohorts.append(("non_insulin", work[work["non_insulin"].fillna(False).astype(bool)]))
    for cohort, df in cohorts:
        for dim_name, col in SUMMARY_DIMENSIONS:
            if col not in df.columns:
                continue
            tab = _summarize_dimension(df, col)
            if tab.empty:
                continue
            tab.insert(0, "cohort", cohort)
            tab.insert(1, "dimension", dim_name)
            frames.append(tab.rename(columns={col: "level"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_study2_interpretability_artifacts(
    forecast_rows: pd.DataFrame,
    source_rows: pd.DataFrame | None,
    event_rows: pd.DataFrame | None,
    out_dir,
    *,
    tau_up: float,
    tau_down: float,
    eps: float = 1e-3,
    bin_minutes: int = 5,
    require_consecutive_anchors: bool = True,
) -> dict:
    """Write the Study 2 failure map, summary tables, figures, and manifest."""
    cfg = Study2InterpretabilityConfig(
        tau_up=float(tau_up),
        tau_down=float(tau_down),
        eps=float(eps),
        bin_minutes=int(bin_minutes),
        require_consecutive_anchors=bool(require_consecutive_anchors),
    )
    out = Path(out_dir)
    table_dir = out / "tables"
    fig_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    failure_map = build_failure_mode_map(forecast_rows, source_rows, event_rows, cfg)
    map_path = out / "study2_failure_mode_map.parquet"
    failure_map.to_parquet(map_path, index=False)
    summary = summarize_failure_mode_map(failure_map)
    summary_path = table_dir / "study2_trigger_surprise_summary.csv"
    summary.to_csv(summary_path, index=False)
    for dim in sorted(summary["dimension"].dropna().unique()) if not summary.empty else []:
        summary[summary["dimension"].eq(dim)].to_csv(
            table_dir / f"study2_trigger_surprise_by_{_safe_name(dim)}.csv",
            index=False,
        )
    figure_paths = _write_summary_figures(summary, fig_dir)
    manifest = {
        "config": asdict(cfg),
        "note": (
            "Descriptive interpretability artifacts only. Trigger thresholds "
            "must be selected before this export and these summaries must not "
            "be used to tune tau after validation selection."
        ),
        "n_anchors": int(len(failure_map)),
        "n_non_insulin_anchors": int(failure_map["non_insulin"].fillna(False).sum()) if "non_insulin" in failure_map else 0,
        "abstain_rate": float(failure_map["abstain"].mean()) if len(failure_map) else np.nan,
        "valid_one_step_rate": float((~failure_map["abstain"].fillna(True)).mean()) if len(failure_map) else np.nan,
        "trigger_up_density": float(failure_map["trigger_up"].mean()) if len(failure_map) else np.nan,
        "trigger_down_density": float(failure_map["trigger_down"].mean()) if len(failure_map) else np.nan,
        "map_path": str(map_path),
        "summary_path": str(summary_path),
        "figure_paths": figure_paths,
    }
    manifest_path = out / "study2_interpretability_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2))
    return manifest


def _forecast_only(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario_mode" not in df.columns:
        return df
    mode = df["scenario_mode"].astype(str)
    keep = mode.eq("forecast_only")
    return df[keep].copy() if keep.any() else df


def _anchor_rows(fc: pd.DataFrame) -> pd.DataFrame:
    order = [c for c in ["participant_id", "segment_id", "split", "anchor_ds", "horizon_step"] if c in fc.columns]
    base = fc.sort_values(order).drop_duplicates(["participant_id", "anchor_ds"]).copy()
    rename = {
        "participants_study_group": "study_group",
        "participants_clinical_site": "clinical_site",
    }
    base = base.rename(columns=rename)
    keep = [
        "participant_id",
        "split",
        "segment_id",
        "anchor_ds",
        "anchor_time_idx",
        "forecast_issue_index",
        "timestamp",
        "issue_timestamp",
        "current_glucose",
        "baseline_glucose",
        "slope",
        "curvature",
        "cgm_delta_5",
        "steps_30",
        "steps_60",
        "sleep_rest",
        "study_group",
        "hba1c_percent_baseline",
        "clinical_site",
        "med_insulin",
        "med_metformin",
        "med_any_diabetes_drug",
        "med_glp1_or_gip_glp1",
        "med_sglt2",
        "med_sulfonylurea",
        "med_thiazolidinedione",
        "eval_in_meal_window",
        "anchor_family",
    ]
    return base[[c for c in keep if c in base.columns]].copy()


def _one_step_rows(fc: pd.DataFrame) -> pd.DataFrame:
    h1 = fc[fc["horizon_step"].eq(1)].copy()
    h1["one_step_target_ds"] = h1["anchor_ds"] + 1
    h1 = h1.sort_values(["participant_id", "anchor_ds"]).drop_duplicates(["participant_id", "one_step_target_ds"])
    return h1.rename(columns={
        "anchor_ds": "one_step_issue_anchor_ds",
        "q10": "q10_one_step",
        "q50": "q50_one_step",
        "q90": "q90_one_step",
        "target": "one_step_target",
    })[[
        "participant_id",
        "one_step_target_ds",
        "one_step_issue_anchor_ds",
        "q10_one_step",
        "q50_one_step",
        "q90_one_step",
        "one_step_target",
    ]]


def _attach_source_columns(out: pd.DataFrame, source_rows: pd.DataFrame | None) -> pd.DataFrame:
    if source_rows is None or source_rows.empty:
        return out
    src = source_rows.copy()
    if "participant_id" not in src.columns or "ds" not in src.columns:
        return out
    src["participant_id"] = src["participant_id"].astype(str)
    src["ds"] = pd.to_numeric(src["ds"], errors="coerce")
    src = src[np.isfinite(src["ds"])].copy()
    src["ds"] = src["ds"].astype(int)
    key_cols = ["participant_id", "ds"]
    if "segment_id" in out.columns and "segment_id" in src.columns:
        key_cols = ["participant_id", "segment_id", "ds"]
        src["segment_id"] = pd.to_numeric(src["segment_id"], errors="coerce")
        out_seg = pd.to_numeric(out["segment_id"], errors="coerce")
        src = src[np.isfinite(src["segment_id"])].copy()
        src["segment_id"] = src["segment_id"].astype(int)
        key = list(zip(out["participant_id"].astype(str), out_seg.astype("Int64"), out["anchor_ds"].astype(int)))
    else:
        key = list(zip(out["participant_id"].astype(str), out["anchor_ds"].astype(int)))
    src = src.drop_duplicates(key_cols).set_index(key_cols)
    aliases = {
        "timestamp": ["ts", "timestamp"],
        "baseline_glucose": ["cgm_glucose", "level", "baseline_glucose"],
        "slope": ["cgm_slope_15", "dy15", "slope"],
        "curvature": ["cgm_accel", "curvature"],
        "tod_sin": ["tod_sin", "hour_sin"],
        "tod_cos": ["tod_cos", "hour_cos"],
        "med_insulin_source": ["med_insulin"],
        "med_any_diabetes_drug_source": ["med_any_diabetes_drug"],
        "med_metformin_source": ["med_metformin"],
        "source_split": ["split"],
        "source_segment_id": ["segment_id"],
    }
    for out_col, choices in aliases.items():
        col = _first_present(src.columns, choices)
        if col is not None:
            out[out_col] = src[col].reindex(key).to_numpy()
    return out


def _attach_event_family(out: pd.DataFrame, event_rows: pd.DataFrame | None, fc: pd.DataFrame) -> pd.DataFrame:
    event = pd.Series(False, index=out.index)
    if event_rows is not None and not event_rows.empty and {"participant_id", "ds"}.issubset(event_rows.columns):
        ev = event_rows.copy()
        ev["participant_id"] = ev["participant_id"].astype(str)
        ev["ds"] = pd.to_numeric(ev["ds"], errors="coerce")
        ev = ev[np.isfinite(ev["ds"])].copy()
        ev["ds"] = ev["ds"].astype(int)
        ev_flag = _event_flag(ev)
        if ev_flag is not None:
            s = ev.assign(_event=ev_flag).drop_duplicates(["participant_id", "ds"]).set_index(["participant_id", "ds"])["_event"]
            key = list(zip(out["participant_id"].astype(str), out["anchor_ds"].astype(int)))
            event = pd.Series(s.reindex(key).fillna(False).to_numpy(dtype=bool), index=out.index)
    elif "anchor_family" in out.columns:
        event = out["anchor_family"].astype(str).eq("event_enriched")
    elif "eval_in_meal_window" in out.columns:
        event = out["eval_in_meal_window"].fillna(False).astype(bool)
    elif "eval_in_meal_window" in fc.columns:
        gf = fc.groupby(["participant_id", "anchor_ds"])["eval_in_meal_window"].max()
        key = list(zip(out["participant_id"].astype(str), out["anchor_ds"].astype(int)))
        event = pd.Series(gf.reindex(key).fillna(False).to_numpy(dtype=bool), index=out.index)
    out["event_enriched"] = event.astype(bool)
    out["anchor_family"] = np.where(out["event_enriched"], "event_enriched", "placebo")
    return out


def _derive_required_columns(out: pd.DataFrame, cfg: Study2InterpretabilityConfig) -> pd.DataFrame:
    if "timestamp" not in out.columns:
        out["timestamp"] = out.get("anchor_time_idx", out["anchor_ds"])
    if "split" not in out.columns:
        out["split"] = out.get("source_split", "unknown")
    if "segment_id" not in out.columns and "source_segment_id" in out.columns:
        out["segment_id"] = out["source_segment_id"]
    out["baseline_glucose"] = pd.to_numeric(
        out.get("baseline_glucose", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    one_step_target = out.get("one_step_target", pd.Series(np.nan, index=out.index))
    out["baseline_glucose"] = out["baseline_glucose"].fillna(pd.to_numeric(one_step_target, errors="coerce"))
    out["slope"] = pd.to_numeric(out.get("slope", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["curvature"] = pd.to_numeric(out.get("curvature", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["interval_width"] = (
        pd.to_numeric(out.get("q90_one_step"), errors="coerce")
        - pd.to_numeric(out.get("q10_one_step"), errors="coerce")
    )
    out["one_step_residual"] = out["baseline_glucose"] - pd.to_numeric(out.get("q50_one_step"), errors="coerce")
    issue = pd.to_numeric(out.get("one_step_issue_anchor_ds"), errors="coerce")
    consecutive = issue.eq(pd.to_numeric(out["anchor_ds"], errors="coerce") - 1)
    valid_interval = np.isfinite(out["interval_width"]) & (out["interval_width"] > 0)
    valid_resid = np.isfinite(out["one_step_residual"])
    if cfg.require_consecutive_anchors:
        valid_step = consecutive.fillna(False)
    else:
        valid_step = issue.notna()
    out["abstain"] = ~(valid_interval & valid_resid & valid_step)
    denom = out["interval_width"] + cfg.eps
    out["standardized_surprise_z_t"] = np.where(
        out["abstain"],
        np.nan,
        out["one_step_residual"] / denom,
    )
    med_insulin = _coalesce_numeric(out, ["med_insulin", "med_insulin_source"], default=0.0)
    med_any = _coalesce_numeric(out, ["med_any_diabetes_drug", "med_any_diabetes_drug_source"], default=0.0)
    med_metformin = _coalesce_numeric(out, ["med_metformin", "med_metformin_source"], default=0.0)
    out["non_insulin"] = med_insulin.fillna(0).eq(0)
    out["medication_group"] = _medication_group(med_insulin, med_any, med_metformin)
    if "study_group" not in out.columns:
        out["study_group"] = "unknown"
    if "clinical_site" not in out.columns:
        out["clinical_site"] = "unknown"
    out["hba1c_quartile"] = _quartile_labels(out.get("hba1c_percent_baseline", pd.Series(np.nan, index=out.index)))
    return out


def _apply_directional_triggers(out: pd.DataFrame, cfg: Study2InterpretabilityConfig) -> pd.DataFrame:
    group_cols = ["participant_id"] + (["segment_id"] if "segment_id" in out.columns else [])
    ordered = out.sort_values(group_cols + ["anchor_ds"]).copy()
    z = pd.to_numeric(ordered["standardized_surprise_z_t"], errors="coerce")
    valid = (~ordered["abstain"].fillna(True).astype(bool)) & np.isfinite(z)
    up_obs = valid & z.gt(cfg.tau_up)
    down_obs = valid & z.lt(-cfg.tau_down)
    ordered["_valid_obs"] = valid.astype(int)
    ordered["_up_obs"] = up_obs.astype(int)
    ordered["_down_obs"] = down_obs.astype(int)
    roll_keys = [ordered[c].astype(str) for c in group_cols]
    valid3 = ordered["_valid_obs"].groupby(roll_keys, sort=False).rolling(3, min_periods=3).sum().reset_index(level=list(range(len(group_cols))), drop=True)
    up3 = ordered["_up_obs"].groupby(roll_keys, sort=False).rolling(3, min_periods=3).sum().reset_index(level=list(range(len(group_cols))), drop=True)
    down3 = ordered["_down_obs"].groupby(roll_keys, sort=False).rolling(3, min_periods=3).sum().reset_index(level=list(range(len(group_cols))), drop=True)
    ds = pd.to_numeric(ordered["anchor_ds"], errors="coerce")
    d1 = ds.groupby(roll_keys, sort=False).diff()
    d2 = ds.groupby(roll_keys, sort=False).diff(2)
    if cfg.require_consecutive_anchors:
        consecutive3 = d1.eq(1) & d2.eq(2)
    else:
        consecutive3 = pd.Series(True, index=ordered.index)
    ordered["trigger_up"] = (valid3.eq(3) & up3.ge(2) & consecutive3).fillna(False).astype(bool)
    ordered["trigger_down"] = (valid3.eq(3) & down3.ge(2) & consecutive3).fillna(False).astype(bool)
    ordered = ordered.drop(columns=["_valid_obs", "_up_obs", "_down_obs"])
    return ordered.sort_index()


def _add_summary_bins(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    out = df.copy()
    out["time_of_day_bin"] = _time_of_day_bin(out, bin_minutes)
    out["baseline_glucose_bin"] = pd.cut(
        pd.to_numeric(out.get("baseline_glucose"), errors="coerce"),
        bins=[-np.inf, 70, 100, 140, 180, 250, np.inf],
        labels=["lt70", "70_99", "100_139", "140_179", "180_249", "ge250"],
    ).astype(object).where(lambda s: s.notna(), "missing")
    out["slope_bin"] = pd.cut(
        pd.to_numeric(out.get("slope"), errors="coerce"),
        bins=[-np.inf, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, np.inf],
        labels=["fast_down", "down", "mild_down", "flat", "mild_up", "up", "fast_up"],
    ).astype(object).where(lambda s: s.notna(), "missing")
    out["interval_width_bin"] = _quartile_labels(pd.to_numeric(out.get("interval_width"), errors="coerce"))
    return out


def _summarize_dimension(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work[col] = work[col].astype(object).where(work[col].notna(), "missing")
    grouped = work.groupby(col, dropna=False, observed=False)
    rows = []
    for level, g in grouped:
        z = pd.to_numeric(g["standardized_surprise_z_t"], errors="coerce")
        rows.append({
            col: str(level),
            "n_anchors": int(len(g)),
            "n_participants": int(g["participant_id"].nunique()) if "participant_id" in g else 0,
            "trigger_up_density": float(g["trigger_up"].fillna(False).mean()),
            "trigger_down_density": float(g["trigger_down"].fillna(False).mean()),
            "trigger_any_density": float((g["trigger_up"].fillna(False) | g["trigger_down"].fillna(False)).mean()),
            "mean_standardized_surprise": float(z.mean()) if z.notna().any() else np.nan,
            "mean_abs_standardized_surprise": float(z.abs().mean()) if z.notna().any() else np.nan,
            "mean_interval_width": float(pd.to_numeric(g["interval_width"], errors="coerce").mean()),
            "abstain_rate": float(g["abstain"].fillna(False).mean()),
            "event_enriched_frac": float(g["event_enriched"].fillna(False).mean()) if "event_enriched" in g else np.nan,
        })
    return pd.DataFrame(rows)


def _write_summary_figures(summary: pd.DataFrame, fig_dir: Path) -> list[str]:
    if summary.empty:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        fail = fig_dir / "study2_figure_generation_failed.txt"
        fail.write_text(str(exc))
        return [str(fail)]
    paths: list[str] = []
    for dim in SUMMARY_DIMENSIONS:
        dim_name = dim[0]
        sub = summary[(summary["cohort"].eq("non_insulin")) & (summary["dimension"].eq(dim_name))].copy()
        if sub.empty:
            sub = summary[(summary["cohort"].eq("all")) & (summary["dimension"].eq(dim_name))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("level")
        width = max(7.0, min(16.0, 0.55 * len(sub) + 3.5))
        fig, ax1 = plt.subplots(figsize=(width, 4.2))
        x = np.arange(len(sub))
        ax1.bar(x - 0.18, sub["trigger_up_density"], width=0.36, label="trigger_up", color="#2b7bba")
        ax1.bar(x + 0.18, sub["trigger_down_density"], width=0.36, label="trigger_down", color="#b44b4b")
        ax1.set_ylim(0, max(0.02, float(sub[["trigger_up_density", "trigger_down_density"]].max().max()) * 1.25))
        ax1.set_ylabel("Trigger density")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub["level"].astype(str), rotation=35, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x, sub["mean_standardized_surprise"], color="#333333", marker="o", linewidth=1.6, label="mean z_t")
        ax2.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("Mean standardized surprise")
        ax1.set_title(f"Study 2 triggers by {dim_name}")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        fig.tight_layout()
        path = fig_dir / f"study2_trigger_surprise_by_{_safe_name(dim_name)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _time_of_day_bin(df: pd.DataFrame, bin_minutes: int) -> pd.Series:
    hour = pd.Series(np.nan, index=df.index, dtype="float64")
    if "timestamp" in df.columns:
        raw = df["timestamp"]
        if not pd.api.types.is_numeric_dtype(raw):
            ts = pd.to_datetime(raw, errors="coerce", utc=True)
            if ts.notna().any():
                hour = ts.dt.hour.astype("float64")
    missing = hour.isna()
    if missing.any() and "anchor_ds" in df.columns:
        anchor_hour = ((pd.to_numeric(df["anchor_ds"], errors="coerce") * int(bin_minutes)) // 60) % 24
        hour.loc[missing] = anchor_hour.loc[missing]
    return hour.round().astype("Int64").astype(str).str.zfill(2).where(hour.notna(), "missing")


def _event_flag(ev: pd.DataFrame) -> pd.Series | None:
    if "anchor_family" in ev.columns:
        return ev["anchor_family"].astype(str).eq("event_enriched")
    if "eval_in_meal_window" in ev.columns:
        return ev["eval_in_meal_window"].fillna(False).astype(bool)
    if "eval_meal_event_id" in ev.columns:
        return pd.to_numeric(ev["eval_meal_event_id"], errors="coerce").fillna(-1).ge(0)
    return None


def _medication_group(med_insulin: pd.Series, med_any: pd.Series, med_metformin: pd.Series) -> pd.Series:
    insulin = med_insulin.fillna(0).astype(float).eq(1)
    any_drug = med_any.fillna(0).astype(float).eq(1)
    metformin = med_metformin.fillna(0).astype(float).eq(1)
    return pd.Series(
        np.select(
            [insulin, (~insulin) & metformin, (~insulin) & any_drug],
            ["insulin_user", "metformin_non_insulin", "other_non_insulin_drug"],
            default="no_recorded_diabetes_drug",
        ),
        index=med_insulin.index,
    )


def _quartile_labels(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series("missing", index=x.index, dtype=object)
    ok = x.notna()
    if ok.sum() < 4 or x[ok].nunique() < 2:
        return out
    try:
        q = pd.qcut(x[ok], 4, labels=False, duplicates="drop")
        out.loc[ok] = q.map(lambda v: f"Q{int(v) + 1}" if pd.notna(v) else "missing").astype(object)
    except Exception:
        pass
    return out


def _coalesce_numeric(df: pd.DataFrame, cols: Iterable[str], default: float) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out.fillna(default)


def _first_present(columns: Iterable[str], choices: Iterable[str]) -> str | None:
    available = set(columns)
    for col in choices:
        if col in available:
            return col
    return None


def _require(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_name(value: object) -> str:
    return "".join(ch if str(ch).isalnum() else "_" for ch in str(value)).strip("_").lower() or "unknown"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj
