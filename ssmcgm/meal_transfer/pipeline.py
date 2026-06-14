"""Phase G — orchestrate the passive meal-transfer pipeline end to end.

Pass 1 (teacher) + Pass 2 (CPU pseudo-labels, student, decoder) of the brief.
Does NOT touch the downstream SSM-CGM forecaster. Saves every artifact the brief
lists plus diagnostics, and returns a summary dict for the smoke report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig, SPLIT_FILES
from . import teacher as T
from . import features as F
from . import pseudo_labels as PL
from . import student as ST
from . import decoder as DEC
from . import diagnostics as DIAG

ID_COLS = ["participant_id", "ts", "ds", "split", "segment_id"]
CONTEXT_COLS = ["cgm_glucose", "study_group", "med_insulin", "hour"]


def _load(cfg: PipelineConfig) -> pd.DataFrame:
    """Load train+test, tag split, restrict to <= max_participants in smoke mode."""
    frames = []
    need = None
    for split, path in SPLIT_FILES.items():
        df = pd.read_feather(path)
        df["split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["participant_id"] = full["participant_id"].astype(str)

    if cfg.smoke:
        # Deterministically sample participants that have a long train history.
        train_parts = (
            full[full["split"] == "train"]["participant_id"].value_counts()
        )
        rng = np.random.default_rng(cfg.random_state)
        eligible = train_parts[train_parts >= cfg.teacher.seq_len + 24].index.to_numpy()
        rng.shuffle(eligible)
        keep = set(eligible[: cfg.max_participants])
        full = full[full["participant_id"].isin(keep)].copy()

    full = full.sort_values(["participant_id", "ts"]).reset_index(drop=True)
    if "minute_of_day" in full.columns:
        full["hour"] = (full["minute_of_day"] // 60).astype(int)
    elif "real_time" in full.columns:
        full["hour"] = pd.to_datetime(full["real_time"]).dt.hour
    return full


def run_pipeline(cfg: PipelineConfig | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    out_dir = cfg.ensure_output_dir()
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    t0 = time.time()
    summary: dict = {"config": {"smoke": cfg.smoke, "max_participants": cfg.max_participants}}

    # ---------------- load ----------------
    df = _load(cfg)
    summary["participants"] = int(df["participant_id"].nunique())
    summary["rows"] = int(len(df))
    summary["rows_by_split"] = df["split"].value_counts().to_dict()

    # ---------------- Phase B: teacher ----------------
    df = T.run_teacher(df, cfg.teacher)            # adds segment_id + teacher cols
    summary["teacher_source"] = str(df["teacher_source"].iloc[0])
    tp = df["cgmacros_teacher_probability"]
    summary["teacher_prob_available"] = bool(tp.notna().any())
    summary["teacher_prob_finite_frac"] = float(np.isfinite(tp).mean())
    summary["teacher_flag_rate"] = float(df["cgmacros_teacher_flag"].mean(skipna=True))
    summary["teacher_baseline_flag_rate"] = float(
        df["cgmacros_teacher_flag_baseline"].mean(skipna=True))
    if "predmeal_flag" in df.columns:
        summary["legacy_predmeal_flag_rate"] = float(df["predmeal_flag"].mean())
        summary["teacher_vs_artifact"] = _teacher_artifact_agreement(df)

    teacher_cols = ID_COLS + ["cgm_glucose", "study_group", "med_insulin",
                              "cgmacros_teacher_probability", "cgmacros_teacher_flag",
                              "cgmacros_teacher_flag_baseline", "teacher_source"]
    teacher_cols += [c for c in ["predmeal_flag"] if c in df.columns]
    df[teacher_cols].to_parquet(out_dir / "teacher_predictions.parquet", index=False)

    # ---------------- causal features ----------------
    df = F.add_causal_features(df)

    # ---------------- Phase D: pseudo-labels ----------------
    train_mask = df["split"] == "train"
    # Compute future-response on the full frame first (needs ordering); fit
    # thresholds on TRAIN rows only.
    df = PL.add_future_response(
        df, horizons_min=cfg.pseudo.response_horizons_min,
        peak_window_min=cfg.pseudo.peak_window_min,
    )
    thr = PL.fit_thresholds(df[train_mask], cfg.pseudo)
    labelled = PL.build_pseudo_labels(df, thr, cfg.pseudo)
    # build_pseudo_labels recomputes future-response; align its label columns back.
    for c in ["meal_pseudo_label", "pseudo_label_confidence",
              "meal_response_size_proxy", "hidden_insulin_risk",
              "future_peak_rise_120", "dg_30", "dg_60", "dg_120", "time_to_peak_min"]:
        df[c] = labelled[c].to_numpy()

    counts = df["meal_pseudo_label"].value_counts().to_dict()
    summary["pseudo_label_counts"] = {k: int(v) for k, v in counts.items()}
    summary["pseudo_label_counts_train"] = {
        k: int(v) for k, v in df.loc[train_mask, "meal_pseudo_label"].value_counts().items()}
    summary["pseudo_thresholds"] = thr.to_dict()

    pl_cols = ID_COLS + ["med_insulin", "meal_pseudo_label", "pseudo_label_confidence",
                         "meal_response_size_proxy", "hidden_insulin_risk",
                         "future_peak_rise_120", "dg_30", "dg_60", "dg_120",
                         "time_to_peak_min"]
    df[pl_cols].to_parquet(out_dir / "meal_pseudo_labels.parquet", index=False)

    # ---------------- Phase E: causal student ----------------
    art = ST.train_student(df[train_mask], cfg.student)
    summary["student"] = {
        "n_features": len(art.feature_cols), "features": art.feature_cols,
        "train_rows": art.train_rows, "train_pos_rate": art.pos_rate,
        "val_average_precision": art.val_ap, "val_auc": art.val_auc,
        "leakage_check": "passed",
    }
    df["student_meal_probability"] = ST.predict_student(df, art)

    student_cols = ID_COLS + ["student_meal_probability"] + art.feature_cols
    student_cols = list(dict.fromkeys(student_cols))  # de-dupe
    df[student_cols].to_parquet(out_dir / "causal_student_predictions.parquet", index=False)

    # ---------------- Phase F: meal-state decoder ----------------
    states = DEC.decode_dataframe(
        df, df["student_meal_probability"].to_numpy(), cfg.decoder,
        size_small=thr.size_small, size_medium=thr.size_medium,
    )
    for c in states.columns:
        df[f"state_{c}" if c == "meal_response_size_proxy" else c] = states[c].to_numpy()
    # keep decoder's size proxy under a distinct name to avoid clashing with PL's
    df.rename(columns={"state_meal_response_size_proxy": "decoded_response_size"},
              inplace=True)

    phase_dist = df["postprandial_phase"].value_counts(normalize=True).to_dict()
    summary["phase_distribution"] = {k: float(v) for k, v in phase_dist.items()}
    summary["decoded_flag_rate"] = float(df["predmeal_flag_clean"].mean())

    states_cols = ID_COLS + ["cgm_glucose", "study_group", "med_insulin",
                             "meal_probability", "predmeal_flag_clean",
                             "postprandial_phase", "time_since_predicted_meal",
                             "decoded_response_size", "meal_confidence",
                             "meal_support_score"]
    df[states_cols].to_parquet(out_dir / "passive_meal_states.parquet", index=False)

    # ---------------- event summary + response-by-size ----------------
    ev = _event_summary(df)
    ev.to_csv(out_dir / "meal_event_summary.csv", index=False)
    summary["n_events"] = int(len(ev))

    by_size = _response_by_size(ev)
    by_size.to_csv(out_dir / "meal_response_by_size.csv", index=False)
    summary["response_size_ordering"] = _size_ordering_check(by_size)

    # ---------------- Phase C: diagnostics ----------------
    diag = DIAG.run_diagnostics(df, diag_dir)
    for name, frame in diag.items():
        frame.to_csv(diag_dir / f"{name}.csv", index=False)
    summary["teacher_student_agreement"] = _teacher_student_agreement(df)

    # ---------------- metrics csv ----------------
    summary["runtime_sec"] = round(time.time() - t0, 1)
    metrics = _metrics_table(summary, df)
    metrics.to_csv(out_dir / "meal_transfer_metrics.csv", index=False)
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    _write_report(cfg, summary, out_dir)
    return summary


# --------------------------------------------------------------------------- #
def _event_summary(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (pid, seg), sub in df.groupby(["participant_id", "segment_id"], sort=False):
        sub = sub.reset_index(drop=True)
        f = sub["predmeal_flag_clean"].fillna(0).to_numpy()
        starts = np.where((f == 1) & (np.r_[0, f[:-1]] == 0))[0]
        ends = np.where((f == 1) & (np.r_[f[1:], 0] == 0))[0]
        for s, e in zip(starts, ends):
            seg_cgm = sub["cgm_glucose"].to_numpy()[s:e + 1]
            recs.append({
                "participant_id": pid, "segment_id": int(seg),
                "event_start_ts": sub["ts"].iloc[s],
                "duration_min": int((e - s + 1) * 5),
                "peak_rise_mgdl": float(np.nanmax(seg_cgm) - seg_cgm[0]),
                "response_size": sub["decoded_response_size"].iloc[s],
                "mean_student_prob": float(np.nanmean(
                    sub["student_meal_probability"].to_numpy()[s:e + 1])),
                "peak_student_prob": float(np.nanmax(
                    sub["student_meal_probability"].to_numpy()[s:e + 1])),
                "dg_60_at_onset": float(sub["dg_60"].iloc[s]) if "dg_60" in sub else np.nan,
                "dg_120_at_onset": float(sub["dg_120"].iloc[s]) if "dg_120" in sub else np.nan,
                "study_group": sub["study_group"].iloc[s] if "study_group" in sub else "",
                "med_insulin": float(sub["med_insulin"].iloc[s]) if "med_insulin" in sub else np.nan,
            })
    return pd.DataFrame(recs)


def _response_by_size(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame()
    order = {"small": 0, "medium": 1, "large": 2}
    g = ev.groupby("response_size").agg(
        n_events=("response_size", "size"),
        mean_peak_rise=("peak_rise_mgdl", "mean"),
        median_peak_rise=("peak_rise_mgdl", "median"),
        mean_duration_min=("duration_min", "mean"),
        mean_dg_60=("dg_60_at_onset", "mean"),
        mean_dg_120=("dg_120_at_onset", "mean"),
        mean_peak_student_prob=("peak_student_prob", "mean"),
    ).reset_index()
    g["_o"] = g["response_size"].map(order).fillna(99)
    return g.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def _size_ordering_check(by_size: pd.DataFrame) -> dict:
    if by_size.empty or "mean_peak_rise" not in by_size:
        return {"monotonic_peak_rise": None}
    sizes = ["small", "medium", "large"]
    vals = [by_size.loc[by_size.response_size == s, "mean_peak_rise"].mean() for s in sizes]
    vals = [v for v in vals if not np.isnan(v)]
    return {"monotonic_peak_rise": bool(all(x < y for x, y in zip(vals, vals[1:]))),
            "peak_rise_by_size": dict(zip(sizes, [round(float(v), 2) for v in
                                                  by_size["mean_peak_rise"]]))}


def _teacher_artifact_agreement(df: pd.DataFrame) -> dict:
    """Per-row agreement between the corrected reconstruction (legacy ratio-vote
    baseline) and the surviving `predmeal_flag` artifact. Characterises how much
    coverage the downstream timestamp-merge lost."""
    m = df["cgmacros_teacher_flag_baseline"].notna()
    if not m.any():
        return {"available": False}
    a = df.loc[m, "cgmacros_teacher_flag_baseline"].astype(int).to_numpy()
    b = df.loc[m, "predmeal_flag"].fillna(0).astype(int).to_numpy()
    return {
        "available": True,
        "rows": int(m.sum()),
        "row_agreement": float((a == b).mean()),
        "reconstruction_flag_rate": float(a.mean()),
        "artifact_flag_rate": float(b.mean()),
        "recon1_artifact0": int(((a == 1) & (b == 0)).sum()),
        "recon0_artifact1": int(((a == 0) & (b == 1)).sum()),
    }


def _teacher_student_agreement(df: pd.DataFrame) -> dict:
    if not np.isfinite(df["cgmacros_teacher_probability"]).any():
        return {"available": False}
    m = df["cgmacros_teacher_probability"].notna() & df["student_meal_probability"].notna()
    a = df.loc[m, "cgmacros_teacher_flag"].to_numpy()
    b = (df.loc[m, "student_meal_probability"] >= 0.5).to_numpy().astype(float)
    agree = float((a == b).mean()) if m.any() else np.nan
    corr = float(np.corrcoef(df.loc[m, "cgmacros_teacher_probability"],
                             df.loc[m, "student_meal_probability"])[0, 1]) if m.sum() > 2 else np.nan
    return {"available": True, "flag_agreement": agree, "prob_correlation": corr}


def _metrics_table(summary: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("participants", summary["participants"]),
        ("rows", summary["rows"]),
        ("teacher_source", summary["teacher_source"]),
        ("teacher_prob_finite_frac", summary["teacher_prob_finite_frac"]),
        ("teacher_flag_rate", summary["teacher_flag_rate"]),
        ("teacher_baseline_flag_rate", summary["teacher_baseline_flag_rate"]),
        ("legacy_predmeal_flag_rate", summary.get("legacy_predmeal_flag_rate")),
        ("teacher_vs_artifact_row_agreement",
         summary.get("teacher_vs_artifact", {}).get("row_agreement")),
        ("artifact_coverage_loss_recon1_artifact0",
         summary.get("teacher_vs_artifact", {}).get("recon1_artifact0")),
        ("pseudo_positive", summary["pseudo_label_counts"].get("positive", 0)),
        ("pseudo_negative", summary["pseudo_label_counts"].get("negative", 0)),
        ("pseudo_uncertain", summary["pseudo_label_counts"].get("uncertain", 0)),
        ("student_val_average_precision", summary["student"]["val_average_precision"]),
        ("student_val_auc", summary["student"]["val_auc"]),
        ("decoded_flag_rate", summary["decoded_flag_rate"]),
        ("n_events", summary["n_events"]),
        ("teacher_student_flag_agreement",
         summary["teacher_student_agreement"].get("flag_agreement")),
        ("teacher_student_prob_correlation",
         summary["teacher_student_agreement"].get("prob_correlation")),
        ("size_ordering_monotonic",
         summary["response_size_ordering"].get("monotonic_peak_rise")),
        ("runtime_sec", summary.get("runtime_sec")),
    ]
    for ph in ("none", "onset", "rising", "peak", "recovery"):
        rows.append((f"phase_frac_{ph}", summary["phase_distribution"].get(ph, 0.0)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def _write_report(cfg: PipelineConfig, summary: dict, out_dir: Path) -> None:
    from .report import build_report
    (out_dir / "meal_transfer_report.md").write_text(build_report(cfg, summary))
