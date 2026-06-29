#!/usr/bin/env python3
"""AI-READI Study 2 interpretability export.

This script mirrors the T1DEXI interpretability outputs using Study 2
assumptions:

* no verified nutrition event timestamps;
* no scheduled medication action covariates;
* event-aware surprise, triggered excursion, unmatched control, medication
  metadata, and diagnostic meal proxy wording only.

All outputs are written below outputs/study2_interpretability by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    build_stream_feature_spec,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.evaluation.study2_interpretability import (
    Study2InterpretabilityConfig,
    build_failure_mode_map,
    summarize_failure_mode_map,
)
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig
from ssmcgm.ops.mes_attribution import compute_hidden_attention

from scripts.run_study2_final_rich_head import (
    add_episode_features,
    build_anchor_table_rich,
    apply_triggers,
    event_enriched_proxy,
    get_triggered_rows,
    pred_ridge_split,
    train_ridge,
)


BIN_MIN = 5
HORIZON = 12
DEFAULT_CACHE = ROOT / "outputs/study2_forecast_cache_5min/study2_forecast_cache.parquet"
DEFAULT_CACHE_MANIFEST = ROOT / "outputs/study2_forecast_cache_5min/MANIFEST.json"
DEFAULT_CONFIG = ROOT / "configs/study2_forecast_cache_5min.yaml"
DEFAULT_FINAL_HEAD_DIR = ROOT / "outputs/study2_final_rich_head"
DEFAULT_OUT = ROOT / "outputs/study2_interpretability"

MODALITY_ORDER = [
    "glucose history",
    "heart rate",
    "steps / activity",
    "sleep",
    "stress",
    "time / calendar",
    "static clinical",
    "medication metadata",
    "site / cohort",
    "data quality",
    "meal proxy, diagnostic only",
]

MODALITY_COLORS = {
    "glucose history": "#2166ac",
    "heart rate": "#1b9e77",
    "steps / activity": "#4daf4a",
    "sleep": "#7570b3",
    "stress": "#e7298a",
    "time / calendar": "#ff7f00",
    "static clinical": "#8c6bb1",
    "medication metadata": "#a6761d",
    "site / cohort": "#666666",
    "data quality": "#999999",
    "meal proxy, diagnostic only": "#17becf",
}

PRETTY_FEATURE = {
    "r_t": "residual",
    "r_t_1": "residual t-1",
    "r_t_2": "residual t-2",
    "r_t_3": "residual t-3",
    "z_t": "z_t",
    "z_t_1": "z_t t-1",
    "z_t_2": "z_t t-2",
    "slope_15": "15 min slope",
    "slope_30": "30 min slope",
    "slope_60": "60 min slope",
    "curvature": "curvature",
    "interval_width": "interval width",
    "current_glucose": "current glucose",
    "time_since_trigger": "time since trigger",
    "trigger_duration_so_far": "trigger duration",
    "cumulative_residual_since_trigger": "cumulative residual",
    "hour_sin": "hour sin",
    "hour_cos": "hour cos",
    "trigger_up_float": "up-trigger flag",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export AI-READI Study 2 interpretability artifacts"
    )
    ap.add_argument("--cache-path", default=str(DEFAULT_CACHE))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--final-head-dir", default=str(DEFAULT_FINAL_HEAD_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-streams", type=int, default=10)
    ap.add_argument("--anchors-per-stream", type=int, default=4)
    ap.add_argument("--max-mes-streams", type=int, default=8)
    ap.add_argument("--max-lags", type=int, default=144)
    ap.add_argument("--episodes-per-figure", type=int, default=2)
    ap.add_argument("--lead-hours", type=float, default=4.0)
    ap.add_argument("--dpi", type=int, default=170)
    return ap.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2))


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def save_figure(fig: plt.Figure, path: Path, dpi: int = 170) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(path), str(pdf)]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def load_cache(path: Path) -> pd.DataFrame:
    wanted = [
        "participant_id",
        "segment_id",
        "split",
        "anchor_time_idx",
        "steps_since_start",
        "hours_since_start",
        "scenario_mode",
        "participants_study_group",
        "hba1c_percent_baseline",
        "bmi_baseline",
        "participants_clinical_site",
        "med_insulin",
        "med_any_diabetes_drug",
        "med_metformin",
        "med_glp1_or_gip_glp1",
        "med_sglt2",
        "med_sulfonylurea",
        "med_thiazolidinedione",
        "horizon_step",
        "horizon_minutes",
        "target",
        "observed",
        "q10",
        "q50",
        "q90",
        "anchor_ds",
        "current_glucose",
    ]
    available = set(parquet_columns(path))
    cols = [c for c in wanted if c in available]
    df = pd.read_parquet(path, columns=cols)
    if "anchor_ds" not in df.columns:
        df["anchor_ds"] = df["anchor_time_idx"]
    df["participant_id"] = df["participant_id"].astype(str)
    if "scenario_mode" in df.columns:
        mode = df["scenario_mode"].astype(str)
        keep = mode.eq("forecast_only")
        if keep.any():
            df = df[keep].copy()
    return df


def load_model(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    cfg = AireadiStreamModelConfig(**md["model_config"])
    model = AireadiStreamModel(spec, pre, cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, spec, pre, ckpt


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_checkpoint(args: argparse.Namespace, cfg: dict) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    manifest = read_json(DEFAULT_CACHE_MANIFEST, {})
    if manifest and manifest.get("base_checkpoint"):
        p = Path(manifest["base_checkpoint"])
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    p = Path(cfg.get("base_checkpoint", ""))
    if not p.is_absolute():
        p = ROOT / p
    return p


def modality_columns(spec: AireadiFeatureSpec) -> OrderedDict[str, dict[str, list[str]]]:
    dyn = list(spec.dynamic_reals)
    tim = list(spec.time_reals)
    scn = list(spec.scenario_reals)
    scont = list(spec.static_reals)
    scat = list(spec.static_categoricals)

    def present(names: Iterable[str], universe: list[str]) -> list[str]:
        have = set(universe)
        return [n for n in names if n in have]

    quality = [
        n
        for n in dyn
        if n.endswith("_count")
        or "device_availability" in n
        or n in {"cgm_count", "sleep_stage_unknown"}
    ]
    heart = [
        n
        for n in dyn
        if (
            n.startswith("heart_rate_")
            or n.startswith("respiratory_rate_")
            or n.startswith("oxygen_saturation_")
        )
        and n not in quality
    ]
    activity = [
        n
        for n in dyn
        if n.startswith("activity_") and n not in quality
    ] + present(["calories_total", "calories_per_min"], dyn)
    sleep = [
        n
        for n in dyn
        if n.startswith("sleep_") and n not in quality
    ]
    stress = [
        n
        for n in dyn
        if n.startswith("stress_level_") and n not in quality
    ]
    clinical = [
        n
        for n in scont
        if not n.startswith("med_")
        and not n.startswith("demo_race_")
        and not n.startswith("demo_ethnicity_")
        and not n.startswith("demo_")
    ]
    meds = [n for n in scont if n.startswith("med_")]
    site = present(["participants_clinical_site", "participants_study_group"], scat)

    groups = OrderedDict()
    for name in MODALITY_ORDER:
        groups[name] = {"dynamic": [], "time": [], "scenario": [], "static_cont": [], "static_cat": []}
    groups["glucose history"]["dynamic"] = present(["cgm_glucose_mean"], dyn)
    groups["heart rate"]["dynamic"] = heart
    groups["steps / activity"]["dynamic"] = activity
    groups["sleep"]["dynamic"] = sleep
    groups["stress"]["dynamic"] = stress
    groups["time / calendar"]["time"] = tim
    groups["static clinical"]["static_cont"] = clinical
    groups["medication metadata"]["static_cont"] = meds
    groups["site / cohort"]["static_cat"] = site
    groups["data quality"]["dynamic"] = quality
    groups["meal proxy, diagnostic only"]["scenario"] = present(["predmeal_flag"], scn)
    return groups


def feature_to_dynamic_group(groups: OrderedDict[str, dict[str, list[str]]]) -> dict[str, str]:
    out = {}
    for group, members in groups.items():
        for name in members.get("dynamic", []):
            out[name] = group
    return out


def pretty_feature(name: str) -> str:
    if name in PRETTY_FEATURE:
        return PRETTY_FEATURE[name]
    return str(name).replace("_", " ")


def build_anchor_state(cache: pd.DataFrame, final_head_dir: Path):
    manifest = read_json(final_head_dir / "study2_final_manifest.json", {})
    tau_up = float(manifest.get("tau_up_selected_on_validation", 0.2))
    tau_down = float(manifest.get("tau_down_selected_on_validation", 0.2))
    at = apply_triggers(build_anchor_table_rich(cache))
    at = add_episode_features(at)
    at["trigger_up_float"] = at["trigger_up"].astype(float)
    at["event_enriched"] = event_enriched_proxy(at)
    at = at.sort_values(["participant_id", "segment_id", "anchor_ds"]).reset_index(drop=True)
    features = list(manifest.get("causal_features_used") or [
        "r_t",
        "r_t_1",
        "r_t_2",
        "r_t_3",
        "z_t",
        "z_t_1",
        "z_t_2",
        "slope_15",
        "slope_30",
        "slope_60",
        "curvature",
        "interval_width",
        "current_glucose",
        "time_since_trigger",
        "trigger_duration_so_far",
        "cumulative_residual_since_trigger",
        "hour_sin",
        "hour_cos",
        "trigger_up_float",
    ])
    return at, features, tau_up, tau_down, manifest


def prepare_correction_head(cache: pd.DataFrame, at: pd.DataFrame, features: list[str]):
    splits_fc = {}
    for split in ("train", "validation", "test"):
        tk = at[
            at["non_insulin"].fillna(True)
            & at["triggered"].fillna(False)
            & at["split"].eq(split)
        ][["participant_id", "segment_id", "anchor_ds", "event_enriched"]].copy()
        fc = get_triggered_rows(cache[cache["split"].eq(split)], tk, at, features)
        fc = fc[fc["non_insulin"].fillna(True)].copy()
        splits_fc[split] = fc

    train_fc = splits_fc["train"]
    usable = [f for f in features if f in train_fc.columns]
    usable = [f for f in usable if float(pd.to_numeric(train_fc[f], errors="coerce").std()) > 0.0]
    ridge_up, scaler_up = train_ridge(train_fc, usable, "up")
    ridge_dn, scaler_dn = train_ridge(train_fc, usable, "down")
    for split, fc in splits_fc.items():
        if fc.empty:
            fc["corrected_pred"] = np.nan
            continue
        fc["corrected_pred"] = pred_ridge_split(
            fc,
            (ridge_up, scaler_up),
            (ridge_dn, scaler_dn),
            usable,
        )
    return splits_fc, usable, (ridge_up, scaler_up), (ridge_dn, scaler_dn)


def export_correction_coefficients(
    out_dir: Path,
    features: list[str],
    up_model,
    down_model,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for direction, bundle in [("trigger_up", up_model), ("trigger_down", down_model)]:
        models, scaler = bundle
        scales = getattr(scaler, "scale_", np.ones(len(features)))
        means = getattr(scaler, "mean_", np.zeros(len(features)))
        for horizon, mdl in sorted(models.items()):
            coef = np.asarray(mdl.coef_, dtype=float)
            for j, feature in enumerate(features):
                rows.append(
                    {
                        "direction": direction,
                        "horizon_step": int(horizon),
                        "horizon_minutes": int(horizon) * BIN_MIN,
                        "feature": feature,
                        "feature_label": pretty_feature(feature),
                        "coef_standardized": float(coef[j]),
                        "coef_raw_scale": float(coef[j] / scales[j]) if scales[j] else np.nan,
                        "scaler_mean": float(means[j]),
                        "scaler_scale": float(scales[j]),
                        "intercept": float(mdl.intercept_),
                    }
                )
    coef_df = pd.DataFrame(rows)
    coef_path = out_dir / "tables/study2_correction_head_coefficients.csv"
    coef_path.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(coef_path, index=False)

    up = coef_df[coef_df["direction"].eq("trigger_up")]
    dn = coef_df[coef_df["direction"].eq("trigger_down")]
    cmp = up.merge(
        dn,
        on=["horizon_step", "horizon_minutes", "feature", "feature_label"],
        suffixes=("_up", "_down"),
    )
    cmp["coef_standardized_delta_up_minus_down"] = (
        cmp["coef_standardized_up"] - cmp["coef_standardized_down"]
    )
    cmp_path = out_dir / "tables/study2_correction_head_up_down_comparison.csv"
    cmp.to_csv(cmp_path, index=False)
    return coef_df, cmp


def correction_contributions(anchor_rows: pd.DataFrame, features: list[str], up_model, down_model) -> pd.Series:
    if anchor_rows.empty:
        return pd.Series(dtype=float)
    direction = "up" if bool(anchor_rows["trigger_up"].iloc[0]) else "down"
    models, scaler = up_model if direction == "up" else down_model
    contribs = []
    for _, row in anchor_rows.iterrows():
        horizon = int(row["horizon_step"])
        if horizon not in models:
            continue
        x = row[features].to_numpy(dtype=float).reshape(1, -1)
        x = np.where(np.isfinite(x), x, 0.0)
        xs = scaler.transform(x)[0]
        contribs.append(pd.Series(xs * np.asarray(models[horizon].coef_, dtype=float), index=features))
    if not contribs:
        return pd.Series(0.0, index=features)
    return pd.concat(contribs, axis=1).mean(axis=1).sort_values(key=lambda s: s.abs(), ascending=False)


def select_local_episodes(test_fc: pd.DataFrame, episodes_per_figure: int) -> dict[str, list[dict]]:
    if test_fc.empty:
        return {}
    key = ["participant_id", "segment_id", "anchor_ds"]
    agg = (
        test_fc.groupby(key, as_index=False)
        .agg(
            split=("split", "first"),
            current_glucose=("current_glucose", "first"),
            z_t=("z_t", "first"),
            one_step_residual=("one_step_residual", "first"),
            interval_width=("interval_width", "first"),
            slope_15=("slope_15", "first"),
            curvature=("curvature", "first"),
            trigger_up=("trigger_up", "first"),
            trigger_down=("trigger_down", "first"),
            event_enriched=("event_enriched", "first"),
            future_max=("target", "max"),
            future_min=("target", "min"),
            base_future_max=("q50", "max"),
            base_future_min=("q50", "min"),
        )
        .copy()
    )
    agg["participant_id"] = agg["participant_id"].astype(str)
    agg["score_up"] = (
        pd.to_numeric(agg["z_t"], errors="coerce").clip(lower=0).fillna(0) * 20.0
        + (agg["future_max"] - agg["current_glucose"]).clip(lower=0).fillna(0)
        + (agg["base_future_max"] - agg["current_glucose"]).clip(lower=0).fillna(0) * 0.5
    )
    agg["score_down"] = (
        (-pd.to_numeric(agg["z_t"], errors="coerce")).clip(lower=0).fillna(0) * 20.0
        + (agg["current_glucose"] - agg["future_min"]).clip(lower=0).fillna(0)
        + (agg["current_glucose"] - agg["base_future_min"]).clip(lower=0).fillna(0) * 0.5
    )

    out: dict[str, list[dict]] = {}
    for direction, flag, score in [
        ("trigger_up", "trigger_up", "score_up"),
        ("trigger_down", "trigger_down", "score_down"),
    ]:
        cand = agg[agg[flag].fillna(False)].sort_values(score, ascending=False).copy()
        picked = []
        for (pid, seg), group in cand.groupby(["participant_id", "segment_id"], sort=False):
            anchors = []
            rows = []
            for _, row in group.iterrows():
                a = int(row["anchor_ds"])
                if all(abs(a - prev) >= 18 for prev in anchors):
                    anchors.append(a)
                    rows.append(row.to_dict())
                if len(rows) >= episodes_per_figure:
                    break
            if len(rows) > len(picked):
                picked = rows
            if len(picked) >= episodes_per_figure:
                break
        if picked:
            out[direction] = picked
    return out


def desired_participants(cache: pd.DataFrame, local: dict[str, list[dict]], split: str, max_streams: int) -> list[str]:
    h1 = cache[(cache["split"].eq(split)) & (cache["horizon_step"].eq(1))]
    pids = h1["participant_id"].astype(str).drop_duplicates().head(max_streams).tolist()
    for rows in local.values():
        for row in rows:
            pid = str(row["participant_id"])
            if pid not in pids:
                pids.append(pid)
    return pids


def load_stream_subset(cfg: dict, spec: AireadiFeatureSpec, pre, pids: list[str], split_name: str):
    data_cfg = cfg["data"]
    df = load_aireadi_panel(
        data_cfg["panel_path"],
        static_path=data_cfg.get("static_path"),
        cohort_path=data_cfg.get("cohort_path"),
    )
    keep = set(str(p) for p in pids)
    df = df[df["participant_id"].astype(str).isin(keep)].copy()
    schema = infer_or_validate_schema(df, data_cfg.get("schema"))
    df = prepare_aireadi_panel(
        df,
        schema,
        bin_minutes=cfg["dataset"].get("bin_minutes", 5),
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    split_cfg = cfg.get("split", {})
    split = make_aireadi_stream_splits(
        df,
        split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", 0.70),
        val=split_cfg.get("val", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg.get("existing_split_path"),
    )
    build_stream_feature_spec(
        df,
        schema,
        horizon_steps=spec.horizon_steps,
        bin_minutes=spec.bin_minutes,
    )
    streams = make_participant_streams(
        df,
        split,
        schema,
        feature_spec=spec,
        preprocessor=pre,
        splits=[split_name],
        min_steps=spec.horizon_steps + 2,
    )
    return streams


def stream_key(stream) -> tuple[str, int]:
    return str(stream.participant_id), int(stream.segment_id)


def build_anchor_selection(
    cache: pd.DataFrame,
    streams,
    split: str,
    anchors_per_stream: int,
    local: dict[str, list[dict]],
) -> dict[tuple[str, int], list[int]]:
    h1 = cache[(cache["split"].eq(split)) & (cache["horizon_step"].eq(1))]
    by_key = {
        k: g["anchor_ds"].drop_duplicates().astype(int).sort_values().to_numpy()
        for k, g in h1.groupby(["participant_id", "segment_id"], sort=False)
    }
    selected: dict[tuple[str, int], set[int]] = defaultdict(set)
    for stream in streams:
        key = stream_key(stream)
        vals = by_key.get(key)
        if vals is None or len(vals) == 0:
            continue
        vals = vals[vals >= 1]
        vals = vals[vals + HORIZON < int(stream.n_steps)]
        if len(vals) == 0:
            continue
        if len(vals) <= anchors_per_stream:
            picks = vals
        else:
            idx = np.linspace(0, len(vals) - 1, anchors_per_stream).round().astype(int)
            picks = vals[idx]
        selected[key].update(int(x) for x in picks)
    for rows in local.values():
        for row in rows:
            key = (str(row["participant_id"]), int(row["segment_id"]))
            selected[key].add(int(row["anchor_ds"]))
    return {k: sorted(v) for k, v in selected.items()}


def clone_stream_tensors(stream, device: str, group=None, groups=None):
    dynamic = stream.dynamic.to(device).clone()
    time_features = stream.time_features.to(device).clone()
    scenario_values = stream.scenario_values.to(device).clone()
    scenario_mask = torch.zeros_like(stream.scenario_mask.to(device))
    static_cont = stream.static_cont.to(device).clone()
    static_cat = stream.static_cat.to(device).clone()
    target = stream.target.to(device)
    time_idx = stream.time_idx.to(device)

    if group is not None and groups is not None:
        spec = stream.feature_spec
        members = groups[group]
        for name in members.get("dynamic", []):
            if name in spec.dynamic_reals:
                dynamic[:, spec.dynamic_reals.index(name)] = 0.0
        for name in members.get("time", []):
            if name in spec.time_reals:
                time_features[:, spec.time_reals.index(name)] = 0.0
        for name in members.get("scenario", []):
            if name in spec.scenario_reals:
                j = spec.scenario_reals.index(name)
                scenario_values[:, j] = 0.0
                scenario_mask[:, j] = 0.0
        for name in members.get("static_cont", []):
            if name in spec.static_reals:
                static_cont[spec.static_reals.index(name)] = 0.0
        for name in members.get("static_cat", []):
            if name in spec.static_categoricals:
                static_cat[spec.static_categoricals.index(name)] = 0
    return dynamic, time_features, scenario_values, scenario_mask, static_cont, static_cat, target, time_idx


@torch.no_grad()
def predict_stream_q(
    model: AireadiStreamModel,
    stream,
    anchors: list[int],
    device: str,
    group=None,
    groups=None,
) -> np.ndarray:
    (
        dynamic,
        time_features,
        scenario_values,
        scenario_mask,
        static_cont,
        static_cat,
        target,
        _time_idx,
    ) = clone_stream_tensors(stream, device, group=group, groups=groups)
    sctx = model.encode_static(static_cat, static_cont)
    state = model.init_stream(sctx)
    state, out = model.scan_chunk(dynamic.unsqueeze(0), sctx, state)
    pos = torch.tensor(anchors, dtype=torch.long, device=device)
    fut = pos[:, None] + 1 + torch.arange(model.feature_spec.horizon_steps, device=device)[None, :]
    raw = model.decode_horizon(out[0, pos], sctx, time_features[fut], scenario_values[fut], scenario_mask[fut])
    pred = target[pos].view(-1, 1, 1) + raw
    return pred.detach().cpu().numpy()


def compute_ablation_attribution(
    model: AireadiStreamModel,
    streams,
    anchors_by_stream: dict[tuple[str, int], list[int]],
    groups: OrderedDict[str, dict[str, list[str]]],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qidx = int(np.argmin(np.abs(np.asarray(model.quantiles) - 0.5)))
    rows = []
    per_anchor = []
    for stream in streams:
        key = stream_key(stream)
        anchors = [a for a in anchors_by_stream.get(key, []) if 0 <= a and a + HORIZON < stream.n_steps]
        if not anchors:
            continue
        base = predict_stream_q(model, stream, anchors, device)[:, :, qidx]
        for group in groups:
            if not any(groups[group].values()):
                delta = np.zeros_like(base)
            else:
                ablated = predict_stream_q(model, stream, anchors, device, group=group, groups=groups)[:, :, qidx]
                delta = base - ablated
            for ai, anchor in enumerate(anchors):
                rows.append(
                    {
                        "participant_id": stream.participant_id,
                        "segment_id": int(stream.segment_id),
                        "anchor_ds": int(anchor),
                        "modality": group,
                        "mean_abs_delta_mgdl": float(np.mean(np.abs(delta[ai]))),
                        "mean_signed_delta_mgdl": float(np.mean(delta[ai])),
                        "terminal_abs_delta_mgdl": float(abs(delta[ai, -1])),
                    }
                )
        for ai, anchor in enumerate(anchors):
            per_anchor.append(
                {
                    "participant_id": stream.participant_id,
                    "segment_id": int(stream.segment_id),
                    "anchor_ds": int(anchor),
                    "base_model_q50_h12": float(base[ai, -1]),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    glob = (
        detail.groupby("modality", as_index=False)
        .agg(
            mean_abs_delta_mgdl=("mean_abs_delta_mgdl", "mean"),
            mean_signed_delta_mgdl=("mean_signed_delta_mgdl", "mean"),
            n_anchor_modality_rows=("mean_abs_delta_mgdl", "size"),
        )
        .copy()
    )
    total = glob["mean_abs_delta_mgdl"].sum()
    glob["frac_influence"] = glob["mean_abs_delta_mgdl"] / total if total > 0 else 0.0
    glob["modality"] = pd.Categorical(glob["modality"], MODALITY_ORDER, ordered=True)
    glob = glob.sort_values("frac_influence", ascending=False)
    return glob, detail


@torch.no_grad()
def scan_with_record(model: AireadiStreamModel, stream, device: str):
    dynamic = stream.dynamic.to(device)
    static_cont = stream.static_cont.to(device)
    static_cat = stream.static_cat.to(device)
    sctx = model.encode_static(static_cat, static_cont)
    state = model.init_stream(sctx)
    feats = model._dynamic_features(dynamic.unsqueeze(0))
    fused, contribs = model.encoder_fusion(feats, return_contributions=True)
    fused = model.film(fused, sctx.embedding)
    out, layer_states, conv_states, caches = model.temporal.scan(
        fused,
        state.layer_states,
        state.conv_states,
        record=True,
        static_embedding=sctx.embedding,
    )
    return out, caches, contribs


def compute_mes_temporal(
    model: AireadiStreamModel,
    streams,
    anchors_by_stream: dict[tuple[str, int], list[int]],
    groups: OrderedDict[str, dict[str, list[str]]],
    device: str,
    out_dir: Path,
    max_lags: int,
    max_streams: int,
) -> tuple[str, pd.DataFrame, pd.DataFrame, Path | None]:
    if model.config.mamba_style != "mes":
        raise RuntimeError(f"MES attribution unavailable for mamba_style={model.config.mamba_style}")
    feature_group = feature_to_dynamic_group(groups)
    group_acc = {g: np.zeros(max_lags + 1, dtype="float64") for g in MODALITY_ORDER}
    feature_acc: dict[str, np.ndarray] = {}
    head_acc = None
    target_maps = []
    n_windows = 0
    used_streams = 0
    lags = np.arange(max_lags + 1) * model.feature_spec.bin_minutes

    for stream in streams:
        if used_streams >= max_streams:
            break
        anchors = [a for a in anchors_by_stream.get(stream_key(stream), []) if 0 <= a < stream.n_steps]
        if not anchors:
            continue
        out, caches, contribs = scan_with_record(model, stream, device)
        layer = len(caches) - 1
        rows_t = torch.tensor(anchors, dtype=torch.long, device=device)
        attn, _ = compute_hidden_attention(
            caches[layer],
            rows=rows_t,
            max_lags=max_lags,
            ngroups=model.temporal.blocks[0].ssm.ngroups,
            normalize="softmax",
            out_device="cpu",
        )
        attn_np = attn.float().cpu().numpy()  # B,H,R,K
        if attn_np.shape[0] != 1:
            attn_np = attn_np[:1]
        target_maps.append(attn_np[0])
        head_mean = attn_np.mean(axis=(0, 2))
        head_acc = head_mean if head_acc is None else head_acc + head_mean

        names = list(contribs)
        u_norm = np.stack(
            [contribs[name].detach().cpu().norm(dim=-1).numpy()[0] for name in names],
            axis=0,
        )  # F,L
        row_arr = np.asarray(anchors, dtype=int)
        lag_idx = np.arange(max_lags + 1, dtype=int)
        src = row_arr[:, None] - lag_idx[None, :]
        valid = src >= 0
        src_clip = np.clip(src, 0, u_norm.shape[1] - 1)
        attn_sum = attn_np[0].sum(axis=0)  # R,K
        for fi, name in enumerate(names):
            vals = u_norm[fi, src_clip]
            vals = np.where(valid, vals, 0.0)
            infl = (attn_sum * vals).sum(axis=0)
            feature_acc.setdefault(name, np.zeros(max_lags + 1, dtype="float64"))
            feature_acc[name] += infl
            group = feature_group.get(name)
            if group is not None:
                group_acc[group] += infl
        n_windows += len(anchors)
        used_streams += 1

    if n_windows == 0 or head_acc is None:
        raise RuntimeError("No MES windows were collected")

    head_lag = head_acc / max(used_streams, 1)
    head_norm = head_lag / np.clip(head_lag.sum(axis=1, keepdims=True), 1e-12, None)
    head_rows = []
    for h in range(head_norm.shape[0]):
        for k, lag in enumerate(lags):
            head_rows.append({"head": h, "lag_min": int(lag), "attention": float(head_norm[h, k])})
    head_df = pd.DataFrame(head_rows)

    group_rows = []
    total_group = sum(float(v.sum()) for v in group_acc.values())
    for group, arr in group_acc.items():
        denom = total_group if total_group > 0 else 1.0
        for k, lag in enumerate(lags):
            group_rows.append(
                {
                    "modality": group,
                    "lag_min": int(lag),
                    "frac_influence": float(arr[k] / denom),
                    "raw_influence": float(arr[k]),
                }
            )
    group_df = pd.DataFrame(group_rows)

    feat_rows = []
    total_feat = sum(float(v.sum()) for v in feature_acc.values())
    for feature, arr in feature_acc.items():
        group = feature_group.get(feature, "unassigned")
        denom = total_feat if total_feat > 0 else 1.0
        for k, lag in enumerate(lags):
            feat_rows.append(
                {
                    "feature": feature,
                    "modality": group,
                    "lag_min": int(lag),
                    "frac_influence": float(arr[k] / denom),
                    "raw_influence": float(arr[k]),
                }
            )
    feat_df = pd.DataFrame(feat_rows)
    overall = (
        feat_df.groupby(["feature", "modality"], as_index=False)["raw_influence"]
        .sum()
        .sort_values("raw_influence", ascending=False)
    )
    total = overall["raw_influence"].sum()
    overall["frac_influence"] = overall["raw_influence"] / total if total > 0 else 0.0

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(tables / "study2_temporal_lag_attribution.csv", index=False)
    feat_df.to_csv(tables / "study2_feature_temporal_attribution.csv", index=False)
    overall.to_csv(tables / "study2_feature_attribution_overall.csv", index=False)
    head_df.to_csv(tables / "study2_mes_head_lag.csv", index=False)

    npz_path = out_dir / "mes_attention.npz"
    head_target_lag = np.concatenate(target_maps, axis=1)
    np.savez_compressed(
        npz_path,
        head_target_lag=head_target_lag,
        head_lag=head_norm,
        lags_min=lags,
        n_windows=n_windows,
        n_streams=used_streams,
    )
    return "mes_hidden_attention", group_df, head_df, npz_path


def lag_block_fallback(
    model: AireadiStreamModel,
    streams,
    anchors_by_stream: dict[tuple[str, int], list[int]],
    device: str,
    out_dir: Path,
) -> tuple[str, pd.DataFrame, pd.DataFrame, None]:
    blocks = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 720)]
    qidx = int(np.argmin(np.abs(np.asarray(model.quantiles) - 0.5)))
    rows = []
    for stream in streams[:4]:
        anchors = anchors_by_stream.get(stream_key(stream), [])[:2]
        for anchor in anchors:
            if anchor + HORIZON >= stream.n_steps:
                continue
            base = predict_stream_q(model, stream, [anchor], device)[:, :, qidx]
            for lo, hi in blocks:
                dynamic = stream.dynamic.to(device).clone()
                lo_s = int(lo // model.feature_spec.bin_minutes)
                hi_s = int(hi // model.feature_spec.bin_minutes)
                start = max(0, anchor - hi_s + 1)
                stop = max(0, anchor - lo_s + 1)
                if stop > start:
                    dynamic[start:stop, :] = 0.0
                static_cont = stream.static_cont.to(device)
                static_cat = stream.static_cat.to(device)
                sctx = model.encode_static(static_cat, static_cont)
                state = model.init_stream(sctx)
                state, out = model.scan_chunk(dynamic.unsqueeze(0), sctx, state)
                pos = torch.tensor([anchor], dtype=torch.long, device=device)
                fut = pos[:, None] + 1 + torch.arange(model.feature_spec.horizon_steps, device=device)[None, :]
                raw = model.decode_horizon(
                    out[0, pos],
                    sctx,
                    stream.time_features.to(device)[fut],
                    stream.scenario_values.to(device)[fut],
                    torch.zeros_like(stream.scenario_mask.to(device)[fut]),
                )
                pred = stream.target.to(device)[pos].view(-1, 1, 1) + raw
                ablated = pred.detach().cpu().numpy()[:, :, qidx]
                rows.append(
                    {
                        "lag_block_min": f"{lo}_{hi}",
                        "lag_lo_min": lo,
                        "lag_hi_min": hi,
                        "mean_abs_delta_mgdl": float(np.mean(np.abs(base - ablated))),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Lag-block fallback also collected no windows")
    out = (
        df.groupby(["lag_block_min", "lag_lo_min", "lag_hi_min"], as_index=False)["mean_abs_delta_mgdl"]
        .mean()
        .sort_values("lag_lo_min")
    )
    total = out["mean_abs_delta_mgdl"].sum()
    out["frac_influence"] = out["mean_abs_delta_mgdl"] / total if total > 0 else 0.0
    path = out_dir / "tables/study2_temporal_lag_ablation_fallback.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return "lag_block_ablation_fallback", out, pd.DataFrame(), None


def build_failure_outputs(
    out_dir: Path,
    cache: pd.DataFrame,
    at: pd.DataFrame,
    test_fc: pd.DataFrame,
    tau_up: float,
    tau_down: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    h1 = cache[cache["horizon_step"].eq(1)].copy()
    h1["_pid_seg"] = h1["participant_id"].astype(str) + "__seg" + h1["segment_id"].astype(str)
    h1["participant_id_orig"] = h1["participant_id"].astype(str)
    h1["participant_id"] = h1["_pid_seg"]
    src = at[
        [
            "participant_id",
            "segment_id",
            "anchor_ds",
            "current_glucose",
            "slope_15",
            "curvature",
            "split",
        ]
    ].copy()
    src["_pid_seg"] = src["participant_id"].astype(str) + "__seg" + src["segment_id"].astype(str)
    src["participant_id"] = src["_pid_seg"]
    src = src.rename(
        columns={
            "anchor_ds": "ds",
            "current_glucose": "cgm_glucose",
            "slope_15": "cgm_slope_15",
            "curvature": "cgm_accel",
        }
    )
    cfg = Study2InterpretabilityConfig(tau_up=tau_up, tau_down=tau_down, bin_minutes=BIN_MIN)
    failure_map = build_failure_mode_map(h1, src, None, cfg)
    failure_map["participant_segment_id"] = failure_map["participant_id"]
    failure_map["participant_id"] = failure_map["participant_segment_id"].astype(str).str.replace(
        r"__seg.*$", "", regex=True
    )
    failure_map.to_parquet(out_dir / "study2_failure_mode_map.parquet", index=False)
    fm_summary = summarize_failure_mode_map(failure_map)
    fm_summary.to_csv(tables / "study2_failure_mode_summary.csv", index=False)

    if test_fc.empty:
        return fm_summary, pd.DataFrame()
    df = test_fc.copy()
    df["base_abs_error"] = (df["target"] - df["q50"]).abs()
    df["corrected_abs_error"] = (df["target"] - df["corrected_pred"]).abs()
    df["correction_gain_mgdl"] = df["base_abs_error"] - df["corrected_abs_error"]
    anchor = (
        df.groupby(["participant_id", "segment_id", "anchor_ds"], as_index=False)
        .agg(
            trigger_up=("trigger_up", "first"),
            trigger_down=("trigger_down", "first"),
            z_t=("z_t", "first"),
            one_step_residual=("one_step_residual", "first"),
            interval_width=("interval_width", "first"),
            slope_15=("slope_15", "first"),
            curvature=("curvature", "first"),
            current_glucose=("current_glucose", "first"),
            event_enriched=("event_enriched", "first"),
            participants_study_group=("participants_study_group", "first"),
            participants_clinical_site=("participants_clinical_site", "first"),
            med_any_diabetes_drug=("med_any_diabetes_drug", "first"),
            base_mae=("base_abs_error", "mean"),
            corrected_mae=("corrected_abs_error", "mean"),
            gain_mgdl=("correction_gain_mgdl", "mean"),
        )
        .copy()
    )
    anchor["trigger_direction"] = np.where(anchor["trigger_up"], "trigger_up", "trigger_down")
    anchor["abs_z_bin"] = pd.qcut(
        pd.to_numeric(anchor["z_t"], errors="coerce").abs(),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    ).astype(object)
    anchor["residual_bin"] = pd.cut(
        pd.to_numeric(anchor["one_step_residual"], errors="coerce"),
        bins=[-np.inf, -20, -10, 0, 10, 20, np.inf],
        labels=["lt_-20", "-20_-10", "-10_0", "0_10", "10_20", "gt_20"],
    ).astype(object)
    anchor["interval_width_bin"] = pd.qcut(
        pd.to_numeric(anchor["interval_width"], errors="coerce"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    ).astype(object)
    anchor["slope_bin"] = pd.cut(
        pd.to_numeric(anchor["slope_15"], errors="coerce"),
        bins=[-np.inf, -2, -1, -0.25, 0.25, 1, 2, np.inf],
        labels=["fast_down", "down", "mild_down", "flat", "mild_up", "up", "fast_up"],
    ).astype(object)
    anchor["curvature_bin"] = pd.qcut(
        pd.to_numeric(anchor["curvature"], errors="coerce"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    ).astype(object)
    anchor["medication_group"] = np.where(
        pd.to_numeric(anchor["med_any_diabetes_drug"], errors="coerce").fillna(0).eq(1),
        "non_insulin_medication_metadata",
        "no_recorded_diabetes_drug",
    )
    dims = OrderedDict(
        [
            ("z_t", "abs_z_bin"),
            ("residual", "residual_bin"),
            ("trigger_direction", "trigger_direction"),
            ("interval_width", "interval_width_bin"),
            ("slope", "slope_bin"),
            ("curvature", "curvature_bin"),
            ("study_group", "participants_study_group"),
            ("clinical_site", "participants_clinical_site"),
            ("medication_metadata", "medication_group"),
            ("event_aware_surprise", "event_enriched"),
        ]
    )
    rows = []
    for dim, col in dims.items():
        for level, g in anchor.groupby(col, dropna=False):
            rows.append(
                {
                    "dimension": dim,
                    "level": str(level),
                    "n_anchors": int(len(g)),
                    "n_participants": int(g["participant_id"].nunique()),
                    "base_mae": float(g["base_mae"].mean()),
                    "corrected_mae": float(g["corrected_mae"].mean()),
                    "correction_gain_mgdl": float(g["gain_mgdl"].mean()),
                    "mean_z_t": float(pd.to_numeric(g["z_t"], errors="coerce").mean()),
                    "mean_abs_z_t": float(pd.to_numeric(g["z_t"], errors="coerce").abs().mean()),
                    "mean_interval_width": float(pd.to_numeric(g["interval_width"], errors="coerce").mean()),
                    "trigger_up_fraction": float(g["trigger_up"].mean()),
                    "trigger_down_fraction": float(g["trigger_down"].mean()),
                }
            )
    failure_attr = pd.DataFrame(rows)
    failure_attr.to_csv(tables / "study2_failure_mode_attribution.csv", index=False)
    anchor.to_csv(tables / "study2_failure_mode_anchor_level.csv", index=False)
    return fm_summary, failure_attr


def fig_global_modality(global_df: pd.DataFrame, out_dir: Path, dpi: int) -> list[str]:
    if global_df.empty:
        return []
    df = global_df.sort_values("frac_influence", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [MODALITY_COLORS.get(str(m), "#777777") for m in df["modality"]]
    ax.barh(df["modality"].astype(str), 100 * df["frac_influence"], color=colors, edgecolor="white")
    ax.set_xlabel("% of model-output ablation influence")
    ax.set_title("Global modality attribution for the frozen Study 2 model")
    ax.grid(alpha=0.25, axis="x")
    for i, v in enumerate(100 * df["frac_influence"]):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8)
    return save_figure(fig, out_dir / "figures/study2_global_modality_attribution.png", dpi)


def fig_temporal_lag(method: str, group_df: pd.DataFrame, head_df: pd.DataFrame, out_dir: Path, dpi: int) -> list[str]:
    if group_df.empty:
        return []
    if method == "mes_hidden_attention":
        pivot = (
            group_df.pivot_table(index="modality", columns="lag_min", values="frac_influence", aggfunc="sum")
            .reindex(MODALITY_ORDER)
            .fillna(0.0)
        )
        keep = pivot.sum(axis=1).sort_values(ascending=False).head(8).index
        pivot = pivot.loc[keep]
        fig = plt.figure(figsize=(11.5, 5.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
        ax0 = fig.add_subplot(gs[0, 0])
        im = ax0.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        cols = list(pivot.columns)
        ticks = [i for i, lag in enumerate(cols) if lag in {0, 30, 60, 120, 240, 360, 720}]
        ax0.set_xticks(ticks)
        ax0.set_xticklabels([str(cols[i]) for i in ticks])
        ax0.set_yticks(np.arange(len(pivot.index)))
        ax0.set_yticklabels(pivot.index)
        ax0.set_xlabel("lag (min)")
        ax0.set_title("Feature-time attribution by modality")
        cb = fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.02)
        cb.set_label("fractional influence")
        ax1 = fig.add_subplot(gs[0, 1])
        if not head_df.empty:
            mean_lag = head_df.groupby("lag_min")["attention"].mean()
            ax1.plot(mean_lag.index, mean_lag.values, color="#222222", marker="o", ms=2)
            ax1.set_title("MES hidden attention over lags")
            ax1.set_xlabel("lag (min)")
            ax1.set_ylabel("mean attention")
            ax1.grid(alpha=0.25)
        fig.suptitle("Temporal lag attribution using MES hidden attention", y=1.02)
    else:
        df = group_df.sort_values("lag_lo_min")
        fig, ax0 = plt.subplots(figsize=(7.5, 4.0))
        ax0.bar(df["lag_block_min"], 100 * df["frac_influence"], color="#4c78a8")
        ax0.set_ylabel("% of lag-block ablation influence")
        ax0.set_xlabel("lag block (min)")
        ax0.set_title("Temporal lag attribution using lag-block ablation fallback")
        ax0.grid(alpha=0.25, axis="y")
    return save_figure(fig, out_dir / "figures/study2_temporal_lag_attribution.png", dpi)


def fig_failure(failure_attr: pd.DataFrame, out_dir: Path, dpi: int) -> list[str]:
    if failure_attr.empty:
        return []
    dims = ["trigger_direction", "z_t", "interval_width", "slope"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for ax, dim in zip(axes.ravel(), dims):
        sub = failure_attr[failure_attr["dimension"].eq(dim)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("level")
        ax.bar(np.arange(len(sub)), sub["base_mae"], color="#bbbbbb", label="base")
        ax.plot(np.arange(len(sub)), sub["corrected_mae"], color="#2166ac", marker="o", label="corrected")
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["level"].astype(str), rotation=35, ha="right")
        ax.set_ylabel("MAE (mg/dL)")
        ax.set_title(dim.replace("_", " "))
        ax.grid(alpha=0.25, axis="y")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Failure-mode attribution on triggered Study 2 anchors", y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_dir / "figures/study2_failure_mode_attribution.png", dpi)


def fig_correction_head(coef_df: pd.DataFrame, cmp_df: pd.DataFrame, out_dir: Path, dpi: int) -> list[str]:
    paths = []
    if not coef_df.empty:
        top = (
            coef_df.groupby("feature")["coef_standardized"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
        for ax, direction in zip(axes, ["trigger_up", "trigger_down"]):
            sub = coef_df[coef_df["direction"].eq(direction) & coef_df["feature"].isin(top)].copy()
            mat = sub.pivot_table(index="feature_label", columns="horizon_minutes", values="coef_standardized")
            mat = mat.reindex([pretty_feature(f) for f in top])
            vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if np.isfinite(mat.to_numpy()).any() else 1.0
            im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(direction.replace("_", " "))
            ax.set_xticks(np.arange(len(mat.columns)))
            ax.set_xticklabels([str(c) for c in mat.columns])
            ax.set_yticks(np.arange(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_xlabel("horizon (min)")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="standardized coefficient")
        fig.suptitle("Correction-head coefficients by horizon", y=1.02)
        paths.extend(save_figure(fig, out_dir / "figures/study2_correction_head_coefficients.png", dpi))
    if not cmp_df.empty:
        top_delta = (
            cmp_df.groupby("feature_label")["coef_standardized_delta_up_minus_down"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        vals = top_delta.sort_values()
        ax.barh(vals.index, vals.values, color="#756bb1")
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel("mean abs standardized coefficient difference")
        ax.set_title("Up-trigger versus down-trigger head contrast")
        ax.grid(alpha=0.25, axis="x")
        paths.extend(save_figure(fig, out_dir / "figures/study2_correction_head_up_down_comparison.png", dpi))
    return paths


def local_modality_series(ablation_detail: pd.DataFrame, row: dict) -> pd.Series:
    sub = ablation_detail[
        ablation_detail["participant_id"].astype(str).eq(str(row["participant_id"]))
        & ablation_detail["segment_id"].astype(int).eq(int(row["segment_id"]))
        & ablation_detail["anchor_ds"].astype(int).eq(int(row["anchor_ds"]))
    ].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    s = sub.set_index("modality")["mean_abs_delta_mgdl"].reindex(MODALITY_ORDER).fillna(0.0)
    total = float(s.sum())
    return s / total if total > 0 else s


def fig_local_episode(
    direction: str,
    rows: list[dict],
    at: pd.DataFrame,
    test_fc: pd.DataFrame,
    ablation_detail: pd.DataFrame,
    features: list[str],
    up_model,
    down_model,
    out_dir: Path,
    lead_steps: int,
    dpi: int,
) -> list[str]:
    if not rows:
        return []
    pid = str(rows[0]["participant_id"])
    seg = int(rows[0]["segment_id"])
    episodes = rows[:]
    sub_at = at[at["participant_id"].astype(str).eq(pid) & at["segment_id"].astype(int).eq(seg)].copy()
    sub_at = sub_at.sort_values("anchor_ds")
    n_ep = len(episodes)
    fig = plt.figure(figsize=(15.0, 2.3 + 3.1 * n_ep))
    gs = fig.add_gridspec(n_ep + 1, 3, height_ratios=[1.0] + [1.5] * n_ep, width_ratios=[2.5, 1.0, 1.0], hspace=0.55, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    xh = sub_at["anchor_ds"].to_numpy(dtype=float) * BIN_MIN / 60.0
    ax_top.plot(xh, sub_at["current_glucose"], color="#777777", lw=1.0)
    up = sub_at[sub_at["trigger_up"].fillna(False)]
    dn = sub_at[sub_at["trigger_down"].fillna(False)]
    ax_top.scatter(up["anchor_ds"] * BIN_MIN / 60.0, up["current_glucose"], color="#b44b4b", s=12, label="trigger_up")
    ax_top.scatter(dn["anchor_ds"] * BIN_MIN / 60.0, dn["current_glucose"], color="#2b7bba", s=12, label="trigger_down")
    ep_colors = ["#6a3d9a", "#c2185b", "#00838f", "#827717"]
    for i, ep in enumerate(episodes):
        anchor = int(ep["anchor_ds"])
        ax_top.axvspan((anchor - lead_steps) * BIN_MIN / 60.0, (anchor + HORIZON) * BIN_MIN / 60.0, color=ep_colors[i % len(ep_colors)], alpha=0.10)
        ax_top.axvline(anchor * BIN_MIN / 60.0, color=ep_colors[i % len(ep_colors)], lw=1.2)
        ax_top.text(anchor * BIN_MIN / 60.0, ax_top.get_ylim()[1], f"ep {i + 1}", ha="center", va="top", color=ep_colors[i % len(ep_colors)], fontweight="bold")
    ax_top.axhline(70, color="#b44b4b", lw=0.7, ls=":")
    ax_top.axhline(180, color="#ff7f00", lw=0.7, ls=":")
    ax_top.set_ylabel("glucose (mg/dL)")
    ax_top.set_xlabel("time (h)")
    ax_top.set_title(
        f"participant {pid}, segment {seg}: {len(up)} up-trigger / {len(dn)} down-trigger anchors; shaded windows shown below"
    )
    ax_top.legend(frameon=False, loc="upper right", ncol=2)
    ax_top.grid(alpha=0.25)

    for i, ep in enumerate(episodes):
        anchor = int(ep["anchor_ds"])
        ep_rows = test_fc[
            test_fc["participant_id"].astype(str).eq(pid)
            & test_fc["segment_id"].astype(int).eq(seg)
            & test_fc["anchor_ds"].astype(int).eq(anchor)
        ].sort_values("horizon_step")
        hist = sub_at[(sub_at["anchor_ds"] >= anchor - lead_steps + 1) & (sub_at["anchor_ds"] <= anchor)]
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.plot((hist["anchor_ds"] - anchor) * BIN_MIN, hist["current_glucose"], color="#6b7f8f", lw=1.4, label="glucose history")
        if not ep_rows.empty:
            xf = ep_rows["horizon_minutes"].to_numpy(dtype=float)
            ax.fill_between(xf, ep_rows["q10"], ep_rows["q90"], color="#d6604d", alpha=0.18, label="base 10-90 percent")
            ax.plot(xf, ep_rows["q50"], color="#d6604d", lw=1.4, ls="--", label="base forecast")
            ax.plot(xf, ep_rows["corrected_pred"], color="#1b9e77", lw=1.6, marker="o", ms=2.5, label="corrected forecast")
            ax.plot(xf, ep_rows["target"], color="#111111", lw=1.5, marker="o", ms=2.5, label="observed future")
        ax.axvline(0, color="#333333", lw=0.9)
        ax.axhline(70, color="#b44b4b", lw=0.7, ls=":")
        ax.axhline(180, color="#ff7f00", lw=0.7, ls=":")
        ax.set_xlabel("minutes relative to forecast time")
        ax.set_ylabel("glucose (mg/dL)")
        z = float(ep.get("z_t", np.nan))
        cur = float(ep.get("current_glucose", np.nan))
        if direction == "trigger_up":
            extreme = float(ep.get("future_max", np.nan))
            title = f"episode {i + 1}: current {cur:.0f} -> observed max {extreme:.0f} mg/dL, z_t={z:.2f}"
        else:
            extreme = float(ep.get("future_min", np.nan))
            title = f"episode {i + 1}: current {cur:.0f} -> observed min {extreme:.0f} mg/dL, z_t={z:.2f}"
        ax.set_title(title, color=ep_colors[i % len(ep_colors)], fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, fontsize=7, loc="upper left", ncol=2)

        axm = fig.add_subplot(gs[i + 1, 1])
        ms = local_modality_series(ablation_detail, ep).sort_values(ascending=True)
        ms = ms[ms > 0].tail(9)
        axm.barh(ms.index, 100 * ms.values, color=[MODALITY_COLORS.get(m, "#777777") for m in ms.index])
        axm.set_xlabel("% of local ablation influence")
        axm.set_title("what drove this forecast")
        axm.grid(alpha=0.25, axis="x")

        axc = fig.add_subplot(gs[i + 1, 2])
        cs = correction_contributions(ep_rows, features, up_model, down_model).head(9).sort_values()
        colors = ["#2b7bba" if v < 0 else "#b44b4b" for v in cs.values]
        axc.barh([pretty_feature(f) for f in cs.index], cs.values, color=colors)
        axc.axvline(0, color="#333333", lw=0.8)
        axc.set_xlabel("mg/dL contribution")
        axc.set_title("correction-head features")
        axc.grid(alpha=0.25, axis="x")

    fig.suptitle(f"Study 2 local {direction} triggered excursion case study", y=0.995)
    stem = f"study2_local_episode_{direction}_{pid}_seg{seg}"
    return save_figure(fig, out_dir / f"figures/individual/{stem}.png", dpi)


def write_reference_manifest(out_dir: Path, checkpoint: Path, args: argparse.Namespace, temporal_method: str, mes_path: Path | None):
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "cache_path": str(Path(args.cache_path)),
        "output_dir": str(out_dir),
        "labels_and_assumptions": {
            "verified_nutrition_event_timestamps": False,
            "scheduled_medication_action_covariates": False,
            "meal_proxy": "diagnostic only when the predmeal_flag channel is present",
            "medication_metadata": "participant metadata, not timed action",
            "control_language": "unmatched control only",
        },
        "temporal_lag_method": temporal_method,
        "mes_attention_path": str(mes_path) if mes_path is not None else None,
        "limits": {
            "max_streams": args.max_streams,
            "anchors_per_stream": args.anchors_per_stream,
            "max_mes_streams": args.max_mes_streams,
            "max_lags": args.max_lags,
        },
    }
    write_json(out_dir / "study2_interpretability_manifest.json", payload)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures/individual").mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.config))
    checkpoint = resolve_checkpoint(args, cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = resolve_device(args.device)
    print(f"[study2-interpret] cache={args.cache_path}")
    print(f"[study2-interpret] checkpoint={checkpoint}")
    print(f"[study2-interpret] device={device}")

    cache = load_cache(Path(args.cache_path))
    at, features, tau_up, tau_down, final_manifest = build_anchor_state(cache, Path(args.final_head_dir))
    splits_fc, usable_features, ridge_up, ridge_down = prepare_correction_head(cache, at, features)
    test_fc = splits_fc.get(args.eval_split, pd.DataFrame())

    coef_df, cmp_df = export_correction_coefficients(out_dir, usable_features, ridge_up, ridge_down)
    local = select_local_episodes(test_fc, args.episodes_per_figure)
    pids = desired_participants(cache, local, args.eval_split, args.max_streams)

    model, spec, pre, ckpt = load_model(checkpoint, device)
    groups = modality_columns(spec)
    streams = load_stream_subset(cfg, spec, pre, pids, args.eval_split)
    for s in streams:
        s.feature_spec = spec
    anchors_by_stream = build_anchor_selection(
        cache,
        streams,
        args.eval_split,
        args.anchors_per_stream,
        local,
    )

    global_df, ablation_detail = compute_ablation_attribution(model, streams, anchors_by_stream, groups, device)
    global_df.to_csv(out_dir / "tables/study2_global_modality_attribution.csv", index=False)
    ablation_detail.to_csv(out_dir / "tables/study2_local_modality_ablation_detail.csv", index=False)

    try:
        temporal_method, lag_df, head_df, mes_path = compute_mes_temporal(
            model,
            streams,
            anchors_by_stream,
            groups,
            device,
            out_dir,
            max_lags=args.max_lags,
            max_streams=args.max_mes_streams,
        )
        print("[study2-interpret] temporal lag method: MES hidden attention")
    except Exception as exc:
        print(f"[study2-interpret] MES attribution failed; falling back to lag-block ablation: {exc}")
        temporal_method, lag_df, head_df, mes_path = lag_block_fallback(
            model,
            streams,
            anchors_by_stream,
            device,
            out_dir,
        )

    fm_summary, failure_attr = build_failure_outputs(
        out_dir,
        cache,
        at,
        test_fc,
        tau_up,
        tau_down,
    )

    figure_paths = []
    figure_paths.extend(fig_global_modality(global_df, out_dir, args.dpi))
    figure_paths.extend(fig_temporal_lag(temporal_method, lag_df, head_df, out_dir, args.dpi))
    figure_paths.extend(fig_failure(failure_attr, out_dir, args.dpi))
    figure_paths.extend(fig_correction_head(coef_df, cmp_df, out_dir, args.dpi))
    lead_steps = int(round(args.lead_hours * 60.0 / BIN_MIN))
    for direction, rows in local.items():
        figure_paths.extend(
            fig_local_episode(
                direction,
                rows,
                at,
                test_fc,
                ablation_detail,
                usable_features,
                ridge_up,
                ridge_down,
                out_dir,
                lead_steps,
                args.dpi,
            )
        )

    write_reference_manifest(out_dir, checkpoint, args, temporal_method, mes_path)
    write_json(
        out_dir / "study2_interpretability_run_summary.json",
        {
            "n_cache_rows": int(len(cache)),
            "n_anchor_rows": int(len(at)),
            "n_test_triggered_forecast_rows": int(len(test_fc)),
            "n_streams_loaded": int(len(streams)),
            "n_attribution_anchors": int(sum(len(v) for v in anchors_by_stream.values())),
            "tau_up": tau_up,
            "tau_down": tau_down,
            "final_head": final_manifest.get("final_head", "ridge_split_dir"),
            "usable_correction_features": usable_features,
            "temporal_lag_method": temporal_method,
            "figures": figure_paths,
        },
    )
    print(f"[study2-interpret] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
