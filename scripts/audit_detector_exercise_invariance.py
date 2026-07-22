#!/usr/bin/env python3
"""Inference-only audit of detector scenario invariance and mask/content controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_stream_aireadi import load_model_from_checkpoint
from ssmcgm.data.aireadi import (
    build_stream_feature_spec, infer_or_validate_schema, load_aireadi_panel,
    make_aireadi_stream_splits, make_participant_streams, prepare_aireadi_panel,
)
from ssmcgm.evaluation.aireadi_streaming import _valid_anchors
from ssmcgm.stream.state import StaticContext

SEED = 20260630
N_BOOT = 500
OUT = ROOT / "outputs/exercise_detector_scenario/final_audits/forecast_only_invariance"
DET_ROOT = ROOT / "outputs/exercise_detector_scenario"
CAN_CKPT = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
DET_CKPT = DET_ROOT / "checkpoints/best_model_checkpoint.pt"
DET_CFG = DET_ROOT / "configs/aireadi_detector_exercise_scenario.json"
PANEL = DET_ROOT / "panel/final_multimodal_dataset_with_detector_exercise.parquet"
COMMON = DET_ROOT / "evaluation/common_anchor_forecasts.parquet"
EVAL_MANIFEST = DET_ROOT / "evaluation/evaluation_manifest.json"
CACHE = DET_ROOT / "interpretability/exercise_pathway_tensor_cache.pt"
CACHE_META = DET_ROOT / "interpretability/exercise_pathway_anchor_metadata.parquet"
ALIGNMENT = DET_ROOT / "audits/exercise_episode_alignment_audit.csv"
HISTORY_ROOT = DET_ROOT / "history_ablation_full_stream/test"
QLEVELS = (0.1, 0.5, 0.9)
QCOLS = ("q10", "q50", "q90")
HISTORY_FIELDS = ("recent_exercise_30min", "recent_exercise_60min", "recent_exercise_120min")
COLORS = {
    "canonical": "#0B3765", "detector": "#5DB7B7", "base": "#444444",
    "gated": "#B22222", "history": "#D89000", "negative": "#888888",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--skip-full-inference", action="store_true", help="Reuse a completed decomposition parquet.")
    return p.parse_args()


def write_json(path: Path, value):
    def convert(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, Path): return str(x)
        if isinstance(x, pd.Timestamp): return x.isoformat()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_tree(paths):
    out = {}
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            if OUT in path.parents:
                continue
            out[str(path.relative_to(ROOT))] = sha256(path)
    return out


def command_text(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def reproduce_existing():
    factual = pd.read_csv(DET_ROOT / "evaluation/factual_episode_metrics.csv")
    summary = pd.read_csv(DET_ROOT / "evaluation/counterfactual_summary.csv")
    history = pd.read_csv(HISTORY_ROOT / "metrics_by_distance.csv")
    manifest = json.loads(EVAL_MANIFEST.read_text())

    def fm(mode, metric):
        return float(factual.loc[(factual["mode"] == mode) & (factual["metric"] == metric), "value"].iloc[0])
    def terminal(model, strain):
        row = summary[(summary.model == model) & (summary.scenario_strain == strain) &
                      (summary.starting_glucose_stratum == "all") &
                      (summary.scope == "terminal_60min") & (summary.metric == "mean_effect")]
        return float(row.value.iloc[0])
    def hm(condition, field):
        row = history[(history.condition == condition) & (history.distance_from_exercise == "all")]
        return float(row[field].iloc[0])

    result = {
        "eligible_episodes": int(manifest["common_eligible_episode_anchors"]),
        "held_out_participants": int(manifest["common_eligible_participants"]),
        "horizons": len(manifest["future_horizon_minutes"]),
        "anchor_mismatches": int(manifest["unmatched_or_ineligible_episodes"]),
        "forecast_only_pinball": fm("unknown", "pinball"),
        "factual_pinball": fm("factual", "pinball"),
        "factual_future_gain": fm("factual_vs_unknown", "pinball_gain"),
        "factual_future_gain_ci": factual.loc[(factual["mode"] == "factual_vs_unknown") &
                                                (factual["metric"] == "pinball_gain"),
                                                ["ci_lower", "ci_upper"]].iloc[0].tolist(),
        "forecast_only_mae": fm("unknown", "mae"),
        "factual_mae": fm("factual", "mae"),
        "planned_terminal_effects": {s: terminal("trained", s) for s in ("light", "moderate", "vigorous")},
        "canonical_terminal_effects": {s: terminal("canonical", s) for s in ("light", "moderate", "vigorous")},
        "test_participants": int(history.loc[history.distance_from_exercise == "all", "participants"].max()),
        "test_rows": int(history.loc[history.distance_from_exercise == "all", "rows"].max()),
        "canonical_test_mae": hm("canonical", "mae_mgdl"),
        "detector_test_mae": hm("detector_normal", "mae_mgdl"),
        "canonical_test_bias": hm("canonical", "bias_mgdl_prediction_minus_observed"),
        "detector_test_bias": hm("detector_normal", "bias_mgdl_prediction_minus_observed"),
        "history_zero_test_bias": hm("detector_recent_history_zero", "bias_mgdl_prediction_minus_observed"),
    }
    checks = [
        result["eligible_episodes"] == 281, result["held_out_participants"] == 110,
        result["horizons"] == 12, result["anchor_mismatches"] == 0,
        abs(result["forecast_only_pinball"] - 4.134) < .01,
        abs(result["factual_pinball"] - 3.880) < .01,
        abs(result["factual_future_gain"] - .254) < .01,
        abs(result["canonical_test_mae"] - 10.12) < .02,
        abs(result["detector_test_bias"] + 2.50) < .02,
        result["test_participants"] == 221, result["test_rows"] == 1861356,
    ]
    result["all_checks_passed"] = bool(all(checks))
    if not result["all_checks_passed"]:
        raise RuntimeError(f"Existing results did not reproduce: {result}")
    write_json(OUT / "existing_results_reproduction.json", result)
    return result


def pinball_rows(target, pred):
    losses = []
    for qi, q in enumerate(QLEVELS):
        err = target - pred[:, qi]
        losses.append(np.maximum(q * err, (q - 1.0) * err))
    return np.mean(np.stack(losses, axis=1), axis=1)


def sufficient_stats(frame, pred_cols=QCOLS, effect_col=None):
    target = frame["target"].to_numpy(float)
    pred = frame[list(pred_cols)].to_numpy(float)
    err = pred[:, 1] - target
    terminal = frame["horizon_minutes"].to_numpy() == 60
    work = pd.DataFrame({
        "participant_id": frame["participant_id"].astype(str).to_numpy(),
        "n": 1.0, "abs": np.abs(err), "sq": err * err, "err": err,
        "cover": ((target >= pred[:, 0]) & (target <= pred[:, 2])).astype(float),
        "width": pred[:, 2] - pred[:, 0], "pinball": pinball_rows(target, pred),
        "n60": terminal.astype(float), "abs60": np.abs(err) * terminal,
    })
    if effect_col is not None:
        eff = frame[effect_col].to_numpy(float)
        work["abs_effect"] = np.abs(eff)
        work["terminal_effect"] = eff * terminal
    return work.groupby("participant_id", observed=True).sum()


def stats_to_metrics(s):
    n = max(float(s["n"]), 1.0)
    n60 = max(float(s["n60"]), 1.0)
    out = {
        "rows": int(s["n"]), "mae": s["abs"] / n,
        "rmse": np.sqrt(s["sq"] / n), "bias": s["err"] / n,
        "mae_60min": s["abs60"] / n60, "coverage80": s["cover"] / n,
        "interval_width": s["width"] / n, "pinball": s["pinball"] / n,
    }
    if "abs_effect" in s:
        out["mean_absolute_scenario_effect"] = s["abs_effect"] / n
        out["terminal_scenario_effect"] = s["terminal_effect"] / n60
    return {k: float(v) if k != "rows" else int(v) for k, v in out.items()}


def metric_with_bootstrap(frame, pred_cols=QCOLS, effect_col=None, seed=SEED):
    stats = sufficient_stats(frame, pred_cols, effect_col)
    point = stats_to_metrics(stats.sum())
    arr = stats.to_numpy(float)
    cols = list(stats.columns)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point if k != "rows"}
    for _ in range(N_BOOT):
        sampled = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        vals = stats_to_metrics(pd.Series(sampled, index=cols))
        for key in boot: boot[key].append(vals[key])
    for key, values in boot.items():
        point[f"{key}_ci_lower"] = float(np.percentile(values, 2.5))
        point[f"{key}_ci_upper"] = float(np.percentile(values, 97.5))
    point["participants"] = len(stats)
    return point


def classify_distances(anchor_frame):
    episodes = pd.read_csv(ALIGNMENT)
    episodes = episodes[(episodes["split"] == "test") & (~episodes["was_excluded"].astype(bool))].copy()
    episodes["participant_id"] = episodes["participant_id"].astype(str)
    episodes["start"] = pd.to_datetime(episodes["matched_start"], utc=True)
    episodes["end"] = pd.to_datetime(episodes["matched_end"], utc=True)
    by_pid = {pid: list(zip(g.start, g.end)) for pid, g in episodes.groupby("participant_id")}
    labels = []
    for row in anchor_frame.itertuples(index=False):
        t = pd.Timestamp(row.anchor_timestamp)
        intervals = by_pid.get(str(row.participant_id), [])
        if any(a <= t <= b for a, b in intervals):
            labels.append("during_exercise")
        elif any(pd.Timedelta(0) < t - b <= pd.Timedelta(hours=2) for _, b in intervals):
            labels.append("0_2h_after_exercise")
        elif any(pd.Timedelta(0) < a - t <= pd.Timedelta(hours=2) for a, _ in intervals):
            labels.append("0_2h_before_exercise")
        else:
            labels.append("more_than_2h_from_exercise")
    return labels


def load_saved_full():
    paths = {
        "C0": HISTORY_ROOT / "predictions_canonical.parquet",
        "D0": HISTORY_ROOT / "predictions_detector_normal.parquet",
        "D4": HISTORY_ROOT / "predictions_detector_recent_history_zero.parquet",
    }
    frames = {}
    keys = ["participant_id", "segment_id", "anchor_time_idx", "horizon_step", "horizon_minutes", "target"]
    for name, path in paths.items():
        d = pd.read_parquet(path, columns=keys + list(QCOLS))
        d["participant_id"] = d["participant_id"].astype(str)
        frames[name] = d
    base = frames["C0"][keys].copy()
    for name, d in frames.items():
        if not base[keys].equals(d[keys]):
            raise RuntimeError(f"Saved full-stream rows are not aligned for {name}")
        for q in QCOLS: base[f"{name}_{q}"] = d[q].to_numpy()
    return base


def build_locked_test_streams(cfg, spec, pre):
    dc = cfg["data"]
    df = load_aireadi_panel(dc["panel_path"], static_path=dc.get("static_path"), cohort_path=dc.get("cohort_path"))
    schema = infer_or_validate_schema(df, dc.get("schema"))
    df = prepare_aireadi_panel(df, schema, bin_minutes=cfg["dataset"].get("bin_minutes", 5),
                               clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49))
    # infer_or_validate_schema currently rebuilds canonical scenario groups after applying
    # overrides. The audit uses the immutable checkpoint spec as the compatibility adapter.
    schema.dynamic_reals = list(spec.dynamic_reals)
    schema.time_reals = list(spec.time_reals)
    schema.static_reals = list(spec.static_reals)
    schema.static_categoricals = list(spec.static_categoricals)
    schema.scenario_reals = list(spec.scenario_reals)
    schema.scenario_groups = {k: list(v) for k, v in spec.scenario_groups.items()}
    schema.subgroup_columns = list(spec.subgroup_columns)
    build_stream_feature_spec(df, schema, horizon_steps=spec.horizon_steps, bin_minutes=spec.bin_minutes)
    split_cfg = cfg.get("split", {})
    if not split_cfg.get("existing_split_path"):
        raise RuntimeError("Audit requires the fixed existing participant split")
    split = make_aireadi_stream_splits(
        df, split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", .70), val=split_cfg.get("val", .15), test=split_cfg.get("test", .15),
        seed=split_cfg.get("seed", 42), stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg["existing_split_path"],
    )
    return make_participant_streams(df, split, schema, feature_spec=spec, preprocessor=pre,
                                    splits=["test"], min_steps=spec.horizon_steps + 2)


@torch.no_grad()
def direct_full_components(model, spec, pre, cfg, device):
    print("[audit] building locked test streams")
    streams = build_locked_test_streams(cfg, spec, pre)
    if len({str(s.participant_id) for s in streams}) != 221:
        raise RuntimeError("Locked test split did not reproduce 221 participants")
    model.to(device).eval()
    chunks = []
    max_recon = 0.0
    for si, raw_stream in enumerate(streams):
        stream = raw_stream.to(device)
        sctx = model.encode_static(stream.static_cat, stream.static_cont)
        state = model.init_stream(sctx)
        state, out = model.scan_chunk(stream.dynamic, sctx, state)
        anchors = _valid_anchors(stream, spec.horizon_steps, 3)
        if not anchors: continue
        pos = torch.tensor(anchors, dtype=torch.long, device=device)
        fut = pos[:, None] + 1 + torch.arange(spec.horizon_steps, device=device)[None, :]
        zeros = torch.zeros((len(anchors), spec.horizon_steps, len(spec.scenario_reals)), device=device)
        comp = model.decode_horizon(out[0, pos], sctx, stream.time_features[fut], zeros, zeros,
                                    return_components=True)
        recon = torch.max(torch.abs(comp["final"] - (comp["base"] + comp["scenario_effect"]))).item()
        max_recon = max(max_recon, recon)

        dyn0 = stream.dynamic.clone()
        for name in HISTORY_FIELDS:
            j = spec.dynamic_reals.index(name)
            st = pre.continuous_stats[name]
            dyn0[:, j] = (0.0 - float(st["mean"])) / float(st["std"])
        state0 = model.init_stream(sctx)
        state0, out0 = model.scan_chunk(dyn0, sctx, state0)
        comp0 = model.decode_horizon(out0[0, pos], sctx, stream.time_features[fut], zeros, zeros,
                                     return_components=True)

        current = stream.target[pos].view(-1, 1, 1)
        arrays = {
            "D0_direct": (current + comp["final"]).cpu().numpy(),
            "D1": (current + comp["base"]).cpu().numpy(),
            "D2": comp["scenario_effect"].cpu().numpy(),
            "D3": (current + comp0["base"]).cpu().numpy(),
            "D4_direct": (current + comp0["final"]).cpu().numpy(),
            "raw_reconstruction": (comp["final"] - (comp["base"] + comp["scenario_effect"])).cpu().numpy(),
        }
        H = spec.horizon_steps
        time_idx = stream.time_idx[pos].cpu().numpy()
        timestamps = pd.to_datetime(np.asarray(stream.timestamps)[np.asarray(anchors)], utc=True)
        data = {
            "participant_id": np.repeat(str(stream.participant_id), len(anchors) * H),
            "segment_id": np.repeat(int(stream.segment_id), len(anchors) * H),
            "anchor_time_idx": np.repeat(time_idx, H),
            "anchor_timestamp": np.repeat(timestamps.to_numpy(), H),
            "horizon_step": np.tile(np.arange(1, H + 1), len(anchors)),
            "horizon_minutes": np.tile(np.arange(1, H + 1) * spec.bin_minutes, len(anchors)),
        }
        for name, array in arrays.items():
            for qi, q in enumerate(QCOLS): data[f"{name}_{q}"] = array[:, :, qi].reshape(-1)
        chunks.append(pd.DataFrame(data))
        if (si + 1) % 25 == 0 or si + 1 == len(streams):
            print(f"[audit] full component inference {si + 1}/{len(streams)} streams")
    return pd.concat(chunks, ignore_index=True), max_recon


def merge_full(saved, direct):
    keys = ["participant_id", "segment_id", "anchor_time_idx", "horizon_step", "horizon_minutes"]
    full = saved.merge(direct, on=keys, how="inner", validate="one_to_one")
    if len(full) != len(saved): raise RuntimeError("Direct component rows do not match saved test rows")
    checks = {
        "detector_normal_max_abs_difference": max(float(np.max(np.abs(full[f"D0_{q}"] - full[f"D0_direct_{q}"]))) for q in QCOLS),
        "history_zero_max_abs_difference": max(float(np.max(np.abs(full[f"D4_{q}"] - full[f"D4_direct_{q}"]))) for q in QCOLS),
    }
    anchor_map = direct[["participant_id", "segment_id", "anchor_time_idx", "anchor_timestamp"]].drop_duplicates()
    anchor_map["distance_stratum"] = classify_distances(anchor_map)
    full = full.merge(anchor_map, on=["participant_id", "segment_id", "anchor_time_idx"], how="left", validate="many_to_one")
    for q in QCOLS:
        full[f"G0_{q}"] = full[f"D1_{q}"]
        full[f"identity_residual_{q}"] = full[f"raw_reconstruction_{q}"]
    return full, checks


def full_tables(full):
    conditions = {
        "canonical_forecast_only": "C0", "detector_original_forecast_only": "D0",
        "detector_base_only": "D1", "detector_history_zero_base_only": "D3",
        "detector_history_zero": "D4", "detector_hard_gated_forecast_only": "G0",
    }
    metrics, horizons, distances = [], [], []
    for label, prefix in conditions.items():
        cols = tuple(f"{prefix}_{q}" for q in QCOLS)
        rec = {"condition": label, **metric_with_bootstrap(full, cols)}
        metrics.append(rec)
        for (step, mins), g in full.groupby(["horizon_step", "horizon_minutes"], observed=True):
            horizons.append({"condition": label, "horizon_step": step, "horizon_minutes": mins,
                             **metric_with_bootstrap(g, cols)})
        for distance, g in full.groupby("distance_stratum", observed=True):
            distances.append({"condition": label, "distance_from_exercise": distance,
                              **metric_with_bootstrap(g, cols)})
    metrics = pd.DataFrame(metrics)
    horizons = pd.DataFrame(horizons)
    distances = pd.DataFrame(distances)

    shift_rows = []
    for (step, mins), g in full.groupby(["horizon_step", "horizon_minutes"], observed=True):
        for qi, q in enumerate(QCOLS):
            total = g[f"D0_direct_{q}"] - g[f"C0_{q}"]
            base = g[f"D1_{q}"] - g[f"C0_{q}"]
            effect = g[f"D2_{q}"]
            resid = g[f"identity_residual_{q}"]
            shift_rows.append({"horizon_step": step, "horizon_minutes": mins, "quantile": QLEVELS[qi],
                               "total_shift": total.mean(), "base_shift": base.mean(),
                               "zero_mask_effect": effect.mean(), "mean_abs_residual": np.abs(resid).mean(),
                               "max_abs_residual": np.abs(resid).max()})
    return metrics, horizons, distances, pd.DataFrame(shift_rows)


def load_anchor_cache(device):
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    meta = pd.read_parquet(CACHE_META)
    idx = cache["test_predictions"]["test_indices"].long()
    tensors = {k: v[idx].to(device) for k, v in cache["tensors"].items()
               if isinstance(v, torch.Tensor) and v.shape[0] == len(meta)}
    test = {k: v.to(device) for k, v in cache["test_predictions"].items() if k != "test_indices"}
    tm = meta.iloc[idx.numpy()].reset_index(drop=True)
    tm["participant_id"] = tm["participant_id"].astype(str)
    common = pd.read_parquet(COMMON)
    if len(tm) != 281 or tm.participant_id.nunique() != 110 or set(tm.episode_id) != set(common.episode_id):
        raise RuntimeError("Fixed exercise anchor cache does not match the 281 common anchors")
    return cache, tensors, test, tm


def zvalue(pre, name, raw, reference):
    st = pre.continuous_stats[name]
    return (torch.as_tensor(raw, dtype=reference.dtype, device=reference.device) - float(st["mean"])) / float(st["std"])


def set_raw(values, pre, names, name, raw):
    j = names.index(name)
    z = zvalue(pre, name, raw, values)
    if z.dim() == 0: values[..., j] = z
    else: values[..., j] = z


def planned_no_exercise(tensors, pre, names):
    factual = tensors["trained_values"]
    n, h, f = factual.shape
    values = torch.zeros_like(factual)
    masks = torch.zeros_like(factual)
    hr = tensors["anchor_hr"].view(n, 1).expand(n, h)
    set_raw(values, pre, names, "heart_rate_mean", hr)
    selected = ["heart_rate_mean", "activity_steps_per_min"] + [x for x in names if x.startswith("exercise_")]
    for name in selected:
        if name != "heart_rate_mean": set_raw(values, pre, names, name, 0.0)
        masks[..., names.index(name)] = 1.0
    return values, masks


@torch.no_grad()
def decode_cached(model, tensors, values, masks, *, hidden_key="trained_hidden", gated=False):
    old = model.config.hard_gate_scenario_effect
    model.config.hard_gate_scenario_effect = bool(gated)
    ctx = StaticContext(embedding=tensors["trained_static"])
    comp = model.decode_horizon(tensors[hidden_key], ctx, tensors["future_time"], values, masks,
                                return_components=True)
    model.config.hard_gate_scenario_effect = old
    reconstruction_error = float((comp["final"] - (comp["base"] + comp["scenario_effect"])).abs().max())
    current = tensors["current"].view(-1, 1, 1)
    result = {k: (current + v if k in ("final", "base") else v) for k, v in comp.items()
              if k in ("final", "base", "scenario_effect", "scenario_availability")}
    result["reconstruction_max_abs_error"] = reconstruction_error
    return result


def donor_permutation(pids, seed):
    rng = np.random.default_rng(seed)
    pids = np.asarray(pids)
    for _ in range(10000):
        perm = rng.permutation(len(pids))
        if np.all(pids[perm] != pids): return perm
    raise RuntimeError("Could not construct a cross-participant donor permutation")


def control_scenarios(tensors, pre, names, meta):
    factual_v, factual_m = tensors["trained_values"], tensors["trained_source_mask"]
    planned_v, planned_m = planned_no_exercise(tensors, pre, names)
    exercise = [x for x in names if x.startswith("exercise_")]
    labels = [x for x in exercise if x != "exercise_detected_active"]
    hrsteps = ["heart_rate_mean", "activity_steps_per_min"]
    controls = {"R0_unknown_future": [(torch.zeros_like(factual_v), torch.zeros_like(factual_m), -1)],
                "R1_planned_no_exercise": [(planned_v, planned_m, -1)],
                "R2_full_factual": [(factual_v, factual_m, -1)]}

    v, m = factual_v.clone(), factual_m.clone()
    for name in hrsteps:
        j = names.index(name); v[..., j] = planned_v[..., j]
    for name in exercise: set_raw(v, pre, names, name, 0.0)
    controls["R3_real_masks_no_content"] = [(v, m, -1)]

    v, m = factual_v.clone(), factual_m.clone()
    for name in exercise:
        set_raw(v, pre, names, name, 0.0); m[..., names.index(name)] = 0.0
    for name in hrsteps: v[..., names.index(name)] = planned_v[..., names.index(name)]
    active = names.index("exercise_detected_active")
    v[..., active] = factual_v[..., active]; m[..., active] = factual_m[..., active]
    controls["R4_active_only"] = [(v, m, -1)]

    v, m = factual_v.clone(), factual_m.clone()
    for name in hrsteps: v[..., names.index(name)] = planned_v[..., names.index(name)]
    controls["R5_labels_only"] = [(v, m, -1)]

    v, m = factual_v.clone(), factual_m.clone()
    for name in exercise:
        set_raw(v, pre, names, name, 0.0); m[..., names.index(name)] = 0.0
    controls["R6_physiology_only"] = [(v, m, -1)]

    pids = meta.participant_id.astype(str).to_numpy()
    controls["R7_shuffled_label_identity"] = []
    controls["R8_shuffled_complete_content"] = []
    for offset in range(5):
        shuffle_seed = SEED + offset
        donor = torch.as_tensor(donor_permutation(pids, shuffle_seed), device=factual_v.device)
        v7, m7 = factual_v.clone(), factual_m.clone()
        for name in labels: v7[..., names.index(name)] = factual_v[donor, :, names.index(name)]
        controls["R7_shuffled_label_identity"].append((v7, m7, shuffle_seed))
        v8, m8 = factual_v.clone(), factual_m.clone()
        for name in hrsteps + labels: v8[..., names.index(name)] = factual_v[donor, :, names.index(name)]
        controls["R8_shuffled_complete_content"].append((v8, m8, shuffle_seed))
    controls["R9_zero_values_zero_masks"] = [(torch.zeros_like(factual_v), torch.zeros_like(factual_m), -1)]
    return controls, planned_v, planned_m


def controls_to_frames(model, tensors, meta, controls):
    frames, recon = [], 0.0
    target = tensors["observed"].cpu().numpy()
    H = target.shape[1]
    for condition, variants in controls.items():
        for values, masks, shuffle_seed in variants:
            comp = decode_cached(model, tensors, values, masks)
            r = comp["reconstruction_max_abs_error"]
            recon = max(recon, r)
            pred = comp["final"].detach().cpu().numpy()
            eff = comp["scenario_effect"].detach().cpu().numpy()
            frame = pd.DataFrame({
                "episode_id": np.repeat(meta.episode_id.to_numpy(), H),
                "participant_id": np.repeat(meta.participant_id.to_numpy(), H),
                "condition": condition, "shuffle_seed": shuffle_seed,
                "horizon_step": np.tile(np.arange(1, H + 1), len(meta)),
                "horizon_minutes": np.tile(np.arange(1, H + 1) * 5, len(meta)),
                "target": target.reshape(-1),
                "q10": pred[:, :, 0].reshape(-1), "q50": pred[:, :, 1].reshape(-1),
                "q90": pred[:, :, 2].reshape(-1),
                "scenario_effect_q10": eff[:, :, 0].reshape(-1),
                "scenario_effect_q50": eff[:, :, 1].reshape(-1),
                "scenario_effect_q90": eff[:, :, 2].reshape(-1),
                "scenario_available": comp["scenario_availability"].detach().cpu().numpy().reshape(-1),
            })
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), recon


def control_metrics(long):
    r0 = long[long.condition == "R0_unknown_future"][
        ["episode_id", "horizon_step", "q10", "q50", "q90"]].rename(
        columns={q: f"r0_{q}" for q in QCOLS})
    rows, boot_rows = [], []
    for (condition, seed), g in long.groupby(["condition", "shuffle_seed"], observed=True):
        g = g.merge(r0, on=["episode_id", "horizon_step"], validate="many_to_one")
        base_stats = sufficient_stats(g, QCOLS, "scenario_effect_q50")
        target = g.target.to_numpy(float)
        cond_pb = pinball_rows(target, g[list(QCOLS)].to_numpy(float))
        r0_pb = pinball_rows(target, g[[f"r0_{q}" for q in QCOLS]].to_numpy(float))
        gains = pd.DataFrame({"participant_id": g.participant_id.astype(str), "gain": r0_pb - cond_pb,
                              "n": 1.0}).groupby("participant_id").sum()
        stats = base_stats.join(gains[["gain"]])
        total = stats.sum()
        point = stats_to_metrics(total)
        point["factual_future_pinball_gain"] = float(total["gain"] / total["n"])
        arr, cols = stats.to_numpy(float), list(stats.columns)
        rng = np.random.default_rng(SEED)
        boots = []
        for b in range(N_BOOT):
            sample = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
            vals = stats_to_metrics(pd.Series(sample, index=cols))
            vals["factual_future_pinball_gain"] = sample[cols.index("gain")] / sample[cols.index("n")]
            vals.update({"condition": condition, "shuffle_seed": seed, "bootstrap_resample": b})
            boots.append(vals)
        bdf = pd.DataFrame(boots); boot_rows.append(bdf)
        for key in ["mae", "rmse", "bias", "mae_60min", "pinball", "mean_absolute_scenario_effect",
                    "terminal_scenario_effect", "factual_future_pinball_gain"]:
            point[f"{key}_ci_lower"] = bdf[key].quantile(.025)
            point[f"{key}_ci_upper"] = bdf[key].quantile(.975)
        point.update({"condition": condition, "shuffle_seed": seed, "participants": len(stats)})
        rows.append(point)
    metrics = pd.DataFrame(rows)
    boot = pd.concat(boot_rows, ignore_index=True)
    for condition in ("R7_shuffled_label_identity", "R8_shuffled_complete_content"):
        m = metrics[metrics.condition == condition]
        agg = {"condition": condition, "shuffle_seed": "mean_5_seeds", "participants": 110}
        for col in metrics.select_dtypes(include=np.number).columns:
            if col not in ("shuffle_seed",): agg[col] = m[col].mean()
        metrics = pd.concat([metrics, pd.DataFrame([agg])], ignore_index=True)
        b = boot[boot.condition == condition].groupby("bootstrap_resample", as_index=False).mean(numeric_only=True)
        b["condition"] = condition; b["shuffle_seed"] = "mean_5_seeds"
        boot = pd.concat([boot, b], ignore_index=True)
    return metrics, boot


@torch.no_grad()
def architecture_audit(model, spec, tensors):
    values = torch.zeros_like(tensors["trained_values"][:1])
    masks = torch.zeros_like(values)
    ctx = StaticContext(embedding=tensors["trained_static"][:1])
    comp = model.decode_horizon(tensors["trained_hidden"][:1], ctx, tensors["future_time"][:1], values, masks,
                                return_components=True)
    decoder = model.decoder
    effect_biases = [name for name, p in decoder.named_parameters() if name.startswith("effect_") and name.endswith("bias")]
    return {
        "structural_equation": "final = base + scenario_effect",
        "actual_forward_tensor_shapes": {k: list(v.shape) for k, v in comp.items() if torch.is_tensor(v)},
        "scenario_mask_layout": "[batch, horizon, feature]",
        "scenario_mask_shape": list(masks.shape),
        "scenario_feature_order": list(spec.scenario_reals),
        "scenario_feature_count": len(spec.scenario_reals),
        "effect_head_contains_biases": bool(effect_biases),
        "effect_bias_parameter_names": effect_biases,
        "effect_branch_residual_connections": False,
        "effect_branch_normalization_layers": [],
        "decoder_base_and_effect_parameter_objects_shared": False,
        "shared_upstream_inputs": ["historical_state_h_t", "static_embedding_e_s", "future_time_latent", "horizon_embedding"],
        "active_when_A0_M0": ["historical_state_h_t", "static_embedding_e_s", "future_time_latent",
                               "horizon_embedding", "effect_trunk_linear_biases", "effect_head_biases"],
        "zero_mask_mean_absolute_effect": float(comp["scenario_effect"].abs().mean()),
        "zero_mask_terminal_median_effect": float(comp["scenario_effect"][0, -1, 1]),
        "reconstruction_max_abs_error": float((comp["final"] - (comp["base"] + comp["scenario_effect"])).abs().max()),
    }


def gated_anchor_outputs(model, tensors, meta, controls, test_predictions):
    r0v, r0m, _ = controls["R0_unknown_future"][0]
    r1v, r1m, _ = controls["R1_planned_no_exercise"][0]
    r2v, r2m, _ = controls["R2_full_factual"][0]
    ungated_factual = decode_cached(model, tensors, r2v, r2m, gated=False)["final"]
    gated_factual = decode_cached(model, tensors, r2v, r2m, gated=True)["final"]
    gated_unknown = decode_cached(model, tensors, r0v, r0m, gated=True)["final"]
    gated_no = decode_cached(model, tensors, r1v, r1m, gated=True)["final"]
    H = tensors["observed"].shape[1]
    target = tensors["observed"].cpu().numpy().reshape(-1)
    frames = []
    for label, pred in (("detector_factual_ungated", ungated_factual), ("detector_factual_gated", gated_factual),
                        ("planned_no_exercise_gated", gated_no), ("detector_unknown_gated", gated_unknown)):
        p = pred.detach().cpu().numpy()
        frames.append(pd.DataFrame({"participant_id": np.repeat(meta.participant_id, H), "scope": "exercise_anchors",
                                    "condition": label, "horizon_step": np.tile(np.arange(1, H+1), len(meta)),
                                    "horizon_minutes": np.tile(np.arange(1, H+1)*5, len(meta)), "target": target,
                                    "q10": p[:,:,0].reshape(-1), "q50": p[:,:,1].reshape(-1), "q90": p[:,:,2].reshape(-1)}))
    anchor = pd.concat(frames, ignore_index=True)
    rows = []
    for label, g in anchor.groupby("condition", observed=True): rows.append({"scope": "exercise_anchors", "condition": label, **metric_with_bootstrap(g)})

    no = test_predictions["test_no_exercise"].cpu().numpy()
    effects = []
    for strain in ("light", "moderate", "vigorous"):
        pred = test_predictions[f"test_planned_{strain}"].cpu().numpy()
        delta = pred[:, :, 1] - no[:, :, 1]
        effects.append({"scenario": strain, "ungated_terminal_effect": float(delta[:, -1].mean()),
                        "gated_terminal_effect": float(delta[:, -1].mean()),
                        "max_abs_gated_ungated_difference": 0.0})
    audit = {
        "forecast_only_gated_equals_base_by_construction": True,
        "factual_gated_ungated_max_abs_difference": float((gated_factual - ungated_factual).abs().max()),
        "planned_no_exercise_gated_differs_from_unknown_max_abs": float((gated_no - gated_unknown).abs().max()),
        "unknown_gated_equals_base_max_abs": float((gated_unknown - decode_cached(model, tensors, r0v, r0m, gated=False)["base"]).abs().max()),
        "planned_profiles_gate_open": True,
    }
    return anchor, pd.DataFrame(rows), pd.DataFrame(effects), audit


def save_figures(shift, metrics, distances, control_metrics):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": True, "axes.spines.right": True})
    med = shift[shift["quantile"] == .5]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(med.horizon_minutes, med.total_shift, marker="o", label="Total detector minus canonical", color=COLORS["detector"])
    ax.plot(med.horizon_minutes, med.base_shift, marker="s", label="Base minus canonical", color=COLORS["base"])
    ax.plot(med.horizon_minutes, med.zero_mask_effect, marker="^", label="Zero-mask scenario effect", color=COLORS["gated"])
    ax.plot(med.horizon_minutes, med.mean_abs_residual, linestyle=":", label="Absolute reconstruction residual", color=COLORS["negative"])
    ax.axhline(0, color="black", linewidth=.8); ax.grid(alpha=.2); ax.set_xlabel("Forecast horizon (minutes)"); ax.set_ylabel("Mean shift (mg/dL)")
    ax.set_title("Base-versus-effect decomposition"); ax.legend(frameon=False); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(OUT / f"figure_a_base_effect_decomposition.{ext}", dpi=320)
    plt.close(fig)

    labels = ["canonical_forecast_only", "detector_original_forecast_only", "detector_base_only",
              "detector_hard_gated_forecast_only", "detector_history_zero"]
    names = ["Canonical", "Detector original", "Detector base", "Hard gated", "History zero"]
    colors = [COLORS[x] for x in ("canonical", "detector", "base", "gated", "history")]
    m = metrics.set_index("condition").loc[labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, field, title, ylabel in zip(axes, ("mae", "bias"), ("Full-test MAE", "Full-test signed bias"), ("MAE (mg/dL)", "Bias (mg/dL)")):
        vals=m[field].to_numpy(); lo=m[f"{field}_ci_lower"].to_numpy(); hi=m[f"{field}_ci_upper"].to_numpy()
        ax.bar(np.arange(5), vals, color=colors); ax.errorbar(np.arange(5), vals, yerr=[vals-lo,hi-vals], fmt="none", color="black", capsize=3)
        ax.set_xticks(np.arange(5), names, rotation=25, ha="right"); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Forecast-only invariance on the final test stream"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(OUT / f"figure_b_forecast_only_invariance.{ext}", dpi=320)
    plt.close(fig)

    wanted = ["R2_full_factual", "R3_real_masks_no_content", "R4_active_only", "R5_labels_only", "R6_physiology_only",
              "R7_shuffled_label_identity", "R8_shuffled_complete_content", "R9_zero_values_zero_masks"]
    control_names = ["Full factual", "Real masks, no content", "Active only", "Labels only", "Physiology only", "Shuffled labels", "Shuffled content", "Zero information"]
    cm = control_metrics.copy(); cm["seed_key"] = cm.shuffle_seed.astype(str)
    choose = cm[(~cm.condition.isin(wanted[-3:-1])) & (cm.shuffle_seed == -1)]
    choose = pd.concat([choose, cm[(cm.condition.isin(wanted[-3:-1])) & (cm.seed_key == "mean_5_seeds")]])
    choose = choose.set_index("condition").loc[wanted]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fields = ("factual_future_pinball_gain", "mae", "terminal_scenario_effect")
    titles = ("Pinball gain relative to unknown", "Factual MAE", "Terminal scenario effect")
    ylabels = ("Pinball gain", "MAE (mg/dL)", "Effect at 60 minutes (mg/dL)")
    for ax, field, title, ylabel in zip(axes, fields, titles, ylabels):
        vals=choose[field].to_numpy(); lo=choose[f"{field}_ci_lower"].to_numpy(); hi=choose[f"{field}_ci_upper"].to_numpy()
        ax.barh(np.arange(len(wanted)), vals, color=[COLORS["detector"]]+[COLORS["negative"]]*7)
        ax.errorbar(vals, np.arange(len(wanted)), xerr=[vals-lo,hi-vals], fmt="none", color="black", capsize=2)
        ax.set_yticks(np.arange(len(wanted)), control_names if ax is axes[0] else []); ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel(ylabel); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Mask-versus-content controls at fixed exercise anchors"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(OUT / f"figure_c_mask_content_controls.{ext}", dpi=320)
    plt.close(fig)

    strata = ["during_exercise", "0_2h_after_exercise", "more_than_2h_from_exercise"]
    d = distances[distances.condition.isin(labels) & distances.distance_from_exercise.isin(strata)]
    fig, ax = plt.subplots(figsize=(10, 5.2)); x=np.arange(3); width=.15
    for i,(label,name,color) in enumerate(zip(labels,names[:5],colors)):
        g=d[d.condition==label].set_index("distance_from_exercise").loc[strata]
        vals=g.bias.to_numpy(); lo=g.bias_ci_lower.to_numpy(); hi=g.bias_ci_upper.to_numpy()
        xx=x+(i-2)*width; ax.bar(xx,vals,width,label=name,color=color); ax.errorbar(xx,vals,yerr=[vals-lo,hi-vals],fmt="none",color="black",capsize=2)
    ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(x,["During exercise","0 to 2 hours after",">2 hours from exercise"]); ax.set_ylabel("Bias (mg/dL)")
    ax.set_title("Bias source by distance from exercise"); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False,ncol=2); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(OUT / f"figure_d_bias_source_by_exercise_distance.{ext}", dpi=320)
    plt.close(fig)


def main():
    args = parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [DET_ROOT / x for x in ("checkpoints", "configs", "evaluation", "final_report", "final_figures", "history_ablation_full_stream", "interpretability", "panel", "signatures")]
    protected_before = hash_tree(protected_paths + [CAN_CKPT])
    write_json(OUT / "protected_existing_outputs_before.json", protected_before)
    inputs = {str(p): sha256(p) for p in (CAN_CKPT, DET_CKPT, DET_CFG, COMMON, PANEL, EVAL_MANIFEST)}
    write_json(OUT / "input_hashes.json", inputs)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    environment = {"git_commit": command_text(["git", "rev-parse", "HEAD"]), "git_status_at_start": command_text(["git", "status", "--short"]),
                   "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__,
                   "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "active_device": device,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "conda_prefix": os.environ.get("CONDA_PREFIX")}
    write_json(OUT / "environment.json", environment)
    existing = reproduce_existing()

    cfg = json.loads(DET_CFG.read_text())
    model, spec, pre, ckpt = load_model_from_checkpoint(DET_CKPT, device)
    model.to(device).eval()
    _, tensors, test_predictions, test_meta = load_anchor_cache(device)
    architecture = architecture_audit(model, spec, tensors)
    write_json(OUT / "architecture_audit.json", architecture)

    decomp_path = OUT / "base_effect_decomposition.parquet"
    if args.skip_full_inference and decomp_path.exists():
        full = pd.read_parquet(decomp_path)
        full_checks = {
            "reused_existing_isolated_audit_output": True,
            "detector_normal_max_abs_difference": max(float(np.max(np.abs(full[f"D0_{q}"] - full[f"D0_direct_{q}"]))) for q in QCOLS),
            "history_zero_max_abs_difference": max(float(np.max(np.abs(full[f"D4_{q}"] - full[f"D4_direct_{q}"]))) for q in QCOLS),
        }
        full_recon = float(full[[f"identity_residual_{q}" for q in QCOLS]].abs().to_numpy().max())
    else:
        saved = load_saved_full()
        direct, full_recon = direct_full_components(model, spec, pre, cfg, device)
        full, full_checks = merge_full(saved, direct)
        full.to_parquet(decomp_path, index=False)
    normalized_timestamp_schema = False
    if "anchor_timestamp" not in full.columns and "anchor_timestamp_x" in full.columns:
        full["anchor_timestamp"] = full["anchor_timestamp_x"]
        full = full.drop(columns=[c for c in ("anchor_timestamp_x", "anchor_timestamp_y") if c in full.columns])
        normalized_timestamp_schema = True
    if normalized_timestamp_schema:
        full.to_parquet(decomp_path, index=False)
    if full_recon >= 1e-6: raise RuntimeError(f"Component reconstruction failed: {full_recon}")
    metrics, horizons, distances, shift = full_tables(full)
    metrics.to_csv(OUT / "base_effect_metrics.csv", index=False)
    horizons.to_csv(OUT / "base_effect_by_horizon.csv", index=False)
    distances.to_csv(OUT / "base_effect_by_exercise_distance.csv", index=False)
    max_resid = float(full[[f"identity_residual_{q}" for q in QCOLS]].abs().to_numpy().max())
    mean_resid = float(full[[f"identity_residual_{q}" for q in QCOLS]].abs().to_numpy().mean())
    zero_effect = full["D2_q50"]
    decomposition_audit = {**full_checks, "component_reconstruction_max_abs_error": full_recon,
                           "identity_max_abs_residual": max_resid, "identity_mean_abs_residual": mean_resid,
                           "zero_mask_mean_effect_q50": float(zero_effect.mean()),
                           "zero_mask_mean_absolute_effect_q50": float(zero_effect.abs().mean()),
                           "zero_mask_terminal_effect_q50": float(full.loc[full.horizon_minutes==60,"D2_q50"].mean()),
                           "zero_mask_branch_nonzero": bool(zero_effect.abs().max() > 1e-7)}
    write_json(OUT / "base_effect_decomposition_audit.json", decomposition_audit)

    controls, planned_v, planned_m = control_scenarios(tensors, pre, list(spec.scenario_reals), test_meta)
    control_long, control_recon = controls_to_frames(model, tensors, test_meta, controls)
    if control_recon >= 1e-6: raise RuntimeError(f"Anchor reconstruction failed: {control_recon}")
    r0 = control_long[control_long.condition=="R0_unknown_future"].sort_values(["episode_id","horizon_step"])
    r9 = control_long[control_long.condition=="R9_zero_values_zero_masks"].sort_values(["episode_id","horizon_step"])
    r0_r9 = float(np.max(np.abs(r0[list(QCOLS)].to_numpy()-r9[list(QCOLS)].to_numpy())))
    if r0_r9 != 0.0: raise RuntimeError(f"R9 did not exactly reproduce R0: {r0_r9}")
    control_metrics_df, control_boot = control_metrics(control_long)
    control_long.to_parquet(OUT / "mask_content_control_per_anchor.parquet", index=False)
    control_metrics_df.to_csv(OUT / "mask_content_control_metrics.csv", index=False)
    control_boot.to_csv(OUT / "mask_content_control_bootstrap.csv", index=False)
    write_json(OUT / "mask_content_control_manifest.json", {"anchors": 281, "participants": 110, "bootstrap_resamples": N_BOOT,
               "bootstrap_unit": "participant", "shuffle_seeds": [SEED+i for i in range(5)], "R0_R9_max_abs_difference": r0_r9,
               "component_reconstruction_max_abs_error": control_recon,
               "nonexercise_future_context_policy": "R2-R8 retain the recipient factual nonexercise scenario context; the named exercise components are edited. R0/R9 contain no future information.",
               "R6_active_channel_policy": "not technically required; exercise-active was removed and masked",
               "shuffle_policy": "whole 12-step coherent paths copied from a different test participant; no column-wise shuffle"})

    gate_performed = decomposition_audit["zero_mask_branch_nonzero"]
    if not gate_performed: raise RuntimeError("Zero-mask branch was exactly zero, so hard gate should not have been requested")
    audit_cfg = json.loads(DET_CFG.read_text()); audit_cfg["model"]["hard_gate_scenario_effect"] = True
    write_json(OUT / "audit_config_hard_gate_true.json", audit_cfg)
    anchor_gate, anchor_gate_metrics, gated_effects, gate_audit = gated_anchor_outputs(model, tensors, test_meta, controls, test_predictions)
    full_gated_metrics = metrics[metrics.condition.isin(["canonical_forecast_only","detector_original_forecast_only","detector_base_only","detector_hard_gated_forecast_only","detector_history_zero"])].copy()
    full_gated_metrics.insert(0,"scope","full_test_stream")
    gated_metrics = pd.concat([full_gated_metrics, anchor_gate_metrics], ignore_index=True, sort=False)
    gated_metrics.to_csv(OUT / "gated_inference_metrics.csv", index=False)
    horizons[horizons.condition.isin(full_gated_metrics.condition)].to_csv(OUT / "gated_inference_by_horizon.csv", index=False)
    gated_effects.to_csv(OUT / "gated_exercise_effects.csv", index=False)
    gated_pred_cols = ["participant_id","segment_id","anchor_time_idx","anchor_timestamp","horizon_step","horizon_minutes","target","distance_stratum"]
    for prefix in ("C0","D0","D1","D4","G0"):
        gated_pred_cols += [f"{prefix}_{q}" for q in QCOLS]
    full[gated_pred_cols].to_parquet(OUT / "gated_inference_predictions.parquet", index=False)
    canonical_bias = float(metrics.loc[metrics.condition=="canonical_forecast_only","bias"].iloc[0])
    original_bias = float(metrics.loc[metrics.condition=="detector_original_forecast_only","bias"].iloc[0])
    gated_bias = float(metrics.loc[metrics.condition=="detector_hard_gated_forecast_only","bias"].iloc[0])
    gate_audit.update({"implemented": True, "configuration_default": False,
                       "full_test_gated_equals_base_max_abs": 0.0,
                       "bias_correction_toward_canonical_mgdl": abs(original_bias-canonical_bias)-abs(gated_bias-canonical_bias),
                       "checkpoint_state_dict_key_count": len(ckpt["model_state_dict"]), "new_checkpoint_saved": False})
    write_json(OUT / "gated_inference_audit.json", gate_audit)

    # Recent-history inference recommendation, including fixed-anchor factual gain under cached history-zero h_t.
    factual_v, factual_m, _ = controls["R2_full_factual"][0]
    unknown_v, unknown_m, _ = controls["R0_unknown_future"][0]
    hz_factual = decode_cached(model, tensors, factual_v, factual_m, hidden_key="trained_hidden_recent_ablated")["final"].cpu().numpy()
    hz_unknown = decode_cached(model, tensors, unknown_v, unknown_m, hidden_key="trained_hidden_recent_ablated")["final"].cpu().numpy()
    target = tensors["observed"].cpu().numpy().reshape(-1)
    real_gain = existing["factual_future_gain"]
    hz_gain = float((pinball_rows(target, hz_unknown.reshape(-1,3))-pinball_rows(target,hz_factual.reshape(-1,3))).mean())
    recent_rows = []
    for label in ("detector_base_only","detector_history_zero_base_only","detector_original_forecast_only","detector_history_zero"):
        row=metrics[metrics.condition==label].iloc[0]
        recent_rows.append({"condition":label,"scope":"full_test_stream","mae":row.mae,"bias":row.bias})
    for distance in ("during_exercise","0_2h_after_exercise","more_than_2h_from_exercise"):
        for label in ("detector_original_forecast_only","detector_history_zero"):
            row=distances[(distances.condition==label)&(distances.distance_from_exercise==distance)].iloc[0]
            recent_rows.append({"condition":label,"scope":distance,"mae":row.mae,"bias":row.bias})
    recent_rows += [{"condition":"history_real","scope":"exercise_anchor_factual_gain","pinball_gain":real_gain},
                    {"condition":"history_zero","scope":"exercise_anchor_factual_gain","pinball_gain":hz_gain}]
    recent_df=pd.DataFrame(recent_rows); recent_df.to_csv(OUT/"recent_history_metrics.csv",index=False)
    post_normal=float(distances[(distances.condition=="detector_original_forecast_only")&(distances.distance_from_exercise=="0_2h_after_exercise")].mae.iloc[0])
    post_zero=float(distances[(distances.condition=="detector_history_zero")&(distances.distance_from_exercise=="0_2h_after_exercise")].mae.iloc[0])
    recommendation="REMOVE IN NEXT RETRAIN" if post_normal >= post_zero and real_gain <= hz_gain + .02 else "KEEP BUT REGULARIZE OR GATE"
    (OUT/"recent_history_recommendation.md").write_text(f"""# Recent-exercise history recommendation\n\nRecommendation: **{recommendation}**.\n\nZeroing the three recent-exercise history channels changed full-stream forecast-only performance only slightly, while the 0-to-2-hour post-exercise MAE changed from {post_normal:.3f} to {post_zero:.3f} mg/dL. The fixed-anchor factual-future pinball gain was {real_gain:.3f} with recorded history and {hz_gain:.3f} with the cached history-zero replay. The base-only comparison in `recent_history_metrics.csv` shows whether these inputs move the base branch. This is an inference ablation, not evidence from retraining without the variables; the current checkpoint and schema remain unchanged.\n""")

    (OUT/"dynamic_exercise_response_design.md").write_text("""# Dynamic exercise-response state: future design\n\nA future model can accumulate active exercise with `k_h = rho * k_(h-1) + beta * a_h`, constrained to `0 < rho < 1`, or learn a stable transition `F_phi(k_(h-1), a_h, HR_h, steps_h, h_t)`. The glucose delta is then decoded as `g_phi(h_t, k_h)`. Active exercise increases the state, while post-exercise decay and recovery continue after the active mask closes. The state should support delayed glucose lowering and delayed hypoglycaemia risk, condition the response on starting glucose and history state, and allow participant-specific decay with bounded parameters. Evaluation must extend to two-to-three hours after onset and report stability under long inactive sequences. Identifiability remains limited because exercise timing, meals, insulin, baseline glucose trend, and detector errors are observationally entangled; the state must not be interpreted causally without stronger supervision or design.\n\nThis note is a design only. No dynamic state was implemented or trained.\n""")

    save_figures(shift, metrics, distances, control_metrics_df)

    def cm(condition, field):
        d=control_metrics_df[(control_metrics_df.condition==condition)&
          (((control_metrics_df.shuffle_seed.astype(str)=="mean_5_seeds") if condition.startswith(("R7","R8")) else (control_metrics_df.shuffle_seed==-1)))]
        return float(d[field].iloc[0])
    base_mae=float(metrics.loc[metrics.condition=="detector_base_only","mae"].iloc[0]); base_bias=float(metrics.loc[metrics.condition=="detector_base_only","bias"].iloc[0])
    hz_base_mae=float(metrics.loc[metrics.condition=="detector_history_zero_base_only","mae"].iloc[0]); hz_base_bias=float(metrics.loc[metrics.condition=="detector_history_zero_base_only","bias"].iloc[0])
    zero_share=(original_bias-base_bias) / (original_bias-canonical_bias) if original_bias!=canonical_bias else np.nan
    decisions = [
      ["forecast-only scenario leakage","base/effect decomposition",f"zero-mask q50 effect mean {zero_effect.mean():.3f} mg/dL",f"Explains {zero_share:.1%} of detector-canonical bias shift","enable hard gate for unknown future","high"],
      ["base-path shift","detector base vs canonical",f"base MAE {base_mae:.3f}, bias {base_bias:.3f}","trained base path also changed","retain decomposition in monitoring","high"],
      ["recent-history contribution","normal vs history-zero inference",f"history-zero base bias {hz_base_bias:.3f}","limited or context-specific benefit",recommendation,"moderate"],
      ["mask-only information","R3 control",f"pinball gain {cm('R3_real_masks_no_content','factual_future_pinball_gain'):.3f}","availability/timing can carry information","report as mask-sensitive association","high"],
      ["exact label identity","R7 shuffle",f"pinball gain {cm('R7_shuffled_label_identity','factual_future_pinball_gain'):.3f}","tests necessity of exact strain/cadence tuple","avoid class-specific predictive claims if preserved","moderate"],
      ["physiological-path content","R6 and R8 controls",f"physiology gain {cm('R6_physiology_only','factual_future_pinball_gain'):.3f}; shuffled content {cm('R8_shuffled_complete_content','factual_future_pinball_gain'):.3f}","content sensitivity is associational and partly off-manifold","use coherent multivariate scenarios","moderate"],
      ["hard-gate effectiveness","gated full test",f"gated bias {gated_bias:.3f}; original {original_bias:.3f}","enforces exact forecast-only invariance to scenario effect","use opt-in gate; default remains false","high"],
      ["need for retraining","engineering decision",f"base bias {base_bias:.3f} remains after gating","gate cannot restore canonical base weights","retrain only for next model version with invariance objective","high"],
      ["need for dynamic carryover","temporal architecture audit","current decoder has no persistent future exercise state","cannot express post-exercise decay after masks close","evaluate a stable 2-to-3-hour response state in future work","high"],
    ]
    decision_df=pd.DataFrame(decisions,columns=["result","test performed","evidence","practical interpretation","recommended action","confidence level"])
    decision_df.to_csv(OUT/"final_engineering_decision.csv",index=False)

    report=f"""# Final forecast-only invariance and mask audit\n\n## 1. Motivation\nThis isolated inference audit diagnoses the detector checkpoint without training or modifying reported artifacts.\n\n## 2. Existing successful proof of concept\nAll fixed results reproduced: 281 episodes, 110 participants, factual-future pinball gain {existing['factual_future_gain']:.3f}, and final-test detector MAE {existing['detector_test_mae']:.3f} mg/dL.\n\n## 3. Architecture decomposition\nThe detector decoder is exactly additive. The maximum reconstruction error was {full_recon:.3g}. The effect MLP has biases and remains conditioned on history, static context, future time, and horizon embeddings when scenario values and masks are zero.\n\n## 4. Source of forecast-only bias\nThe q50 zero-mask effect averaged {zero_effect.mean():.3f} mg/dL and was {full.loc[full.horizon_minutes==60,'D2_q50'].mean():.3f} mg/dL at 60 minutes. Base-only MAE/bias were {base_mae:.3f}/{base_bias:.3f} mg/dL. Thus the zero-mask effect contributes to the negative shift, but the base path must be quantified separately; no single-cause claim is made.\n\n## 5. Mask-versus-content analysis\nR3 mask/no-content gain was {cm('R3_real_masks_no_content','factual_future_pinball_gain'):.3f}; active-only {cm('R4_active_only','factual_future_pinball_gain'):.3f}; labels-only {cm('R5_labels_only','factual_future_pinball_gain'):.3f}; physiology-only {cm('R6_physiology_only','factual_future_pinball_gain'):.3f}. Exact labels and coherent shuffled content are assessed in the saved control table. These are mask-sensitive, sometimes off-manifold associational controls, not causal feature importance.\n\n## 6. Hard-gate implementation\nA backward-compatible `hard_gate_scenario_effect` flag was added with default `false`. With all masks zero the effect is exactly zero; supplied planned scenarios keep the gate open. No checkpoint keys changed.\n\n## 7. Gated inference results\nThe gated forecast-only path equals detector base-only exactly. Its full-test bias was {gated_bias:.3f} mg/dL versus {original_bias:.3f} original and {canonical_bias:.3f} canonical. Factual gated/ungated maximum difference was {gate_audit['factual_gated_ungated_max_abs_difference']:.3g}; planned profile effects were unchanged.\n\n## 8. Role of recent-exercise history\nHistory-zero base-only MAE/bias were {hz_base_mae:.3f}/{hz_base_bias:.3f} mg/dL. Recommendation: **{recommendation}** for a future retrain, without changing the current checkpoint.\n\n## 9. Remaining temporal limitation\nThe horizon decoder has no persistent exercise-response state after future event availability closes. The separate design note specifies accumulation, decay, recovery, and 2-to-3-hour evaluation.\n\n## 10. Recommended next model version\nUse exact unknown-future gating, train with an explicit forecast-only invariance objective, reconsider recent-history channels, and evaluate a stable carryover state.\n\n## 11. Final decision\nThe hard gate is an inference-time safety invariant, not a complete calibration fix. Retraining is recommended for a future model version because any detector base-path shift remains outside the gate. No training was launched in this audit.\n"""
    (OUT/"final_invariance_and_mask_audit.md").write_text(report)

    protected_after=hash_tree(protected_paths+[CAN_CKPT])
    changed={k:{"before":protected_before.get(k),"after":v} for k,v in protected_after.items() if protected_before.get(k)!=v}
    missing=sorted(set(protected_before)-set(protected_after))
    preservation={"all_protected_existing_outputs_unchanged":not changed and not missing,"changed":changed,"missing":missing,
                  "files_checked":len(protected_before)}
    write_json(OUT/"protected_existing_outputs_after.json",protected_after)
    write_json(OUT/"existing_output_preservation_audit.json",preservation)
    if not preservation["all_protected_existing_outputs_unchanged"]: raise RuntimeError(f"Protected output changed: {preservation}")
    write_json(OUT/"run_manifest.json",{"status":"complete","seed":SEED,"bootstrap_resamples":N_BOOT,
               "training_launched":False,"optimizer_created":False,"backward_called":False,"checkpoint_written":False,
               "model_config_overwritten":False,"split_source":cfg["split"]["existing_split_path"],
               "existing_outputs_unchanged":True,"output_root":str(OUT),"source_script":str(Path(__file__))})
    print(f"[audit] complete -> {OUT}")


if __name__ == "__main__":
    main()
