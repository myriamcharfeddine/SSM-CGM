#!/usr/bin/env python3
"""AI-READI per-feature h_t attribution export (Step 2, base pass).

Adapts T1DEXI's ``interpret_export.py`` to the AI-READI checkpoint. Computes,
for every INDIVIDUAL dynamic-real feature (the only pathway with per-feature
contribution vectors u_f into the SSM hidden state h_t -- see
scripts/interpret_airedi_groups.py), the banded MES hidden-attention
attribution onto h_t, over a dense anchor grid spanning the FULL held-out
test split (not a small triggered-anchor subset).

This is the per-feature ingredient both scripts/interpret_airedi_grouped_export.py
(rigorous joint-group norm) and its post-hoc-sum comparison need; it does not
itself group features.

AI-READI has no time-varying insulin dose or carbohydrate intake column in
the model's feature schema (confirmed by scripts/interpret_airedi_groups.py --
med_insulin is a STATIC therapy flag in "medication metadata"). There is
therefore nothing to remove or disable here: this script never treats any
feature as a time-varying insulin/carb causal covariate.

PAUSE: by default this only COUNTS streams/anchors and exits (no GPU pass).
Pass --execute to actually run attribution after reviewing the printed count.

Usage:
  python scripts/interpret_airedi_export.py                 # count only
  python scripts/interpret_airedi_export.py --execute        # run + write CSVs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig
from ssmcgm.ops.mes_attribution import compute_hidden_attention

BIN_MIN = 5
HORIZON = 12                      # steps; must match the checkpoint's horizon_steps
ANCHOR_STRIDE_MIN = 15            # dense anchor grid spacing (config evaluation.anchor_stride_minutes)
ANCHOR_STRIDE_STEPS = ANCHOR_STRIDE_MIN // BIN_MIN
WARMUP_STEPS = 12                 # skip the first hour of each stream (state still settling)
MAX_LAG_STEPS = 72                # banded attribution depth (6h); matches T1DEXI's --max-lags default
FINE_LAG_EDGES = list(range(0, 65, 5)) + [72]   # coarse lag bins (minutes) for the temporal CSV, capped at MAX_LAG_STEPS

CHECKPOINT_PATH = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
CONFIG_PATH = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/config_resolved.yaml"
DEFAULT_OUT_DIR = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/interpret"


def load_model(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    raw_model_config = md["model_config"]
    valid_fields = {f.name for f in __import__("dataclasses").fields(AireadiStreamModelConfig)}
    dropped = {k: v for k, v in raw_model_config.items() if k not in valid_fields}
    if dropped:
        # Older checkpoints saved config fields for components (e.g. the disabled
        # hr_exercise head) that no longer exist in the current model class. Safe to
        # drop only when the component was inactive at save time (no matching weights
        # in model_state_dict); caller is responsible for that check if it matters.
        print(f"[load_model] dropping checkpoint config keys not in current AireadiStreamModelConfig: "
              f"{sorted(dropped)}")
    cfg = AireadiStreamModelConfig(**{k: v for k, v in raw_model_config.items() if k in valid_fields})
    model = AireadiStreamModel(spec, pre, cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec, pre


def load_test_streams(cfg: dict, spec: AireadiFeatureSpec, pre: AireadiPreprocessor):
    data_cfg = cfg["data"]
    df = load_aireadi_panel(
        data_cfg["panel_path"],
        static_path=data_cfg.get("static_path"),
        cohort_path=data_cfg.get("cohort_path"),
    )
    schema = infer_or_validate_schema(df, data_cfg.get("schema"))
    df = prepare_aireadi_panel(
        df, schema,
        bin_minutes=cfg["dataset"].get("bin_minutes", BIN_MIN),
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    split_cfg = cfg["split"]
    split = make_aireadi_stream_splits(
        df,
        split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", 0.70), val=split_cfg.get("val", 0.15),
        test=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg.get("existing_split_path"),
    )
    streams = make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=["test"], min_steps=spec.horizon_steps + WARMUP_STEPS + 2,
    )
    return streams


def anchors_for_stream(n_steps: int) -> np.ndarray:
    lo, hi = WARMUP_STEPS, n_steps - HORIZON
    if hi <= lo:
        return np.array([], dtype=int)
    return np.arange(lo, hi, ANCHOR_STRIDE_STEPS, dtype=int)


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
    _out, _layer_states, _conv_states, caches = model.temporal.scan(
        fused, state.layer_states, state.conv_states, record=True, static_embedding=sctx.embedding,
    )
    return caches, contribs


def count_only(streams):
    n_anchors = 0
    per_stream = []
    for s in streams:
        a = anchors_for_stream(s.n_steps)
        n_anchors += len(a)
        per_stream.append(len(a))
    print(f"[interpret-export] test streams: {len(streams)}")
    if per_stream:
        print(f"[interpret-export] anchors/stream: min={min(per_stream)} "
              f"median={int(np.median(per_stream))} max={max(per_stream)}")
    print(f"[interpret-export] TOTAL anchors (windows) across the full test split: {n_anchors}")
    print(f"[interpret-export] GPU passes required: {len(streams)} stream scans "
          f"(one banded hidden-attention call per stream, covering all its anchors at once)")
    print("[interpret-export] PAUSE: this was a count-only dry run. Re-run with --execute to "
          "actually run the GPU attribution pass.")
    return n_anchors


@torch.no_grad()
def run_export(model, streams, device, out_dir: Path):
    names = list(model.feature_spec.dynamic_reals)
    feature_acc = {n: np.zeros(MAX_LAG_STEPS + 1, dtype="float64") for n in names}
    n_windows = 0
    n_streams_used = 0

    for stream in streams:
        anchors = anchors_for_stream(stream.n_steps)
        if len(anchors) == 0:
            continue
        caches, contribs = scan_with_record(model, stream, device)
        layer = len(caches) - 1
        rows_t = torch.tensor(anchors, dtype=torch.long, device=device)
        attn, _ = compute_hidden_attention(
            caches[layer], rows=rows_t, max_lags=MAX_LAG_STEPS,
            ngroups=model.temporal.blocks[0].ssm.ngroups, normalize="softmax", out_device="cpu",
        )
        attn_np = attn.float().numpy()[0].sum(axis=0)          # (R, K+1), summed over heads

        lag_idx = np.arange(MAX_LAG_STEPS + 1, dtype=int)
        src = anchors[:, None] - lag_idx[None, :]
        valid = src >= 0
        src_clip = np.clip(src, 0, stream.n_steps - 1)

        for name in names:
            u_norm = contribs[name].detach().cpu().norm(dim=-1).numpy()[0]   # (L,)
            vals = np.where(valid, u_norm[src_clip], 0.0)                    # (R, K+1)
            feature_acc[name] += (attn_np * vals).sum(axis=0)

        n_windows += len(anchors)
        n_streams_used += 1

    total = sum(float(v.sum()) for v in feature_acc.values())
    lags_min = np.arange(MAX_LAG_STEPS + 1) * BIN_MIN

    overall_rows = [{"feature": n, "raw_influence": float(v.sum()),
                      "frac_influence": float(v.sum()) / total if total > 0 else 0.0}
                     for n, v in feature_acc.items()]
    overall = pd.DataFrame(overall_rows).sort_values("frac_influence", ascending=False)

    bin_labels = pd.cut(lags_min, FINE_LAG_EDGES, right=False, include_lowest=True)
    temporal_rows = []
    for name, arr in feature_acc.items():
        s = pd.Series(arr, index=bin_labels)
        binned = s.groupby(level=0, observed=True).sum()
        denom = arr.sum() if arr.sum() > 0 else 1.0
        for lag_bin, val in binned.items():
            temporal_rows.append({"feature": name, "lag_bin": str(lag_bin),
                                  "raw_influence": float(val), "frac_influence": float(val) / denom})
    temporal = pd.DataFrame(temporal_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_dir / "attribution_feature_overall.csv", index=False)
    temporal.to_csv(out_dir / "attribution_temporal.csv", index=False)
    print(f"[interpret-export] attribution over {n_windows} windows across {n_streams_used} "
          f"test streams -> {out_dir}")
    print(overall.head(10).to_string(index=False))
    return overall, temporal, n_windows, n_streams_used


def main():
    ap = argparse.ArgumentParser(description="AI-READI per-feature h_t attribution export")
    ap.add_argument("--execute", action="store_true", help="run the GPU attribution pass (default: count only)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    print(f"[interpret-export] checkpoint={CHECKPOINT_PATH}")
    print(f"[interpret-export] config={CONFIG_PATH}")
    print(f"[interpret-export] split=existing_split_participants seed=42 "
          f"source={cfg['split']['existing_split_path']}")

    model, spec, pre = load_model(CHECKPOINT_PATH, device)
    print(f"[interpret-export] dynamic_reals={len(spec.dynamic_reals)} device={device}")

    streams = load_test_streams(cfg, spec, pre)
    count_only(streams)

    if not args.execute:
        return

    run_export(model, streams, device, out_dir)


if __name__ == "__main__":
    main()
