#!/usr/bin/env python3
"""Generate Study 2 Block A/B rolling-forecast cache with 5-minute anchor stride.

Runs the frozen base SSM-CGM checkpoint in forecast-only mode with
anchor_stride_steps = 1 (every 5-minute bin is an anchor) for train,
validation, and test participant-disjoint splits.

Post-processing adds:
  anchor_ds      -- alias for anchor_time_idx (used by study2_interpretability)
  current_glucose -- G_t at the forecast issue time (derived from horizon-1
                     target of the previous anchor, valid because stride = 1)

A MANIFEST.json is written next to the cache that records all provenance fields
required by Study 2.

Usage (on a VM with torch):
    python scripts/generate_study2_forecast_cache.py \\
        --config configs/study2_forecast_cache_5min.yaml \\
        --ckpt   outputs/aireadi_stream_full/checkpoints/best_model_checkpoint.pt \\
        --output-dir outputs/study2_forecast_cache_5min

Add --smoke for a fast end-to-end check (6 participants, 4 anchors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from ssmcgm.evaluation.aireadi_streaming import evaluate_aireadi_streams


_SPLITS = ("train", "validation", "test")
_MODE = "forecast_only"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate Study 2 frozen-base forecast cache with 5-min anchor stride"
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs/study2_forecast_cache_5min.yaml"),
        help="Path to the Study 2 cache-generation config (YAML)",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Path to frozen base checkpoint; overrides config.base_checkpoint",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; overrides config.output.base_dir",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity run: few participants and max 4 anchors per stream",
    )
    ap.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Limit number of participants per split (for testing)",
    )
    return ap.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return requested


def load_frozen_checkpoint(ckpt_path: Path, device: str):
    from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    mcfg = AireadiStreamModelConfig(**md["model_config"])
    model = AireadiStreamModel(spec, pre, mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec, pre


def build_streams_for_split(cfg: dict, spec: AireadiFeatureSpec, pre, split_name: str,
                             *, smoke: bool, max_participants: int | None):
    dc = cfg["data"]
    n_smoke = max_participants or (cfg.get("smoke", {}).get("eval_participants") if smoke else None)
    df = load_aireadi_panel(
        dc["panel_path"],
        static_path=dc.get("static_path"),
        cohort_path=dc.get("cohort_path"),
        smoke_participants=n_smoke,
    )
    schema = infer_or_validate_schema(df, dc.get("schema"))
    df = prepare_aireadi_panel(
        df, schema,
        bin_minutes=cfg["dataset"].get("bin_minutes", 5),
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    split_cfg = cfg.get("split", {})
    split = make_aireadi_stream_splits(
        df,
        split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", 0.70),
        val=split_cfg.get("val", 0.15),
        test=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg.get("existing_split_path"),
    )
    build_stream_feature_spec(df, schema, horizon_steps=spec.horizon_steps, bin_minutes=spec.bin_minutes)
    return make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=[split_name], min_steps=spec.horizon_steps + 2,
        max_participants=(cfg.get("smoke", {}).get("eval_participants") if smoke else max_participants),
    )


def _add_anchor_ds_and_current_glucose(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process the flat forecast dataframe to add anchor_ds and current_glucose.

    anchor_ds = anchor_time_idx  (monotonically increasing integer per participant-segment;
                                   with stride=1, consecutive anchors differ by exactly 1)

    current_glucose = G_t at the forecast issue time.
        At anchor_ds = t the fresh 12-step forecast covers [t+1 .. t+12].
        The one-step forecast issued at anchor_ds = t-1 predicts G_t at horizon_step = 1.
        Its target column stores the actual G_t, which is the current glucose at anchor t.
        Because stride_steps = 1, the row (anchor_ds = t-1, horizon_step = 1) is always
        present for every anchor t > min(anchor_ds) within the same (participant, segment).
    """
    df = df.copy()
    df["anchor_ds"] = df["anchor_time_idx"]

    h1 = (
        df[df["horizon_step"].eq(1)][["participant_id", "segment_id", "anchor_ds", "target"]]
        .copy()
        .rename(columns={"target": "current_glucose", "anchor_ds": "_prev_ds"})
    )
    h1["anchor_ds"] = h1["_prev_ds"] + 1
    h1 = h1.drop_duplicates(["participant_id", "segment_id", "anchor_ds"])

    df = df.merge(
        h1[["participant_id", "segment_id", "anchor_ds", "current_glucose"]],
        on=["participant_id", "segment_id", "anchor_ds"],
        how="left",
    )
    return df


def _enforce_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every required Study 2 column is present (NaN if not in cache)."""
    required = [
        "participant_id", "segment_id", "split",
        "anchor_ds", "anchor_time_idx",
        "steps_since_start", "hours_since_start",
        "horizon_step", "horizon_minutes",
        "q10", "q50", "q90",
        "target",
        "current_glucose",
        "med_insulin",
        "participants_study_group",
        "hba1c_percent_baseline",
        "participants_clinical_site",
        "med_any_diabetes_drug",
        "scenario_mode",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def write_manifest(out_dir: Path, *, ckpt_path: Path, config_path: Path,
                   cache_path: Path, command: str, cfg: dict) -> Path:
    eval_cfg = cfg.get("evaluation", {})
    bin_minutes = cfg["dataset"].get("bin_minutes", 5)
    horizon_steps = cfg["dataset"].get("horizon_steps", 12)
    manifest = {
        "anchor_stride_minutes": eval_cfg.get("anchor_stride_minutes", 5),
        "anchor_stride_steps": eval_cfg.get("anchor_stride_steps", 1),
        "sampling_minutes": bin_minutes,
        "horizon_steps": horizon_steps,
        "horizon_minutes": bin_minutes * horizon_steps,
        "base_checkpoint": str(ckpt_path),
        "mode": "forecast_only",
        "base_model_frozen": True,
        "splits": "participant_disjoint",
        "future_labels_used_as_inputs": False,
        "future_covariates_revealed": False,
        "two_forecast_convention": {
            "one_step_for_surprise": "horizon_step=1 row at anchor_ds=t-1, target=G_t",
            "fresh_12step_at_t": "all 12 horizon rows at anchor_ds=t",
        },
        "config_path": str(config_path),
        "command": command,
        "cache_path": str(cache_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[study2_cache] manifest -> {manifest_path}")
    return manifest_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)

    ckpt_path = Path(
        args.ckpt
        or cfg.get("base_checkpoint")
        or (Path(cfg["output"]["base_dir"]).parent / "aireadi_stream_full/checkpoints/best_model_checkpoint.pt")
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found: {ckpt_path}")

    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})
    bin_minutes = cfg["dataset"].get("bin_minutes", 5)
    stride_steps = max(1, round(eval_cfg.get("anchor_stride_minutes", 5) / bin_minutes))
    if stride_steps != 1:
        raise ValueError(
            f"Expected anchor_stride_steps = 1 for Study 2, got {stride_steps}. "
            f"Check anchor_stride_minutes in the config."
        )

    max_anchors = cfg.get("smoke", {}).get("max_anchors_per_stream") if args.smoke else None

    print(f"[study2_cache] Loading frozen checkpoint: {ckpt_path}")
    model, spec, pre = load_frozen_checkpoint(ckpt_path, device)
    print(f"[study2_cache] device={device}  stride_steps={stride_steps}  mode={_MODE}")

    split_frames: list[pd.DataFrame] = []
    for split_name in _SPLITS:
        print(f"\n[study2_cache] === split={split_name} ===")
        streams = build_streams_for_split(
            cfg, spec, pre, split_name,
            smoke=args.smoke, max_participants=args.max_participants,
        )
        if not streams:
            print(f"[study2_cache] WARNING: no {split_name} streams - skipping")
            continue

        res = evaluate_aireadi_streams(
            model, streams,
            scenario_mode=_MODE,
            anchor_stride_steps=stride_steps,
            warmup_steps=0,
            max_anchors_per_stream=max_anchors,
            device=device,
            record_latency=False,
        )
        preds = res["predictions"]
        print(f"[study2_cache] {split_name}: {len(preds):,} rows, "
              f"{preds['participant_id'].nunique()} participants")
        split_frames.append(preds)

    if not split_frames:
        raise RuntimeError("No predictions were generated for any split.")

    combined = pd.concat(split_frames, ignore_index=True)
    combined = _add_anchor_ds_and_current_glucose(combined)
    combined = _enforce_required_columns(combined)

    cache_path = out_dir / "study2_forecast_cache.parquet"
    combined.to_parquet(cache_path, index=False)
    print(f"\n[study2_cache] Saved {len(combined):,} rows -> {cache_path}")
    print(f"[study2_cache] Columns: {list(combined.columns)}")
    print(f"[study2_cache] Splits: {combined['split'].value_counts().to_dict()}")

    command = (
        f"python scripts/generate_study2_forecast_cache.py"
        f" --config {args.config}"
        f" --ckpt {ckpt_path}"
        f" --output-dir {out_dir}"
        f" --device {args.device}"
        + (" --smoke" if args.smoke else "")
    )
    write_manifest(
        out_dir,
        ckpt_path=ckpt_path,
        config_path=Path(args.config),
        cache_path=cache_path,
        command=command,
        cfg=cfg,
    )
    print(f"\n[study2_cache] Done.  Next step:")
    print(f"  python scripts/run_study2_blocks_ab_diagnostics.py \\")
    print(f"    --cache-path {cache_path}")


if __name__ == "__main__":
    main()
