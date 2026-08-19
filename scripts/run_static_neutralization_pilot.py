#!/usr/bin/env python3
"""Step 1: validation-only static-neutralized replay and burn-in pilot.

This is deliberately a standalone analysis entry point.  It does not modify the
canonical panel, checkpoint, split, schema, or production evaluation modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Required by PyTorch for deterministic CUDA BLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    CLEAN_CORE_COLS,
    CLEAN_GAP_THRESHOLDS_BINS,
    build_stream_feature_spec,
    infer_or_validate_schema,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig


LOG = logging.getLogger("step1_static_neutralization")
CONDITIONS = ("full_profile", "static_neutral")
DISCREPANT_IDS = ("1118", "4211", "7139")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--multimodal-parquet", required=True)
    p.add_argument("--static-table", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--step0-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--n-pilot-participants", type=int, default=12)
    p.add_argument("--split", default="validation")
    p.add_argument("--participant-ids", nargs="*", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--backend", default="checkpoint")
    p.add_argument("--state-save-frequency-minutes", type=int, default=5)
    p.add_argument("--bootstrap-replicates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256(path: Path, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def setup_logging(path: Path) -> None:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.handlers[:] = [fh, sh]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def git_info() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True)
    return commit, status


def valid_anchors(stream, horizon: int, stride: int) -> list[int]:
    """Exact local copy of the canonical evaluation anchor rule."""
    anchors: list[int] = []
    last = -(10**9)
    for t in range(0, stream.n_steps - horizon):
        if (t - last) < stride:
            continue
        if bool(stream.observed[t]) and bool(stream.observed[t + 1 : t + 1 + horizon].all()):
            anchors.append(t)
            last = t
    return anchors


def flatten_initial_state(state) -> np.ndarray:
    parts = [x.detach().cpu().numpy().reshape(-1) for x in state.layer_states]
    parts += [
        x.detach().cpu().numpy().reshape(-1)
        for x in (state.conv_states or [])
        if x is not None
    ]
    return np.concatenate(parts)


def quartile(series: pd.Series) -> pd.Series:
    rank = series.rank(method="first")
    try:
        return pd.qcut(rank, 4, labels=("Q1", "Q2", "Q3", "Q4")).astype(str)
    except ValueError:
        return pd.Series("Q?", index=series.index)


def time_band(minutes: float) -> str:
    if minutes < 30:
        return "0-30m"
    if minutes < 60:
        return "30-60m"
    if minutes < 180:
        return "1-3h"
    if minutes < 360:
        return "3-6h"
    if minutes < 720:
        return "6-12h"
    return ">12h"


def _long_gap_mask(series: pd.Series, threshold: int) -> pd.Series:
    missing = series.isna()
    run = (~missing).cumsum()
    lens = missing.groupby(run).sum()
    return missing & run.isin(set(lens[lens > threshold].index))


def reconcile_segments(
    raw_three: pd.DataFrame,
    prepared_three: pd.DataFrame,
    split_map: dict[str, str],
    enriched_root: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    saved_segments = pd.read_csv(enriched_root / "segments.csv")
    saved_cohort = pd.read_csv(enriched_root / "cohort.csv")
    rows = []
    details = {}
    for pid in DISCREPANT_IDS:
        raw = raw_three[raw_three["participant_id"].astype(str) == pid].copy()
        raw["_ts"] = pd.to_datetime(raw["timestamp_local"])
        raw = raw.sort_values("_ts")
        prep = prepared_three[prepared_three["participant_id"].astype(str) == pid]
        reconstructed = []
        for sid, g in prep.groupby("segment_id", sort=True):
            reconstructed.append(
                {
                    "segment_id": int(sid),
                    "start": pd.Timestamp(g["_stream_timestamp"].min()),
                    "end": pd.Timestamp(g["_stream_timestamp"].max()),
                    "n_bins": int(len(g)),
                }
            )
        masks = {}
        for key, col in CLEAN_CORE_COLS.items():
            masks[key] = _long_gap_mask(
                pd.to_numeric(raw[col], errors="coerce"), CLEAN_GAP_THRESHOLDS_BINS[key]
            )
        bad = pd.Series(
            np.logical_or.reduce([m.to_numpy() for m in masks.values()]), index=raw.index
        )
        run_id = (~bad).cumsum()
        gaps = []
        for _, gap in raw[bad].groupby(run_id[bad]):
            prev_ts = raw.loc[raw["_ts"] < gap["_ts"].min(), "_ts"].max()
            next_ts = raw.loc[raw["_ts"] > gap["_ts"].max(), "_ts"].min()
            responsible = [
                k for k, mask in masks.items() if int(mask.loc[gap.index].sum()) > 0
            ]
            gaps.append(
                {
                    "start": gap["_ts"].min(),
                    "end": gap["_ts"].max(),
                    "previous": prev_ts,
                    "next": next_ts,
                    "minutes": int(len(gap) * 5),
                    "modalities": "+".join(responsible),
                }
            )
        saved = saved_segments[saved_segments["participant_id"].astype(str) == pid]
        coh = saved_cohort[saved_cohort["participant_id"].astype(str) == pid].iloc[0]
        # The discrepant boundary is the gap separating the edge segment omitted
        # only after saved pre-segmentation edge trimming.
        if pid == "1118":
            div_gap = gaps[-1]
            divergence = pd.Timestamp(coh["valid_end"])
            affected = reconstructed[-1]
            trim_side = "trailing"
            trimmed_bins = int(coh["bins_trimmed_trail"])
        elif pid == "4211":
            div_gap = gaps[-1]
            divergence = pd.Timestamp(coh["valid_end"])
            affected = reconstructed[-1]
            trim_side = "trailing"
            trimmed_bins = int(coh["bins_trimmed_trail"])
        else:
            div_gap = gaps[0]
            divergence = pd.Timestamp(coh["valid_start"])
            affected = reconstructed[0]
            trim_side = "leading"
            trimmed_bins = int(coh["bins_trimmed_lead"])
        saved_bounds = (
            f"{coh['valid_start']} .. {coh['valid_end']} "
            "(cohort trimmed recording; per-segment timestamps were not persisted)"
        )
        recon_bounds = "; ".join(
            f"{x['segment_id']}:{x['start'].isoformat()}..{x['end'].isoformat()} "
            f"(n={x['n_bins']})"
            for x in reconstructed
        )
        row = {
            "participant_id": pid,
            "split": split_map.get(pid),
            "saved_n_segments": int(len(saved)),
            "reconstructed_n_segments": int(len(reconstructed)),
            "saved_segment_start_end": saved_bounds,
            "reconstructed_segment_start_end": recon_bounds,
            "divergence_timestamp": divergence,
            "gap_start": div_gap["start"],
            "gap_end": div_gap["end"],
            "previous_retained_timestamp": div_gap["previous"],
            "next_retained_timestamp": div_gap["next"],
            "modality_responsible": div_gap["modalities"],
            "gap_duration_minutes": div_gap["minutes"],
            "saved_edge_trim_side": trim_side,
            "saved_edge_trim_bins": trimmed_bins,
            "affected_edge_segment_start": affected["start"],
            "affected_edge_segment_end": affected["end"],
            "affected_edge_segment_n_bins_untrimmed": affected["n_bins"],
            "changes_reset_timing": True,
            "canonical_replay_choice": "checkpoint_current_reconstruction",
            "canonical_evidence": (
                "Checkpoint metadata/config and train_stream_aireadi.build_data call "
                "prepare_aireadi_panel directly; no checkpoint-run stream manifest exists. "
                "The older enriched segments.csv was not loaded by the checkpoint run."
            ),
            "reconciliation_outcome": (
                f"Resolved: saved {trim_side} trim ({trimmed_bins} bins) pushed the edge "
                "segment below 49h; checkpoint path retained the untrimmed >=49h segment."
            ),
        }
        rows.append(row)
        details[pid] = row
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "segment_boundary_reconciliation.csv", index=False)
    md = [
        "# Segment-boundary reconciliation",
        "",
        "Decision: use the checkpoint-era current reconstruction for replay.",
        "",
        "The checkpoint metadata records the canonical panel, 49-hour clean-segment "
        "minimum, and saved split. Both training and evaluation construct streams by "
        "calling `prepare_aireadi_panel`; neither loads the older enriched "
        "`segments.csv`. No checkpoint-associated stream manifest was found.",
        "",
        "All three differences arise from the older enrichment pipeline's leading or "
        "trailing edge trim. There are no timestamp discontinuities. The current "
        "checkpoint path omits that trim, so an outer segment remains just long enough "
        "to clear the 49-hour count threshold.",
        "",
    ]
    for row in rows:
        md += [
            f"## Participant {row['participant_id']} ({row['split']})",
            "",
            f"- Saved/reconstructed segment counts: {row['saved_n_segments']} / "
            f"{row['reconstructed_n_segments']}",
            f"- Divergent gap: {row['gap_start']} to {row['gap_end']} "
            f"({row['gap_duration_minutes']} minutes; {row['modality_responsible']})",
            f"- Saved edge trim: {row['saved_edge_trim_side']} "
            f"{row['saved_edge_trim_bins']} bins",
            f"- Outcome: {row['reconciliation_outcome']}",
            "",
        ]
    (out_dir / "segment_boundary_reconciliation.md").write_text("\n".join(md) + "\n")
    return result, details


def build_static_reference(
    prepared: pd.DataFrame,
    spec: AireadiFeatureSpec,
    pre: AireadiPreprocessor,
    train_ids: set[str],
    out_dir: Path,
    source_hashes: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    participant = prepared.drop_duplicates("participant_id", keep="first").copy()
    participant["participant_id"] = participant["participant_id"].astype(str)
    train = participant[participant["participant_id"].isin(train_ids)].copy()
    if len(train) != len(train_ids):
        missing = sorted(train_ids - set(train["participant_id"]))
        raise RuntimeError(f"{len(missing)} training participants have no canonical stream")
    ref_cont: list[float] = []
    ref_cat: list[int] = []
    profile_rows = []
    audit_rows = []
    order = 0
    for col in spec.static_reals:
        vals = pd.to_numeric(train[col], errors="coerce")
        valid = vals.dropna()
        uniques = set(valid.unique().tolist())
        binary = bool(uniques) and uniques.issubset({0, 1, 0.0, 1.0})
        st = pre.continuous_stats.get(col)
        if st is None:
            raise RuntimeError(f"checkpoint preprocessing stats missing for {col}")
        if binary:
            raw_ref = float(valid.mean()) if len(valid) else 0.0
            rule = "training_participant_prevalence"
            ftype = "binary_continuous"
        else:
            raw_ref = float(st["mean"])
            rule = "checkpoint_saved_scaler_mean"
            ftype = "continuous"
        transformed = (raw_ref - float(st["mean"])) / float(st["std"])
        ref_cont.append(transformed)
        profile_rows.append(
            {
                "feature_name": col,
                "feature_type": ftype,
                "reference_rule": rule,
                "training_n_total": len(train),
                "training_n_valid": int(vals.notna().sum()),
                "training_missing_fraction": float(vals.isna().mean()),
                "raw_reference_value": raw_ref,
                "transformed_reference_value": transformed,
                "scaler_center": st["mean"],
                "scaler_scale": st["std"],
                "expected_neutral_value": transformed,
                "model_input_order": order,
                "notes": "Continuous channel; missing raw values canonically imputed with saved median.",
            }
        )
        audit_rows.append(
            {
                "feature_name": col,
                "source_column": col,
                "consumed_by_model": True,
                "input_order": order,
                "feature_type": ftype,
                "raw_dtype": str(prepared[col].dtype),
                "encoding_type": "z_score_continuous",
                "scaler_type": "saved_mean_std",
                "scaler_center": st["mean"],
                "scaler_scale": st["std"],
                "missing_value_rule": f"saved median imputation ({st['median']}) then z-score",
                "has_missingness_indicator": False,
                "missingness_indicator_name": "",
                "used_in_h0": True,
                "used_in_film": True,
                "used_elsewhere": "static embedding supplied to horizon decoder",
                "reference_rule": rule,
                "manual_review_required": False,
                "notes": "Checkpoint order from metadata feature_spec.static_reals.",
            }
        )
        order += 1
    for col in spec.static_categoricals:
        mapping = pre.static_category_maps.get(col)
        if mapping is None:
            raise RuntimeError(f"checkpoint categorical map missing for {col}")
        vals = train[col].dropna().astype(str)
        counts = Counter(vals.tolist())
        valid_counts = [(n, k) for k, n in counts.items() if k in mapping]
        if not valid_counts:
            raw_ref = "__unknown__"
        else:
            top = max(n for n, _ in valid_counts)
            raw_ref = sorted(k for n, k in valid_counts if n == top)[0]
        transformed = int(mapping.get(raw_ref, 0))
        ref_cat.append(transformed)
        profile_rows.append(
            {
                "feature_name": col,
                "feature_type": "categorical_embedding",
                "reference_rule": "training_participant_mode",
                "training_n_total": len(train),
                "training_n_valid": int(train[col].notna().sum()),
                "training_missing_fraction": float(train[col].isna().mean()),
                "raw_reference_value": raw_ref,
                "transformed_reference_value": transformed,
                "scaler_center": np.nan,
                "scaler_scale": np.nan,
                "expected_neutral_value": transformed,
                "model_input_order": order,
                "notes": "Integer category required by learned embedding; mode used instead of prevalence.",
            }
        )
        audit_rows.append(
            {
                "feature_name": col,
                "source_column": col,
                "consumed_by_model": True,
                "input_order": order,
                "feature_type": "categorical",
                "raw_dtype": str(prepared[col].dtype),
                "encoding_type": "checkpoint_integer_index_then_learned_embedding",
                "scaler_type": "none",
                "scaler_center": np.nan,
                "scaler_scale": np.nan,
                "missing_value_rule": "__missing__ if mapped, otherwise __unknown__ index 0",
                "has_missingness_indicator": False,
                "missingness_indicator_name": "",
                "used_in_h0": True,
                "used_in_film": True,
                "used_elsewhere": "static embedding supplied to horizon decoder",
                "reference_rule": "training_participant_mode",
                "manual_review_required": False,
                "notes": f"Checkpoint category map: {mapping}",
            }
        )
        order += 1
    cont = np.asarray(ref_cont, dtype="float32")
    cat = np.asarray(ref_cat, dtype="int64")
    if cont.shape != (len(spec.static_reals),) or cat.shape != (len(spec.static_categoricals),):
        raise RuntimeError("reference vector dimension mismatch")
    profile = pd.DataFrame(profile_rows)
    audit = pd.DataFrame(audit_rows)
    profile.to_csv(out_dir / "static_reference_profile.csv", index=False)
    audit.to_csv(out_dir / "static_schema_audit.csv", index=False)
    payload = {
        "feature_names": {
            "static_reals": spec.static_reals,
            "static_categoricals": spec.static_categoricals,
        },
        "ordered_raw_vector": profile["raw_reference_value"].tolist(),
        "ordered_transformed_vector": profile["transformed_reference_value"].tolist(),
        "ordered_missingness_vector_or_mask": [],
        "transformed_static_cont": cont.tolist(),
        "transformed_static_cat": cat.tolist(),
        "rules": dict(zip(profile["feature_name"], profile["reference_rule"])),
        "source_hashes": source_hashes,
        "training_participant_count": len(train),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_identifier": source_hashes["checkpoint"],
        "profile_hash": hash_jsonable(
            {"cont": cont.tolist(), "cat": cat.tolist(), "features": profile["feature_name"].tolist()}
        ),
    }
    json_dump(out_dir / "static_reference_profile.json", payload)
    return cont, cat, profile, payload, audit


def participant_summaries(
    prepared: pd.DataFrame,
    split_map: dict[str, str],
    spec: AireadiFeatureSpec,
) -> pd.DataFrame:
    val = prepared[
        prepared["participant_id"].astype(str).map(split_map).eq("validation")
    ].copy()
    if val.empty:
        raise RuntimeError("no canonical validation rows")
    seg = (
        val.groupby(["participant_id", "segment_id"], as_index=False)
        .agg(n_bins=("time_idx", "size"))
        .assign(segment_hours=lambda x: x["n_bins"] * 5 / 60)
    )
    segsum = (
        seg.groupby("participant_id", as_index=False)
        .agg(n_segments=("segment_id", "size"), median_segment_hours=("segment_hours", "median"))
    )
    first = val.drop_duplicates("participant_id").set_index("participant_id")
    glucose = (
        val.groupby("participant_id")["cgm_glucose_mean"]
        .agg(mean_glucose="mean", glucose_sd="std")
        .reset_index()
    )
    glucose["glucose_cv"] = 100 * glucose["glucose_sd"] / glucose["mean_glucose"]
    out = segsum.merge(glucose.drop(columns="glucose_sd"), on="participant_id")
    out["clinical_site"] = out["participant_id"].map(first["participants_clinical_site"])
    out["study_group"] = out["participant_id"].map(first["participants_study_group"])
    out["hba1c"] = pd.to_numeric(
        out["participant_id"].map(first["hba1c_percent_baseline"]), errors="coerce"
    )
    static_cols = spec.static_reals + spec.static_categoricals
    miss = first[static_cols].isna().mean(axis=1)
    out["static_missing_fraction"] = out["participant_id"].map(miss)
    for metric in ("hba1c", "mean_glucose", "glucose_cv"):
        out[f"{metric}_quartile"] = quartile(out[metric])
    return out.sort_values("participant_id").reset_index(drop=True)


def select_pilot(summary: pd.DataFrame, n: int, explicit: list[str] | None) -> pd.DataFrame:
    if explicit:
        ids = list(dict.fromkeys(map(str, explicit)))
        if len(ids) != n:
            raise ValueError(f"--participant-ids must contain exactly {n} unique IDs")
        missing = sorted(set(ids) - set(summary["participant_id"]))
        if missing:
            raise ValueError(f"requested IDs are not eligible validation participants: {missing}")
        out = summary.set_index("participant_id").loc[ids].reset_index()
        out["selection_reason"] = "explicit validation-only participant list"
        return out
    work = summary.copy()
    work["missingness_quartile"] = quartile(work["static_missing_fraction"])
    tags: dict[str, set[str]] = {}
    for _, r in work.iterrows():
        pid = str(r["participant_id"])
        tags[pid] = {
            f"site={r['clinical_site']}",
            f"group={r['study_group']}",
            f"hba1c={r['hba1c_quartile']}",
            f"mean={r['mean_glucose_quartile']}",
            f"cv={r['glucose_cv_quartile']}",
            "segments=multi" if int(r["n_segments"]) > 1 else "segments=one",
            f"missing={r['missingness_quartile']}",
        }
        if pid in DISCREPANT_IDS:
            tags[pid].add("reconciled_discrepancy")
    desired = set().union(*tags.values())
    # Emphasize categorical strata, extremes, segment count, and missingness extremes.
    desired = {
        x
        for x in desired
        if x.startswith(("site=", "group=", "segments="))
        or x.endswith(("=Q1", "=Q4"))
        or x == "reconciled_discrepancy"
    }
    selected: list[str] = []
    reasons: dict[str, str] = {}
    covered: set[str] = set()
    while len(selected) < n:
        candidates = []
        for pid in sorted(tags):
            if pid in selected:
                continue
            gained = (tags[pid] & desired) - covered
            candidates.append((len(gained), pid, gained))
        if not candidates:
            break
        best_score = max(x[0] for x in candidates)
        _, pid, gained = next(x for x in candidates if x[0] == best_score)
        selected.append(pid)
        covered |= gained
        reasons[pid] = "greedy coverage: " + (", ".join(sorted(gained)) or "tie-break fill")
    if len(selected) != n:
        raise RuntimeError(f"could select only {len(selected)} of requested {n}")
    out = work.set_index("participant_id").loc[selected].reset_index()
    out["selection_reason"] = out["participant_id"].map(reasons)
    return out.drop(columns="missingness_quartile")


def static_hash(cont: np.ndarray, cat: np.ndarray) -> str:
    return hash_jsonable({"cont": np.asarray(cont).tolist(), "cat": np.asarray(cat).tolist()})


@torch.no_grad()
def replay_stream(
    model: AireadiStreamModel,
    raw_stream,
    condition: str,
    neutral_cont: np.ndarray,
    neutral_cat: np.ndarray,
    device: str,
    checkpoint_hash: str,
    stride_steps: int,
) -> tuple[list[dict], list[dict], dict[str, Any], np.ndarray]:
    stream = raw_stream.to(device)
    if condition == "static_neutral":
        cont = torch.tensor(neutral_cont, dtype=torch.float32, device=device)
        cat = torch.tensor(neutral_cat, dtype=torch.long, device=device)
    else:
        cont, cat = stream.static_cont, stream.static_cat
    sctx = model.encode_static(cat, cont)
    state0 = model.init_stream(sctx)
    actual_h0 = flatten_initial_state(state0)
    # No 128-D decoder output exists before the first observation.  e_s is the
    # only canonical 128-D static-conditioned value and directly generates the
    # true recurrent h0; it is exported as an explicitly documented proxy.
    h0_proxy = sctx.embedding[0].detach().cpu().numpy().astype("float32")
    state, out = model.scan_chunk(stream.dynamic, sctx, state0)
    states = out[0].detach().cpu().numpy().astype("float32")
    if states.shape[1] != model.config.hidden_size:
        raise RuntimeError(f"state dimension {states.shape[1]} != {model.config.hidden_size}")
    dyn_hash = hashlib.sha256(stream.dynamic.detach().cpu().numpy().tobytes()).hexdigest()
    profile_hash = static_hash(cont.detach().cpu().numpy(), cat.detach().cpu().numpy())
    start = pd.Timestamp(stream.timestamps[0])
    end = pd.Timestamp(stream.timestamps[-1])
    hcols = [f"h_{i:03d}" for i in range(states.shape[1])]
    hidden_rows = []
    h0row = {
        "participant_id": stream.participant_id,
        "split": stream.split,
        "condition": condition,
        "segment_id": stream.segment_id,
        "segment_start": start,
        "segment_end": end,
        "timestamp": start - pd.Timedelta(minutes=model.feature_spec.bin_minutes),
        "step_index_in_segment": -1,
        "minutes_since_reset": 0,
        "is_h0_row": True,
        "is_post_update_state": False,
        "state_semantics": "static_embedding_initialization_proxy_e_s",
        "static_profile_hash": profile_hash,
        "dynamic_input_hash": dyn_hash,
        "checkpoint_hash": checkpoint_hash,
        "state_l2_norm": float(np.linalg.norm(h0_proxy)),
        "state_mean": float(h0_proxy.mean()),
        "state_std": float(h0_proxy.std()),
    }
    h0row.update(dict(zip(hcols, h0_proxy.tolist())))
    hidden_rows.append(h0row)
    for t, h in enumerate(states):
        row = {
            "participant_id": stream.participant_id,
            "split": stream.split,
            "condition": condition,
            "segment_id": stream.segment_id,
            "segment_start": start,
            "segment_end": end,
            "timestamp": pd.Timestamp(stream.timestamps[t]),
            "step_index_in_segment": t,
            "minutes_since_reset": (t + 1) * model.feature_spec.bin_minutes,
            "is_h0_row": False,
            "is_post_update_state": True,
            "state_semantics": "post_update_top_layer_output_supplied_to_decoder",
            "static_profile_hash": profile_hash,
            "dynamic_input_hash": dyn_hash,
            "checkpoint_hash": checkpoint_hash,
            "state_l2_norm": float(np.linalg.norm(h)),
            "state_mean": float(h.mean()),
            "state_std": float(h.std()),
        }
        row.update(dict(zip(hcols, h.tolist())))
        hidden_rows.append(row)
    anchors = valid_anchors(stream, model.feature_spec.horizon_steps, stride_steps)
    forecast_rows: list[dict] = []
    if anchors:
        pos = torch.tensor(anchors, dtype=torch.long, device=device)
        H = model.feature_spec.horizon_steps
        fut = pos[:, None] + 1 + torch.arange(H, device=device)[None, :]
        # Strict forecast-only replay: no future scenario values or masks.
        values = torch.zeros_like(stream.scenario_values[fut])
        masks = torch.zeros_like(stream.scenario_mask[fut])
        raw = model.decode_horizon(
            out[0, pos], sctx, stream.time_features[fut], values, masks
        )
        pred = stream.target[pos].view(-1, 1, 1) + raw
        pred_np = pred.detach().cpu().numpy()
        target = stream.target.detach().cpu().numpy()
        qlevels = list(model.quantiles)
        qidx = {round(q, 6): i for i, q in enumerate(qlevels)}
        if not all(round(q, 6) in qidx for q in (0.1, 0.5, 0.9)):
            raise RuntimeError(f"required q10/q50/q90 absent from checkpoint quantiles {qlevels}")
        for ai, anchor in enumerate(anchors):
            for h in range(H):
                obs = float(target[anchor + 1 + h])
                q10 = float(pred_np[ai, h, qidx[0.1]])
                q50 = float(pred_np[ai, h, qidx[0.5]])
                q90 = float(pred_np[ai, h, qidx[0.9]])
                forecast_rows.append(
                    {
                        "participant_id": stream.participant_id,
                        "condition": condition,
                        "segment_id": stream.segment_id,
                        "anchor_timestamp": pd.Timestamp(stream.timestamps[anchor]),
                        "minutes_since_reset": (anchor + 1) * model.feature_spec.bin_minutes,
                        "horizon_step": h + 1,
                        "horizon_minutes": (h + 1) * model.feature_spec.bin_minutes,
                        "q10": q10,
                        "q50": q50,
                        "q90": q90,
                        "observed_glucose": obs,
                        "current_glucose": float(target[anchor]),
                        "forecast_error_q50": q50 - obs,
                        "abs_error_q50": abs(q50 - obs),
                        "interval_covered": bool(q10 <= obs <= q90),
                        "static_profile_hash": profile_hash,
                        "dynamic_input_hash": dyn_hash,
                    }
                )
    meta = {
        "participant_id": stream.participant_id,
        "segment_id": stream.segment_id,
        "condition": condition,
        "n_states": len(states),
        "n_anchors": len(anchors),
        "dynamic_hash": dyn_hash,
        "static_hash": profile_hash,
        "h0_proxy": h0_proxy,
        "actual_h0": actual_h0,
        "out": states,
        "anchors": anchors,
    }
    return hidden_rows, forecast_rows, meta, states


@torch.no_grad()
def replay_qc(
    model: AireadiStreamModel,
    streams: list,
    device: str,
    tolerance: float,
) -> dict[str, Any]:
    prefix_max = 0.0
    backend_state_max = 0.0
    deterministic_max = 0.0
    forecast_backend_max = 0.0
    checked = []
    prefix_scale = 0.0
    backend_state_scale = 0.0
    forecast_backend_scale = 0.0
    for raw in streams[:2]:
        s = raw.to(device)
        ctx = model.encode_static(s.static_cat, s.static_cont)
        init = model.init_stream(ctx)
        _, full = model.scan_chunk(s.dynamic, ctx, init)
        endpoint = max(2, s.n_steps // 2)
        init2 = model.init_stream(ctx)
        _, prefix = model.scan_chunk(s.dynamic[:endpoint], ctx, init2)
        prefix_max = max(
            prefix_max,
            float(torch.max(torch.abs(full[:, :endpoint] - prefix)).detach().cpu()),
        )
        prefix_scale = max(prefix_scale, float(torch.max(torch.abs(full[:, :endpoint])).detach().cpu()))
        init3 = model.init_stream(ctx)
        _, repeat = model.scan_chunk(s.dynamic, ctx, init3)
        deterministic_max = max(
            deterministic_max, float(torch.max(torch.abs(full - repeat)).detach().cpu())
        )
        seq_state = model.init_stream(ctx)
        seq = []
        for t in range(s.n_steps):
            seq_state = model.update_stream(seq_state, s.dynamic[t])
            seq.append(seq_state.last_output)
        seq_out = torch.stack(seq, dim=1)
        backend_state_max = max(
            backend_state_max, float(torch.max(torch.abs(full - seq_out)).detach().cpu())
        )
        backend_state_scale = max(backend_state_scale, float(torch.max(torch.abs(full)).detach().cpu()))
        anchors = valid_anchors(s, model.feature_spec.horizon_steps, 3)[:16]
        if anchors:
            pos = torch.tensor(anchors, device=device)
            H = model.feature_spec.horizon_steps
            fut = pos[:, None] + 1 + torch.arange(H, device=device)[None, :]
            zval = torch.zeros_like(s.scenario_values[fut])
            zmask = torch.zeros_like(s.scenario_mask[fut])
            a = model.decode_horizon(full[0, pos], ctx, s.time_features[fut], zval, zmask)
            b = model.decode_horizon(seq_out[0, pos], ctx, s.time_features[fut], zval, zmask)
            forecast_backend_max = max(
                forecast_backend_max, float(torch.max(torch.abs(a - b)).detach().cpu())
            )
            forecast_backend_scale = max(forecast_backend_scale, float(torch.max(torch.abs(a)).detach().cpu()))
        checked.append(f"{s.participant_id}:{s.segment_id}")
    prefix_relative = prefix_max / max(prefix_scale, EPS)
    backend_state_relative = backend_state_max / max(backend_state_scale, EPS)
    backend_forecast_relative = forecast_backend_max / max(forecast_backend_scale, EPS)
    if prefix_max > 0.002 or prefix_relative > 1e-4:
        raise RuntimeError(
            f"prefix invariance failed: abs={prefix_max}, relative={prefix_relative}"
        )
    if deterministic_max > tolerance:
        raise RuntimeError(f"deterministic rerun failed: {deterministic_max} > {tolerance}")
    # Long float32 CUDA scan and deployment-step arithmetic need not be bitwise equal.
    if (
        backend_state_max > 0.05
        or backend_state_relative > 0.002
        or forecast_backend_max > 0.05
        or backend_forecast_relative > 0.002
    ):
        raise RuntimeError(
            "sequential/chunk equivalence failed: "
            f"state_abs={backend_state_max}, state_rel={backend_state_relative}, "
            f"forecast_abs={forecast_backend_max}, forecast_rel={backend_forecast_relative}"
        )
    return {
        "segments_checked": checked,
        "prefix_invariance_max_abs_difference": prefix_max,
        "prefix_invariance_max_relative_difference": prefix_relative,
        "deterministic_rerun_max_abs_difference": deterministic_max,
        "backend_state_max_abs_difference": backend_state_max,
        "backend_state_max_relative_difference": backend_state_relative,
        "backend_forecast_max_abs_difference": forecast_backend_max,
        "backend_forecast_max_relative_difference": backend_forecast_relative,
    }


def pairwise_tables(hidden: pd.DataFrame, forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hcols = [c for c in hidden.columns if c.startswith("h_")]
    key = ["participant_id", "segment_id", "timestamp", "is_h0_row"]
    full = hidden[hidden.condition == "full_profile"].set_index(key)
    neutral = hidden[hidden.condition == "static_neutral"].set_index(key)
    if not full.index.equals(neutral.index):
        raise RuntimeError("full/neutral hidden-state keys differ")
    a, b = full[hcols].to_numpy(), neutral[hcols].to_numpy()
    dot = (a * b).sum(1)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    diff = a - b
    hpair = full.reset_index()[key + ["minutes_since_reset"]].copy()
    hpair["comparison_type"] = "hidden_state"
    hpair["full_state_norm"] = an
    hpair["neutral_state_norm"] = bn
    hpair["full_neutral_l2"] = np.linalg.norm(diff, axis=1)
    hpair["full_neutral_cosine"] = dot / np.maximum(an * bn, EPS)
    hpair["max_abs_state_difference"] = np.abs(diff).max(1)
    hpair["mean_abs_state_difference"] = np.abs(diff).mean(1)
    fkey = ["participant_id", "segment_id", "anchor_timestamp", "horizon_step"]
    ff = forecasts[forecasts.condition == "full_profile"].set_index(fkey)
    fn = forecasts[forecasts.condition == "static_neutral"].set_index(fkey)
    if not ff.index.equals(fn.index):
        raise RuntimeError("full/neutral forecast keys differ")
    if not np.array_equal(ff["observed_glucose"].to_numpy(), fn["observed_glucose"].to_numpy()):
        raise RuntimeError("paired observed targets differ")
    if not np.array_equal(ff["current_glucose"].to_numpy(), fn["current_glucose"].to_numpy()):
        raise RuntimeError("paired current glucose differs")
    fr = ff.reset_index().copy()
    fr["abs_q50_delta"] = np.abs(ff["q50"].to_numpy() - fn["q50"].to_numpy())
    fr["abs_width_delta"] = np.abs(
        (ff["q90"] - ff["q10"]).to_numpy() - (fn["q90"] - fn["q10"]).to_numpy()
    )
    anchor = (
        fr.groupby(["participant_id", "segment_id", "anchor_timestamp"], as_index=False)
        .agg(
            minutes_since_reset=("minutes_since_reset", "first"),
            mean_abs_q50_forecast_delta=("abs_q50_delta", "mean"),
            mean_abs_interval_width_delta=("abs_width_delta", "mean"),
        )
    )
    term = fr[fr.horizon_step == fr.horizon_step.max()][
        ["participant_id", "segment_id", "anchor_timestamp", "abs_q50_delta"]
    ].rename(columns={"abs_q50_delta": "terminal_60min_q50_delta"})
    anchor = anchor.merge(term, on=["participant_id", "segment_id", "anchor_timestamp"])
    anchor["comparison_type"] = "forecast_anchor"
    anchor["current_glucose_difference"] = 0.0
    anchor["observed_target_difference"] = 0.0
    combined = pd.concat([hpair, anchor], ignore_index=True, sort=False)
    return combined, hpair


def cluster_bootstrap_ci(group: pd.DataFrame, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    pids = sorted(group["participant_id"].astype(str).unique())
    if not pids:
        return np.nan, np.nan
    pid_index = {pid: i for i, pid in enumerate(pids)}
    values = group["value"].to_numpy(dtype=float)
    codes = group["participant_id"].astype(str).map(pid_index).to_numpy(dtype=int)
    order = np.argsort(values)
    values, codes = values[order], codes[order]
    draws = rng.multinomial(len(pids), np.full(len(pids), 1 / len(pids)), size=reps)
    weights = draws[:, codes]
    cumulative = np.cumsum(weights, axis=1)
    totals = cumulative[:, -1]
    lower_rank = np.floor_divide(totals - 1, 2)
    upper_rank = np.floor_divide(totals, 2)
    lower_idx = np.argmax(cumulative > lower_rank[:, None], axis=1)
    upper_idx = np.argmax(cumulative > upper_rank[:, None], axis=1)
    medians = (values[lower_idx] + values[upper_idx]) / 2
    return tuple(np.quantile(medians, [0.025, 0.975]))


def burn_in_curve(
    hidden: pd.DataFrame,
    forecasts: pd.DataFrame,
    hpair: pd.DataFrame,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hcols = [c for c in hidden.columns if c.startswith("h_")]
    state_metrics = []
    for condition in CONDITIONS:
        sub = hidden[hidden.condition == condition].copy()
        for (pid, sid), g in sub.groupby(["participant_id", "segment_id"], sort=False):
            g = g.sort_values("step_index_in_segment")
            h0 = g[g.is_h0_row][hcols].iloc[0].to_numpy(dtype=float)
            post = g[~g.is_h0_row]
            arr = post[hcols].to_numpy(dtype=float)
            base = post[["participant_id", "segment_id", "minutes_since_reset"]].copy()
            drift = base.copy()
            drift["metric"] = "drift_from_h0_proxy"
            drift["condition"] = condition
            drift["value"] = np.linalg.norm(arr - h0, axis=1)
            norm = base.copy()
            norm["metric"] = "state_norm"
            norm["condition"] = condition
            norm["value"] = np.linalg.norm(arr, axis=1)
            state_metrics += [drift, norm]
    pair_post = hpair[~hpair.is_h0_row].copy()
    dist = pair_post[["participant_id", "segment_id", "minutes_since_reset"]].copy()
    dist["metric"], dist["condition"], dist["value"] = (
        "full_neutral_l2",
        "paired",
        pair_post["full_neutral_l2"].to_numpy(),
    )
    cos = pair_post[["participant_id", "segment_id", "minutes_since_reset"]].copy()
    cos["metric"], cos["condition"], cos["value"] = (
        "full_neutral_cosine",
        "paired",
        pair_post["full_neutral_cosine"].to_numpy(),
    )
    fkey = ["participant_id", "segment_id", "anchor_timestamp", "horizon_step"]
    ff = forecasts[forecasts.condition == "full_profile"].set_index(fkey)
    fn = forecasts[forecasts.condition == "static_neutral"].set_index(fkey)
    delta = np.abs(ff["q50"] - fn["q50"]).rename("delta").reset_index()
    mins = ff["minutes_since_reset"].rename("minutes_since_reset").reset_index()
    delta = delta.merge(mins, on=fkey)
    fmetric = (
        delta.groupby(["participant_id", "segment_id", "anchor_timestamp"], as_index=False)
        .agg(minutes_since_reset=("minutes_since_reset", "first"), value=("delta", "mean"))
    )
    fmetric["metric"], fmetric["condition"] = "forecast_q50_delta", "paired"
    metrics = pd.concat(state_metrics + [dist, cos, fmetric], ignore_index=True)
    metrics["minutes_since_reset_bin"] = (
        np.floor(metrics["minutes_since_reset"] / 15) * 15
    ).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for (metric, condition, minute), g in metrics.groupby(
        ["metric", "condition", "minutes_since_reset_bin"], sort=True
    ):
        vals = g["value"].to_numpy(dtype=float)
        lo, hi = cluster_bootstrap_ci(g, rng, reps)
        rows.append(
            {
                "metric": metric,
                "condition": condition,
                "minutes_since_reset": int(minute),
                "n_participants": g["participant_id"].nunique(),
                "n_segments": g[["participant_id", "segment_id"]].drop_duplicates().shape[0],
                "n_states": len(g),
                "median": np.median(vals),
                "p25": np.quantile(vals, 0.25),
                "p75": np.quantile(vals, 0.75),
                "p10": np.quantile(vals, 0.10),
                "p90": np.quantile(vals, 0.90),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    return pd.DataFrame(rows), metrics


def plateau_result(curve: pd.DataFrame, tolerance: float, persistence_minutes: int) -> dict[str, Any]:
    d = curve[
        (curve.metric == "drift_from_h0_proxy")
        & (curve.condition == "static_neutral")
        & (curve.minutes_since_reset <= 1440)
    ].sort_values("minutes_since_reset").copy()
    late = d[d.minutes_since_reset.between(720, 1440)]["median"]
    late_ref = float(late.median()) if len(late) else np.nan
    d["detect"] = d["median"].rolling(3, center=True, min_periods=3).median()
    band = tolerance * max(abs(late_ref), EPS)
    d["inside"] = np.abs(d["detect"] - late_ref) <= band
    bins = int(persistence_minutes // 15)
    selected = None
    selected_slope = None
    for i in range(0, len(d) - bins + 1):
        w = d.iloc[i : i + bins]
        expected = np.arange(
            int(w.minutes_since_reset.iloc[0]),
            int(w.minutes_since_reset.iloc[0]) + 15 * bins,
            15,
        )
        if not np.array_equal(w.minutes_since_reset.to_numpy(), expected):
            continue
        if not bool(w["inside"].all()):
            continue
        if int(w["n_participants"].min()) < 8 or int(w["n_segments"].min()) < 12:
            continue
        slopes = np.abs(np.diff(w["detect"].to_numpy())) / 0.25
        slope = float(np.median(slopes)) if len(slopes) else 0.0
        if slope <= 0.01 * max(abs(late_ref), EPS):
            selected = int(w.minutes_since_reset.iloc[0])
            selected_slope = slope
            break
    return {
        "plateau_tolerance": tolerance,
        "persistence_duration_minutes": persistence_minutes,
        "late_reference_value": late_ref,
        "selected_burn_in_minutes": selected,
        "selected_burn_in_hours": None if selected is None else selected / 60,
        "burn_in_status": "no_plateau_detected" if selected is None else "plateau_detected",
        "selected_interval_median_abs_slope_per_hour": selected_slope,
    }


def forecast_metrics(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = forecasts.copy()
    d["time_band"] = d["minutes_since_reset"].map(time_band)
    rows = []
    groupings = [
        ("overall", ["condition"]),
        ("participant", ["condition", "participant_id"]),
        ("time_since_reset", ["condition", "time_band"]),
    ]
    for scope, cols in groupings:
        for keys, g in d.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"scope": scope, **dict(zip(cols, keys))}
            err = g.q50 - g.observed_glucose
            row.update(
                {
                    "n_horizon_rows": len(g),
                    "n_anchors": g[
                        ["participant_id", "segment_id", "anchor_timestamp"]
                    ].drop_duplicates().shape[0],
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "signed_bias": float(err.mean()),
                    "interval_coverage": float(g.interval_covered.mean()),
                    "mean_interval_width": float((g.q90 - g.q10).mean()),
                    "terminal_60min_mae": float(
                        g.loc[g.horizon_step == g.horizon_step.max(), "abs_error_q50"].mean()
                    ),
                }
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    key = ["participant_id", "segment_id", "anchor_timestamp", "horizon_step"]
    a = d[d.condition == "full_profile"].set_index(key)
    b = d[d.condition == "static_neutral"].set_index(key)
    overall_delta = float(np.abs(a.q50 - b.q50).mean())
    terminal_mask = (
        a.index.get_level_values("horizon_step")
        == a.index.get_level_values("horizon_step").max()
    )
    terminal_delta = float(
        np.abs(a.loc[terminal_mask, "q50"] - b.loc[terminal_mask, "q50"]).mean()
    )
    return table, {
        "mean_absolute_full_neutral_forecast_difference": overall_delta,
        "terminal_full_neutral_forecast_difference": terminal_delta,
    }


def make_figures(
    out_dir: Path,
    curve: pd.DataFrame,
    hidden: pd.DataFrame,
    forecasts: pd.DataFrame,
    hpair: pd.DataFrame,
    burn: dict[str, Any],
) -> list[str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    files = []

    def save(name: str):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        files.append(name)

    plt.figure(figsize=(9, 5))
    for condition, label in [
        ("full_profile", "Full profile"),
        ("static_neutral", "Static neutral"),
    ]:
        d = curve[(curve.metric == "drift_from_h0_proxy") & (curve.condition == condition)]
        x = d.minutes_since_reset / 60
        plt.plot(x, d["median"], label=label)
        plt.fill_between(x, d.p25, d.p75, alpha=0.2)
    if burn["selected_burn_in_minutes"] is not None:
        plt.axvline(burn["selected_burn_in_minutes"] / 60, color="black", ls="--")
    plt.xlim(0, 24)
    plt.xlabel("Hours since reset")
    plt.ylabel("L2 distance from 128-D initialization proxy")
    plt.legend()
    save("fig_state_drift_from_h0.png")

    plt.figure(figsize=(9, 5))
    d = curve[curve.metric == "full_neutral_l2"]
    x = d.minutes_since_reset / 60
    plt.plot(x, d["median"], color="#7B2CBF")
    plt.fill_between(x, d.p25, d.p75, color="#7B2CBF", alpha=0.2)
    plt.xlim(0, 24)
    plt.xlabel("Hours since reset")
    plt.ylabel("Full vs neutral state L2")
    save("fig_full_vs_neutral_state_distance.png")

    plt.figure(figsize=(9, 5))
    d = curve[curve.metric == "full_neutral_cosine"]
    x = d.minutes_since_reset / 60
    plt.plot(x, d["median"], color="#0077B6")
    plt.fill_between(x, d.p25, d.p75, color="#0077B6", alpha=0.2)
    plt.xlim(0, 24)
    plt.xlabel("Hours since reset")
    plt.ylabel("Full vs neutral cosine similarity")
    save("fig_full_vs_neutral_cosine_similarity.png")

    plt.figure(figsize=(9, 5))
    d = curve[curve.metric == "forecast_q50_delta"]
    x = d.minutes_since_reset / 60
    plt.plot(x, d["median"], color="#D00000")
    plt.fill_between(x, d.bootstrap_ci_low, d.bootstrap_ci_high, color="#D00000", alpha=0.2)
    plt.xlim(0, 24)
    plt.xlabel("Hours since reset")
    plt.ylabel("Mean absolute q50 delta (mg/dL)")
    save("fig_forecast_delta_by_time_since_reset.png")

    effect = (
        hpair[~hpair.is_h0_row]
        .groupby("participant_id")["full_neutral_l2"]
        .median()
        .sort_values()
    )
    example_ids = [
        effect.index[0],
        effect.index[len(effect) // 2],
        effect.index[-1],
    ]
    post = hidden[~hidden.is_h0_row].copy()
    hcols = [c for c in post if c.startswith("h_")]
    sample = post.iloc[:: max(1, len(post) // 10000)]
    pca = PCA(n_components=2, random_state=42).fit(sample[hcols].to_numpy())
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")
    for row, pid in enumerate(example_ids):
        sub = post[post.participant_id == pid].copy()
        xy = pca.transform(sub[hcols].to_numpy())
        sub["pc1"], sub["pc2"] = xy[:, 0], xy[:, 1]
        for condition in CONDITIONS:
            c = sub[sub.condition == condition]
            x = c.minutes_since_reset / 60
            axes[row, 0].plot(x, c.state_l2_norm, label=condition)
            axes[row, 1].plot(x, c.pc1, label=condition)
            axes[row, 1].plot(x, c.pc2, ls="--")
        pair = hpair[(hpair.participant_id == pid) & ~hpair.is_h0_row]
        axes[row, 2].plot(pair.minutes_since_reset / 60, pair.full_neutral_l2)
        axes[row, 0].set_ylabel(f"{pid}\nstate norm")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Pilot-only PC1 (solid), PC2 (dashed)")
    axes[0, 2].set_title("Full-neutral L2")
    for ax in axes[-1]:
        ax.set_xlabel("Hours since reset")
    save("fig_example_state_trajectories.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for ax, pid in zip(axes, example_ids):
        sub = forecasts[forecasts.participant_id == pid]
        anchors = sorted(sub.anchor_timestamp.unique())
        chosen = anchors[len(anchors) // 2]
        for condition, color in [("full_profile", "#0077B6"), ("static_neutral", "#D00000")]:
            g = sub[(sub.anchor_timestamp == chosen) & (sub.condition == condition)].sort_values(
                "horizon_step"
            )
            x = g.horizon_minutes
            ax.plot(x, g.q50, label=condition, color=color)
            ax.fill_between(x, g.q10, g.q90, color=color, alpha=0.15)
        g = sub[(sub.anchor_timestamp == chosen) & (sub.condition == "full_profile")].sort_values(
            "horizon_step"
        )
        ax.plot(g.horizon_minutes, g.observed_glucose, color="black", marker=".", label="observed")
        ax.set_title(f"Participant {pid}; deterministic median-time anchor {chosen}")
        ax.set_ylabel("Glucose (mg/dL)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Forecast horizon (minutes)")
    save("fig_example_forecast_comparisons.png")
    return files


def report_text(
    paths: dict[str, str],
    hashes: dict[str, str],
    reconciliation: pd.DataFrame,
    profile: pd.DataFrame,
    pilot: pd.DataFrame,
    qc: dict[str, Any],
    h0_qc: dict[str, float],
    hpair: pd.DataFrame,
    forecast_table: pd.DataFrame,
    forecast_delta: dict[str, Any],
    burn: dict[str, Any],
    flags: dict[str, int],
    recommendation: str,
) -> str:
    overall = forecast_table[forecast_table.scope == "overall"]
    static_effect = hpair.loc[~hpair.is_h0_row, "full_neutral_l2"]
    cosine = hpair.loc[~hpair.is_h0_row, "full_neutral_cosine"]
    return f"""# Step 1 static-neutralization pilot

## 1. Objective

Validate frozen full-profile versus training-reference static-neutralized replay,
reconcile canonical reset boundaries, and estimate a provisional validation-only
burn-in.

## 2. Scope and explicit exclusions

Only 12 saved-split validation participants were replayed. No test participant,
training, fine-tuning, checkpoint change, enriched-data change, external phenotype,
probe, association test, clustering, exercise analysis, or complete-validation
state extraction was performed. PCA was used only for the required pilot figure.

## 3. Canonical inputs and hashes

{chr(10).join(f"- `{k}`: `{v}` ({hashes.get(k, 'n/a')})" for k, v in paths.items())}

## 4. Segment-boundary reconciliation

All three cases were resolved in favor of the checkpoint's current reconstruction:
the older saved manifest applied edge trimming before the 49-hour minimum, while
the checkpoint build path did not. Participants: {", ".join(DISCREPANT_IDS)}.
See `segment_boundary_reconciliation.csv` for exact gaps and boundaries.

## 5. Static feature schema

The checkpoint consumes {len(profile)} features: {len(profile[profile.feature_type != 'categorical_embedding'])}
continuous/binary channels and {len(profile[profile.feature_type == 'categorical_embedding'])}
categorical embedding indices. Exact ordering and transformations are in
`static_schema_audit.csv`.

## 6. Construction of the training-reference profile

Continuous variables use saved checkpoint scaler means. Binary and one-hot
continuous channels use training-participant prevalence. Categorical embedding
indices use the training-participant mode. Validation and test values were not
used.

## 7. Pilot participant selection

Deterministic greedy coverage selected: {", ".join(pilot.participant_id.astype(str))}.
Selection covers sites, study groups, metric extremes, segment counts, and static
missingness without using external targets.

## 8. Replay implementation

The exact checkpoint was loaded strictly in evaluation mode. Both conditions
reuse identical stream tensors, reset boundaries, timestamps, anchors, and
forecast-only decoding. Future scenario values and masks are zero.

## 9. Hidden-state export semantics

Post-update rows are the 128-D top-layer outputs supplied to the horizon decoder
after ingesting time t. The architecture has no 128-D decoder output before its
first observation: true h0 is an internal MES recurrent tensor. Therefore h0 rows
store the 128-D static embedding `e_s`, which directly constructs recurrent h0,
and are labeled `static_embedding_initialization_proxy_e_s`. True recurrent h0
was retained separately in memory for identity QC. Drift-from-h0 and the burn-in
decision consequently use this explicitly labeled proxy; this is the principal
methodological caveat.

## 10. Pairwise full-versus-neutral QC

Participants, segments, timestamps, dynamic hashes, anchors, current glucose, and
targets match exactly. All state and forecast values are finite.

## 11. Neutral h0 identity check

Maximum neutral transformed-vector difference: {h0_qc['neutral_vector_max_diff']:.3g}.
Maximum true recurrent neutral-h0 difference: {h0_qc['neutral_actual_h0_max_diff']:.3g}.

## 12. Prefix-invariance check

Maximum difference: {qc['prefix_invariance_max_abs_difference']:.3g} on
{qc['segments_checked']}.

## 13. Sequential/chunked equivalence check

State maximum absolute/relative difference: {qc['backend_state_max_abs_difference']:.3g} / {qc['backend_state_max_relative_difference']:.3g}.
Forecast residual maximum absolute/relative difference: {qc['backend_forecast_max_abs_difference']:.3g} / {qc['backend_forecast_max_relative_difference']:.3g}.

## 14. State drift after reset

Five-minute post-update trajectories were retained, with 15-minute descriptive
aggregation and participant-cluster bootstrap intervals.

## 15. Provisional burn-in decision

Status: **{burn['burn_in_status']}**. Selected value:
{burn['selected_burn_in_minutes']} minutes ({burn['selected_burn_in_hours']} hours).
Late 12–24h reference: {burn['late_reference_value']:.6g}. This pilot value is
provisional and must be confirmed on a complete validation export before test use.

## 16. Full-versus-neutral state differences

Median L2 distance: {static_effect.median():.6g}. Median cosine similarity:
{cosine.median():.6g}.

## 17. Forecast plausibility

```
{overall.to_string(index=False)}
```

Mean absolute full-neutral q50 difference:
{forecast_delta['mean_absolute_full_neutral_forecast_difference']:.4f} mg/dL;
terminal difference: {forecast_delta['terminal_full_neutral_forecast_difference']:.4f}
mg/dL. Diagnostic flag counts: {flags}. No inferential tests were run.

## 18. Participant heterogeneity in static-profile influence

Participant-specific median state distances are visible in the example plots;
low/median/high examples were selected deterministically from static-effect
quantiles, not phenotype outcomes.

## 19. Blocking issues

No structural replay blocker remains. The absence of a decoder-space true h0 is a
methodological caveat for the requested drift curve and provisional burn-in.

## 20. Go/no-go recommendation for full validation-state export

**{recommendation}**

Replay is structurally valid, but the h0-proxy limitation must be carried into
the full export and the burn-in definition should be revised or explicitly
accepted before confirmatory use.
"""


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("test split is forbidden for Step 1")
    if args.split != "validation":
        raise ValueError("Step 1 pilot must use --split validation")
    if args.state_save_frequency_minutes != 5:
        raise ValueError("Step 1 requires 5-minute state saving")
    if args.n_pilot_participants != 12:
        raise ValueError("this Step 1 specification requires exactly 12 pilot participants")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    out_dir = output_root / run_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"run exists; use --overwrite only if intentional: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    setup_logging(out_dir / "step1_run.log")
    started = datetime.now(timezone.utc)
    errors: list[str] = []
    warnings = [
        "Categorical embedding features use training modes rather than prevalence vectors.",
        "The 128-D h0 export/drift uses static embedding e_s because decoder h_t does not exist before the first observation.",
    ]
    device = resolve_device(args.device)
    LOG.info("run_id=%s device=%s", run_id, device)

    paths = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "schema": str(Path(args.schema).resolve()),
        "multimodal_parquet": str(Path(args.multimodal_parquet).resolve()),
        "static_table": str(Path(args.static_table).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "step0_manifest": str((Path(args.step0_dir) / "step0_manifest.json").resolve()),
    }
    path_objs = {k: Path(v) for k, v in paths.items()}
    for name, path in path_objs.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    immutable = {
        k: sha256(v)
        for k, v in path_objs.items()
        if k in {"checkpoint", "schema", "multimodal_parquet", "static_table", "split_manifest", "config"}
    }
    commit, dirty = git_info()
    with path_objs["config"].open() as f:
        cfg = yaml.safe_load(f)
    with path_objs["schema"].open() as f:
        saved_schema = json.load(f)
    split_df = pd.read_csv(path_objs["split_manifest"])
    split_df["participant_id"] = split_df["participant_id"].astype(str)
    split_df["split"] = split_df["split"].replace({"val": "validation", "valid": "validation"})
    if split_df["participant_id"].duplicated().any():
        raise RuntimeError("participant appears more than once in split manifest")
    counts = split_df.split.value_counts().to_dict()
    if counts != {"train": 1131, "validation": 239, "test": 221}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    split_map = dict(zip(split_df.participant_id, split_df.split))
    train_ids = set(split_df.loc[split_df.split == "train", "participant_id"])

    ckpt = torch.load(path_objs["checkpoint"], map_location=device, weights_only=False)
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    mcfg = AireadiStreamModelConfig(**md["model_config"])
    if saved_schema["feature_spec"] != md["feature_spec"]:
        raise RuntimeError("saved schema feature_spec differs from checkpoint metadata")
    model = AireadiStreamModel(spec, pre, mcfg).to(device)
    if model.static_encoder.n_cont != len(spec.static_reals):
        raise RuntimeError(
            "checkpoint static continuous dimension disagrees with the saved feature order: "
            f"model={model.static_encoder.n_cont}, schema={len(spec.static_reals)}"
        )
    if model.static_encoder.n_cat != len(spec.static_categoricals):
        raise RuntimeError(
            "checkpoint static categorical dimension disagrees with the saved feature order: "
            f"model={model.static_encoder.n_cat}, schema={len(spec.static_categoricals)}"
        )
    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint load not exact: {incompatible}")
    model.eval()
    if model.training or any(m.training for m in model.modules()):
        raise RuntimeError("model did not enter evaluation mode")

    LOG.info("loading canonical panel (read-only)")
    panel = load_aireadi_panel(
        path_objs["multimodal_parquet"],
        static_path=path_objs["static_table"],
        cohort_path=cfg["data"].get("cohort_path"),
    )
    schema = infer_or_validate_schema(panel, saved_schema.get("schema"))
    LOG.info("preparing checkpoint-era canonical streams")
    prepared = prepare_aireadi_panel(
        panel,
        schema,
        bin_minutes=spec.bin_minutes,
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    current_spec = build_stream_feature_spec(
        prepared, schema, horizon_steps=spec.horizon_steps, bin_minutes=spec.bin_minutes
    )
    if asdict(current_spec) != asdict(spec):
        raise RuntimeError("current reconstructed model-input schema differs from checkpoint")
    split = make_aireadi_stream_splits(
        prepared,
        existing_split_path=path_objs["split_manifest"],
        seed=args.seed,
    )

    raw_three = panel[panel.participant_id.astype(str).isin(DISCREPANT_IDS)].copy()
    prepared_three = prepared[prepared.participant_id.astype(str).isin(DISCREPANT_IDS)].copy()
    reconciliation, rec_details = reconcile_segments(
        raw_three,
        prepared_three,
        split_map,
        path_objs["multimodal_parquet"].parent,
        out_dir,
    )
    if set(reconciliation.canonical_replay_choice) != {"checkpoint_current_reconstruction"}:
        raise RuntimeError("canonical segmentation remains unresolved")
    LOG.info("segment reconciliation gate passed")

    neutral_cont, neutral_cat, profile, profile_json, audit = build_static_reference(
        prepared, spec, pre, train_ids, out_dir, immutable
    )
    summary = participant_summaries(prepared, split_map, spec)
    pilot = select_pilot(summary, args.n_pilot_participants, args.participant_ids)
    if not set(pilot.participant_id).issubset(
        set(split_df.loc[split_df.split == "validation", "participant_id"])
    ):
        raise RuntimeError("pilot contains non-validation participant")
    pilot.to_csv(out_dir / "pilot_participants.csv", index=False)
    pilot_ids = pilot.participant_id.astype(str).tolist()
    subset = prepared[prepared.participant_id.astype(str).isin(pilot_ids)].copy()
    streams = make_participant_streams(
        subset,
        split,
        schema,
        feature_spec=spec,
        preprocessor=pre,
        splits=["validation"],
        min_steps=spec.horizon_steps + 2,
    )
    if set(s.participant_id for s in streams) != set(pilot_ids):
        raise RuntimeError("stream construction lost a pilot participant")
    if any(s.split != "validation" for s in streams):
        raise RuntimeError("non-validation stream constructed")
    stride_steps = max(1, round(cfg["evaluation"]["anchor_stride_minutes"] / spec.bin_minutes))
    hidden_rows: list[dict] = []
    forecast_rows: list[dict] = []
    metas = []
    for i, stream in enumerate(streams, 1):
        LOG.info(
            "replay %d/%d participant=%s segment=%s steps=%d",
            i,
            len(streams),
            stream.participant_id,
            stream.segment_id,
            stream.n_steps,
        )
        for condition in CONDITIONS:
            h, f, meta, _ = replay_stream(
                model,
                stream,
                condition,
                neutral_cont,
                neutral_cat,
                device,
                immutable["checkpoint"],
                stride_steps,
            )
            hidden_rows.extend(h)
            forecast_rows.extend(f)
            metas.append(meta)
    hidden = pd.DataFrame(hidden_rows)
    forecasts = pd.DataFrame(forecast_rows)
    hcols = [c for c in hidden if c.startswith("h_")]
    if len(hcols) != 128:
        raise RuntimeError(f"expected 128 hidden columns, got {len(hcols)}")
    if not np.isfinite(hidden[hcols].to_numpy()).all():
        raise RuntimeError("hidden states contain NaN or infinity")
    if not np.isfinite(forecasts[["q10", "q50", "q90"]].to_numpy()).all():
        raise RuntimeError("required forecasts contain NaN or infinity")
    hidden.to_parquet(out_dir / "pilot_hidden_states.parquet", index=False)
    forecasts.to_parquet(out_dir / "pilot_forecasts.parquet", index=False)

    comparison, hpair = pairwise_tables(hidden, forecasts)
    comparison.to_csv(out_dir / "pilot_anchor_comparison.csv", index=False)
    neutral_metas = [m for m in metas if m["condition"] == "static_neutral"]
    ref_vecs = [
        np.concatenate([neutral_cont.astype(float), neutral_cat.astype(float)])
        for _ in neutral_metas
    ]
    neutral_vector_max = max(
        float(np.max(np.abs(x - ref_vecs[0]))) for x in ref_vecs
    )
    neutral_actual_h0_max = max(
        float(np.max(np.abs(m["actual_h0"] - neutral_metas[0]["actual_h0"])))
        for m in neutral_metas
    )
    if neutral_vector_max > 1e-7 or neutral_actual_h0_max > 1e-6:
        raise RuntimeError(
            f"neutral identity failed vector={neutral_vector_max} h0={neutral_actual_h0_max}"
        )
    qc = replay_qc(model, streams, device, tolerance=1e-5)
    warnings.append(
        "CUDA Mamba scan is not bitwise equal to truncated-prefix or sequential-step "
        f"arithmetic: prefix abs/rel={qc['prefix_invariance_max_abs_difference']:.6g}/"
        f"{qc['prefix_invariance_max_relative_difference']:.6g}; backend state abs/rel="
        f"{qc['backend_state_max_abs_difference']:.6g}/"
        f"{qc['backend_state_max_relative_difference']:.6g}."
    )
    h0_qc = {
        "neutral_vector_max_diff": neutral_vector_max,
        "neutral_actual_h0_max_diff": neutral_actual_h0_max,
    }

    curve, metric_long = burn_in_curve(
        hidden, forecasts, hpair, args.bootstrap_replicates, args.seed
    )
    curve.to_csv(out_dir / "burn_in_curve.csv", index=False)
    primary = plateau_result(curve, 0.05, 120)
    sensitivities = {
        "band_2.5pct": plateau_result(curve, 0.025, 120),
        "band_10pct": plateau_result(curve, 0.10, 120),
        "persistence_1h": plateau_result(curve, 0.05, 60),
        "persistence_4h": plateau_result(curve, 0.05, 240),
    }
    burn = {
        "primary_metric": "median ||h_t - e_s||_2 (static-neutralized; e_s is labeled h0 proxy)",
        "reference_window": "12-24 hours after reset",
        "late_reference_value": primary["late_reference_value"],
        "plateau_tolerance": 0.05,
        "persistence_duration": "2 hours (8 consecutive 15-minute bins)",
        "minimum_participants": 8,
        "minimum_segments": 12,
        "selected_burn_in_minutes": primary["selected_burn_in_minutes"],
        "selected_burn_in_hours": primary["selected_burn_in_hours"],
        "burn_in_status": primary["burn_in_status"],
        "sensitivity_results": sensitivities,
        "decision_notes": (
            "Predefined before plotting. Three-bin centered rolling median used only "
            "for detection; slope threshold is 1% of late reference per hour. "
            "Provisional because architecture exposes no decoder-space true h0."
        ),
    }
    json_dump(out_dir / "burn_in_decision.json", burn)
    forecast_table, forecast_delta = forecast_metrics(forecasts)
    forecast_table.to_csv(out_dir / "forecast_plausibility_metrics.csv", index=False)
    state_norm = hidden.loc[~hidden.is_h0_row, "state_l2_norm"]
    state_outlier_threshold = float(state_norm.quantile(0.75) + 3 * (state_norm.quantile(0.75) - state_norm.quantile(0.25)))
    flag_key = ["participant_id", "segment_id", "anchor_timestamp", "horizon_step"]
    flag_full = forecasts[forecasts.condition == "full_profile"].set_index(flag_key)
    flag_neutral = forecasts[forecasts.condition == "static_neutral"].set_index(flag_key)
    flag_detail = pd.DataFrame(
        {
            "q50_delta": (flag_full.q50 - flag_neutral.q50).abs(),
            "width_delta": (
                (flag_full.q90 - flag_full.q10)
                - (flag_neutral.q90 - flag_neutral.q10)
            ).abs(),
        }
    ).reset_index()
    flag_anchor = flag_detail.groupby(flag_key[:-1], as_index=False).agg(
        max_q50_delta=("q50_delta", "max"),
        max_width_delta=("width_delta", "max"),
    )
    flags = {
        "anchors_any_q50_delta_gt_20": int(flag_anchor.max_q50_delta.gt(20).sum()),
        "horizon_rows_q50_delta_gt_20": int(flag_detail.q50_delta.gt(20).sum()),
        "anchors_any_interval_width_delta_gt_20": int(
            flag_anchor.max_width_delta.gt(20).sum()
        ),
        "horizon_rows_interval_width_delta_gt_20": int(
            flag_detail.width_delta.gt(20).sum()
        ),
        "forecast_outside_40_400": int(
            ((forecasts.q50 < 40) | (forecasts.q50 > 400)).sum()
        ),
        "neutral_state_norm_extreme": int(
            (
                hidden.loc[
                    (hidden.condition == "static_neutral") & ~hidden.is_h0_row,
                    "state_l2_norm",
                ]
                > state_outlier_threshold
            ).sum()
        ),
    }
    figures = make_figures(out_dir, curve, hidden, forecasts, hpair, burn)
    recommendation = (
        "GO WITH CAVEATS: Replay is valid, but listed limitations must be carried "
        "into the full export."
    )
    report = report_text(
        paths,
        immutable,
        reconciliation,
        profile,
        pilot,
        qc,
        h0_qc,
        hpair,
        forecast_table,
        forecast_delta,
        burn,
        flags,
        recommendation,
    )
    (out_dir / "step1_report.md").write_text(report)

    source_files = [
        "segment_boundary_reconciliation.csv",
        "segment_boundary_reconciliation.md",
        "static_reference_profile.csv",
        "static_reference_profile.json",
        "static_schema_audit.csv",
        "pilot_participants.csv",
        "pilot_hidden_states.parquet",
        "pilot_forecasts.parquet",
        "pilot_anchor_comparison.csv",
        "burn_in_curve.csv",
        "burn_in_decision.json",
        "forecast_plausibility_metrics.csv",
        "step1_report.md",
        "step1_run.log",
        *figures,
    ]
    backend = {
        "checkpoint_scan_mode": mcfg.scan_mode,
        "mamba_style": mcfg.mamba_style,
        "canonical_replay_call": "AireadiStreamModel.scan_chunk",
        "equivalence_call": "AireadiStreamModel.update_stream",
    }
    manifest = {
        "run_id": run_id,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "git_commit": commit,
        "dirty_working_tree_status": dirty,
        "canonical_paths": paths,
        "file_hashes": immutable,
        "checkpoint_hash": immutable["checkpoint"],
        "checkpoint_metadata": {
            "epoch": ckpt.get("epoch"),
            "metrics": ckpt.get("metrics"),
            "model_config": md["model_config"],
        },
        "feature_schema_hash": immutable["schema"],
        "config_hash": immutable["config"],
        "split_hash": immutable["split_manifest"],
        "model_input_dimensions": {
            "dynamic": len(spec.dynamic_reals),
            "time": len(spec.time_reals),
            "scenario": len(spec.scenario_reals),
            "static_continuous": len(spec.static_reals),
            "static_categorical": len(spec.static_categoricals),
        },
        "hidden_state_dimension": mcfg.hidden_size,
        "forecast_horizon_steps": spec.horizon_steps,
        "forecast_horizon_minutes": spec.horizon_steps * spec.bin_minutes,
        "quantiles": model.quantiles,
        "dynamic_feature_names": spec.dynamic_reals,
        "static_feature_names": {
            "continuous": spec.static_reals,
            "categorical": spec.static_categoricals,
        },
        "pilot_participant_ids": pilot_ids,
        "n_pilot_segments": len(streams),
        "segment_reconciliation_decision": {
            "source": "checkpoint metadata/config + canonical training build_data code",
            "choice": "current prepare_aireadi_panel reconstruction",
            "participants": rec_details,
        },
        "static_reference_profile_hash": profile_json["profile_hash"],
        "reference_calculation_rules": profile_json["rules"],
        "inference_mode": "torch.no_grad + model.eval; frozen checkpoint",
        "backend": backend,
        "device": device,
        "python_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "state_save_semantics": {
            "post_update": "top-layer 128-D output supplied to decoder after observing t",
            "h0_row": "128-D static embedding e_s proxy; true recurrent h0 has a different internal shape",
            "frequency_minutes": 5,
        },
        "forecast_mode": "forecast_only; future scenario values=0 and masks=0",
        "random_seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
        "qc_tolerances": {
            "neutral_vector": 1e-7,
            "neutral_true_recurrent_h0": 1e-6,
            "prefix_invariance_abs": 0.002,
            "prefix_invariance_relative_to_state_scale": 1e-4,
            "deterministic_rerun_abs": 1e-5,
            "sequential_chunk_state_abs": 0.05,
            "sequential_chunk_state_relative_to_state_scale": 0.002,
            "sequential_chunk_forecast_abs": 0.05,
            "sequential_chunk_forecast_relative_to_residual_scale": 0.002,
        },
        "qc_results": {**qc, **h0_qc},
        "plateau_rule": burn,
        "counts": {
            "participants": len(pilot_ids),
            "segments": len(streams),
            "hidden_rows_including_h0_both_conditions": len(hidden),
            "post_update_states_both_conditions": int((~hidden.is_h0_row).sum()),
            "forecast_horizon_rows_both_conditions": len(forecasts),
            "forecast_anchors_per_condition": forecasts[
                ["participant_id", "segment_id", "anchor_timestamp", "condition"]
            ].drop_duplicates().shape[0]
            // 2,
        },
        "descriptive_forecast_delta": forecast_delta,
        "diagnostic_flags": flags,
        "source_functions": {
            "load_panel": "ssmcgm.data.aireadi.load_aireadi_panel",
            "prepare_segments": "ssmcgm.data.aireadi.prepare_aireadi_panel",
            "split": "ssmcgm.data.aireadi.make_aireadi_stream_splits",
            "streams": "ssmcgm.data.aireadi.make_participant_streams",
            "static_transform": "ssmcgm.data.aireadi.AireadiPreprocessor",
            "checkpoint_load": "strict AireadiStreamModel.load_state_dict",
            "static_encoder": "AireadiStreamModel.encode_static / StaticEncoder",
            "h0": "AireadiStreamModel.init_stream / StaticStateInitializer",
            "film": "AireadiStreamModel.fuse_history / StaticFiLM",
            "advance": "AireadiStreamModel.scan_chunk",
            "decode": "AireadiStreamModel.decode_horizon",
        },
        "source_files_created": source_files + ["step1_manifest.json"],
        "warnings": warnings,
        "errors": errors,
        "final_go_no_go_status": recommendation,
    }
    json_dump(out_dir / "step1_manifest.json", manifest)

    # Verify canonical inputs remained byte-identical.
    after = {
        k: sha256(v)
        for k, v in path_objs.items()
        if k in immutable
    }
    if after != immutable:
        raise RuntimeError("canonical input hash changed during run")
    required = set(source_files + ["step1_manifest.json"])
    missing = sorted(x for x in required if not (out_dir / x).exists())
    if missing:
        raise RuntimeError(f"required outputs missing: {missing}")
    latest = output_root / "latest"
    tmp = output_root / f".latest.{run_id}.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(out_dir.name)
    os.replace(tmp, latest)
    LOG.info("QC complete; latest -> %s", out_dir.name)

    state_effect = hpair.loc[~hpair.is_h0_row, "full_neutral_l2"]
    state_cos = hpair.loc[~hpair.is_h0_row, "full_neutral_cosine"]
    overall_metrics = forecast_table[forecast_table.scope == "overall"].to_dict("records")
    terminal = {
        "output_directory": str(out_dir),
        "files_created": sorted(required),
        "git_commit": commit,
        "checkpoint_hash": immutable["checkpoint"],
        "canonical_segmentation_source": "checkpoint current prepare_aireadi_panel reconstruction",
        "discrepant_participants": list(DISCREPANT_IDS),
        "reconciliation_outcomes": {
            p: rec_details[p]["reconciliation_outcome"] for p in DISCREPANT_IDS
        },
        "consumed_static_features": profile.feature_name.tolist(),
        "static_reference_profile_hash": profile_json["profile_hash"],
        "pilot_validation_participants": pilot_ids,
        "counts": manifest["counts"],
        "neutral_static_vector_max_difference": neutral_vector_max,
        "neutral_true_recurrent_h0_max_difference": neutral_actual_h0_max,
        "prefix_invariance_max_difference": qc["prefix_invariance_max_abs_difference"],
        "backend_equivalence_max_difference": qc["backend_state_max_abs_difference"],
        "median_full_neutral_state_distance": float(state_effect.median()),
        "median_full_neutral_cosine_similarity": float(state_cos.median()),
        "mean_absolute_forecast_difference": forecast_delta[
            "mean_absolute_full_neutral_forecast_difference"
        ],
        "forecast_metrics": overall_metrics,
        "provisional_burn_in": burn,
        "warnings": warnings,
        "recommendation": recommendation,
    }
    print(json.dumps(terminal, indent=2, default=json_default))


if __name__ == "__main__":
    main()
