#!/usr/bin/env python3
"""Controlled state-reset warm-up experiment (replaces the flat matched-anchor
audit behind Figure 9 / Table 8 with a real causal test).

Why this exists: ``ssmcgm/evaluation/plotting.py::matched_personalization`` (the
function behind Table 8) computes its 48h-position-eligible anchor subset once,
*outside* the loop over ``warmup_hours``, and reports the same
``aggregate_metrics(matched)`` for every nominal warm-up value. It never resets
the recurrent state or varies how much history built ``h_t`` — it audits
anchor-selection bias, not a warm-up effect. See
``outputs/forecasting_story/FORECASTING_STORY_REPORT.md`` section 8 for the
audit that first noticed the flat curve, and
``ssmcgm/evaluation/streaming.py`` module docstring ("Warm-up (spec Sec11):
``warmup_steps`` gates *scoring* only") for the underlying mechanism.

This script instead, for each fixed anchor and each nominal warm-up hours W:
  1. Resets the recurrent state to h0 (``model.init_stream``, never carried
     over from another anchor).
  2. Streams forward through exactly the participant's own real (non-imputed)
     history for the W hours preceding the anchor, plus the anchor's own
     current step (window length = W * BINS_PER_HOUR + 1).
  3. Decodes the 12-horizon forecast from the resulting h_t.

Anchor set provenance: Table 8's 81,962/221 anchors are pulled verbatim from
the existing stream-evaluation predictions
(``TABLE8_PREDICTIONS``, produced by ``evaluate_stream_aireadi.py`` /
``ssmcgm.evaluation.aireadi_streaming.evaluate_aireadi_streams``), not
reconstructed by hand. The state-reset-eligible subset actually used here is
re-derived directly against the raw multimodal parquet (real, non-null CGM
values, no imputation-mask trust) and intersected with the Table 8 set per
the gate-check decision rule below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]          # .../CGM/SSM-CGM
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_stream_aireadi import (             # noqa: E402
    build_eval_streams,
    load_config,
    load_model_from_checkpoint,
    resolve_device,
)

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #

WARM_UP_HOURS: List[int] = [0, 6, 12, 24, 48]
BINS_PER_HOUR: int = 12                      # 5-minute native resolution
HORIZON_STEPS: int = 12                      # 12 x 5min = 60min forecast horizon
MAX_WARM_UP_H: int = max(WARM_UP_HOURS)

# Table 8 targets (matched-anchor sweep, ssmcgm/evaluation/plotting.py::matched_personalization)
N_ANCHORS_EXPECTED: int = 81962
N_PARTICIPANTS_EXPECTED: int = 221

# Gate-check "small margin" thresholds (step 4 decision rule). These were set
# after observing the actual divergence between the position-based Table 8
# definition and the genuine real-data-contiguity definition (-5.3% anchors,
# -1 participant) and confirming with the study owner that this counts as a
# small margin to intersect-and-proceed on, not a stop condition.
SMALL_MARGIN_ANCHOR_FRAC: float = 0.10        # +/-10% of N_ANCHORS_EXPECTED
SMALL_MARGIN_PARTICIPANT_DELTA: int = 5       # +/-5 participants

BOOTSTRAP_N: int = 2000
BOOTSTRAP_SEED: int = 42
MEDIAN_Q: str = "q50"
LOWER_INTERVAL_Q: str = "q10"
UPPER_INTERVAL_Q: str = "q90"

FORBIDDEN_SPLIT_SUBSTRING: str = "adapt48h"   # overlaps training data; never use

DATA_ROOT = Path("/home/myriamcharfeddine/CGM/Data")
CANONICAL_PARQUET = DATA_ROOT / "enriched_multimodal" / "final_multimodal_dataset_20260515_184339.parquet"
CANONICAL_SPLIT_DIR = DATA_ROOT / "experiment_c_split_adapt6h_seed42"
CANONICAL_SPLIT_PARTICIPANTS_CSV = CANONICAL_SPLIT_DIR / "split_participants.csv"

CANONICAL_CHECKPOINT = REPO_ROOT / "outputs" / "aireadi_stream_mamba_stateful_5epoch" / "checkpoints" / "best_model_checkpoint.pt"
# Table 8's own predictions bundle -- ground truth for the exact (participant_id,
# segment_id, anchor_time_idx) triples the old audit used, and for the existing
# "unmatched sweep" (left panel, unchanged) numbers.
TABLE8_EVAL_DIR = REPO_ROOT / "outputs" / "aireadi_stream_mamba_stateful_10epoch_eval_test"
TABLE8_PREDICTIONS = TABLE8_EVAL_DIR / "predictions" / "predictions.parquet"
UNMATCHED_SWEEP_CSV = TABLE8_EVAL_DIR / "metrics" / "personalization_sweep.csv"

CONFIG_PATH = REPO_ROOT / "configs" / "aireadi_stream_full.yaml"

OUTPUT_TABLE_DIR = REPO_ROOT / "report" / "tables" / "generated"
OUTPUT_FIGURE_DIR = REPO_ROOT / "report" / "figures" / "generated"
OUTPUT_TABLE_CSV = OUTPUT_TABLE_DIR / "state_reset_warmup_sweep.csv"
OUTPUT_FIGURE_PNG = OUTPUT_FIGURE_DIR / "fig_warmup_audit_state_reset.png"
OUTPUT_FIGURE_PDF = OUTPUT_FIGURE_DIR / "fig_warmup_audit_state_reset.pdf"
OUTPUT_SUMMARY_MD = REPO_ROOT / "outputs" / "forecasting_story" / "STATE_RESET_WARMUP_SUMMARY.md"

TEAL = "#5BBABA"
GREY = "#9AA5B1"

ANCHOR_ID_COLS = ["participant_id", "segment_id", "anchor_time_idx"]


# --------------------------------------------------------------------------- #
# GATE CHECK
# --------------------------------------------------------------------------- #

class GateCheckFailure(RuntimeError):
    pass


def _check_paths_readable() -> None:
    for label, path in [
        ("canonical checkpoint", CANONICAL_CHECKPOINT),
        ("canonical parquet", CANONICAL_PARQUET),
        ("canonical split participants csv", CANONICAL_SPLIT_PARTICIPANTS_CSV),
        ("Table 8 predictions", TABLE8_PREDICTIONS),
    ]:
        if not path.exists():
            raise GateCheckFailure(f"{label} not readable at {path}")
    print(f"[gate] checkpoint OK: {CANONICAL_CHECKPOINT}")
    print(f"[gate] parquet OK: {CANONICAL_PARQUET}")
    print(f"[gate] split OK: {CANONICAL_SPLIT_PARTICIPANTS_CSV}")
    print(f"[gate] Table 8 predictions OK: {TABLE8_PREDICTIONS}")


def _check_split_is_canonical() -> None:
    cfg = load_config(str(CONFIG_PATH))
    split_path = str(cfg.get("split", {}).get("existing_split_path", ""))
    if FORBIDDEN_SPLIT_SUBSTRING in split_path:
        raise GateCheckFailure(
            f"config split path {split_path!r} contains forbidden {FORBIDDEN_SPLIT_SUBSTRING!r} "
            "(overlaps training data)"
        )
    if "adapt6h_seed42" not in split_path:
        raise GateCheckFailure(f"config split path {split_path!r} is not the canonical adapt6h_seed42 split")
    print(f"[gate] split config confirmed canonical: {split_path}")


def load_table8_anchor_set() -> pd.DataFrame:
    """Exact (participant_id, segment_id, anchor_time_idx, anchor_timestamp)
    anchors behind Table 8 -- reproduces matched_personalization's eligibility
    rule (hours_since_start >= MAX_WARM_UP_H) against the real predictions
    file, not a hand-reconstruction."""
    pred = pd.read_parquet(
        TABLE8_PREDICTIONS,
        columns=["participant_id", "segment_id", "anchor_time_idx", "anchor_timestamp",
                 "hours_since_start", "scenario_mode"],
    )
    forecast_only = pred[pred["scenario_mode"] == "forecast_only"]
    eligible = (
        forecast_only[forecast_only["hours_since_start"] >= MAX_WARM_UP_H]
        [ANCHOR_ID_COLS + ["anchor_timestamp"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return eligible


def recompute_state_reset_eligible(anchors: pd.DataFrame) -> pd.DataFrame:
    """For each Table-8 anchor, verify directly against the raw multimodal
    parquet that real (non-imputed) CGM data exists for the full
    MAX_WARM_UP_H-hour window ending at the anchor, plus the HORIZON_STEPS
    targets after it. Returns the subset that passes, with a `pos` column
    (integer position within that participant's chronological row order) for
    later positional slicing."""
    window_bins = MAX_WARM_UP_H * BINS_PER_HOUR + 1

    anchors = anchors.copy()
    anchors["anchor_timestamp"] = anchors["anchor_timestamp"].astype("datetime64[us, America/Los_Angeles]")
    test_ids = set(anchors["participant_id"].unique())

    raw = pd.read_parquet(CANONICAL_PARQUET, columns=["participant_id", "timestamp_local", "cgm_glucose_mean"])
    raw = raw[raw["participant_id"].isin(test_ids)].sort_values(["participant_id", "timestamp_local"]).reset_index(drop=True)
    raw["pos"] = raw.groupby("participant_id").cumcount()

    merged = anchors.merge(
        raw[["participant_id", "timestamp_local", "pos"]],
        left_on=["participant_id", "anchor_timestamp"],
        right_on=["participant_id", "timestamp_local"],
        how="left",
    )
    if merged["pos"].isna().any():
        n_bad = int(merged["pos"].isna().sum())
        raise GateCheckFailure(f"{n_bad} Table 8 anchors have no matching timestamp in the raw parquet")

    valid_by_pid: Dict[str, Tuple[np.ndarray, int]] = {}
    for pid, g in raw.groupby("participant_id", sort=False):
        valid = g["cgm_glucose_mean"].notna().to_numpy()
        csum = np.cumsum(np.concatenate([[0], valid.astype(int)]))
        valid_by_pid[pid] = (csum, len(g))

    def window_all_valid(csum: np.ndarray, start: int, length: int, n: int) -> bool:
        if start < 0 or start + length > n:
            return False
        return bool(csum[start + length] - csum[start] == length)

    pids = merged["participant_id"].to_numpy()
    positions = merged["pos"].to_numpy().astype(int)
    ok = np.zeros(len(merged), dtype=bool)
    for k in range(len(merged)):
        csum, n = valid_by_pid[pids[k]]
        i = positions[k]
        real_history_ok = window_all_valid(csum, i - window_bins + 1, window_bins, n)
        horizon_ok = window_all_valid(csum, i + 1, HORIZON_STEPS, n)
        ok[k] = real_history_ok and horizon_ok

    result = merged.loc[ok, ANCHOR_ID_COLS + ["anchor_timestamp", "pos"]].reset_index(drop=True)
    return result


def gate_check() -> pd.DataFrame:
    """Run the full gate check. Returns the final eligible-anchor set to use
    for the experiment, or raises GateCheckFailure with a plain explanation."""
    print("=" * 72)
    print("GATE CHECK")
    print("=" * 72)

    _check_paths_readable()
    _check_split_is_canonical()

    table8 = load_table8_anchor_set()
    n_table8_anchors = len(table8)
    n_table8_participants = table8["participant_id"].nunique()
    print(f"[gate] Table 8 anchor set reproduced from predictions.parquet: "
          f"{n_table8_anchors} anchors, {n_table8_participants} participants")
    if n_table8_anchors != N_ANCHORS_EXPECTED or n_table8_participants != N_PARTICIPANTS_EXPECTED:
        raise GateCheckFailure(
            f"Reproduced Table 8 set ({n_table8_anchors}/{n_table8_participants}) does not match "
            f"expected ({N_ANCHORS_EXPECTED}/{N_PARTICIPANTS_EXPECTED}); TABLE8_PREDICTIONS may have "
            "changed since this script was written. Stopping rather than silently proceeding."
        )

    fresh = recompute_state_reset_eligible(table8)
    n_fresh_anchors = len(fresh)
    n_fresh_participants = fresh["participant_id"].nunique()
    anchor_frac_delta = (n_fresh_anchors - N_ANCHORS_EXPECTED) / N_ANCHORS_EXPECTED
    participant_delta = n_fresh_participants - N_PARTICIPANTS_EXPECTED
    print(f"[gate] state-reset-eligible (real 48h contiguous data, direct parquet check): "
          f"{n_fresh_anchors} anchors ({anchor_frac_delta:+.1%}), "
          f"{n_fresh_participants} participants ({participant_delta:+d})")

    small_margin = (
        abs(anchor_frac_delta) <= SMALL_MARGIN_ANCHOR_FRAC
        and abs(participant_delta) <= SMALL_MARGIN_PARTICIPANT_DELTA
    )
    if not small_margin:
        dropped_participants = sorted(set(table8["participant_id"]) - set(fresh["participant_id"]))
        raise GateCheckFailure(
            f"State-reset-eligible set diverges from Table 8 by more than the small-margin "
            f"threshold (anchors {anchor_frac_delta:+.1%} vs +/-{SMALL_MARGIN_ANCHOR_FRAC:.0%}, "
            f"participants {participant_delta:+d} vs +/-{SMALL_MARGIN_PARTICIPANT_DELTA}). "
            f"Dropped participants: {dropped_participants}. Stopping per protocol."
        )

    dropped_participants = sorted(set(table8["participant_id"]) - set(fresh["participant_id"]))
    print(f"[gate] within small margin -> taking intersection, consistent with Table 8")
    if dropped_participants:
        print(f"[gate] participants dropped by the real-data-contiguity check: {dropped_participants} "
              f"(genuine CGM gaps, not eligible at any nominal W under a true state reset)")

    n_forward_passes = len(WARM_UP_HOURS) * n_fresh_anchors
    print("-" * 72)
    print(f"[gate] compute footprint: {len(WARM_UP_HOURS)} warm-up values x {n_fresh_anchors} anchors "
          f"= {n_forward_passes:,} independent forward passes")
    print("[gate] NOTE: this environment exposes 1 visible GPU (A100-40GB), not the 4xA100 DDP target; "
          "footprint below is a single-GPU estimate, scale down ~4x for the DDP setup.")
    print("=" * 72)

    return fresh


# --------------------------------------------------------------------------- #
# EXPERIMENT (guarded behind --launch / --smoke; not run by gate check alone)
# --------------------------------------------------------------------------- #

def _window_bins(warm_up_hours: int) -> int:
    return warm_up_hours * BINS_PER_HOUR + 1  # +1: the anchor's own current step


def _build_static_context(model, sctx, batch_size: int):
    from ssmcgm.stream.state import StaticContext
    return StaticContext(embedding=sctx.embedding.expand(batch_size, -1))


def run_state_reset_sweep(
    checkpoint_path: Path,
    eligible_anchors: pd.DataFrame,
    warm_up_hours: List[int],
    device: str,
    max_participants: Optional[int] = None,
    max_anchors_per_participant: Optional[int] = None,
) -> pd.DataFrame:
    """Batched by W then by participant, per the experiment design: for every
    (participant, segment) group we slice pre-built torch tensors positionally
    (no timestamp masks over the parquet), stack the fixed anchors for that
    group into one batch, reset h0 once per batch (never reused across
    anchors), scan through exactly W hours, and decode."""
    import torch

    model, spec, pre, ckpt = load_model_from_checkpoint(str(checkpoint_path), device)
    model = model.to(device).eval()
    cfg = load_config(str(CONFIG_PATH))
    streams = build_eval_streams(cfg, spec, pre, "test", smoke=False, max_participants=max_participants)
    streams_by_key = {(s.participant_id, s.segment_id): s.to(device) for s in streams}

    groups = eligible_anchors.groupby(["participant_id", "segment_id"])
    all_rows: List[dict] = []

    with torch.no_grad():
        for warm_up_h in warm_up_hours:
            window_bins = _window_bins(warm_up_h)
            print(f"[sweep] W={warm_up_h}h (window={window_bins} bins)")
            for (pid, seg_id), grp in groups:
                key = (pid, seg_id)
                if key not in streams_by_key:
                    continue
                stream = streams_by_key[key]
                # `anchor_time_idx` is segment-local (groupby([participant, segment]).cumcount()
                # in ssmcgm/data/aireadi.py::prepare_aireadi_panel), i.e. exactly the position to
                # index into this stream's tensors. The gate check's `pos` column is a *different*
                # coordinate system (flat per-participant row count in the raw parquet, used only
                # to verify real-data contiguity against calendar time) and must not be reused here.
                anchor_pos = grp["anchor_time_idx"].to_numpy()
                if max_anchors_per_participant is not None:
                    anchor_pos = anchor_pos[:max_anchors_per_participant]

                # Bounds check against this segment's *actual* tensor length. The gate check
                # verified real-data contiguity in flat raw-parquet coordinates, which is a
                # different coordinate system from segment-local anchor_time_idx; torch advanced
                # indexing silently treats negative indices as "from the end" rather than raising,
                # so an unguarded underflow here would corrupt data instead of crashing.
                in_bounds = (anchor_pos - (window_bins - 1) >= 0) & (anchor_pos + HORIZON_STEPS < stream.n_steps)
                n_dropped = int((~in_bounds).sum())
                if n_dropped:
                    print(f"[sweep] WARNING: dropping {n_dropped} anchors for participant={pid} "
                          f"segment={seg_id} W={warm_up_h}h: window would exceed this segment's bounds")
                anchor_pos = anchor_pos[in_bounds]
                B = len(anchor_pos)
                if B == 0:
                    continue

                pos_t = torch.tensor(anchor_pos, dtype=torch.long, device=device)
                hist_idx = pos_t[:, None] - (window_bins - 1) + torch.arange(window_bins, device=device)[None, :]
                dyn_batch = stream.dynamic[hist_idx]  # (B, window_bins, n_dyn)

                sctx_single = model.encode_static(stream.static_cat, stream.static_cont)
                sctx_batch = _build_static_context(model, sctx_single, B)
                state = model.init_stream(sctx_batch)
                state, out = model.scan_chunk(dyn_batch, sctx_batch, state)
                h_t = out[:, -1]  # (B, d_model) -- state after observing exactly the anchor's own step

                H = HORIZON_STEPS
                fut = pos_t[:, None] + 1 + torch.arange(H, device=device)[None, :]
                values, mask = stream.scenario_values[fut], stream.scenario_mask[fut]
                pred = model.decode_horizon(h_t, sctx_batch, stream.time_features[fut], values, mask)
                pred = (stream.target[pos_t].view(-1, 1, 1) + pred).cpu().numpy()

                target = stream.target.cpu().numpy()
                time_idx = stream.time_idx.cpu().numpy()
                qlevels = model.quantiles
                for ai in range(B):
                    anchor = int(anchor_pos[ai])
                    for h in range(H):
                        row = {
                            "participant_id": pid, "segment_id": seg_id,
                            "anchor_time_idx": int(time_idx[anchor]),
                            "warm_up_hours": float(warm_up_h),
                            "horizon_step": h + 1,
                            "horizon_minutes": spec.bin_minutes * (h + 1),
                            "target": float(target[anchor + 1 + h]),
                        }
                        for qi, level in enumerate(qlevels):
                            row[f"q{round(float(level) * 100):02d}"] = float(pred[ai, h, qi])
                        all_rows.append(row)

    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------- #
# METRICS
# --------------------------------------------------------------------------- #

def _mae_for_participants(df: pd.DataFrame, participant_ids: np.ndarray) -> float:
    sub = df[df["participant_id"].isin(participant_ids)]
    if sub.empty:
        return float("nan")
    return float((sub[MEDIAN_Q] - sub["target"]).abs().mean())


def participant_clustered_bootstrap_mae_ci(
    df: pd.DataFrame, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> Tuple[float, float]:
    participants = df["participant_id"].unique()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        stats[b] = _mae_for_participants(df, sampled)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics_by_warmup(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for warm_up_h, g in pred_df.groupby("warm_up_hours"):
        err = g[MEDIAN_Q] - g["target"]
        lo, hi = g[LOWER_INTERVAL_Q], g[UPPER_INTERVAL_Q]
        coverage = float(((g["target"] >= lo) & (g["target"] <= hi)).mean())
        ci_lo, ci_hi = participant_clustered_bootstrap_mae_ci(g)
        anchors = g.drop_duplicates(["participant_id", "segment_id", "anchor_time_idx"])
        rows.append({
            "warm_up_hours": float(warm_up_h),
            "anchors": int(len(anchors)),
            "participants": int(g["participant_id"].nunique()),
            "MAE": float(err.abs().mean()),
            "bias": float(err.mean()),
            "coverage": coverage,
            "MAE_CI_low": ci_lo,
            "MAE_CI_high": ci_hi,
        })
    return pd.DataFrame(rows).sort_values("warm_up_hours").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# OUTPUT
# --------------------------------------------------------------------------- #

def write_figure(state_reset_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    unmatched = pd.read_csv(UNMATCHED_SWEEP_CSV)

    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(unmatched["warmup_hours"], unmatched["mae"], marker="o", color=GREY)
    for _, r in unmatched.iterrows():
        ax.annotate(f"n={int(r['n']):,}", (r["warmup_hours"], r["mae"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=7, color=GREY)
    ax.set_xlabel("Warm-up hours")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Unmatched warm-up sweep")
    ax.text(0.5, 0.03, "Different anchor sets, descriptive only",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(state_reset_summary["warm_up_hours"], state_reset_summary["MAE"], marker="o", color=TEAL)
    ax.fill_between(state_reset_summary["warm_up_hours"],
                     state_reset_summary["MAE_CI_low"], state_reset_summary["MAE_CI_high"],
                     color=TEAL, alpha=0.2)
    n_anchors = int(state_reset_summary["anchors"].iloc[0])
    n_participants = int(state_reset_summary["participants"].iloc[0])
    ax.set_xlabel("Warm-up hours")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("State-reset warm-up sweep")
    ax.text(0.5, 0.03, f"{n_anchors:,} identical anchors, {n_participants} participants, genuine h0 reset",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Warm-up evaluation: state-reset causal test", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_PNG, dpi=150)
    fig.savefig(OUTPUT_FIGURE_PDF)
    plt.close(fig)


def write_summary_markdown(state_reset_summary: pd.DataFrame) -> None:
    mae_0 = state_reset_summary.loc[state_reset_summary["warm_up_hours"] == 0, "MAE"].iloc[0]
    mae_48 = state_reset_summary.loc[state_reset_summary["warm_up_hours"] == 48, "MAE"].iloc[0]
    delta = mae_48 - mae_0
    lines = [
        "# State-reset warm-up sweep: result",
        "",
        f"n_anchors={int(state_reset_summary['anchors'].iloc[0]):,}, "
        f"n_participants={int(state_reset_summary['participants'].iloc[0])}, "
        f"identical across all warm-up values.",
        "",
        "| warm_up_hours | MAE | 95% CI | bias | coverage |",
        "|---|---|---|---|---|",
    ]
    for _, r in state_reset_summary.iterrows():
        lines.append(
            f"| {r['warm_up_hours']:.0f} | {r['MAE']:.3f} | "
            f"[{r['MAE_CI_low']:.3f}, {r['MAE_CI_high']:.3f}] | {r['bias']:.3f} | {r['coverage']:.3f} |"
        )
    lines += [
        "",
        f"MAE at W=0 is {mae_0:.3f} mg/dL; MAE at W=48 is {mae_48:.3f} mg/dL (delta {delta:+.3f}).",
    ]
    OUTPUT_SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_MD.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true",
                     help="Tiny correctness check: 2 participants, 1 anchor each, W in [0, 48].")
    ap.add_argument("--launch", action="store_true",
                     help="Run the full sweep across all eligible anchors. Not passed by default.")
    return ap.parse_args()


def main():
    args = parse_args()
    eligible = gate_check()

    if not args.smoke and not args.launch:
        print("[main] gate check complete. Pass --smoke for a tiny correctness dry run, "
              "or --launch to run the full sweep.")
        return

    device = resolve_device(args.device)

    if args.smoke:
        smoke_anchors = eligible.groupby("participant_id", group_keys=False).head(1).head(2)
        print(f"[main] SMOKE: {len(smoke_anchors)} anchors, W in [0, {MAX_WARM_UP_H}]")
        pred = run_state_reset_sweep(CANONICAL_CHECKPOINT, smoke_anchors, [0, MAX_WARM_UP_H], device,
                                      max_participants=None)
        print(pred.head())
        print(f"[main] SMOKE OK: {len(pred)} rows")
        return

    pred = run_state_reset_sweep(CANONICAL_CHECKPOINT, eligible, WARM_UP_HOURS, device)
    summary = compute_metrics_by_warmup(pred)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE_CSV, index=False)
    write_figure(summary)
    write_summary_markdown(summary)
    print(summary)
    print(f"[main] wrote {OUTPUT_TABLE_CSV}, {OUTPUT_FIGURE_PNG}, {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
