"""
Stage 0: Model-free matched-control de-confounding of exercise-onset glucose effect (AI-READI).

Reads:
  - Panel data (raw 5-min bins) for glucose, HR, steps, TOD, valid flags
  - Detected episode table (v2, high-confidence) for exercise onsets

Produces:
  - outputs/study2_exercise_stage0/fig_deconfounded_s22.png / .pdf
  - outputs/study2_exercise_stage0/deconfounded_effect_targets.csv

Run (Steps 1-2 only, for review):
    python scripts/study2_exercise_stage0.py

Run (Steps 1-5, after reviewing pause output):
    python scripts/study2_exercise_stage0.py --confirm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# NAMED CONSTANTS
# ============================================================
G0_BINS        = [(130, 160), (160, 200), (200, 300)]
G0_BINS_FULL   = [(70, 100), (100, 130), (130, 160), (160, 200), (200, 300)]
HORIZON_MIN    = 60
HORIZON_STEPS  = HORIZON_MIN // 5          # 12 steps at 5-min resolution
NADIR_WINDOW   = (45, 75)                  # minutes after onset
NADIR_LO       = NADIR_WINDOW[0] // 5              # step lower bound (inclusive) = 9
NADIR_HI       = min(NADIR_WINDOW[1] // 5, HORIZON_STEPS)  # clipped to horizon = 12
ONSET_REF      = "detected_onset"
CONTROL_RATIO  = 30
G0_MATCH_TOL   = 5                         # mg/dL
TOD_MATCH_TOL  = 90                        # minutes
EXCLUSION_PRE  = 60                        # minutes before onset to exclude from controls
EXCLUSION_PRE_STEPS = EXCLUSION_PRE // 5  # steps
SUBGROUP       = "non_insulin"
SEED           = 0
BIN_MIN        = 5                         # minutes per step
AWAKE_THRESH   = 0.3                       # raw sleep_stage_awake fraction
N_BOOTSTRAP    = 2000
MIN_BOUTS_PER_BIN   = 20
MAX_CROSS_PATIENT_FRAC = 0.5
PRIMARY_STRAINS = ("moderate", "vigorous")

ROOT = Path(__file__).resolve().parents[1]

# ---- Paths ----
PANEL_PATH   = ROOT.parent / "Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
EPISODE_PATH = ROOT / "notebooks/outputs/exercise_episode_detection_v2/exercise_episodes_features.parquet"
STATIC_PATH  = ROOT.parent / "Data/enriched_multimodal/participant_static_features.parquet"
COHORT_PATH  = ROOT.parent / "Data/enriched_multimodal/cohort.csv"
OUT_DIR      = ROOT / "outputs/study2_exercise_stage0"

PANEL_COLS = [
    "participant_id",
    "timestamp_local",
    "cgm_glucose_mean",
    "heart_rate_mean",
    "activity_steps_per_min",
    "sleep_stage_awake",
]


# ============================================================
# 0. ARGUMENT PARSING + PATH CHECKS
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm", action="store_true",
                   help="Proceed past the matching-quality pause to compute curves and write files.")
    p.add_argument(
        "--g0-match-tol",
        type=float,
        default=G0_MATCH_TOL,
        help=f"Starting-glucose matching caliper in mg/dL. Default: {G0_MATCH_TOL}.",
    )
    p.add_argument(
        "--within-participant-only",
        action="store_true",
        help="Disable cross-participant fallback and use same-participant controls only.",
    )
    return p.parse_args()


def output_dir_for_run(g0_match_tol: float, within_participant_only: bool) -> Path:
    """Keep sensitivity outputs separate from the prespecified primary analysis."""
    tol_label = f"{g0_match_tol:g}".replace(".", "p")
    if within_participant_only:
        return OUT_DIR.parent / f"{OUT_DIR.name}_within_g0tol{tol_label}"
    if np.isclose(g0_match_tol, G0_MATCH_TOL):
        return OUT_DIR
    return OUT_DIR.parent / f"{OUT_DIR.name}_g0tol{tol_label}"


def check_paths(out_dir: Path) -> None:
    ok = True
    for label, path in [
        ("Panel data", PANEL_PATH),
        ("Episode table", EPISODE_PATH),
        ("Static features", STATIC_PATH),
        ("Cohort file", COHORT_PATH),
    ]:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if status == "MISSING":
            ok = False
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Output dir: {out_dir}")
    if not ok:
        sys.exit("ERROR: required input files missing.")


# ============================================================
# 1. LOAD DATA + BUILD PER-PARTICIPANT ARRAYS
# ============================================================

def load_and_build_arrays(non_insulin_pids: set[str], cohort_pids: set[str]):
    """
    Load panel parquet and filter to COHORT participants before sorting. The episode
    table's start_orig was computed from this same cohort-filtered, sorted, reset_index'd
    DataFrame, so we must replicate the exact same filter to get matching global offsets.

    Build per-pid NumPy arrays for non-insulin cohort participants only.
    """
    print("\n[1/5] Loading panel data ...")
    panel = pd.read_parquet(PANEL_PATH, columns=PANEL_COLS)
    panel["participant_id"] = panel["participant_id"].astype(str)
    panel["timestamp_local"] = pd.to_datetime(panel["timestamp_local"], utc=False)
    # Filter to cohort participants (matches detection notebook's cohort guard at cell 9)
    panel = panel[panel["participant_id"].isin(cohort_pids)].copy()
    panel.sort_values(["participant_id", "timestamp_local"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    print(f"    Panel rows (cohort participants): {len(panel):,}")
    print(f"    Non-insulin in cohort panel: {panel['participant_id'].isin(non_insulin_pids).sum():,} rows")

    # Time-of-day in minutes
    panel["tod_min"] = (
        panel["timestamp_local"].dt.hour * 60 + panel["timestamp_local"].dt.minute
    )

    # Global offsets per participant (from full sorted panel, matching start_orig indexing)
    pid_arr = panel["participant_id"].values
    unique_pids, pid_starts = np.unique(pid_arr, return_index=True)

    glc = panel["cgm_glucose_mean"].values.astype(float)
    hr  = panel["heart_rate_mean"].values.astype(float)
    stp = panel["activity_steps_per_min"].values.astype(float)
    awk = panel["sleep_stage_awake"].values.astype(float)
    tod = panel["tod_min"].values.astype(float)

    glucose_by_pid: dict[str, np.ndarray] = {}
    hr_by_pid: dict[str, np.ndarray] = {}
    steps_by_pid: dict[str, np.ndarray] = {}
    awake_by_pid: dict[str, np.ndarray] = {}
    tod_by_pid: dict[str, np.ndarray] = {}
    global_offset_by_pid: dict[str, int] = {}
    # valid_fwd[t] = True if glucose is non-NaN at t and all of t+1..t+HORIZON_STEPS
    valid_fwd_by_pid: dict[str, np.ndarray] = {}

    for i, pid in enumerate(unique_pids):
        # Only build full arrays for non-insulin participants (saves memory)
        s = int(pid_starts[i])
        e = int(pid_starts[i + 1]) if i + 1 < len(unique_pids) else len(panel)
        global_offset_by_pid[pid] = s   # always record offset; needed for start_orig mapping
        if pid not in non_insulin_pids:
            continue
        g = glc[s:e]
        # Vectorized valid_fwd: position t is valid if glucose finite from t to t+HORIZON_STEPS
        g_finite = np.isfinite(g)
        # Compute rolling all-finite over a window of (HORIZON_STEPS+1) using cumsum trick
        cs = np.concatenate([[0], np.cumsum(g_finite.astype(np.int32))])
        window_sum = cs[HORIZON_STEPS + 1:] - cs[:len(g) - HORIZON_STEPS]
        valid_fwd = np.zeros(len(g), dtype=bool)
        valid_fwd[:len(window_sum)] = (window_sum == HORIZON_STEPS + 1)
        glucose_by_pid[pid]       = g
        hr_by_pid[pid]            = hr[s:e]
        steps_by_pid[pid]         = stp[s:e]
        awake_by_pid[pid]         = awk[s:e]
        tod_by_pid[pid]           = tod[s:e]
        valid_fwd_by_pid[pid]     = valid_fwd

    arrays = dict(
        glucose=glucose_by_pid,
        hr=hr_by_pid,
        steps=steps_by_pid,
        awake=awake_by_pid,
        tod=tod_by_pid,
        valid_fwd=valid_fwd_by_pid,
        global_offset=global_offset_by_pid,
    )
    return arrays


def build_exercise_masks(arrays: dict, episodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    For each participant: build a boolean mask where True means this position is
    contaminated (within or near a detected exercise episode) and cannot be a
    clean control anchor.

    Contaminated range for episode [s_local, e_local]:
        [s_local - HORIZON_STEPS, e_local + EXCLUSION_PRE_STEPS]
    (so the control's clean window [t - EXCLUSION_PRE, t + HORIZON] won't overlap).
    """
    offset = arrays["global_offset"]
    ex_mask: dict[str, np.ndarray] = {}

    for pid, g in arrays["glucose"].items():
        ex_mask[pid] = np.zeros(len(g), dtype=bool)

    for _, ep in episodes.iterrows():
        pid = str(ep["participant_id"])
        if pid not in offset:
            continue
        s_local = int(ep["start_orig"]) - offset[pid]
        e_local = int(ep["end_orig"]) - offset[pid]
        n = len(ex_mask[pid])
        lo = max(0, s_local - HORIZON_STEPS)
        hi = min(n, e_local + EXCLUSION_PRE_STEPS + 1)
        if lo < hi:
            ex_mask[pid][lo:hi] = True

    return ex_mask


# ============================================================
# STEP 1: BUILD ALIGNED EXERCISE MATRIX
# ============================================================

def build_exercise_matrix(
    episodes: pd.DataFrame,
    arrays: dict,
) -> pd.DataFrame:
    """Extract aligned glucose trajectory for each usable exercise bout."""
    offset = arrays["global_offset"]
    glucose = arrays["glucose"]
    tod = arrays["tod"]

    rows = []
    for _, ep in episodes.iterrows():
        pid = str(ep["participant_id"])
        if pid not in offset:
            continue
        s_local = int(ep["start_orig"]) - offset[pid]
        n = len(glucose[pid])
        if s_local < 0 or s_local + HORIZON_STEPS >= n:
            continue
        traj = glucose[pid][s_local: s_local + HORIZON_STEPS + 1]
        if not np.all(np.isfinite(traj)):
            continue

        g0     = float(traj[0])
        g_at_60 = float(traj[HORIZON_STEPS])
        # steps 9-15 are 45-75 min (indices NADIR_LO to NADIR_HI)
        window = traj[NADIR_LO: NADIR_HI + 1]
        nadir  = float(np.nanmin(window)) if len(window) > 0 else np.nan

        rows.append(dict(
            participant_id   = pid,
            start_orig       = int(ep["start_orig"]),
            s_local          = s_local,
            strain_class     = str(ep["strain_class"]),
            g0               = g0,
            g_at_60          = g_at_60,
            nadir            = nadir,
            tod_min          = float(tod[pid][s_local]),
            traj             = traj.tolist(),
        ))

    return pd.DataFrame(rows)


# ============================================================
# STEP 2: BUILD MATCHED NON-EXERCISE CONTROLS
# ============================================================

def find_valid_control_positions(
    arrays: dict,
    ex_mask: dict,
    non_insulin_pids: set[str],
) -> dict[str, np.ndarray]:
    """
    For each participant: array of valid non-exercise control positions.
    Valid = valid_fwd AND awake AND NOT in exercise mask.
    """
    valid_ctrl: dict[str, np.ndarray] = {}
    for pid in non_insulin_pids:
        if pid not in arrays["glucose"]:
            continue
        awake = arrays["awake"][pid]
        vf    = arrays["valid_fwd"][pid]
        ex    = ex_mask[pid]
        positions = np.where(vf & ~ex & (awake >= AWAKE_THRESH))[0]
        valid_ctrl[pid] = positions
    return valid_ctrl


def match_controls(
    ex_matrix: pd.DataFrame,
    valid_ctrl: dict[str, np.ndarray],
    arrays: dict,
    rng: np.random.Generator,
    g0_match_tol: float,
    within_participant_only: bool,
) -> pd.DataFrame:
    """
    For each exercise bout, draw up to CONTROL_RATIO matched control anchors.
    Matching criteria: same participant (preferred), |g0 diff| <= G0_MATCH_TOL,
    |TOD diff| <= TOD_MATCH_TOL (circular difference for TOD).
    """
    glucose = arrays["glucose"]
    tod     = arrays["tod"]
    all_pids = sorted(valid_ctrl.keys())

    ctrl_rows = []
    n_cross_patient = 0
    n_total_bouts   = 0

    for _, bout in ex_matrix.iterrows():
        pid   = str(bout["participant_id"])
        g0_ex = bout["g0"]
        tod_ex = bout["tod_min"]
        n_total_bouts += 1

        collected = []  # list of (local_idx, pid, is_cross)

        # Same-participant candidates first
        if pid in valid_ctrl and len(valid_ctrl[pid]) > 0:
            pos = valid_ctrl[pid]
            g0_ctrl = glucose[pid][pos]
            tod_ctrl = tod[pid][pos]
            tod_diff = np.abs(tod_ctrl - tod_ex)
            tod_diff = np.minimum(tod_diff, 1440 - tod_diff)  # circular
            same_pid_mask = (
                (np.abs(g0_ctrl - g0_ex) <= g0_match_tol) &
                (tod_diff <= TOD_MATCH_TOL)
            )
            candidates = pos[same_pid_mask]
            if len(candidates) > 0:
                chosen = rng.choice(
                    candidates,
                    size=min(CONTROL_RATIO, len(candidates)),
                    replace=False,
                )
                for idx in chosen:
                    collected.append((int(idx), pid, False))

        needed = CONTROL_RATIO - len(collected)
        if needed > 0 and not within_participant_only:
            # Cross-participant fill
            for xpid in rng.permutation(all_pids):
                if xpid == pid:
                    continue
                if xpid not in valid_ctrl or len(valid_ctrl[xpid]) == 0:
                    continue
                pos = valid_ctrl[xpid]
                g0_ctrl = glucose[xpid][pos]
                tod_ctrl = tod[xpid][pos]
                tod_diff = np.abs(tod_ctrl - tod_ex)
                tod_diff = np.minimum(tod_diff, 1440 - tod_diff)
                mask = (
                    (np.abs(g0_ctrl - g0_ex) <= g0_match_tol) &
                    (tod_diff <= TOD_MATCH_TOL)
                )
                candidates = pos[mask]
                if len(candidates) == 0:
                    continue
                take = min(needed, len(candidates))
                chosen = rng.choice(candidates, size=take, replace=False)
                for idx in chosen:
                    collected.append((int(idx), str(xpid), True))
                needed -= take
                if needed <= 0:
                    break

        if not collected:
            continue

        cross = sum(1 for _, _, is_cross in collected if is_cross)
        if cross > 0:
            n_cross_patient += 1

        for s_local, ctrl_pid, is_cross in collected:
            g = glucose[ctrl_pid]
            traj = g[s_local: s_local + HORIZON_STEPS + 1]
            if not np.all(np.isfinite(traj)):
                continue
            g0_c     = float(traj[0])
            g_at_60_c = float(traj[HORIZON_STEPS])
            window_c = traj[NADIR_LO: NADIR_HI + 1]
            nadir_c  = float(np.nanmin(window_c)) if len(window_c) > 0 else np.nan
            ctrl_rows.append(dict(
                ex_participant_id = str(bout["participant_id"]),
                ex_start_orig     = int(bout["start_orig"]),
                ex_strain         = str(bout["strain_class"]),
                ex_g0             = g0_ex,
                ctrl_participant_id = ctrl_pid,
                ctrl_s_local        = s_local,
                is_cross_patient    = bool(is_cross),
                g0                  = g0_c,
                g_at_60             = g_at_60_c,
                nadir               = nadir_c,
                traj                = traj.tolist(),
            ))

    frac_cross = n_cross_patient / max(1, n_total_bouts)
    return pd.DataFrame(ctrl_rows), frac_cross


def summarize_matching_by_bin(
    ex_matrix: pd.DataFrame,
    ctrl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize control support and cross-patient reliance within each g0 bin."""
    rows = []
    for glo, ghi in G0_BINS_FULL:
        ex_bin = ex_matrix[(ex_matrix["g0"] >= glo) & (ex_matrix["g0"] < ghi)]
        bout_keys = set(ex_bin["start_orig"].astype(int))
        controls = ctrl_df[ctrl_df["ex_start_orig"].isin(bout_keys)]
        controls_per_bout = controls.groupby("ex_start_orig").size()
        cross_by_bout = controls.groupby("ex_start_orig")["is_cross_patient"].any()
        n_bouts = len(ex_bin)
        n_matched_bouts = int(controls_per_bout.size)
        rows.append(dict(
            g0_bin_low=glo,
            g0_bin_high=ghi,
            n_bouts=n_bouts,
            n_matched_bouts=n_matched_bouts,
            n_unmatched_bouts=n_bouts - n_matched_bouts,
            n_controls=len(controls),
            mean_controls_per_bout=len(controls) / max(1, n_bouts),
            frac_control_cross_patient=(
                float(controls["is_cross_patient"].mean()) if len(controls) else np.nan
            ),
            frac_bouts_with_cross_patient=(
                float(cross_by_bout.mean()) if len(cross_by_bout) else np.nan
            ),
        ))
    return pd.DataFrame(rows)


# ============================================================
# STEP 3: DE-CONFOUNDED EFFECT + BOOTSTRAP CIs
# ============================================================

def bootstrap_paired(
    ex_g60: np.ndarray,
    ctrl_g60_per_bout: list[np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Paired bootstrap: resample bouts (with replacement) and for each boot,
    compute exercise_mean - mean(ctrl_means per bout).
    Returns (point_estimate, ci_lo, ci_hi).
    """
    n = len(ex_g60)
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ex_b = ex_g60[idx]
        ctrl_b = np.array([ctrl_g60_per_bout[i].mean() for i in idx])
        boot_diffs[b] = ex_b.mean() - ctrl_b.mean()
    point = ex_g60.mean() - np.array([c.mean() for c in ctrl_g60_per_bout]).mean()
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    return float(point), ci_lo, ci_hi


def compute_deconfounded_effect(
    ex_matrix: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result_rows = []
    ctrl_grouped = ctrl_df.groupby("ex_start_orig")

    for glo, ghi in G0_BINS_FULL:
        ex_bin = ex_matrix[
            (ex_matrix["g0"] >= glo) & (ex_matrix["g0"] < ghi)
        ]
        if len(ex_bin) == 0:
            continue

        ex_g60_vals  = []
        ex_nadir_vals = []
        ctrl_g60_per = []
        ctrl_all_g60 = []
        ctrl_nadir_per = []

        for _, bout in ex_bin.iterrows():
            key = int(bout["start_orig"])
            if key not in ctrl_grouped.groups:
                continue
            ctrls = ctrl_grouped.get_group(key)
            if len(ctrls) == 0:
                continue
            ex_g60_vals.append(bout["g_at_60"])
            ctrl_g60_per.append(ctrls["g_at_60"].values)
            ctrl_all_g60.extend(ctrls["g_at_60"].values)
            ex_nadir_vals.append(bout["nadir"])
            ctrl_nadir_per.append(ctrls["nadir"].values)

        if len(ex_g60_vals) == 0:
            continue

        ex_g60  = np.array(ex_g60_vals)
        ctrl_mu = np.array([c.mean() for c in ctrl_g60_per])

        ex_mean_g60    = float(ex_g60.mean())
        ctrl_mean_g60  = float(ctrl_mu.mean())
        ex_nadir       = float(np.mean(ex_nadir_vals))
        ctrl_nadir_mu  = float(np.mean([c.mean() for c in ctrl_nadir_per]))

        point, ci_lo, ci_hi = bootstrap_paired(ex_g60, ctrl_g60_per, N_BOOTSTRAP, rng)

        n_bouts    = len(ex_g60_vals)
        n_controls = len(ctrl_all_g60)
        cross_frac = float(ctrl_df[
            ctrl_df["ex_start_orig"].isin(ex_bin["start_orig"])
        ]["is_cross_patient"].mean()) if n_controls > 0 else np.nan

        result_rows.append(dict(
            g0_bin_low              = glo,
            g0_bin_high             = ghi,
            n_bouts                 = n_bouts,
            n_controls              = n_controls,
            frac_control_cross_patient = cross_frac,
            exercise_g60_mean       = ex_mean_g60,
            control_g60_mean        = ctrl_mean_g60,
            deconf_effect_g60       = point,
            deconf_ci_low           = ci_lo,
            deconf_ci_high          = ci_hi,
            deconf_effect_nadir     = ex_nadir - ctrl_nadir_mu,
        ))

    return pd.DataFrame(result_rows)


# ============================================================
# STEP 4: FIGURE
# ============================================================

def make_figure(effect_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bin_labels = [
        f"{r.g0_bin_low}-{r.g0_bin_high}\n(n={r.n_bouts})"
        for _, r in effect_df.iterrows()
    ]
    x = np.arange(len(effect_df))

    # Left panel: exercise vs control at +60 min
    ax = axes[0]
    ax.errorbar(
        x, effect_df["exercise_g60_mean"],
        fmt="o-", color="#2ca02c", label="Exercise", linewidth=1.5, capsize=4,
    )
    ax.errorbar(
        x, effect_df["control_g60_mean"],
        fmt="s--", color="#636363", label="Matched control", linewidth=1.5, capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_xlabel("Starting glucose g0 (mg/dL)", fontsize=9)
    ax.set_ylabel("Mean glucose at +60 min (mg/dL)", fontsize=9)
    ax.set_title("Exercise vs Matched Control at +60 min", fontsize=9)
    ax.legend(fontsize=8)

    # Right panel: de-confounded effect with 95% CI
    ax = axes[1]
    effects = effect_df["deconf_effect_g60"].values
    ci_lo   = effect_df["deconf_ci_low"].values
    ci_hi   = effect_df["deconf_ci_high"].values
    yerr    = np.array([effects - ci_lo, ci_hi - effects])
    ax.bar(x, effects, color="#2ca02c", alpha=0.7, zorder=2)
    ax.errorbar(x, effects, yerr=yerr, fmt="none", color="black", capsize=5, linewidth=1.2)
    ax.axhline(0, color="#d62728", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_xlabel("Starting glucose g0 (mg/dL)", fontsize=9)
    ax.set_ylabel("De-confounded effect at +60 min (mg/dL)", fontsize=9)
    ax.set_title("De-confounded Exercise Effect (exercise minus control)", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"fig_deconfounded_s22.{ext}"
        fig.savefig(path, dpi=150)
        print(f"    Saved: {path}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    if args.g0_match_tol <= 0:
        sys.exit("ERROR: --g0-match-tol must be positive.")
    out_dir = output_dir_for_run(args.g0_match_tol, args.within_participant_only)
    rng  = np.random.default_rng(SEED)

    print("=" * 70)
    print("STUDY 2 EXERCISE STAGE 0: MODEL-FREE MATCHED-CONTROL DE-CONFOUNDING")
    print(f"G0 matching caliper: +/-{args.g0_match_tol:g} mg/dL")
    print(f"Control scope: {'within-participant only' if args.within_participant_only else 'same-participant preferred'}")
    print("=" * 70)

    # ---- 0. Path checks ----
    print("\nInput paths:")
    check_paths(out_dir)

    # ---- Cohort pids ----
    cohort_df = pd.read_csv(COHORT_PATH)
    cohort_df["participant_id"] = cohort_df["participant_id"].astype(str)
    cohort_pids: set[str] = set(cohort_df["participant_id"])
    print(f"\nCohort participants: {len(cohort_pids)}")

    # ---- Non-insulin pids ----
    static = pd.read_parquet(STATIC_PATH, columns=["participant_id", "med_insulin"])
    static["participant_id"] = static["participant_id"].astype(str)
    non_insulin_pids: set[str] = set(
        static.loc[static["med_insulin"] == 0, "participant_id"]
    )
    ni_cohort_pids: set[str] = non_insulin_pids & cohort_pids
    print(f"Non-insulin participants (static): {len(non_insulin_pids)}")
    print(f"Non-insulin cohort participants: {len(ni_cohort_pids)}")

    # ---- Load episode table ----
    episodes_all = pd.read_parquet(EPISODE_PATH)
    episodes_all["participant_id"] = episodes_all["participant_id"].astype(str)
    episodes_hi = episodes_all[episodes_all["episode_confidence_class"] == "high"].copy()
    episodes_ni = episodes_hi[episodes_hi["participant_id"].isin(ni_cohort_pids)].copy()
    episodes    = episodes_ni[episodes_ni["strain_class"].isin(PRIMARY_STRAINS)].copy()

    print(f"Episodes (all confidence): {len(episodes_all)}")
    print(f"Episodes (high confidence): {len(episodes_hi)}")
    print(f"Episodes (high conf, non-insulin): {len(episodes_ni)}")
    print(f"Episodes (high conf, non-insulin, moderate/vigorous): {len(episodes)}")
    print("  Strain counts:")
    print(episodes["strain_class"].value_counts().to_string(index=True))

    # ---- Step 1: Build per-pid arrays ----
    arrays = load_and_build_arrays(ni_cohort_pids, cohort_pids)

    # ---- Build exercise mask (uses ALL non-insulin cohort episodes for exclusion) ----
    print("\n[2/5] Building exercise exclusion mask ...")
    ex_mask = build_exercise_masks(arrays, episodes_ni)

    # ---- Step 1: Exercise matrix ----
    print("\n[3/5] Building aligned exercise matrix ...")
    ex_matrix = build_exercise_matrix(episodes, arrays)
    print(f"    Usable exercise bouts: {len(ex_matrix)}")
    print("    Bouts per G0_BINS_FULL:")
    for glo, ghi in G0_BINS_FULL:
        n = ((ex_matrix["g0"] >= glo) & (ex_matrix["g0"] < ghi)).sum()
        flag = "  [WARNING: low support]" if n < MIN_BOUTS_PER_BIN else ""
        print(f"      g0 [{glo},{ghi}): n={n}{flag}")

    # ---- Step 2: Valid control positions ----
    print("\n[4/5] Finding valid control positions ...")
    valid_ctrl = find_valid_control_positions(arrays, ex_mask, ni_cohort_pids)
    total_ctrl_positions = sum(len(v) for v in valid_ctrl.values())
    print(f"    Total valid control anchor positions: {total_ctrl_positions:,}")

    # ---- Step 2: Match controls ----
    print("    Matching controls to exercise bouts ...")
    ctrl_df, frac_cross = match_controls(
        ex_matrix, valid_ctrl, arrays, rng, args.g0_match_tol,
        args.within_participant_only,
    )
    n_ctrl_total = len(ctrl_df)
    mean_ctrl_per_bout = n_ctrl_total / max(1, len(ex_matrix))
    print(f"    Total matched control windows: {n_ctrl_total:,}")
    matching_df = summarize_matching_by_bin(ex_matrix, ctrl_df)
    print("    Matching quality by g0 bin:")
    print(matching_df[[
        "g0_bin_low", "g0_bin_high", "n_bouts", "n_matched_bouts",
        "n_unmatched_bouts", "n_controls", "mean_controls_per_bout",
        "frac_control_cross_patient", "frac_bouts_with_cross_patient",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"    Mean controls per bout: {mean_ctrl_per_bout:.1f}")
    print(f"    Fraction of bouts with any cross-participant control: {frac_cross:.3f}")

    if frac_cross > MAX_CROSS_PATIENT_FRAC:
        print(
            f"    [POSITIVITY WARNING] Bouts using cross-patient controls {frac_cross:.3f} > "
            f"{MAX_CROSS_PATIENT_FRAC}. Inspect the per-bin control fractions above."
        )

    # ---- PAUSE ----
    print()
    print("=" * 70)
    print("PAUSE: review matching quality before computing effect.")
    print()
    print("Check above for:")
    print("  - Any g0 bin with n < 20 bouts (low support, flag in output above)")
    print(f"  - Cross-participant fraction > {MAX_CROSS_PATIENT_FRAC} (positivity concern)")
    print()
    if not args.confirm:
        print("Re-run with --confirm to proceed to Steps 3-5 (curves + figure + CSV).")
        print("=" * 70)
        return
    print("--confirm passed. Proceeding to Steps 3-5.")
    print("=" * 70)
    matching_path = out_dir / "matching_quality_by_g0.csv"
    matching_df.to_csv(matching_path, index=False)
    print(f"    Saved: {matching_path}")

    # ---- Step 3: De-confounded effect ----
    print("\n[5/5] Computing de-confounded effect curves ...")
    effect_df = compute_deconfounded_effect(ex_matrix, ctrl_df, rng)
    print(effect_df[[
        "g0_bin_low", "g0_bin_high", "n_bouts", "n_controls",
        "exercise_g60_mean", "control_g60_mean", "deconf_effect_g60",
        "deconf_ci_low", "deconf_ci_high",
    ]].to_string(index=False))

    # ---- Step 4: Figure ----
    print("\nGenerating figure ...")
    make_figure(effect_df, out_dir)

    # ---- Step 5: Save CSV ----
    csv_path = out_dir / "deconfounded_effect_targets.csv"
    effect_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    # ---- Interpretation rule ----
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    nonempty = effect_df[effect_df["n_bouts"] >= MIN_BOUTS_PER_BIN]
    if len(nonempty) == 0:
        print("INCONCLUSIVE: no g0 bins with sufficient bout support.")
    else:
        effects = nonempty["deconf_effect_g60"].values
        ci_lo   = nonempty["deconf_ci_low"].values
        ci_hi   = nonempty["deconf_ci_high"].values
        monotone = all(effects[i] <= effects[i + 1] for i in range(len(effects) - 1))
        all_ci_zero = all(lo <= 0 <= hi for lo, hi in zip(ci_lo, ci_hi))
        sign_flip  = len(np.unique(np.sign(effects))) > 1

        print(f"De-confounded effect at +60 min (mg/dL) per g0 bin:")
        for _, row in nonempty.iterrows():
            print(
                f"  g0 [{row.g0_bin_low},{row.g0_bin_high}): "
                f"effect={row.deconf_effect_g60:+.1f}  "
                f"CI=[{row.deconf_ci_low:+.1f}, {row.deconf_ci_high:+.1f}]  "
                f"n_bouts={row.n_bouts}"
            )
        print()
        if sign_flip:
            print(
                "RESULT: NON-MONOTONIC or sign-flipping effect. Matching or positivity "
                "problem. Do not calibrate against this. Investigate before Stage 1."
            )
        elif all_ci_zero:
            print(
                "RESULT: De-confounded effect is FLAT / within CI of zero across all bins. "
                "Exercise effect does not survive de-confounding on AI-READI. "
                "Clean NOT-CONSTRUCTIBLE signal at the data level."
            )
        else:
            if monotone:
                print(
                    "RESULT: De-confounded effect is present and MONOTONICALLY increasing "
                    "with g0 (or at least not sign-flipping). "
                    "This CONFIRMS baseline glucose as the real conditioning axis, "
                    "validates the one-sided ramp R(t), and the CSV is a usable "
                    "calibration target. PROCEED to Stage 1."
                )
            else:
                print(
                    "RESULT: De-confounded effect is present but non-monotonic. "
                    "Inspect the figure. Possible positivity issue at extreme bins."
                )
    print("=" * 70)


if __name__ == "__main__":
    main()
