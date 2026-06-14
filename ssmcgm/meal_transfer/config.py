"""Configuration for the passive meal-transfer pipeline.

All tunables live here so the smoke test, a future full run, and a later
threshold/model sweep share one source of truth. Thresholds that the brief
requires to be *data-driven* (Phase D) are expressed as **quantiles** here, not
as hardcoded universal glucose values; the absolute cut points are fit on the
training split at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Repository-relative locations -----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "no_log_scenarios" / "meal_transfer"

# AI-READI enriched + mealflag feathers (the target domain). The mealflag copy
# carries the surviving teacher `predmeal_flag` plus all wearable/clinical cols.
DATA_DIR = Path("/home/myriamcharfeddine/CGM/Data/ssmcgm_ready_enriched_mealflag")
SPLIT_FILES = {
    "train": DATA_DIR / "train_timeseries_static_mealflag.feather",
    "test": DATA_DIR / "test_timeseries_static_mealflag.feather",
}

# Immutable baseline asset supplied by the user (do NOT modify these files).
ASSET_DIR = REPO_ROOT / "assets" / "meal_transfer"
SOURCE_NOTEBOOK = ASSET_DIR / "TimeseriesFinalMealDetect (3).ipynb"

# Candidate locations for the real CGMacros checkpoint, searched in order.
# The genuine `.pth` lives in the immutable asset dir; the OUTPUT_DIR path is a
# convenience override slot.
TEACHER_CHECKPOINT_CANDIDATES = [
    ASSET_DIR / "meal_detection_model_final (4).pth",
    OUTPUT_DIR / "teacher_checkpoint.pth",
    REPO_ROOT / "MealDetection" / "meal_detection_model_final.pth",
]


@dataclass
class TeacherConfig:
    """Source CGMacros model architecture + inference settings (Phase A/B)."""

    # Architecture (recovered from MealDetection/mealdetection.ipynb cell 9).
    in_channels: int = 1
    c_out: int = 16
    hidden: int = 64
    dropout: float = 0.4
    seq_len: int = 72            # window length in steps
    stride: int = 1             # dense overlapping windows (stride=1)

    # Resolution is confirmed 5-min for BOTH CGMacros (training) and AI-READI
    # (deployment): 72 samples = 6 h in both domains (SSM-CGM paper
    # arXiv:2510.04386, Methods / Appendix A.2). No resampling is needed; the
    # stale notebook "1 min" docstring is documentation drift.
    target_resolution_min: int = 5
    source_resolution_min: int = 5

    # Baseline-only decision rule retained from the original notebook.
    baseline_threshold: float = 0.40
    baseline_ratio: float = 0.65

    batch_size: int = 256


@dataclass
class PseudoLabelConfig:
    """Phase D weak-label construction (quantile-based, insulin-aware)."""

    # Future response horizons in minutes (offline labelling only).
    response_horizons_min: tuple = (30, 60, 120)
    peak_window_min: int = 120          # R_t = max future rise over this window

    # Quantiles fit on the TRAIN split (per the brief), not absolute glucose.
    pos_response_q: float = 0.85        # strong future rise -> meal-like
    neg_response_q: float = 0.45        # flat-ish -> candidate non-meal
    teacher_pos_q: float = 0.80         # teacher prob high (if available)
    teacher_low_q: float = 0.40         # teacher prob low

    # Competing-event exclusion thresholds.
    activity_steps_per_5min: float = 250.0   # high activity at/around t
    hr_zscore_high: float = 1.5              # endurance-like HR elevation
    hypo_recovery_mgdl: float = 70.0         # rising out of hypo, not a meal

    # Response-size proxy: training-split tertiles of the peak rise.
    size_quantiles: tuple = (1 / 3, 2 / 3)

    # Insulin cohort gets wider uncertainty and separate rules.
    insulin_confidence_scale: float = 0.6


@dataclass
class StudentConfig:
    """Phase E causal gradient-boosted student (CPU)."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int | None = 6
    l2_regularization: float = 1.0
    max_leaf_nodes: int = 31
    early_stopping: bool = True
    validation_fraction: float = 0.15
    random_state: int = 42


@dataclass
class DecoderConfig:
    """Phase F duration-constrained meal-state decoder.

    States: 0=none, 1=onset, 2=rising, 3=peak, 4=recovery.
    Minimum dwell times (in 5-min steps) impose physiological durations and
    stop the point-wise flag from flickering.
    """

    states: tuple = ("none", "onset", "rising", "peak", "recovery")
    min_dwell_steps: tuple = (3, 1, 2, 1, 2)   # >=15min none, >=10min rising...
    max_dwell_steps: tuple = (10_000, 4, 18, 6, 24)
    # Transition penalty for leaving `none` (discourages spurious events).
    onset_penalty: float = 1.2
    self_transition_bonus: float = 0.2
    # Bias added to the `none` log-emission: meal vs none is driven by the
    # student probability; a positive bias suppresses spurious brief events and
    # keeps the postprandial-state fraction physiologically plausible.
    none_bias: float = 0.85
    # Phase-shape terms are CENTERED across the 4 meal phases and scaled by this,
    # so they only redistribute mass among onset/rising/peak/recovery and never
    # inflate total meal evidence relative to `none`.
    phase_shape_scale: float = 0.7


@dataclass
class PipelineConfig:
    output_dir: Path = OUTPUT_DIR
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    pseudo: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Smoke-test controls.
    smoke: bool = True
    max_participants: int = 50
    random_state: int = 42

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
