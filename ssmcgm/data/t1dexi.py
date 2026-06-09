"""T1DEXI data loader for SSM-CGM v2.

Turns the preprocessed T1DEXI *model-ready panel*
(``outputs/t1dexi_preprocessed*/pytorch_forecasting/modeling_panel_pytorch.parquet``)
into pytorch-forecasting ``TimeSeriesDataSet`` objects that
:class:`ssmcgm.models.ssmcgm.SSMCGM` consumes via ``from_dataset``.

The feature schema is the **counterfactual-curated** one (ported from the v1
``T1DEXI_SSM/train.py``): action streams — insulin, meals, exercise, wearables —
are placed in ``time_varying_known_*`` so they are *decoder-visible* and can be
intervened on. That is exactly the contract v2's scenario interface expects
(``scenario_reals`` / ``scenario_categoricals`` must be decoder covariates), so
those columns can be wired straight into ``SSMCGM.from_dataset(...)``.

Typical use::

    from ssmcgm.data import t1dexi
    df = t1dexi.load_t1dexi_data(t1dexi.default_panel_path())
    training, validation, test = t1dexi.make_datasets(df, context_length=288)
    model = ssmcgm.SSMCGM.from_dataset(training, scenario_reals=t1dexi.SCENARIO_REALS)

Counterfactual design (mirrors SSM-CGM paper §3): ``time_varying_known_*`` appear
in both encoder and decoder — at counterfactual inference the decoder values are
replaced with the hypothetical intervention sequence; ``time_varying_unknown_*``
are encoder-only (raw CGM, timing-from-history, sensor-validity) and are not
directly intervene-able.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# pytorch-forecasting only — these do not pull mamba_ssm, and importing this
# module always runs ``ssmcgm/__init__`` first (which binds the conda libstdc++
# via the eager mamba_ssm import), so the usual mamba-before-ptf order holds.
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder


# ---------------------------------------------------------------------------
# default panel locations (relative to the repo root: .../T1DEXI)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRY_RUN_PANEL = _REPO_ROOT / "outputs/t1dexi_preprocessed/pytorch_forecasting/modeling_panel_pytorch.parquet"
_FULL_PANEL = _REPO_ROOT / "outputs/t1dexi_preprocessed_full/pytorch_forecasting/modeling_panel_pytorch.parquet"

TARGET = "glucose_current_mg_dl"
GROUP_ID = "USUBJID"
TIME_IDX = "time_idx"


def default_panel_path(full: bool = False) -> str:
    """Path to the model-ready panel. ``full=False`` -> 4-participant dry run."""
    return str(_FULL_PANEL if full else _DRY_RUN_PANEL)


# ---------------------------------------------------------------------------
# feature schema (counterfactual-curated)
# ---------------------------------------------------------------------------
def get_t1dexi_feature_schema() -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Return the six feature-role lists for ``TimeSeriesDataSet``.

    Order: ``(static_categoricals, static_reals, time_varying_known_reals,
    time_varying_known_categoricals, time_varying_unknown_categoricals,
    time_varying_unknown_reals)``.
    """
    static_categoricals = [
        "sex",
        "race",
        "ethnicity",
        "randomized_arm",
        "insulin_modality",
        "insulin_delivery_group",
        "lifetime_hypoglycemia_category",
        "lifetime_dka_category",
        "education_level",
        "income_level",
        "health_insurance",
    ]
    static_reals = [
        "age_years",
        "height_in",
        "weight_lb",
        "bmi_kg_m2",
        "baseline_hba1c_value",
        "age_at_diabetes_onset_years",
        "diabetes_duration_years",
    ]
    # Known reals: time-of-day covariates (always known in future) plus
    # action/intervention features that appear in the decoder for counterfactuals.
    time_varying_known_reals = [
        # Time-of-day (calendar features, always known future)
        "minute_of_day",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "sin_hour_of_day",
        "cos_hour_of_day",
        "sin_day_of_week",
        "cos_day_of_week",
        "tod_sin",
        "tod_cos",
        # Wearable signals — future covariates; counterfactual targets
        "heart_rate_value_for_model",
        "steps_value_for_model",
        # Exercise — planned activity interventions
        "exercise_active_flag",
        "exercise_minutes_at_t",
        # Meals / nutrition — planned eating interventions
        "carbs_g_at_t",
        "carbs_absorption_fast",
        "carbs_absorption_slow",
        "fat_protein_delayed_absorption",
        "meal_absorption_total",
        "meal_event_count_last_60min",
        # Insulin — planned dosing interventions
        "bolus_units_at_t",
        "basal_units_at_t",
        "basal_rate_value_for_model",
        "basal_delivered_units_5min",
        "basal_rate_u_per_hr_current",
        "bolus_iob_units_fast",
        "bolus_iob_units_slow",
        "bolus_iob_units_total",
        "basal_iob_proxy",
        "insulin_activity_curve",
    ]
    # Known categoricals: behavioral/state flags that are planned (decoder-visible).
    time_varying_known_categoricals = [
        "recent_sleep_flag_12h",
        "recent_exercise_flag_2h",
        "exercise_intensity_current_or_recent",
        "competitive_exercise_flag_current_or_recent",
        "snack_before_exercise_flag_current_or_recent",
    ]
    # Unknown categoricals: none remain (all moved to known for counterfactual design).
    time_varying_unknown_categoricals: List[str] = []
    # Unknown reals: encoder-only — raw CGM signal, timing-from-history features
    # derived from past events, sleep-quality history, and sensor-validity flags.
    time_varying_unknown_reals = [
        # TARGET — must be listed here for pytorch-forecasting
        "glucose_current_mg_dl",
        # CGM signal (encoder input; target is the observed reading)
        "glucose_model_input_mg_dl",
        "cgm_gap_minutes_since_last_observed",
        # Timing-from-history (derived from past events; not directly intervene-able)
        "time_since_last_bolus_min",
        "time_since_last_meal_min",
        "time_since_last_exercise_min",
        # Sleep quality history (past episode; updated only after a sleep event)
        "previous_sleep_total_sleep_time_min",
        "previous_sleep_rem_duration_min",
        "previous_sleep_efficiency",
        "poor_sleep_flag",
        "sleep_feature_age_hours",
        # Validity / missingness indicators (reflect sensor state, not interventions)
        "glucose_model_input_is_observed",
        "glucose_model_input_is_ffill",
        "missing_glucose",
        "basal_rate_is_valid",
        "heart_rate_is_valid",
        "steps_is_valid",
        "wearable_any_stream_valid_flag",
        "wearable_device_is_valid",
    ]
    return (
        static_categoricals,
        static_reals,
        time_varying_known_reals,
        time_varying_known_categoricals,
        time_varying_unknown_categoricals,
        time_varying_unknown_reals,
    )


# Convenient scenario-variable presets (subset of the decoder-visible columns).
# These are valid ``scenario_reals`` / ``scenario_categoricals`` for SSMCGM
# because every one is a ``time_varying_known_*`` covariate above.
SCENARIO_REALS = [
    "heart_rate_value_for_model",
    "steps_value_for_model",
    "exercise_minutes_at_t",
    "carbs_g_at_t",
    "bolus_units_at_t",
    "basal_units_at_t",
]
SCENARIO_CATEGORICALS = [
    "recent_exercise_flag_2h",
    "exercise_intensity_current_or_recent",
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_t1dexi_data(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """Load only the columns the dataset needs from the model-ready panel.

    Casts categoricals to string (pytorch-forecasting requires it) and backfills
    the target with the preprocessor's already-imputed ``glucose_model_input_mg_dl``
    (linear interp <=15 min, ffill 15-30 min); gaps >30 min stay NaN and are
    dropped by the per-window quality filters baked into the ``split`` column.
    """
    parquet_path = parquet_path or default_panel_path()
    (static_cats, static_reals, tv_known_reals, tv_known_cats,
     tv_unk_cats, tv_unk_reals) = get_t1dexi_feature_schema()
    needed_cols = list(dict.fromkeys(
        [GROUP_ID, TIME_IDX, "split"]
        + static_cats + static_reals
        + tv_known_reals + tv_known_cats + tv_unk_cats + tv_unk_reals
    ))
    print(f"[t1dexi] loading {len(needed_cols)} columns from {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    print(f"[t1dexi] loaded {len(df):,} rows x {df.shape[1]} columns, "
          f"{df[GROUP_ID].nunique()} participants")

    all_categoricals = static_cats + tv_known_cats + tv_unk_cats
    for col in all_categoricals:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    if "glucose_model_input_mg_dl" in df.columns:
        df[TARGET] = df[TARGET].fillna(df["glucose_model_input_mg_dl"])

    return df


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------
def _split_slices(df: pd.DataFrame, context_length: int, horizon: int
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve per-participant train / val / test row slices with enough lookback.

    Each slice extends ``context_length`` rows before the split's first anchor
    (so the earliest window has full encoder context) and ``horizon`` rows past
    its last anchor (so the final decoder is covered). Participants that appear
    only in val/test still contribute their pre-val context rows to the *train*
    slice, so ``from_dataset`` sees every group id and never KeyErrors.
    """
    val_min = df[df["split"] == "validation"].groupby(GROUP_ID)[TIME_IDX].min()
    val_max = df[df["split"] == "validation"].groupby(GROUP_ID)[TIME_IDX].max()
    test_min = df[df["split"] == "test"].groupby(GROUP_ID)[TIME_IDX].min()
    test_max = df[df["split"] == "test"].groupby(GROUP_ID)[TIME_IDX].max()
    train_max = df[df["split"] == "train"].groupby(GROUP_ID)[TIME_IDX].max()

    train_parts, val_parts, test_parts = [], [], []
    for uid, group in df.groupby(GROUP_ID):
        gmin = int(group[TIME_IDX].min())
        if uid in train_max.index:
            cutoff = int(train_max[uid]) + horizon
            train_parts.append(group[group[TIME_IDX] <= cutoff])
        elif uid in val_min.index:
            lookback_end = int(val_min[uid]) - 1
            lookback_start = max(gmin, lookback_end - context_length + 1)
            chunk = group[(group[TIME_IDX] >= lookback_start) & (group[TIME_IDX] <= lookback_end)]
            if len(chunk):
                train_parts.append(chunk)
        if uid in val_max.index:
            lo = max(gmin, int(val_min[uid]) - context_length)
            hi = int(val_max[uid]) + horizon
            val_parts.append(group[(group[TIME_IDX] >= lo) & (group[TIME_IDX] <= hi)])
        if uid in test_max.index:
            lo = max(gmin, int(test_min[uid]) - context_length)
            hi = int(test_max[uid]) + horizon
            test_parts.append(group[(group[TIME_IDX] >= lo) & (group[TIME_IDX] <= hi)])

    training_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return training_df, val_df, test_df


def _fill_missing(df_part: pd.DataFrame, real_cols: List[str], cat_cols: List[str]) -> None:
    """In-place NaN fill: reals ffill->bfill->0 within group; cats -> "unknown".

    ``TimeSeriesDataSet`` raises on any NaN in continuous columns regardless of
    ``allow_missing_timesteps``; the target gets its best fill at load time.
    """
    real_present = [c for c in real_cols if c in df_part.columns]
    df_part.sort_values([GROUP_ID, TIME_IDX], inplace=True)
    grouped = df_part.groupby(GROUP_ID, sort=False)[real_present]
    df_part[real_present] = grouped.ffill()
    df_part[real_present] = df_part.groupby(GROUP_ID, sort=False)[real_present].bfill()
    df_part[real_present] = df_part[real_present].fillna(0.0)
    for col in cat_cols:
        if col in df_part.columns:
            df_part[col] = df_part[col].fillna("unknown").astype(str)


def make_datasets(
    df: pd.DataFrame,
    *,
    context_length: int = 576,
    min_encoder_length: int = 288,
    horizon: int = 12,
    train_stride: int = 1,
    val_stride: int = 1,
    target_norm: str = "group",
    restrict_to_participants: Optional[set] = None,
) -> Tuple[TimeSeriesDataSet, Optional[TimeSeriesDataSet], Optional[TimeSeriesDataSet]]:
    """Build train/val/test ``TimeSeriesDataSet`` from the loaded panel.

    ``context_length`` 576 = 48h, 288 = 24h (5-min bins). ``*_stride`` keeps every
    Nth anchor in the corresponding index — e.g. ``train_stride=12`` -> one window
    per hour, ~12x fewer windows, good for smoke tests. Returns
    ``(training, validation, test)``; validation/test are ``None`` if absent.

    ``target_norm``: ``"group"`` = per-participant ``GroupNormalizer`` (good for seen
    participants but biased at held-out cold start), ``"global"`` = a single train-fit
    center/scale shared by everyone (leakage-safe, no per-person center) — see
    :mod:`ssmcgm.evaluation.target_transform`.
    """
    (static_categoricals, static_reals, tv_known_reals, tv_known_cats,
     tv_unk_cats, tv_unk_reals) = get_t1dexi_feature_schema()

    # Restrict the scaler/normalizer fit to an explicit participant set (the *stream*
    # train participants). This decouples the fit from the panel-`split` lookback rule
    # (which otherwise pulls held-out validation participants' pre-window rows into the
    # train slice and fits their per-participant GroupNormalizer center) — guaranteeing
    # held-out val/test participants are genuinely unseen by the scalers.
    if restrict_to_participants is not None:
        keep = {str(p) for p in restrict_to_participants}
        df = df[df[GROUP_ID].astype(str).isin(keep)]

    training_df, val_df, test_df = _split_slices(df, context_length, horizon)
    print(f"[t1dexi] train slice: {len(training_df):,} rows / "
          f"{training_df[GROUP_ID].nunique()} participants | "
          f"val: {len(val_df):,} | test: {len(test_df):,}")
    del df
    gc.collect()

    real_cols = static_reals + tv_known_reals + tv_unk_reals
    cat_cols = tv_known_cats + tv_unk_cats
    for part in (training_df, val_df, test_df):
        if len(part):
            _fill_missing(part, real_cols, cat_cols)

    training = TimeSeriesDataSet(
        training_df,
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        max_encoder_length=context_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=tv_known_reals,
        time_varying_known_categoricals=tv_known_cats,
        time_varying_unknown_categoricals=tv_unk_cats,
        time_varying_unknown_reals=tv_unk_reals,
        target_normalizer=GroupNormalizer(
            groups=([GROUP_ID] if target_norm == "group" else []), transformation="softplus"),
        # add_nan=True on *every* categorical (not just the group id): under
        # participant-held-out splits a categorical level can occur only in held-out
        # val/test (e.g. an 'Unknown' static value), and a default encoder raises on
        # the unseen level. Mapping it to the nan code is the correct held-out behavior.
        categorical_encoders={
            c: NaNLabelEncoder(add_nan=True)
            for c in [GROUP_ID, *static_categoricals, *tv_known_cats, *tv_unk_cats]
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    if train_stride > 1:
        training.index = training.index.iloc[::train_stride].reset_index(drop=True)
    print(f"[t1dexi] training windows: {len(training):,}"
          + (f" (stride={train_stride})" if train_stride > 1 else ""))

    validation = test = None
    if len(val_df):
        validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
        if val_stride > 1:
            validation.index = validation.index.iloc[::val_stride].reset_index(drop=True)
        print(f"[t1dexi] validation windows: {len(validation):,}"
              + (f" (stride={val_stride})" if val_stride > 1 else ""))
    if len(test_df):
        test = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
        if val_stride > 1:
            test.index = test.index.iloc[::val_stride].reset_index(drop=True)
        print(f"[t1dexi] test windows: {len(test):,}"
              + (f" (stride={val_stride})" if val_stride > 1 else ""))

    return training, validation, test


def make_dataloaders(
    df: pd.DataFrame,
    *,
    context_length: int = 576,
    min_encoder_length: int = 288,
    horizon: int = 12,
    batch_size: int = 16,
    num_workers: int = 0,
    train_stride: int = 1,
    val_stride: int = 1,
):
    """Build datasets and their dataloaders in one call.

    Returns ``(training, train_dl, validation, val_dl, test, test_dl)`` —
    datasets are ``TimeSeriesDataSet`` (validation/test may be ``None``) and the
    dataloaders mirror them. ``num_workers=0`` (default) keeps it single-process
    and CPU/smoke-test friendly; bump it (with ``multiprocessing_context="spawn"``
    handled internally) for real training.
    """
    training, validation, test = make_datasets(
        df, context_length=context_length, min_encoder_length=min_encoder_length,
        horizon=horizon, train_stride=train_stride, val_stride=val_stride,
    )
    mp_ctx = {"multiprocessing_context": "spawn", "persistent_workers": True} if num_workers > 0 else {}
    train_dl = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, **mp_ctx,
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=num_workers,
        pin_memory=True, **mp_ctx,
    ) if validation is not None else None
    test_dl = test.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=num_workers,
        pin_memory=False, **mp_ctx,
    ) if test is not None else None
    return training, train_dl, validation, val_dl, test, test_dl


# ---------------------------------------------------------------------------
# production-style chronological streams (deployment data path) — see
# ssmcgm.data.streaming. Re-exported here so callers can use a single namespace.
# ---------------------------------------------------------------------------
from .streaming import (  # noqa: E402,F401  (deferred to avoid a circular import)
    ParticipantStream,
    StreamFeatureSpec,
    StreamSplit,
    build_stream_feature_spec,
    iter_participant_streams,
    make_participant_streams,
    make_stream_splits,
    split_assignments_frame,
)
