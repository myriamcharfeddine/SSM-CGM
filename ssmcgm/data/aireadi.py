"""AI-READI stateful-stream data adapter.

This module keeps the AI-READI preprocessing boundary narrow: it reads the existing
``final_multimodal_dataset_*.parquet`` and optional
``participant_static_features.parquet`` files, preserves participant-level split
semantics, and builds chronological participant+segment streams. It does not
materialize overlapping 24h/48h windows.
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
VAL_ALIASES = {"val": "validation", "valid": "validation", "validation": "validation"}
DEFAULT_BIN_MINUTES = 5
CLEAN_CORE_COLS = {
    "cgm": "cgm_glucose_mean",
    "hr": "heart_rate_mean",
    "rr": "respiratory_rate_mean",
    "activity": "activity_steps_per_min",
}
CLEAN_GAP_THRESHOLDS_BINS = {"cgm": 6, "hr": 12, "rr": 12, "activity": 12}
DEFAULT_CLEAN_MIN_SEGMENT_HOURS = 49


def _long_missing_run_mask(series: pd.Series, threshold_bins: int) -> pd.Series:
    is_null = series.isna()
    if not bool(is_null.any()):
        return pd.Series(False, index=series.index)
    run_id = (~is_null).cumsum()
    run_lens = is_null.groupby(run_id).sum()
    bad_runs = set(run_lens[run_lens > threshold_bins].index)
    if not bad_runs:
        return pd.Series(False, index=series.index)
    return is_null & run_id.isin(bad_runs)


def _impute_clean_segment(seg: pd.DataFrame) -> pd.DataFrame:
    seg = seg.copy()
    if "cgm_glucose_mean" in seg.columns:
        seg["cgm_glucose_mean"] = seg["cgm_glucose_mean"].interpolate("linear", limit=CLEAN_GAP_THRESHOLDS_BINS["cgm"])
    if "heart_rate_mean" in seg.columns:
        seg["heart_rate_mean"] = seg["heart_rate_mean"].interpolate("linear", limit=CLEAN_GAP_THRESHOLDS_BINS["hr"])
    if "respiratory_rate_mean" in seg.columns:
        seg["respiratory_rate_mean"] = seg["respiratory_rate_mean"].interpolate("linear", limit=CLEAN_GAP_THRESHOLDS_BINS["rr"])
    if "activity_steps_per_min" in seg.columns:
        seg["activity_steps_per_min"] = seg["activity_steps_per_min"].fillna(0.0)
    return seg


@dataclass
class AireadiSchema:
    participant_id: str = "participant_id"
    timestamp: str = "timestamp_local"
    target: str = "cgm_glucose_mean"
    target_count: str = "cgm_count"
    segment_id: str = "segment_id"
    time_idx: str = "time_idx"
    target_observed: str = "target_observed"
    dynamic_reals: List[str] = field(default_factory=list)
    time_reals: List[str] = field(default_factory=list)
    static_reals: List[str] = field(default_factory=list)
    static_categoricals: List[str] = field(default_factory=list)
    scenario_reals: List[str] = field(default_factory=list)
    scenario_groups: Dict[str, List[str]] = field(default_factory=dict)
    subgroup_columns: List[str] = field(default_factory=list)
    derived_columns: Dict[str, str] = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return asdict(self)


@dataclass
class AireadiStreamSplit:
    mode: str
    participant_split: Dict[str, str]
    fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 42
    stratify_col: Optional[str] = None
    source: str = "generated"

    def participants(self, split: str) -> List[str]:
        split = VAL_ALIASES.get(split, split)
        return sorted(pid for pid, sp in self.participant_split.items() if sp == split)


@dataclass
class AireadiFeatureSpec:
    dynamic_reals: List[str]
    time_reals: List[str]
    static_reals: List[str]
    static_categoricals: List[str]
    scenario_reals: List[str]
    scenario_groups: Dict[str, List[str]]
    subgroup_columns: List[str]
    horizon_steps: int = 12
    bin_minutes: int = DEFAULT_BIN_MINUTES
    target_name: str = "cgm_glucose_mean"

    @property
    def horizon_minutes(self) -> List[int]:
        return [self.bin_minutes * (i + 1) for i in range(self.horizon_steps)]


@dataclass
class AireadiPreprocessor:
    dynamic_reals: List[str]
    time_reals: List[str]
    static_reals: List[str]
    static_categoricals: List[str]
    scenario_reals: List[str]
    continuous_stats: Dict[str, Dict[str, float]]
    static_category_maps: Dict[str, Dict[str, int]]

    @classmethod
    def fit(cls, df: pd.DataFrame, spec: AireadiFeatureSpec, train_participants: Iterable[str]):
        train_ids = {str(p) for p in train_participants}
        train_df = df[df["participant_id"].astype(str).isin(train_ids)]
        if train_df.empty:
            raise ValueError("Cannot fit AI-READI stream preprocessor: no train rows found.")

        cont_cols = list(dict.fromkeys(
            spec.dynamic_reals + spec.time_reals + spec.scenario_reals + spec.static_reals
        ))
        stats: Dict[str, Dict[str, float]] = {}
        for col in cont_cols:
            if col not in train_df.columns:
                continue
            values = pd.to_numeric(train_df[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean()) if len(filled) else 0.0
            std = float(filled.std(ddof=0)) if len(filled) else 1.0
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            stats[col] = {"median": median, "mean": mean, "std": std}

        cat_maps: Dict[str, Dict[str, int]] = {}
        for col in spec.static_categoricals:
            vals = train_df[col].fillna("__missing__").astype(str) if col in train_df.columns else pd.Series([], dtype=str)
            cats = ["__unknown__"] + sorted(v for v in vals.unique().tolist() if v != "__unknown__")
            cat_maps[col] = {cat: i for i, cat in enumerate(cats)}

        return cls(
            dynamic_reals=list(spec.dynamic_reals),
            time_reals=list(spec.time_reals),
            static_reals=list(spec.static_reals),
            static_categoricals=list(spec.static_categoricals),
            scenario_reals=list(spec.scenario_reals),
            continuous_stats=stats,
            static_category_maps=cat_maps,
        )

    def _scale_frame(self, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        arrs = []
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column {col!r} is missing during stream transformation.")
            st = self.continuous_stats.get(col, {"median": 0.0, "mean": 0.0, "std": 1.0})
            values = pd.to_numeric(df[col], errors="coerce").fillna(st["median"]).to_numpy(dtype="float32")
            arrs.append(((values - st["mean"]) / st["std"]).astype("float32"))
        if not arrs:
            return np.zeros((len(df), 0), dtype="float32")
        return np.stack(arrs, axis=1)

    def transform_dynamic(self, df: pd.DataFrame) -> np.ndarray:
        return self._scale_frame(df, self.dynamic_reals)

    def transform_time(self, df: pd.DataFrame) -> np.ndarray:
        return self._scale_frame(df, self.time_reals)

    def transform_scenario(self, df: pd.DataFrame) -> np.ndarray:
        return self._scale_frame(df, self.scenario_reals)

    def transform_static_reals(self, participant_row: pd.Series) -> np.ndarray:
        vals = []
        for col in self.static_reals:
            st = self.continuous_stats.get(col, {"median": 0.0, "mean": 0.0, "std": 1.0})
            raw = pd.to_numeric(pd.Series([participant_row.get(col, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(raw):
                raw = st["median"]
            vals.append((float(raw) - st["mean"]) / st["std"])
        return np.asarray(vals, dtype="float32")

    def transform_static_categoricals(self, participant_row: pd.Series) -> np.ndarray:
        vals = []
        for col in self.static_categoricals:
            mapping = self.static_category_maps.get(col, {"__unknown__": 0})
            raw = participant_row.get(col, "__missing__")
            key = "__missing__" if pd.isna(raw) else str(raw)
            vals.append(mapping.get(key, 0))
        return np.asarray(vals, dtype="int64")

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, d: Mapping[str, object]) -> "AireadiPreprocessor":
        return cls(
            dynamic_reals=list(d["dynamic_reals"]),
            time_reals=list(d["time_reals"]),
            static_reals=list(d["static_reals"]),
            static_categoricals=list(d["static_categoricals"]),
            scenario_reals=list(d["scenario_reals"]),
            continuous_stats={str(k): {str(kk): float(vv) for kk, vv in dict(v).items()}
                              for k, v in dict(d["continuous_stats"]).items()},
            static_category_maps={str(k): {str(kk): int(vv) for kk, vv in dict(v).items()}
                                  for k, v in dict(d["static_category_maps"]).items()},
        )


@dataclass
class AireadiParticipantStream:
    participant_id: str
    segment_id: int
    split: str
    dynamic: object
    time_features: object
    scenario_values: object
    scenario_mask: object
    static_cont: object
    static_cat: object
    target: object
    observed: object
    time_idx: object
    timestamps: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.target.shape[0])

    def to(self, device):
        import torch

        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return AireadiParticipantStream(
            participant_id=self.participant_id,
            segment_id=self.segment_id,
            split=self.split,
            dynamic=mv(self.dynamic),
            time_features=mv(self.time_features),
            scenario_values=mv(self.scenario_values),
            scenario_mask=mv(self.scenario_mask),
            static_cont=mv(self.static_cont),
            static_cat=mv(self.static_cat),
            target=mv(self.target),
            observed=mv(self.observed),
            time_idx=mv(self.time_idx),
            timestamps=self.timestamps,
            metadata=dict(self.metadata),
        )


def _similar_columns(df: pd.DataFrame, col: str, n: int = 12) -> List[str]:
    names = list(map(str, df.columns))
    close = difflib.get_close_matches(col, names, n=n, cutoff=0.25)
    tokens = [p for p in col.lower().replace("_", " ").split() if len(p) > 2]
    token_hits = [c for c in names if any(t in c.lower() for t in tokens)]
    return list(dict.fromkeys(close + token_hits))[:n]


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return
    details = []
    for col in missing:
        details.append(f"{col}: similar={_similar_columns(df, col)}")
    raise ValueError(
        "AI-READI schema validation failed. Missing required columns:\n  "
        + "\n  ".join(details)
        + f"\nAvailable columns ({len(df.columns)}): {list(df.columns)}"
    )


def _available(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def load_aireadi_panel(
    path,
    *,
    static_path=None,
    cohort_path=None,
    smoke_participants: Optional[int] = None,
) -> pd.DataFrame:
    """Load the existing AI-READI multimodal parquet and optional static table.

    Static rows are left-joined one row per participant. Time-series rows are never
    dropped by the static join.
    """
    df = pd.read_parquet(path)
    _require_columns(df, ["participant_id"])
    df["participant_id"] = df["participant_id"].astype(str)

    if cohort_path:
        cohort = pd.read_csv(cohort_path)
        _require_columns(cohort, ["participant_id"])
        retained = set(cohort["participant_id"].astype(str))
        df = df[df["participant_id"].isin(retained)].copy()

    if static_path:
        static = pd.read_parquet(static_path)
        _require_columns(static, ["participant_id"])
        static["participant_id"] = static["participant_id"].astype(str)
        static = static.drop_duplicates("participant_id", keep="first")
        add_cols = [c for c in static.columns if c == "participant_id" or c not in df.columns]
        if len(add_cols) > 1:
            before = len(df)
            df = df.merge(static[add_cols], on="participant_id", how="left", validate="many_to_one")
            if len(df) != before:
                raise RuntimeError("Static feature join changed the number of time-series rows.")

    if smoke_participants:
        keep = sorted(df["participant_id"].unique())[: int(smoke_participants)]
        df = df[df["participant_id"].isin(keep)].copy()

    return df


def infer_or_validate_schema(df: pd.DataFrame, schema: Optional[Mapping[str, object]] = None) -> AireadiSchema:
    """Infer the AI-READI stream schema from the parquet column names."""
    base = AireadiSchema()
    if schema:
        for key, value in schema.items():
            if hasattr(base, key):
                setattr(base, key, value)
    _require_columns(df, [base.participant_id, base.timestamp, base.target, base.target_count])

    dynamic_candidates = [
        base.target, base.target_count, "calories_total", "calories_per_min",
        "heart_rate_mean", "heart_rate_std", "heart_rate_min", "heart_rate_max",
        "heart_rate_count", "heart_rate_device_availability",
        "respiratory_rate_mean", "respiratory_rate_std", "respiratory_rate_min",
        "respiratory_rate_max", "respiratory_rate_count", "respiratory_rate_device_availability",
        "oxygen_saturation_mean", "oxygen_saturation_count", "oxygen_saturation_device_availability",
        "stress_level_mean", "stress_level_std", "stress_level_min", "stress_level_max",
        "stress_level_count", "stress_level_device_availability",
        "sleep_stage_awake", "sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem",
        "sleep_stage_unknown", "sleep_transitions", "sleep_efficiency",
        "activity_stage_walking", "activity_stage_sedentary", "activity_stage_generic",
        "activity_stage_running", "activity_steps_per_min", "activity_transitions",
        "activity_intensity_score",
    ]
    time_reals = ["minute_of_day", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]

    static_reals = _available(df, [
        "participants_age", "bmi_baseline", "hba1c_percent_baseline",
        "waist_to_hip_ratio_baseline", "clinical_systolic_bp_mmhg_baseline",
        "clinical_diastolic_bp_mmhg_baseline", "clinical_resting_hr_bpm_baseline",
        "serum_glucose_mgdl_baseline", "c_peptide_ngml_baseline",
        "hdl_cholesterol_mgdl_baseline", "ldl_cholesterol_mgdl_baseline",
        "triglycerides_mgdl_baseline", "demo_currently_pregnant",
        "demo_race_american_indian_or_alaska_native", "demo_race_asian",
        "demo_race_black_or_african_american", "demo_race_middle_eastern",
        "demo_race_native_hawaiian_or_pacific_islander", "demo_race_north_african",
        "demo_race_white_or_caucasian", "demo_race_other_race_ethnicity_or_origin",
        "demo_race_prefer_not_to_say", "demo_ethnicity_hispanic_or_latino",
        "demo_ethnicity_not_hispanic_or_latino", "demo_ethnicity_central_american",
        "demo_ethnicity_cuban", "demo_ethnicity_mexican_or_mexican_american",
        "demo_ethnicity_puerto_rican", "demo_ethnicity_south_american",
        "demo_ethnicity_other_hispanic_or_latino", "demo_ethnicity_dont_know",
        "demo_ethnicity_prefer_not_to_answer", "med_metformin", "med_insulin",
        "med_glp1_or_gip_glp1", "med_sglt2", "med_sulfonylurea",
        "med_thiazolidinedione", "med_any_diabetes_drug",
    ])
    static_cats = _available(df, [
        "participants_clinical_site", "participants_study_group",
        "demo_sex_at_birth", "demo_race_primary", "demo_marital_status",
    ])

    scenario_groups = {
        "meal_proxy": ["predmeal_flag"],
        "activity_proxy": _available(df, ["heart_rate_mean", "activity_steps_per_min", "stress_level_mean"]),
        "sleep_rest_proxy": _available(df, [
            "sleep_stage_awake", "sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem",
            "activity_stage_sedentary",
        ]),
    }
    scenario_reals = list(dict.fromkeys(sum(scenario_groups.values(), [])))
    subgroup_cols = _available(df, [
        "participants_study_group", "hba1c_percent_baseline", "bmi_baseline",
        "participants_clinical_site", "med_insulin", "med_any_diabetes_drug",
    ])

    base.dynamic_reals = _available(df, dynamic_candidates)
    base.time_reals = time_reals
    base.static_reals = static_reals
    base.static_categoricals = static_cats
    base.scenario_reals = scenario_reals
    base.scenario_groups = scenario_groups
    base.subgroup_columns = subgroup_cols
    base.derived_columns = {
        "predmeal_flag": "derived constant 0.0 because AI-READI has no observed meal annotations; future mask defaults to 0",
        "minute_of_day/tod_sin/tod_cos/dow_sin/dow_cos": "derived from timestamp_local",
        "segment_id/time_idx": "derived after dropping missing CGM by participant timestamp gaps",
    }
    return base


def prepare_aireadi_panel(
    df: pd.DataFrame,
    schema: AireadiSchema,
    *,
    bin_minutes: int = DEFAULT_BIN_MINUTES,
    clean_min_segment_hours: float = DEFAULT_CLEAN_MIN_SEGMENT_HOURS,
    use_clean_segment_logic: bool = True,
) -> pd.DataFrame:
    """Create stream-only derived columns and clean participant segments.

    The default segmenter mirrors ``Preprocessing/cohort_selection.py``: split at
    long core-modality gaps (CGM >30 min, HR/RR/activity >60 min), keep long clean
    segments, then interpolate only the short gaps that the existing preprocessing
    allowed. This keeps the streaming unit aligned with the saved clean segment
    artifacts while avoiding rolling-window materialization.
    """
    _require_columns(df, [schema.participant_id, schema.timestamp, schema.target, schema.target_count])
    out = df.copy()
    out[schema.participant_id] = out[schema.participant_id].astype(str)
    ts = pd.to_datetime(out[schema.timestamp])
    out["_stream_timestamp"] = ts
    local_naive = ts.dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts
    seconds = local_naive.dt.hour * 3600 + local_naive.dt.minute * 60 + local_naive.dt.second
    out["minute_of_day"] = (seconds // 60).astype("float32")
    angle = 2.0 * np.pi * seconds / 86400.0
    out["tod_sin"] = np.sin(angle).astype("float32")
    out["tod_cos"] = np.cos(angle).astype("float32")
    dow_angle = 2.0 * np.pi * local_naive.dt.dayofweek.astype(float) / 7.0
    out["dow_sin"] = np.sin(dow_angle).astype("float32")
    out["dow_cos"] = np.cos(dow_angle).astype("float32")
    if "predmeal_flag" not in out.columns:
        out["predmeal_flag"] = 0.0

    out["_target_observed_raw"] = (
        pd.to_numeric(out[schema.target], errors="coerce").notna()
        & (pd.to_numeric(out[schema.target_count], errors="coerce").fillna(0) > 0)
    )
    out = out.sort_values([schema.participant_id, "_stream_timestamp"]).reset_index(drop=True)

    if not use_clean_segment_logic:
        out = out[out["_target_observed_raw"]].copy()
        gap = pd.Timedelta(minutes=bin_minutes)
        delta = out.groupby(schema.participant_id)["_stream_timestamp"].diff()
        new_seg = delta.isna() | (delta != gap)
        out[schema.segment_id] = new_seg.groupby(out[schema.participant_id]).cumsum().astype("int64") - 1
        out[schema.time_idx] = out.groupby([schema.participant_id, schema.segment_id]).cumcount().astype("int64")
        out[schema.target_observed] = True
        return out

    min_steps = max(1, round(float(clean_min_segment_hours) * 60.0 / float(bin_minutes)))
    expected_gap = pd.Timedelta(minutes=bin_minutes)
    pieces = []
    for pid, grp in out.groupby(schema.participant_id, sort=True):
        grp = grp.sort_values("_stream_timestamp").copy()
        # Timestamp discontinuities are hard stream boundaries independent of modality gaps.
        ts_break = grp["_stream_timestamp"].diff().isna() | (grp["_stream_timestamp"].diff() != expected_gap)
        contiguous_id = ts_break.cumsum()
        next_seg_id = 0
        for _, block in grp.groupby(contiguous_id, sort=True):
            bad = pd.Series(False, index=block.index)
            for key, col in CLEAN_CORE_COLS.items():
                if col in block.columns:
                    bad |= _long_missing_run_mask(pd.to_numeric(block[col], errors="coerce"), CLEAN_GAP_THRESHOLDS_BINS[key])
            good = ~bad
            if not bool(good.any()):
                continue
            seg_label = (good & (good != good.shift(fill_value=False))).cumsum()
            seg_label[bad] = 0
            for _, seg in block[good].groupby(seg_label[good], sort=True):
                if len(seg) < min_steps:
                    continue
                seg = _impute_clean_segment(seg)
                seg[schema.segment_id] = next_seg_id
                seg[schema.time_idx] = np.arange(len(seg), dtype="int64")
                seg[schema.target_observed] = pd.to_numeric(seg[schema.target], errors="coerce").notna()
                pieces.append(seg)
                next_seg_id += 1
    if not pieces:
        raise ValueError(
            "AI-READI clean segment logic produced no streams. "
            f"Try lowering clean_min_segment_hours (currently {clean_min_segment_hours})."
        )
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values([schema.participant_id, schema.segment_id, schema.time_idx]).reset_index(drop=True)
    return out

def _normalize_split_name(value: object) -> str:
    s = str(value).strip().lower()
    return VAL_ALIASES.get(s, s)


def _stratified_participant_split(
    participants: pd.DataFrame,
    stratify_col: Optional[str],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    mapping: Dict[str, str] = {}
    if stratify_col and stratify_col in participants.columns:
        groups = participants.groupby(stratify_col, dropna=False)
        grouped = [(str(k), g["participant_id"].astype(str).tolist()) for k, g in groups]
    else:
        grouped = [("__all__", participants["participant_id"].astype(str).tolist())]
    for _name, ids in grouped:
        ids = list(rng.permutation(np.asarray(ids, dtype=object)))
        n = len(ids)
        if n == 0:
            continue
        n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
        n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = max(0, n - 2)
            n_val = max(0, n - 1 - n_test)
        for pid in ids[:n_test]:
            mapping[str(pid)] = "test"
        for pid in ids[n_test:n_test + n_val]:
            mapping[str(pid)] = "validation"
        for pid in ids[n_test + n_val:]:
            mapping[str(pid)] = "train"
    return mapping


def make_aireadi_stream_splits(
    df: pd.DataFrame,
    split_mode: str = "participant_heldout",
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    *,
    seed: int = 42,
    stratify_col: Optional[str] = None,
    existing_split_path=None,
) -> AireadiStreamSplit:
    """Make participant-level stream splits, optionally reusing split_participants.csv."""
    fractions = (float(train), float(val), float(test))
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-4):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}")

    if existing_split_path:
        sp = pd.read_csv(existing_split_path)
        _require_columns(sp, ["participant_id", "split"])
        mapping = {
            str(row["participant_id"]): _normalize_split_name(row["split"])
            for _, row in sp.iterrows()
            if _normalize_split_name(row["split"]) in SPLITS
        }
        return AireadiStreamSplit(
            mode="existing_split_participants",
            participant_split=mapping,
            fractions=fractions,
            seed=seed,
            stratify_col=stratify_col,
            source=str(existing_split_path),
        )

    if split_mode != "participant_heldout":
        raise ValueError("AI-READI stream path currently supports participant_heldout or existing_split_path.")

    participant_cols = ["participant_id"]
    if stratify_col is None:
        for cand in ("participants_study_group", "stratum", "study_group"):
            if cand in df.columns:
                stratify_col = cand
                break
    if stratify_col and stratify_col in df.columns:
        participant_cols.append(stratify_col)
    participants = df[participant_cols].drop_duplicates("participant_id")
    mapping = _stratified_participant_split(participants, stratify_col, fractions, seed)
    return AireadiStreamSplit(
        mode=split_mode,
        participant_split=mapping,
        fractions=fractions,
        seed=seed,
        stratify_col=stratify_col,
        source="generated",
    )


def build_stream_feature_spec(df: pd.DataFrame, schema: AireadiSchema, *, horizon_steps: int = 12,
                              bin_minutes: int = DEFAULT_BIN_MINUTES) -> AireadiFeatureSpec:
    missing = [c for c in schema.dynamic_reals + schema.static_reals + schema.static_categoricals
               if c not in df.columns]
    if missing:
        raise ValueError(f"Feature spec contains columns missing from panel: {missing}")
    return AireadiFeatureSpec(
        dynamic_reals=list(schema.dynamic_reals),
        time_reals=[c for c in schema.time_reals if c in df.columns],
        static_reals=list(schema.static_reals),
        static_categoricals=list(schema.static_categoricals),
        scenario_reals=[c for c in schema.scenario_reals if c in df.columns],
        scenario_groups={k: [c for c in v if c in df.columns] for k, v in schema.scenario_groups.items()},
        subgroup_columns=[c for c in schema.subgroup_columns if c in df.columns],
        horizon_steps=int(horizon_steps),
        bin_minutes=int(bin_minutes),
        target_name=schema.target,
    )


def create_scenario_values_and_masks(df: pd.DataFrame, schema: AireadiSchema) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw scenario values and factual availability masks.

    ``predmeal_flag`` is intentionally mask=0 by default: AI-READI has no meal
    annotations, so a zero value is not evidence of known no-meal.
    """
    if "predmeal_flag" not in df.columns:
        df = df.assign(predmeal_flag=0.0)
    values = pd.DataFrame(index=df.index)
    masks = pd.DataFrame(index=df.index)
    for col in schema.scenario_reals:
        if col not in df.columns:
            continue
        values[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if col == "predmeal_flag":
            masks[col] = 0.0
        else:
            masks[col] = pd.to_numeric(df[col], errors="coerce").notna().astype("float32")
    return values, masks


def make_participant_streams(
    df: pd.DataFrame,
    split: AireadiStreamSplit,
    schema: AireadiSchema,
    *,
    feature_spec: Optional[AireadiFeatureSpec] = None,
    preprocessor: Optional[AireadiPreprocessor] = None,
    splits: Sequence[str] = SPLITS,
    min_steps: Optional[int] = None,
    max_participants: Optional[int] = None,
) -> List[AireadiParticipantStream]:
    """Build chronological participant+segment streams with train-fit scalers."""
    import torch

    feature_spec = feature_spec or build_stream_feature_spec(df, schema)
    if preprocessor is None:
        preprocessor = AireadiPreprocessor.fit(df, feature_spec, split.participants("train"))
    wanted = {_normalize_split_name(s) for s in splits}
    min_steps = min_steps or (feature_spec.horizon_steps + 2)
    scenario_values, scenario_masks = create_scenario_values_and_masks(df, schema)
    work = df.copy()
    streams: List[AireadiParticipantStream] = []
    participant_seen = 0

    for pid in sorted(work[schema.participant_id].astype(str).unique()):
        sp = split.participant_split.get(str(pid))
        if sp not in wanted:
            continue
        participant_seen += 1
        if max_participants is not None and participant_seen > max_participants:
            break
        p_df = work[work[schema.participant_id].astype(str) == str(pid)]
        participant_row = p_df.iloc[0]
        static_cont = torch.tensor(preprocessor.transform_static_reals(participant_row), dtype=torch.float32)
        static_cat = torch.tensor(preprocessor.transform_static_categoricals(participant_row), dtype=torch.long)
        for seg_id, seg_df in p_df.groupby(schema.segment_id, sort=True):
            seg_df = seg_df.sort_values(schema.time_idx).copy()
            if len(seg_df) < min_steps:
                continue
            idx = seg_df.index
            dyn = torch.tensor(preprocessor.transform_dynamic(seg_df), dtype=torch.float32)
            tim = torch.tensor(preprocessor.transform_time(seg_df), dtype=torch.float32)
            scn = torch.tensor(preprocessor.transform_scenario(seg_df), dtype=torch.float32)
            mask = torch.tensor(scenario_masks.loc[idx, feature_spec.scenario_reals].to_numpy(dtype="float32"),
                                dtype=torch.float32)
            tgt = torch.tensor(pd.to_numeric(seg_df[schema.target], errors="coerce").to_numpy(dtype="float32"))
            obs = torch.tensor(seg_df[schema.target_observed].to_numpy(dtype=bool), dtype=torch.bool)
            t_idx = torch.tensor(seg_df[schema.time_idx].to_numpy(dtype="int64"), dtype=torch.long)
            meta = {c: participant_row.get(c) for c in feature_spec.subgroup_columns if c in p_df.columns}
            streams.append(AireadiParticipantStream(
                participant_id=str(pid),
                segment_id=int(seg_id),
                split=sp,
                dynamic=dyn,
                time_features=tim,
                scenario_values=scn,
                scenario_mask=mask,
                static_cont=static_cont,
                static_cat=static_cat,
                target=tgt,
                observed=obs,
                time_idx=t_idx,
                timestamps=seg_df["_stream_timestamp"].to_numpy(),
                metadata=meta,
            ))
    streams.sort(key=lambda s: (s.participant_id, s.segment_id, int(s.time_idx[0])))
    return streams


def write_schema_mapping(schema: AireadiSchema, feature_spec: AireadiFeatureSpec, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"schema": schema.to_jsonable(), "feature_spec": asdict(feature_spec)}, f, indent=2)
