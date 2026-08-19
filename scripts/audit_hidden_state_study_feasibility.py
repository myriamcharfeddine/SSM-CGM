#!/usr/bin/env python3
"""Step 0 feasibility audit for hidden-state clinical phenotyping.

This script is deliberately read-only with respect to the enriched AI-READI
dataset and model artifacts. It inventories raw clinical targets and audits
context coverage; it does not load a model, extract hidden states, or run any
inferential analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_GCS_ROOT = "gs://cgmproject2025/AIREADI"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/hidden_state_phenotype/step0_feasibility"
DEFAULT_ENRICHED_ROOT = ROOT.parent / "Data/enriched_multimodal"
DEFAULT_PANEL = DEFAULT_ENRICHED_ROOT / "final_multimodal_dataset_20260515_184339.parquet"
DEFAULT_STATIC = DEFAULT_ENRICHED_ROOT / "participant_static_features.parquet"
DEFAULT_COHORT = DEFAULT_ENRICHED_ROOT / "cohort.csv"
DEFAULT_SEGMENTS = DEFAULT_ENRICHED_ROOT / "segments.csv"
DEFAULT_SPLIT = ROOT.parent / "Data/experiment_c_split_adapt6h_seed42/split_participants.csv"
DEFAULT_CONFIG = ROOT / "configs/aireadi_stream_full.yaml"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
)
DEFAULT_SCHEMA = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/schema_mapping.json"
DEFAULT_DETECTOR_NOTEBOOK = ROOT / "notebooks/exercise_episode_detection.ipynb"
DEFAULT_DETECTOR_CONFIG = (
    ROOT / "notebooks/outputs/exercise_episode_detection_v2/detector_config.json"
)
DEFAULT_RAW_EPISODES = (
    ROOT / "notebooks/outputs/exercise_episode_detection_v2/exercise_episodes_all.parquet"
)
DEFAULT_CLEAN_EPISODES = (
    ROOT / "notebooks/outputs/exercise_episode_detection_v2/exercise_episodes_clean.parquet"
)

CLINICAL_FILES = (
    "measurement.csv",
    "observation.csv",
    "condition_occurrence.csv",
    "person.csv",
    "visit_occurrence.csv",
)
OPTIONAL_CLINICAL_FILES = ("concept.csv", "drug_exposure.csv")
DISCOVERY_ONLY_FILES = ("procedure_occurrence.csv", "dqd_omop.json")
REQUIRED_OUTPUTS = (
    "clinical_target_inventory.csv",
    "context_coverage_audit.csv",
    "context_coverage_by_participant.csv",
    "step0_feasibility_report.md",
    "step0_manifest.json",
    "step0_run.log",
)
BIN_MINUTES = 5
RESET_NEAR_MINUTES = 30
EXERCISE_PRE_MINUTES = 30
EXERCISE_RECOVERY_MINUTES = 30
EXERCISE_WEARABLE_MIN_FRACTION = 0.80

# Exact mappings used by Preprocessing/create_multimodal_with_clinical.py.
CONCEPT_MODEL_ALIASES: dict[str, str] = {
    "3004249": "clinical_systolic_bp_mmhg_baseline",
    "3012888": "clinical_diastolic_bp_mmhg_baseline",
    "4239408": "clinical_resting_hr_bpm_baseline",
    "4245997": "bmi_baseline",
    "44809433": "waist_to_hip_ratio_baseline",
    "3004410": "hba1c_percent_baseline",
    "3004501": "serum_glucose_mgdl_baseline",
    "3010084": "c_peptide_ngml_baseline",
    "3016244": "serum_insulin_uuml_baseline",
    "3007070": "hdl_cholesterol_mgdl_baseline",
    "3028288": "ldl_cholesterol_mgdl_baseline",
    "3022192": "triglycerides_mgdl_baseline",
}
EXPECTED_UNITS: dict[str, set[str]] = {
    "3004249": {"mmhg", ""},
    "3012888": {"mmhg", ""},
    "4239408": {"beats/min", "/min", "bpm", ""},
    "4245997": {"kg/m2", "kg/m²", ""},
    "44809433": {"ratio", ""},
    "3004410": {"%", "percent"},
    "3004501": {"mg/dl"},
    "3010084": {"ng/ml", "ng/l"},
    "3016244": {"uiu/ml", "uu/ml", "μiu/ml"},
    "3007070": {"mg/dl"},
    "3028288": {"mg/dl"},
    "3022192": {"mg/dl"},
}
CONVERTIBLE_UNITS: dict[str, set[str]] = {
    "3004410": {"mmol/mol"},
    "3004501": {"mmol/l"},
    "3010084": {"ng/l"},
    "3007070": {"mmol/l"},
    "3028288": {"mmol/l"},
    "3022192": {"mmol/l"},
}
OBSERVATION_SUMMARY_PREFIXES = ("cestl,", "paidscore,", "dietscore,")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--clinical-data-dir", type=Path)
    p.add_argument("--multimodal-parquet", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--static-parquet", type=Path, default=DEFAULT_STATIC)
    p.add_argument("--cohort-path", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--segments-path", type=Path, default=DEFAULT_SEGMENTS)
    p.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--feature-schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--exercise-detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    p.add_argument("--exercise-detector-notebook", type=Path, default=DEFAULT_DETECTOR_NOTEBOOK)
    p.add_argument("--raw-exercise-episodes", type=Path, default=DEFAULT_RAW_EPISODES)
    p.add_argument("--clean-exercise-episodes", type=Path, default=DEFAULT_CLEAN_EPISODES)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_id(value: Any) -> str:
    """Normalize an identifier through string operations only."""
    if value is None or pd.isna(value):
        return ""
    out = str(value).strip()
    out = re.sub(r"^AIREADI-", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\.0$", "", out)
    return out


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        st = path.stat()
        rec.update(
            size_bytes=st.st_size,
            modified_utc=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        )
        if hash_file:
            rec["sha256"] = sha256_file(path)
    return rec


def run_command(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), check=check, text=True, capture_output=True)


def setup_output(args: argparse.Namespace) -> tuple[Path, Path, str]:
    output = args.output_dir.resolve()
    cache = (args.cache_dir or output / "cache").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in REQUIRED_OUTPUTS if (output / name).exists()]
    if existing and not args.overwrite:
        names = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Refusing to overwrite an existing Step 0 run. Pass --overwrite explicitly "
            f"only after verifying the run identifier.\n{names}"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output, cache, run_id


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step0_feasibility")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_inputs(args: argparse.Namespace, cache: Path, logger: logging.Logger) -> tuple[Path, list[dict]]:
    clinical = (args.clinical_data_dir or cache).resolve()
    clinical.mkdir(parents=True, exist_ok=True)
    gcs_meta: list[dict] = []
    for name in CLINICAL_FILES:
        local = clinical / name
        uri = f"{args.gcs_root.rstrip('/')}/clinical_data/{name}"
        if not local.exists():
            if shutil.which("gcloud") is None:
                raise FileNotFoundError(f"{local} is missing and gcloud is unavailable.")
            logger.info("Downloading required small clinical table %s", uri)
            run_command(["gcloud", "storage", "cp", uri, str(local)])
        try:
            proc = run_command(
                ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                check=False,
            )
            metadata = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
        except Exception as exc:  # manifest remains useful offline
            logger.warning("Could not query GCS metadata for %s: %s", uri, exc)
            metadata = {}
        metadata.update(gcs_uri=uri, local_cache_path=str(local))
        gcs_meta.append(metadata)
    for name in OPTIONAL_CLINICAL_FILES:
        uri = f"{args.gcs_root.rstrip('/')}/clinical_data/{name}"
        exists = False
        if shutil.which("gcloud"):
            proc = run_command(["gcloud", "storage", "ls", uri], check=False)
            exists = proc.returncode == 0
        gcs_meta.append({"gcs_uri": uri, "present": exists, "local_cache_path": None})
    for name in DISCOVERY_ONLY_FILES:
        uri = f"{args.gcs_root.rstrip(chr(47))}/clinical_data/{name}"
        metadata: dict[str, Any] = {}
        if shutil.which("gcloud"):
            proc = run_command(
                ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                metadata = json.loads(proc.stdout)
        metadata.update(
            gcs_uri=uri, present=bool(metadata), local_cache_path=None, downloaded=False
        )
        gcs_meta.append(metadata)
    return clinical, gcs_meta


def validate_paths(args: argparse.Namespace) -> None:
    required = {
        "multimodal parquet": args.multimodal_parquet,
        "static parquet": args.static_parquet,
        "cohort": args.cohort_path,
        "segments": args.segments_path,
        "split manifest": args.split_manifest,
        "model config": args.model_config,
        "checkpoint": args.checkpoint,
        "feature schema": args.feature_schema,
        "exercise detector config": args.exercise_detector_config,
        "exercise detector notebook": args.exercise_detector_notebook,
        "raw exercise episodes": args.raw_exercise_episodes,
        "clean exercise episodes": args.clean_exercise_episodes,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required canonical artifact(s):\n" + "\n".join(missing))


def load_split(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    split = pd.read_csv(path, dtype=str)
    if not {"participant_id", "split"}.issubset(split.columns):
        raise ValueError(f"Canonical split lacks participant_id/split: {path}")
    split["participant_id"] = split["participant_id"].map(normalize_id)
    split["split"] = (
        split["split"].str.strip().str.lower().replace({"val": "validation", "valid": "validation"})
    )
    split = split[split["split"].isin(["train", "validation", "test"])].copy()
    if split.empty:
        raise ValueError("No canonical split could be resolved.")
    dup = split.groupby("participant_id")["split"].nunique()
    if (dup > 1).any():
        raise ValueError(f"Participants appear in multiple splits: {dup[dup > 1].index.tolist()[:20]}")
    if split["participant_id"].duplicated().any():
        split = split.drop_duplicates("participant_id", keep="first")
    mapping = dict(zip(split["participant_id"], split["split"]))
    return split, mapping


def load_model_schema(path: Path) -> tuple[dict, set[str], dict[str, str]]:
    data = json.loads(path.read_text())
    schema = data.get("schema", data)
    status: dict[str, str] = {}
    for col in schema.get("static_reals", []):
        status[str(col)] = "static_input"
    for col in schema.get("static_categoricals", []):
        status[str(col)] = "static_input"
    for col in schema.get("dynamic_reals", []):
        status[str(col)] = "dynamic_input"
    for col in schema.get("time_reals", []):
        status[str(col)] = "derived_input"
    for col in schema.get("derived_columns", {}):
        for token in str(col).split("/"):
            status[token] = "derived_input"
    if not status or not schema.get("dynamic_reals") or not schema.get("static_reals"):
        raise ValueError("Model-input feature classification cannot be resolved from schema metadata.")
    return schema, set(status), status


def local_naive_from_timestamp(series: pd.Series) -> pd.Series:
    """Drop timezone while preserving displayed wall time."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_localize(None)
    return pd.to_datetime(series, errors="coerce")


def participant_local_time(frame: pd.DataFrame) -> pd.Series:
    """Convert stored instants to each row's declared participant timezone."""
    ts = pd.to_datetime(frame["timestamp_local"], errors="coerce", utc=True)
    out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for tz_name, idx in frame.groupby("timezone", dropna=False).groups.items():
        try:
            if pd.isna(tz_name) or not str(tz_name).strip():
                converted = ts.loc[idx].dt.tz_localize(None)
            else:
                converted = ts.loc[idx].dt.tz_convert(str(tz_name)).dt.tz_localize(None)
        except Exception:
            converted = ts.loc[idx].dt.tz_localize(None)
        out.loc[idx] = converted.to_numpy(dtype="datetime64[ns]")
    return out


def read_panel_context(
    args: argparse.Namespace,
    cohort_ids: set[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    desired = [
        "participant_id",
        "timezone",
        "timestamp_local",
        "cgm_glucose_mean",
        "cgm_count",
        "heart_rate_mean",
        "respiratory_rate_mean",
        "activity_steps_per_min",
        "sleep_stage_awake",
        "sleep_stage_light",
        "sleep_stage_deep",
        "sleep_stage_rem",
        "sleep_stage_unknown",
        "participants_clinical_site",
        "participants_study_group",
    ]
    pf = pq.ParquetFile(args.multimodal_parquet)
    all_columns = pf.schema_arrow.names
    missing = [c for c in desired if c not in all_columns]
    if missing:
        raise ValueError(f"Context audit columns missing from canonical panel: {missing}")
    logger.info("Reading %d context columns from canonical enriched panel", len(desired))
    panel = pq.read_table(args.multimodal_parquet, columns=desired).to_pandas()
    panel["participant_id"] = panel["participant_id"].map(normalize_id)
    panel = panel[panel["participant_id"].isin(cohort_ids)].copy()
    return panel, all_columns


def get_cgm_bounds(panel: pd.DataFrame) -> pd.DataFrame:
    valid = (
        pd.to_numeric(panel["cgm_glucose_mean"], errors="coerce").notna()
        & (pd.to_numeric(panel["cgm_count"], errors="coerce").fillna(0) > 0)
    )
    work = panel.loc[valid, ["participant_id", "timezone", "timestamp_local"]].copy()
    work["_local_naive"] = participant_local_time(work)
    bounds = work.groupby("participant_id")["_local_naive"].agg(cgm_start="min", cgm_end="max")
    return bounds


def prepare_clean_panel(panel: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    panel = panel.copy()
    panel["_hr_observed_raw"] = pd.to_numeric(panel["heart_rate_mean"], errors="coerce").notna()
    panel["_rr_observed_raw"] = pd.to_numeric(panel["respiratory_rate_mean"], errors="coerce").notna()
    panel["_activity_observed_raw"] = pd.to_numeric(panel["activity_steps_per_min"], errors="coerce").notna()
    # Import the canonical implementation rather than copying/reinterpreting its rules.
    from ssmcgm.data.aireadi import AireadiSchema, prepare_aireadi_panel

    schema = AireadiSchema(
        participant_id="participant_id",
        timestamp="timestamp_local",
        target="cgm_glucose_mean",
        target_count="cgm_count",
    )
    logger.info("Applying canonical 49-hour clean-segment logic from ssmcgm/data/aireadi.py")
    clean = prepare_aireadi_panel(
        panel,
        schema,
        bin_minutes=BIN_MINUTES,
        clean_min_segment_hours=49,
        use_clean_segment_logic=True,
    )
    clean["_participant_local_time"] = participant_local_time(clean)
    clean["_valid_cgm"] = pd.to_numeric(clean["cgm_glucose_mean"], errors="coerce").notna()
    clean["_near_reset"] = clean["time_idx"] < math.ceil(RESET_NEAR_MINUTES / BIN_MINUTES)
    return clean


def segment_summary(clean: pd.DataFrame) -> pd.DataFrame:
    seg = (
        clean.groupby(["participant_id", "segment_id"], as_index=False)
        .agg(n_rows=("time_idx", "size"))
    )
    seg["segment_hours"] = seg["n_rows"] * BIN_MINUTES / 60.0
    return seg.groupby("participant_id").agg(
        total_clean_hours=("segment_hours", "sum"),
        n_segments=("segment_id", "nunique"),
        median_segment_hours=("segment_hours", "median"),
    )


def snake_name(source: str, concept_id: str) -> str:
    text = str(source or "").strip()
    if "," in text:
        prefix, label = text.split(",", 1)
        text = label.strip() or prefix.strip()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    if not text:
        text = f"concept_{concept_id}"
    return text[:120]


def clinical_domain(label: str) -> str:
    s = label.lower()
    rules = [
        ("inflammatory", ("c reactive", "crp", "inflamm")),
        ("renal", ("creatinine", "urea nitrogen", "bun", "kidney", "renal", "urine albumin")),
        ("hepatic", ("alanine", "aspartate", "bilirubin", "albumin", "globulin", "alkaline phosphatase", "protein total")),
        ("cardiovascular", ("troponin", "natriuretic", "probnp", "heart", "stroke", "circulation", "blood pressure", "systolic", "diastolic")),
        ("cognitive", ("moca", "cognit", "dementia", "parkinson", "memory", "visuospatial", "orientation", "abstraction")),
        ("neuropathy", ("foot", "neurop", "felt")),
        ("hematologic", ("white blood", "red blood", "platelet", "hematocrit", "hemoglobin", "mcv", "mch", "rdw")),
        ("metabolic_glycemic", ("glucose", "insulin", "c peptide", "hba1c", "a1c", "diabetes", "bmi", "obesity", "weight", "waist", "hip")),
        ("lipid", ("cholesterol", "triglyceride", "hdl", "ldl")),
        ("electrolyte", ("sodium", "potassium", "chloride", "calcium", "carbon dioxide")),
        ("ophthalmic", ("eye", "retina", "visual", "photopic", "mesopic", "logmar", "contrast sensitivity", "glaucoma", "cataract", "autorefractor")),
        ("mental_health", ("cesd", "depress", "paid score")),
        ("respiratory", ("pulmonary", "lung")),
        ("musculoskeletal", ("arthritis", "osteoporosis")),
    ]
    for domain, tokens in rules:
        if any(t in s for t in tokens):
            return domain
    return "other_clinical"


def all_table_participant_alignment(clinical: Path, split_map: Mapping[str, str]) -> dict[str, Any]:
    cohort_ids = set(split_map)
    out: dict[str, Any] = {}
    for filename in CLINICAL_FILES:
        table = filename.removesuffix(".csv")
        ids = pd.read_csv(clinical / filename, dtype=str, usecols=["person_id"])["person_id"].map(normalize_id)
        raw_ids = set(ids) - {""}
        overlap = raw_ids & cohort_ids
        out[table] = {
            "n_unique_participants": len(raw_ids),
            "n_model_cohort_overlap": len(overlap),
            "n_raw_not_model_cohort": len(raw_ids - cohort_ids),
            "n_model_cohort_not_in_table": len(cohort_ids - raw_ids),
            "model_cohort_overlap_fraction": len(overlap) / len(cohort_ids) if cohort_ids else np.nan,
            "sample_raw_not_model_cohort": sorted(raw_ids - cohort_ids)[:20],
            "sample_model_cohort_not_in_table": sorted(cohort_ids - raw_ids)[:20],
        }
    return out


def load_clinical_tables(clinical: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    measurement_cols = [
        "person_id",
        "measurement_concept_id",
        "measurement_source_value",
        "measurement_type_concept_id",
        "value_as_number",
        "value_as_concept_id",
        "unit_concept_id",
        "unit_source_value",
        "measurement_date",
        "measurement_datetime",
        "visit_occurrence_id",
        "range_low",
        "range_high",
        "qualifier_source_value",
    ]
    observation_cols = [
        "person_id",
        "observation_concept_id",
        "observation_source_value",
        "observation_type_concept_id",
        "value_as_number",
        "value_as_string",
        "value_as_concept_id",
        "unit_concept_id",
        "unit_source_value",
        "observation_date",
        "observation_datetime",
        "visit_occurrence_id",
    ]
    condition_cols = [
        "person_id",
        "condition_concept_id",
        "condition_source_value",
        "condition_start_date",
        "condition_start_datetime",
        "visit_occurrence_id",
    ]
    logger.info("Reading raw measurement table")
    measurement = pd.read_csv(
        clinical / "measurement.csv", dtype=str, usecols=measurement_cols, low_memory=False
    )
    logger.info("Reading raw observation table and retaining summary outcomes only")
    observation = pd.read_csv(
        clinical / "observation.csv", dtype=str, usecols=observation_cols, low_memory=False
    )
    src = observation["observation_source_value"].fillna("").str.strip().str.lower()
    observation = observation[src.str.startswith(OBSERVATION_SUMMARY_PREFIXES)].copy()
    logger.info("Reading raw condition-occurrence indicators")
    condition = pd.read_csv(
        clinical / "condition_occurrence.csv", dtype=str, usecols=condition_cols, low_memory=False
    )
    for df in (measurement, observation, condition):
        df["person_id"] = df["person_id"].map(normalize_id)
    return {
        "measurement": measurement,
        "observation": observation,
        "condition_occurrence": condition,
    }


def quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def pct(n: int | float, d: int | float) -> float:
    return 100.0 * float(n) / float(d) if d else np.nan


def frac(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def unit_assessment(
    group: pd.DataFrame,
    concept_id: str,
    target_type: str,
    label: str,
) -> dict[str, Any]:
    if target_type == "binary_indicator":
        return {
            "units_observed": "",
            "dominant_unit": "",
            "dominant_unit_fraction": np.nan,
            "n_distinct_units": 0,
            "missing_unit_fraction": 1.0,
            "unit_status": "nonnumeric_or_concept_value",
            "conversion_rule_available": False,
            "units_directly_compatible": True,
            "mixed_units_require_conversion": False,
            "unsafe_to_analyze": False,
        }
    units = group.get("unit_source_value", pd.Series("", index=group.index)).fillna("").astype(str).str.strip()
    counts = units.value_counts(dropna=False)
    observed_nonblank = counts[counts.index != ""]
    units_text = json.dumps(
        {("<missing>" if str(k) == "" else str(k)): int(v) for k, v in counts.items()},
        sort_keys=True,
    )
    dominant = str(counts.index[0]) if len(counts) else ""
    missing_fraction = float((units == "").mean()) if len(units) else 1.0
    expected = EXPECTED_UNITS.get(concept_id, set())
    convertible = CONVERTIBLE_UNITS.get(concept_id, set())
    normalized = {u.lower() for u in units.unique()}
    dimensionless_hint = any(
        token in label.lower()
        for token in ("score", "ratio", "number of", "felt", "misses", "orientation", "memory", "naming", "subtraction", "repetition", "fluency", "abstraction")
    )
    if not observed_nonblank.size:
        if "" in expected or dimensionless_hint:
            status = "single_valid_unit"
            direct = True
            unsafe = False
        else:
            status = "missing_units"
            direct = False
            unsafe = True
    elif len(observed_nonblank) == 1:
        status = "single_valid_unit"
        direct = True
        unsafe = False
    elif expected and normalized.issubset(expected | convertible):
        status = "convertible_mixed_units"
        direct = normalized.issubset(expected)
        unsafe = False
    elif len(observed_nonblank) == 1 and (missing_fraction == 0 or "" in expected):
        status = "single_valid_unit"
        direct = True
        unsafe = False
    else:
        status = "incompatible_or_unknown_units"
        direct = False
        unsafe = True
    return {
        "units_observed": units_text,
        "dominant_unit": dominant or "<missing>",
        "dominant_unit_fraction": float(counts.iloc[0] / len(units)) if len(units) else np.nan,
        "n_distinct_units": int(len(observed_nonblank)),
        "missing_unit_fraction": missing_fraction,
        "unit_status": status,
        "conversion_rule_available": bool(expected or convertible),
        "units_directly_compatible": direct,
        "mixed_units_require_conversion": status == "convertible_mixed_units",
        "unsafe_to_analyze": unsafe,
    }


def published_cca(label: str) -> tuple[str, str, str]:
    s = label.lower()
    if "troponin" in s:
        return (
            "explicitly_named",
            "static_embedding",
            "Published Appendix A.13 text names troponin; exact local source figure was not available.",
        )
    if "hba1c" in s or "hemoglobin a1c" in s:
        return ("explicitly_named", "static_embedding", "Published Appendix A.13 text names HbA1c.")
    if "insulin" in s and "resistance" not in s:
        return ("explicitly_named", "static_embedding", "Published Appendix A.13 text names insulin.")
    if "glucose" in s and "urine" not in s:
        return ("explicitly_named", "static_embedding", "Published Appendix A.13 text names glucose.")
    if any(t in s for t in ("total cholesterol", "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "cholesterol in hdl", "cholesterol in ldl", "triglyceride")):
        return (
            "family_match",
            "static_embedding",
            "The paper says 'cholesterol features'; the exact variable is not locally confirmed.",
        )
    return ("not_explicitly_reported", "not_reported", "Not explicitly named in the available Appendix A.13 description.")


def input_classification(
    source_table: str,
    concept_id: str,
    panel_columns: set[str],
    feature_status: Mapping[str, str],
    schema_path: Path,
) -> dict[str, Any]:
    alias = CONCEPT_MODEL_ALIASES.get(concept_id, "") if source_table == "measurement" else ""
    present = bool(alias and alias in panel_columns)
    consumed = bool(alias and alias in feature_status)
    if consumed:
        status = feature_status[alias]
        features = alias
        evidence = f"{schema_path}: exact concept-to-enrichment alias and saved feature schema"
    elif alias and present:
        status = "not_model_input"
        features = ""
        evidence = f"{schema_path}: present in panel but absent from saved consumed-feature lists"
    elif alias:
        status = "not_model_input"
        features = ""
        evidence = f"{schema_path}: mapped enrichment alias absent from panel and consumed-feature lists"
    else:
        status = "not_model_input"
        features = ""
        evidence = f"{schema_path}: no exact concept alias in saved consumed-feature lists"
    return {
        "present_in_enriched_dataset": present,
        "enriched_column_names": alias if present else "",
        "model_input_status": status,
        "model_input_feature_names": features,
        "model_input_evidence_source": evidence,
    }


def inventory_targets(
    tables: Mapping[str, pd.DataFrame],
    split_map: Mapping[str, str],
    cgm_bounds: pd.DataFrame,
    panel_columns: set[str],
    feature_status: Mapping[str, str],
    schema_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort_ids = set(split_map)
    split_sizes = Counter(split_map.values())
    cgm_start = cgm_bounds["cgm_start"].to_dict()
    rows: list[dict[str, Any]] = []
    seen_names: Counter[str] = Counter()
    table_alignment: dict[str, Any] = {}

    specs = {
        "measurement": {
            "concept": "measurement_concept_id",
            "source": "measurement_source_value",
            "date": "measurement_date",
            "datetime": "measurement_datetime",
            "value": "value_as_number",
        },
        "observation": {
            "concept": "observation_concept_id",
            "source": "observation_source_value",
            "date": "observation_date",
            "datetime": "observation_datetime",
            "value": "value_as_number",
        },
        "condition_occurrence": {
            "concept": "condition_concept_id",
            "source": "condition_source_value",
            "date": "condition_start_date",
            "datetime": "condition_start_datetime",
            "value": None,
        },
    }

    for table_name, df in tables.items():
        spec = specs[table_name]
        raw_ids = set(df["person_id"]) - {""}
        overlap = raw_ids & cohort_ids
        table_alignment[table_name] = {
            "n_unique_participants": len(raw_ids),
            "n_model_cohort_overlap": len(overlap),
            "n_raw_not_model_cohort": len(raw_ids - cohort_ids),
            "n_model_cohort_not_in_table": len(cohort_ids - raw_ids),
            "model_cohort_overlap_fraction": len(overlap) / len(cohort_ids) if cohort_ids else np.nan,
            "sample_raw_not_model_cohort": sorted(raw_ids - cohort_ids)[:20],
            "sample_model_cohort_not_in_table": sorted(cohort_ids - raw_ids)[:20],
        }
        if table_name in {"measurement", "observation"} and len(overlap) / max(1, len(cohort_ids)) < 0.80:
            raise ValueError(
                f"Participant ID overlap unexpectedly low for {table_name}: "
                f"{len(overlap)}/{len(cohort_ids)}"
            )

        group_cols = [spec["concept"], spec["source"]]
        for (concept_raw, source_raw), group in df.groupby(group_cols, dropna=False, sort=True):
            concept_id = normalize_id(concept_raw)
            source = "" if pd.isna(source_raw) else str(source_raw).strip()
            base_name = snake_name(source, concept_id)
            seen_names[base_name] += 1
            normalized = base_name
            if seen_names[base_name] > 1:
                normalized = f"{base_name}__{concept_id}"
                if seen_names[base_name] > 2:
                    normalized += f"_{seen_names[base_name]}"

            target_type = "binary_indicator" if table_name == "condition_occurrence" else "continuous"
            value_rep = "condition_presence" if target_type == "binary_indicator" else "value_as_number"
            work = group.copy()
            work["_pid"] = work["person_id"].map(normalize_id)
            work["_split"] = work["_pid"].map(split_map)
            cohort = work[work["_pid"].isin(cohort_ids)].copy()
            if spec["value"]:
                work["_numeric"] = pd.to_numeric(work[spec["value"]], errors="coerce")
                cohort["_numeric"] = pd.to_numeric(cohort[spec["value"]], errors="coerce")
            else:
                work["_numeric"] = 1.0
                cohort["_numeric"] = 1.0

            dt = pd.to_datetime(cohort[spec["datetime"]], errors="coerce")
            date_fallback = pd.to_datetime(cohort[spec["date"]], errors="coerce")
            cohort["_measurement_dt"] = dt.fillna(date_fallback)
            parse_fraction = float(cohort["_measurement_dt"].notna().mean()) if len(cohort) else np.nan
            valid = cohort[cohort["_numeric"].notna()].copy()
            valid["_cgm_start"] = valid["_pid"].map(cgm_start)
            valid["_days"] = (
                valid["_measurement_dt"] - pd.to_datetime(valid["_cgm_start"], errors="coerce")
            ).dt.total_seconds() / 86400.0
            dated = valid["_days"].dropna()

            raw_participants = set(work["_pid"]) - {""}
            cohort_participants = set(cohort["_pid"]) - {""}
            valid_participants = set(valid["_pid"]) - {""}
            n_by_split = {
                sp: len(set(cohort.loc[cohort["_split"].eq(sp), "_pid"]) - {""})
                for sp in ("train", "validation", "test")
            }
            nv_by_split = {
                sp: len(set(valid.loc[valid["_split"].eq(sp), "_pid"]) - {""})
                for sp in ("train", "validation", "test")
            }
            numeric_values = valid["_numeric"].dropna()
            value_counts = numeric_values.value_counts(normalize=True)
            almost_constant = (
                len(numeric_values) == 0
                or numeric_values.nunique() <= 1
                or (len(value_counts) and float(value_counts.iloc[0]) >= 0.95)
            )
            units = unit_assessment(group, concept_id, target_type, source)
            inp = input_classification(
                table_name, concept_id, panel_columns, feature_status, schema_path
            )
            cca_status, cca_scope, cca_note = published_cca(source)
            within_180 = frac(dated.abs() <= 180)
            within_90 = frac(dated.abs() <= 90)
            timing_ok = bool(pd.notna(within_180) and within_180 >= 0.80)
            timing90 = bool(pd.notna(within_90) and within_90 >= 0.80)
            unit_ok = not units["unsafe_to_analyze"]

            test_valid = nv_by_split["test"]
            vt_valid = nv_by_split["validation"] + nv_by_split["test"]
            pos_test = len(set(valid.loc[valid["_split"].eq("test"), "_pid"]))
            pos_vt = len(set(valid.loc[valid["_split"].isin(["validation", "test"]), "_pid"]))
            neg_test = split_sizes["test"] - pos_test
            neg_vt = split_sizes["validation"] + split_sizes["test"] - pos_vt
            if target_type == "binary_indicator":
                strict = unit_ok and timing_ok and pos_test >= 40 and neg_test >= 40
                combined = unit_ok and timing_ok and pos_vt >= 40 and neg_vt >= 40
            else:
                strict = (
                    inp["model_input_status"] == "not_model_input"
                    and test_valid >= 100
                    and unit_ok
                    and timing_ok
                    and not almost_constant
                )
                combined = (
                    inp["model_input_status"] == "not_model_input"
                    and vt_valid >= 100
                    and unit_ok
                    and timing_ok
                    and not almost_constant
                )

            if inp["model_input_status"] in {"static_input", "dynamic_input", "derived_input"}:
                role = "direct_input_positive_control"
                reason = "Exact saved-schema model input; not an external target."
            elif inp["model_input_status"] == "ambiguous":
                role = "ambiguous_target"
                reason = "Model-input mapping is ambiguous."
            elif strict:
                role = "external_primary_candidate"
                reason = "Meets strict untouched-test coverage, unit, timing, and non-input rules."
            elif combined:
                role = "exploratory_candidate"
                reason = "Meets combined validation-plus-test exploratory rules, not strict test rules."
            elif cca_status in {"explicitly_named", "family_match"}:
                role = "published_replication_reference"
                reason = "Previously named or family-matched in static-embedding CCA; current feasibility rules not met."
            elif not unit_ok:
                role = "invalid_units"
                reason = f"Unit status is {units['unit_status']}."
            elif not timing_ok:
                role = "invalid_timing"
                reason = "Fewer than 80% of dated values are within ±180 days of CGM start."
            else:
                role = "insufficient_coverage"
                reason = "Does not meet strict or combined participant coverage thresholds."

            manual = bool(
                units["unit_status"] in {"incompatible_or_unknown_units", "missing_units"}
                or parse_fraction < 0.80
                or cca_status == "family_match"
                or (target_type == "binary_indicator")
            )
            row = {
                "source_table": table_name,
                "source_field_or_concept": f"{spec['source']}|{concept_id}",
                "measurement_concept_id": concept_id,
                "measurement_source_value": source,
                "concept_name": "",
                "normalized_target_name": normalized,
                "clinical_domain": clinical_domain(source),
                "target_type": target_type,
                "value_representation": value_rep,
                "n_rows_raw": len(group),
                "n_unique_participants_raw": len(raw_participants),
                "n_model_cohort": len(cohort_participants),
                "n_train": n_by_split["train"],
                "n_validation": n_by_split["validation"],
                "n_test": n_by_split["test"],
                "n_validation_plus_test": n_by_split["validation"] + n_by_split["test"],
                "coverage_model_cohort_pct": pct(len(cohort_participants), len(cohort_ids)),
                "coverage_train_pct": pct(n_by_split["train"], split_sizes["train"]),
                "coverage_validation_pct": pct(n_by_split["validation"], split_sizes["validation"]),
                "coverage_test_pct": pct(n_by_split["test"], split_sizes["test"]),
                "coverage_validation_plus_test_pct": pct(
                    n_by_split["validation"] + n_by_split["test"],
                    split_sizes["validation"] + split_sizes["test"],
                ),
                "n_valid_numeric_total": len(valid_participants),
                "n_valid_numeric_train": nv_by_split["train"],
                "n_valid_numeric_validation": nv_by_split["validation"],
                "n_valid_numeric_test": nv_by_split["test"],
                "n_valid_numeric_rows": int(valid["_numeric"].notna().sum()),
                "n_missing_numeric": int(cohort["_numeric"].isna().sum()),
                "minimum": float(numeric_values.min()) if len(numeric_values) else np.nan,
                "maximum": float(numeric_values.max()) if len(numeric_values) else np.nan,
                "median": float(numeric_values.median()) if len(numeric_values) else np.nan,
                "IQR": quantile(numeric_values, 0.75) - quantile(numeric_values, 0.25),
                "almost_constant": almost_constant,
                **units,
                "measurement_date_min": cohort["_measurement_dt"].min(),
                "measurement_date_max": cohort["_measurement_dt"].max(),
                "median_days_to_cgm_start": float(dated.median()) if len(dated) else np.nan,
                "iqr_days_to_cgm_start": quantile(dated, 0.75) - quantile(dated, 0.25),
                "p10_days_to_cgm_start": quantile(dated, 0.10),
                "p90_days_to_cgm_start": quantile(dated, 0.90),
                "median_absolute_days_to_cgm_start": float(dated.abs().median()) if len(dated) else np.nan,
                "fraction_before_cgm": frac(dated < 0),
                "fraction_after_cgm": frac(dated > 0),
                "fraction_same_day_as_cgm": frac(dated == 0),
                "fraction_within_30d": frac(dated.abs() <= 30),
                "fraction_within_90d": within_90,
                "fraction_within_180d": within_180,
                "missing_measurement_date_fraction": 1.0 - parse_fraction if pd.notna(parse_fraction) else np.nan,
                "baseline_candidate_available_fraction": (
                    valid.dropna(subset=["_measurement_dt"])["_pid"].nunique()
                    / max(1, valid["_pid"].nunique())
                ),
                **inp,
                "published_cca_status": cca_status,
                "published_cca_scope": cca_scope,
                "published_cca_note": cca_note,
                "n_positive_test": pos_test if target_type == "binary_indicator" else np.nan,
                "n_negative_test": neg_test if target_type == "binary_indicator" else np.nan,
                "n_positive_validation_plus_test": pos_vt if target_type == "binary_indicator" else np.nan,
                "n_negative_validation_plus_test": neg_vt if target_type == "binary_indicator" else np.nan,
                "strict_test_eligible": strict,
                "combined_exploratory_eligible": combined,
                "timing_90d_sensitivity_eligible": timing90,
                "target_role": role,
                "eligibility_reason": reason,
                "manual_review_required": manual,
            }
            rows.append(row)

    inventory = pd.DataFrame(rows)
    if inventory["normalized_target_name"].duplicated().any():
        dup = inventory.loc[
            inventory["normalized_target_name"].duplicated(False), "normalized_target_name"
        ].tolist()
        raise ValueError(f"Duplicate target identifiers remain after normalization: {dup}")
    date_parse = 1.0 - inventory.loc[
        inventory["source_table"].eq("measurement"), "missing_measurement_date_fraction"
    ].median()
    if pd.isna(date_parse) or date_parse < 0.80:
        raise ValueError(f"Clinical dates cannot be parsed for most measurement records: {date_parse}")
    role_order = {
        "external_primary_candidate": 0,
        "direct_input_positive_control": 1,
        "published_replication_reference": 2,
        "exploratory_candidate": 3,
        "insufficient_coverage": 4,
        "invalid_units": 5,
        "invalid_timing": 6,
        "ambiguous_target": 7,
        "exclude": 8,
    }
    inventory["_role_order"] = inventory["target_role"].map(role_order).fillna(99)
    inventory = inventory.sort_values(
        ["_role_order", "coverage_test_pct", "normalized_target_name"],
        ascending=[True, False, True],
    ).drop(columns="_role_order")
    logger.info("Inventoried %d distinct candidate targets", len(inventory))
    return inventory, table_alignment


def context_participant_metrics(
    clean: pd.DataFrame,
    split_map: Mapping[str, str],
    cgm_bounds: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    work = clean.copy()
    sleep_cols = ["sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem"]
    asleep = work[sleep_cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
    unknown = pd.to_numeric(work["sleep_stage_unknown"], errors="coerce").fillna(0).gt(0)
    hour = work["_participant_local_time"].dt.hour
    work["_night"] = asleep & ~unknown & work["_valid_cgm"]
    work["_night_clock"] = hour.between(0, 5) & work["_valid_cgm"]
    work["_day"] = hour.between(8, 19) & ~asleep & work["_valid_cgm"]
    work["_night_label"] = (work["_participant_local_time"] - pd.Timedelta(hours=12)).dt.date
    work["_day_label"] = work["_participant_local_time"].dt.date
    work["_hr_avail"] = work["_hr_observed_raw"].astype(bool)
    work["_rr_avail"] = work["_rr_observed_raw"].astype(bool)
    work["_activity_avail"] = work["_activity_observed_raw"].astype(bool)

    seg_stats = segment_summary(work)
    participant_base = (
        work.groupby("participant_id")
        .agg(
            fraction_rows_near_reset=("_near_reset", "mean"),
            clinical_site=("participants_clinical_site", "first"),
            study_group=("participants_study_group", "first"),
        )
        .join(seg_stats)
        .join(cgm_bounds)
    )
    participant_base["split"] = participant_base.index.map(split_map)

    context_frames: dict[str, pd.DataFrame] = {}
    for name, mask_col, label_col in (
        ("night", "_night", "_night_label"),
        ("night_clock", "_night_clock", "_night_label"),
        ("day", "_day", "_day_label"),
    ):
        ctx = work[work[mask_col]].copy()
        if ctx.empty:
            per = pd.DataFrame(index=participant_base.index)
            by_period = pd.DataFrame(columns=["participant_id", label_col, "hours"])
        else:
            by_period = (
                ctx.groupby(["participant_id", label_col], as_index=False)
                .agg(hours=("_valid_cgm", lambda x: len(x) * BIN_MINUTES / 60.0))
            )
            per = ctx.groupby("participant_id").agg(
                valid_rows=("_valid_cgm", "size"),
                distinct_periods=(label_col, "nunique"),
                wearable_hr=("_hr_avail", "mean"),
                wearable_rr=("_rr_avail", "mean"),
                wearable_activity=("_activity_avail", "mean"),
                fraction_near_reset=("_near_reset", "mean"),
            )
        per = participant_base[[]].join(per)
        per["valid_rows"] = per["valid_rows"].fillna(0)
        per["valid_hours"] = per["valid_rows"] * BIN_MINUTES / 60.0
        per["distinct_periods"] = per["distinct_periods"].fillna(0).astype(int)
        period_stats = by_period.groupby("participant_id")["hours"].agg(
            median_hours_per_period="median",
            periods_ge_2h=lambda x: int((x >= 2).sum()),
        )
        per = per.join(period_stats)
        per["median_hours_per_period"] = per["median_hours_per_period"].fillna(0)
        per["periods_ge_2h"] = per["periods_ge_2h"].fillna(0).astype(int)
        per["wearable_availability"] = per[
            ["wearable_hr", "wearable_rr", "wearable_activity"]
        ].mean(axis=1)
        context_frames[name] = per

    out = participant_base.copy()
    night = context_frames["night"]
    clock = context_frames["night_clock"]
    day = context_frames["day"]
    out["night_valid_hours"] = night["valid_hours"]
    out["night_distinct_nights"] = night["distinct_periods"]
    out["night_nights_ge_2h"] = night["periods_ge_2h"]
    out["night_median_valid_hours_per_night"] = night["median_hours_per_period"]
    out["night_hr_availability"] = night["wearable_hr"]
    out["night_rr_availability"] = night["wearable_rr"]
    out["night_activity_availability"] = night["wearable_activity"]
    out["night_fraction_near_reset"] = night["fraction_near_reset"]
    out["night_eligible_any"] = out["night_valid_hours"] > 0
    out["night_eligible_2h"] = out["night_nights_ge_2h"] >= 1
    out["night_eligible_2nights"] = out["night_nights_ge_2h"] >= 2
    out["night_eligible_6h_total"] = out["night_valid_hours"] >= 6
    out["night_clock_valid_hours"] = clock["valid_hours"]
    out["night_clock_distinct_nights"] = clock["distinct_periods"]
    out["night_clock_eligible_any"] = out["night_clock_valid_hours"] > 0
    out["day_valid_hours"] = day["valid_hours"]
    out["day_distinct_days"] = day["distinct_periods"]
    out["day_days_ge_2h"] = day["periods_ge_2h"]
    out["day_median_valid_hours_per_day"] = day["median_hours_per_period"]
    out["day_wearable_availability"] = day["wearable_availability"]
    out["day_fraction_near_reset"] = day["fraction_near_reset"]
    out["day_eligible_any"] = out["day_valid_hours"] > 0
    out["day_eligible_6h_total"] = out["day_valid_hours"] >= 6
    out["day_eligible_2days"] = out["day_days_ge_2h"] >= 2
    logger.info("Calculated sleep-derived night, clock-night, and daytime participant coverage")
    return out, context_frames


def timestamp_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, errors="coerce", utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")


def exercise_metrics(
    clean: pd.DataFrame,
    base: pd.DataFrame,
    raw_path: Path,
    clean_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(raw_path).copy()
    detector_clean = pd.read_parquet(clean_path).copy()
    for frame in (raw, detector_clean):
        frame["participant_id"] = frame["participant_id"].map(normalize_id)
        for col in ("start_time", "end_time", "refined_start_time", "refined_end_time"):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
        frame["_start"] = frame.get("refined_start_time", frame.get("start_time")).fillna(
            frame.get("start_time")
        )
        frame["_end"] = frame.get("refined_end_time", frame.get("end_time")).fillna(
            frame.get("end_time")
        )
        frame["_duration_min"] = (
            (frame["_end"] - frame["_start"]).dt.total_seconds() / 60.0
        )

    raw = raw[raw["participant_id"].isin(base.index)].copy()
    detector_clean = detector_clean[detector_clean["participant_id"].isin(base.index)].copy()
    # Any temporal overlap with another raw episode is disqualifying.
    raw = raw.sort_values(["participant_id", "_start", "_end"])
    raw["_overlap"] = False
    for _, idx in raw.groupby("participant_id").groups.items():
        indices = list(idx)
        starts = raw.loc[indices, "_start"].tolist()
        ends = raw.loc[indices, "_end"].tolist()
        for i in range(len(indices)):
            if (i > 0 and starts[i] <= ends[i - 1]) or (
                i + 1 < len(indices) and ends[i] >= starts[i + 1]
            ):
                raw.loc[indices[i], "_overlap"] = True
    overlap_keys = set(
        zip(
            raw.loc[raw["_overlap"], "participant_id"],
            raw.loc[raw["_overlap"], "_start"].astype(str),
            raw.loc[raw["_overlap"], "_end"].astype(str),
        )
    )

    segments_by_pid: dict[str, list[pd.DataFrame]] = {}
    for (pid, _seg_id), group in clean.groupby(["participant_id", "segment_id"], sort=False):
        segments_by_pid.setdefault(str(pid), []).append(
            group.sort_values("_stream_timestamp").copy()
        )

    audits: list[dict[str, Any]] = []
    for episode_id, ep in detector_clean.reset_index(drop=True).iterrows():
        pid = str(ep["participant_id"])
        detector_seg_id = int(ep["segment_id"]) if pd.notna(ep.get("segment_id")) else -1
        start = ep["_start"]
        end = ep["_end"]
        seg = None
        if pd.notna(start) and pd.notna(end):
            for candidate in segments_by_pid.get(pid, []):
                candidate_ts = pd.to_datetime(candidate["_stream_timestamp"], utc=True)
                if start >= candidate_ts.min() and end <= candidate_ts.max():
                    seg = candidate
                    break
        seg_id = int(seg["segment_id"].iloc[0]) if seg is not None else -1
        rec: dict[str, Any] = {
            "episode_id": int(episode_id),
            "participant_id": pid,
            "segment_id": seg_id,
            "detector_segment_id": detector_seg_id,
            "start": start,
            "end": end,
            "duration_min": ep["_duration_min"],
            "detector_clean": True,
        }
        rec["segment_found"] = seg is not None
        if seg is None or pd.isna(start) or pd.isna(end):
            rec.update(
                entirely_in_one_clean_segment=False,
                valid_pre_period=False,
                valid_active_period=False,
                valid_recovery_period=False,
                wearable_missingness_ok=False,
                overlaps_other_episode=False,
                analysis_clean=False,
                exclusion_reason="clean_segment_or_timestamp_missing",
            )
            audits.append(rec)
            continue
        ts = pd.to_datetime(seg["_stream_timestamp"], utc=True)
        seg_start, seg_end = ts.min(), ts.max()
        pre_start = start - pd.Timedelta(minutes=EXERCISE_PRE_MINUTES)
        recovery_end = end + pd.Timedelta(minutes=EXERCISE_RECOVERY_MINUTES)
        in_segment = start >= seg_start and end <= seg_end
        pre_in_segment = pre_start >= seg_start
        pre = seg[(ts >= pre_start) & (ts < start)]
        active = seg[(ts >= start) & (ts <= end)]
        recovery = seg[(ts > end) & (ts <= recovery_end)]
        expected_pre = EXERCISE_PRE_MINUTES // BIN_MINUTES
        expected_active = max(1, int(round(float(ep["_duration_min"]) / BIN_MINUTES)) + 1)
        expected_recovery = EXERCISE_RECOVERY_MINUTES // BIN_MINUTES
        pre_cgm = len(pre) >= expected_pre and pre["_valid_cgm"].all()
        active_cgm = len(active) >= expected_active and active["_valid_cgm"].all()
        recovery_cgm = len(recovery) >= expected_recovery and recovery["_valid_cgm"].all()
        active_wear = min(
            active["_hr_observed_raw"].astype(bool).mean() if len(active) else 0,
            active["_activity_observed_raw"].astype(bool).mean() if len(active) else 0,
        )
        pre_wear = min(
            pre["_hr_observed_raw"].astype(bool).mean() if len(pre) else 0,
            pre["_activity_observed_raw"].astype(bool).mean() if len(pre) else 0,
        )
        wearable_ok = min(active_wear, pre_wear) >= EXERCISE_WEARABLE_MIN_FRACTION
        overlap = (pid, str(start), str(end)) in overlap_keys
        analysis_clean = bool(in_segment and pre_in_segment and pre_cgm and active_cgm and wearable_ok and not overlap)
        reasons = []
        if not in_segment or not pre_in_segment:
            reasons.append("segment_boundary_or_pre_context")
        if not pre_cgm:
            reasons.append("invalid_pre_cgm")
        if not active_cgm:
            reasons.append("invalid_active_cgm")
        if not wearable_ok:
            reasons.append("wearable_missingness")
        if overlap:
            reasons.append("overlap")
        local_start = pd.NaT
        tz_name = base.loc[pid, "timezone"] if "timezone" in base.columns else None
        try:
            local_start = start.tz_convert(str(tz_name)).tz_localize(None)
        except Exception:
            if pd.notna(start):
                local_start = start.tz_localize(None)
        rec.update(
            entirely_in_one_clean_segment=in_segment,
            pre_context_in_same_segment=pre_in_segment,
            valid_pre_period=pre_cgm,
            valid_active_period=active_cgm,
            valid_recovery_period=recovery_cgm,
            active_wearable_availability=active_wear,
            pre_wearable_availability=pre_wear,
            wearable_missingness_ok=wearable_ok,
            overlaps_other_episode=overlap,
            analysis_clean=analysis_clean,
            exclusion_reason=";".join(reasons),
            local_start=local_start,
            day_night=("day" if pd.notna(local_start) and 8 <= local_start.hour < 20 else "night"),
        )
        audits.append(rec)
    audit = pd.DataFrame(audits)

    raw_stats = raw.groupby("participant_id").agg(
        exercise_raw_episodes=("_start", "size"),
        exercise_raw_minutes=("_duration_min", "sum"),
    )
    clean_only = audit[audit["analysis_clean"]].copy()
    clean_stats = audit.groupby("participant_id").agg(
        exercise_detector_clean_episodes=("episode_id", "size"),
        exercise_episodes_with_valid_pre=("valid_pre_period", "sum"),
        exercise_episodes_with_valid_active=("valid_active_period", "sum"),
        exercise_episodes_with_valid_recovery=("valid_recovery_period", "sum"),
    )
    if clean_only.empty:
        eligible_stats = pd.DataFrame(index=base.index)
    else:
        eligible_stats = clean_only.groupby("participant_id").agg(
            exercise_clean_episodes=("episode_id", "size"),
            exercise_total_clean_minutes=("duration_min", "sum"),
            exercise_median_episode_duration=("duration_min", "median"),
            exercise_day_episodes=("day_night", lambda x: int((x == "day").sum())),
            exercise_night_episodes=("day_night", lambda x: int((x == "night").sum())),
            exercise_segments_with_episode=("segment_id", "nunique"),
        )
    out = base.join(raw_stats).join(clean_stats).join(eligible_stats)
    count_cols = [
        "exercise_raw_episodes",
        "exercise_detector_clean_episodes",
        "exercise_clean_episodes",
        "exercise_episodes_with_valid_pre",
        "exercise_episodes_with_valid_active",
        "exercise_episodes_with_valid_recovery",
        "exercise_day_episodes",
        "exercise_night_episodes",
        "exercise_segments_with_episode",
    ]
    for col in count_cols:
        out[col] = out[col].fillna(0).astype(int)
    for col in ("exercise_raw_minutes", "exercise_total_clean_minutes"):
        out[col] = out[col].fillna(0.0)
    out["exercise_eligible_ge_1_raw"] = out["exercise_raw_episodes"] >= 1
    out["exercise_eligible_ge_1"] = out["exercise_clean_episodes"] >= 1
    out["exercise_eligible_ge_3"] = out["exercise_clean_episodes"] >= 3
    logger.info(
        "Exercise audit: %d raw detector episodes, %d detector-clean, %d analysis-clean",
        len(raw),
        len(detector_clean),
        int(audit["analysis_clean"].sum()) if len(audit) else 0,
    )
    return out, audit


def safe_stats(series: pd.Series) -> tuple[float, float, float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if not len(x):
        return np.nan, np.nan, np.nan, np.nan
    return float(x.sum()), float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))


def context_audit_rows(participants: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "night_any": ("night", "night_eligible_any", "night_valid_hours", "night_distinct_nights", "≥1 valid sleep-derived night row"),
        "night_2h": ("night", "night_eligible_2h", "night_valid_hours", "night_nights_ge_2h", "≥1 night with ≥2 valid hours"),
        "night_2nights": ("night", "night_eligible_2nights", "night_valid_hours", "night_nights_ge_2h", "≥2 nights with ≥2 valid hours"),
        "night_6h_total": ("night", "night_eligible_6h_total", "night_valid_hours", "night_distinct_nights", "≥6 total sleep-derived hours"),
        "night_clock_00_06_sensitivity": ("night", "night_clock_eligible_any", "night_clock_valid_hours", "night_clock_distinct_nights", "≥1 valid row from 00:00–06:00 local time"),
        "day_any": ("day", "day_eligible_any", "day_valid_hours", "day_distinct_days", "≥1 valid daytime row"),
        "day_6h_total": ("day", "day_eligible_6h_total", "day_valid_hours", "day_distinct_days", "≥6 total daytime hours"),
        "day_2days": ("day", "day_eligible_2days", "day_valid_hours", "day_days_ge_2h", "≥2 days with ≥2 valid daytime hours"),
        "exercise_ge_1_raw": ("exercise", "exercise_eligible_ge_1_raw", "exercise_raw_minutes", "exercise_raw_episodes", "≥1 raw detector episode"),
        "exercise_ge_1_clean": ("exercise", "exercise_eligible_ge_1", "exercise_total_clean_minutes", "exercise_clean_episodes", "≥1 analysis-clean episode"),
        "exercise_ge_3_clean": ("exercise", "exercise_eligible_ge_3", "exercise_total_clean_minutes", "exercise_clean_episodes", "≥3 analysis-clean episodes"),
    }
    splits: dict[str, set[str]] = {
        "train": set(participants.index[participants["split"].eq("train")]),
        "validation": set(participants.index[participants["split"].eq("validation")]),
        "test": set(participants.index[participants["split"].eq("test")]),
        "validation_plus_test": set(
            participants.index[participants["split"].isin(["validation", "test"])]
        ),
        "all": set(participants.index),
    }
    groupings = [(None, None)]
    for col in ("clinical_site", "study_group"):
        for level in sorted(participants[col].dropna().astype(str).unique()):
            groupings.append((col, level))
    rows = []
    for split_name, ids in splits.items():
        for subgroup_variable, subgroup_level in groupings:
            sub = participants.loc[participants.index.intersection(ids)].copy()
            if subgroup_variable:
                sub = sub[sub[subgroup_variable].astype(str).eq(str(subgroup_level))]
            if sub.empty:
                continue
            for def_name, (ctx_type, flag, hours_col, events_col, threshold) in definitions.items():
                eligible = sub[flag].fillna(False).astype(bool)
                hours = pd.to_numeric(sub[hours_col], errors="coerce").fillna(0)
                if ctx_type == "exercise":
                    hours = hours / 60.0
                    wearable_col = None
                    reset_col = None
                elif ctx_type == "night":
                    wearable_col = "night_hr_availability"
                    reset_col = "night_fraction_near_reset"
                else:
                    wearable_col = "day_wearable_availability"
                    reset_col = "day_fraction_near_reset"
                events = pd.to_numeric(sub[events_col], errors="coerce").fillna(0)
                total_h, med_h, p25_h, p75_h = safe_stats(hours)
                total_e, med_e, p25_e, p75_e = safe_stats(events)
                rows.append(
                    {
                        "split": split_name,
                        "subgroup_variable": subgroup_variable or "",
                        "subgroup_level": subgroup_level or "",
                        "context_type": ctx_type,
                        "context_definition": def_name,
                        "eligibility_threshold": threshold,
                        "n_participants_in_split": len(sub),
                        "n_participants_eligible": int(eligible.sum()),
                        "eligible_fraction": float(eligible.mean()),
                        "total_valid_hours": total_h,
                        "median_valid_hours_per_participant": med_h,
                        "p25_valid_hours_per_participant": p25_h,
                        "p75_valid_hours_per_participant": p75_h,
                        "total_events": total_e,
                        "median_events_per_participant": med_e,
                        "p25_events_per_participant": p25_e,
                        "p75_events_per_participant": p75_e,
                        "n_with_ge_1_event": int((events >= 1).sum()),
                        "n_with_ge_3_events": int((events >= 3).sum()),
                        "median_distinct_days_or_nights": float(events.median()),
                        "wearable_availability_median": (
                            float(pd.to_numeric(sub[wearable_col], errors="coerce").median())
                            if wearable_col
                            else np.nan
                        ),
                        "fraction_near_segment_reset_median": (
                            float(pd.to_numeric(sub[reset_col], errors="coerce").median())
                            if reset_col
                            else np.nan
                        ),
                        "notes": (
                            f"Rows use {BIN_MINUTES}-minute bins; near reset means first "
                            f"{RESET_NEAR_MINUTES} minutes of a canonical clean segment."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def choose_recommendations(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = inventory[
        inventory["strict_test_eligible"].map(as_bool)
        & inventory["model_input_status"].eq("not_model_input")
        & inventory["target_type"].eq("continuous")
    ].copy()
    unpublished = eligible[eligible["published_cca_status"].eq("not_explicitly_reported")].copy()
    priority = {
        "cardiovascular": 0,
        "inflammatory": 1,
        "renal": 2,
        "hepatic": 3,
        "neuropathy": 4,
        "cognitive": 5,
        "hematologic": 6,
        "mental_health": 7,
        "ophthalmic": 8,
        "other_clinical": 9,
        "electrolyte": 10,
        "metabolic_glycemic": 11,
        "lipid": 12,
    }
    unpublished["_priority"] = unpublished["clinical_domain"].map(priority).fillna(99)
    unpublished = unpublished.sort_values(
        ["_priority", "n_valid_numeric_test", "normalized_target_name"],
        ascending=[True, False, True],
    )
    picked = []
    domains = set()
    for idx, row in unpublished.iterrows():
        if row["clinical_domain"] in domains:
            continue
        picked.append(idx)
        domains.add(row["clinical_domain"])
        if len(picked) == 3:
            break
    if len(picked) < 3:
        for idx in unpublished.index:
            if idx not in picked:
                picked.append(idx)
            if len(picked) == 3:
                break
    primary = inventory.loc[picked].copy() if picked else inventory.iloc[0:0].copy()
    backup = inventory[
        inventory["combined_exploratory_eligible"].map(as_bool)
        & inventory["model_input_status"].eq("not_model_input")
        & ~inventory.index.isin(picked)
    ].sort_values(["n_valid_numeric_test", "normalized_target_name"], ascending=[False, True]).head(10)
    return primary, backup


def aggregate_lookup(audit: pd.DataFrame, split: str, definition: str) -> pd.Series:
    q = audit[
        audit["split"].eq(split)
        & audit["context_definition"].eq(definition)
        & audit["subgroup_variable"].eq("")
    ]
    return q.iloc[0] if len(q) else pd.Series(dtype=object)


def write_report(
    path: Path,
    args: argparse.Namespace,
    run_id: str,
    split_counts: Mapping[str, int],
    alignment: Mapping[str, Any],
    inventory: pd.DataFrame,
    context: pd.DataFrame,
    participants: pd.DataFrame,
    primary: pd.DataFrame,
    backup: pd.DataFrame,
    detector_config: Mapping[str, Any],
    paper_evidence: Mapping[str, Any],
) -> None:
    model_inputs = inventory[inventory["model_input_status"].isin(["static_input", "dynamic_input", "derived_input"])]
    published = inventory[inventory["published_cca_status"].isin(["explicitly_named", "family_match"])]
    external = inventory[inventory["strict_test_eligible"].map(as_bool)]
    invalid_units = inventory[inventory["target_role"].eq("invalid_units")]
    invalid_timing = inventory[inventory["target_role"].eq("invalid_timing")]
    manual = inventory[inventory["manual_review_required"].map(as_bool)]
    vt = participants[participants["split"].isin(["validation", "test"])]
    saved_segments = pd.read_csv(args.segments_path, dtype={"participant_id": str})
    saved_segments["participant_id"] = saved_segments["participant_id"].map(normalize_id)
    saved_counts = saved_segments.groupby("participant_id")["segment_id"].nunique()
    computed_counts = pd.to_numeric(participants["n_segments"], errors="coerce")
    segment_mismatch_count = int(
        computed_counts.to_frame("computed").join(saved_counts.rename("saved"), how="left").query("computed != saved").shape[0]
    )

    def names(df: pd.DataFrame, n: int = 20) -> str:
        if df.empty:
            return "None."
        return ", ".join(df["normalized_target_name"].head(n).astype(str)) + ("." if len(df) <= n else ", …")

    night_any = aggregate_lookup(context, "validation_plus_test", "night_any")
    day_any = aggregate_lookup(context, "validation_plus_test", "day_any")
    ex1 = aggregate_lookup(context, "validation_plus_test", "exercise_ge_1_clean")
    ex3 = aggregate_lookup(context, "validation_plus_test", "exercise_ge_3_clean")
    lines = [
        "# Step 0 hidden-state phenotype feasibility report",
        "",
        f"Run ID: `{run_id}`. This is a descriptive feasibility/coverage audit only.",
        "",
        "## 1. Data paths and versions",
        "",
        f"- Canonical enriched panel (read-only): `{args.multimodal_parquet}`",
        f"- Canonical static table (read-only): `{args.static_parquet}`",
        f"- Canonical configuration: `{args.model_config}`",
        f"- Canonical split: `{args.split_manifest}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Saved feature schema: `{args.feature_schema}`",
        f"- Exercise detector: `{args.exercise_detector_notebook}` with `{args.exercise_detector_config}`",
        f"- Local manuscript inspection: {paper_evidence.get('note', 'not available')}",
        "",
        "## 2. Canonical model cohort and split counts",
        "",
        f"The exact saved participant split contains {sum(split_counts.values()):,} participants: "
        f"{split_counts.get('train', 0):,} train, {split_counts.get('validation', 0):,} validation, "
        f"and {split_counts.get('test', 0):,} test. No split was regenerated.",
        "",
        "## 3. Participant-ID alignment",
        "",
    ]
    for table, rec in alignment.items():
        lines.append(
            f"- `{table}`: {rec['n_unique_participants']:,} raw participants; "
            f"{rec['n_model_cohort_overlap']:,}/{sum(split_counts.values()):,} overlap the model cohort; "
            f"{rec['n_model_cohort_not_in_table']:,} model participants unmatched."
        )
    lines += [
        "",
        "Identifiers were read as strings, whitespace-stripped, optional `AIREADI-` prefixes removed, "
        "and textual trailing `.0` removed. No floating-point identifier conversion or inferred crosswalk was used.",
        "",
        "## 4. Clinical measurement inventory summary",
        "",
        f"The inventory contains {len(inventory):,} targets: "
        f"{(inventory.source_table == 'measurement').sum():,} measurement concepts, "
        f"{(inventory.source_table == 'observation').sum():,} selected summary observations, and "
        f"{(inventory.source_table == 'condition_occurrence').sum():,} defined comorbidity indicators. "
        "Questionnaire items were not expanded automatically.",
        "",
        "## 5. Existing model-input measurements",
        "",
        names(model_inputs),
        "",
        "A target is marked as an input only when its exact enrichment alias is present in the saved consumed-feature schema.",
        "",
        "## 6. External phenotype candidates not used by the model",
        "",
        f"{len(external):,} targets meet the preliminary strict untouched-test rules. "
        f"Top candidates by valid test-participant coverage: {names(external.sort_values('n_valid_numeric_test', ascending=False), 15)}",
        "",
        "## 7. Measurements previously represented in the published static-embedding CCA",
        "",
        names(published),
        "",
        "These labels refer to the prior individual static-embedding CCA, not streaming hidden states. "
        "Cholesterol variables are family matches unless an exact local source is later recovered.",
        "",
        "## 8. Lab timing and unit-quality problems",
        "",
        f"{len(invalid_units):,} targets have invalid/unknown units and {len(invalid_timing):,} fail the "
        "80%-within-±180-day timing rule. Raw rows were neither converted nor discarded. "
        f"{len(manual):,} targets are flagged for manual review.",
        "",
        "## 9. Night/day coverage",
        "",
        f"Validation+test sleep-derived night coverage: {int(night_any.get('n_participants_eligible', 0)):,}/"
        f"{int(night_any.get('n_participants_in_split', len(vt))):,} participants with any valid night data. "
        f"Day coverage: {int(day_any.get('n_participants_eligible', 0)):,}/"
        f"{int(day_any.get('n_participants_in_split', len(vt))):,} with any valid day data.",
        "",
        "Sleep-derived night and clock-based 00:00–06:00 sensitivity definitions are reported separately. "
        "No final hidden-state study definition is selected here.",
        "",
        "## 10. Exercise coverage",
        "",
        f"The saved detector defines raw episodes and primary detector-clean episodes as "
        f"`{detector_config.get('primary_definition', 'see detector config')}`. "
        f"For the future Δh design, this audit additionally requires 30 minutes of pre-context, "
        f"complete model-valid CGM in pre/active windows, ≥{EXERCISE_WEARABLE_MIN_FRACTION:.0%} "
        "HR and activity availability in both windows, one clean segment, and no overlap.",
        "",
        f"Validation+test: {int(ex1.get('n_participants_eligible', 0)):,} participants have ≥1 "
        f"analysis-clean episode and {int(ex3.get('n_participants_eligible', 0)):,} have ≥3.",
        "",
        "## 11. Recommended three primary clinical targets",
        "",
        names(primary, 3) if len(primary) == 3 else (
            f"Only {len(primary)} target(s) satisfy the recommendation rules: {names(primary, 3)}"
        ),
        "",
        "Recommendations exclude direct model inputs and prefer distinct, previously unreported clinical domains. "
        "They are feasibility priorities, not statistically selected features.",
        "",
        "## 12. Backup exploratory targets",
        "",
        names(backup, 10),
        "",
        "## 13. Targets to exclude and why",
        "",
        f"Invalid units: {names(invalid_units, 12)} Low coverage/timing failures remain in the CSV with explicit reasons.",
        "",
        "## 14. Blocking issues requiring manual review",
        "",
        "- `concept.csv` is absent from this release, so readable names come from source values; concept IDs are preserved.",
        "- The local manuscript and report archive did not contain recoverable Appendix A.13 text; published CCA labels follow the study specification and cholesterol remains family-level.",
        f"- Canonical model-code segment reconstruction differs from the saved segment manifest for {segment_mismatch_count} participants; inspect these boundaries before replay.",
        "- Saved detector segment labels are global and unrelated to canonical per-participant segment IDs; Step 0 aligns episodes by participant and timestamp containment.",
        f"- The detector config has no explicit active/pre-window wearable-missingness ceiling; Step 0 uses an explicitly logged {EXERCISE_WEARABLE_MIN_FRACTION:.0%} threshold that must be locked before Step 1.",
        "- Condition-occurrence non-records are treated as indicator negatives for feasibility counts; confirm this semantic choice against the release dictionary.",
        "",
        "## 15. Go/no-go recommendation",
        "",
        f"- All-recording clustering: **GO** for future planning; {len(participants):,} canonical participants have clean segments. Do not extract states in Step 0.",
        f"- Nighttime clustering: **{'GO' if int(night_any.get('n_participants_eligible', 0)) >= 100 else 'NO-GO'}** based on validation+test any-night coverage; retain threshold sensitivity definitions.",
        f"- Daytime clustering: **{'GO' if int(day_any.get('n_participants_eligible', 0)) >= 100 else 'NO-GO'}** based on validation+test any-day coverage.",
        f"- Exercise-response clustering: **{'GO' if int(ex3.get('n_participants_eligible', 0)) >= 100 else 'NO-GO'}** for a ≥3 clean-episode participant design; ≥1-episode counts are also reported.",
        f"- External clinical probe analysis: **{'GO' if len(primary) >= 1 else 'NO-GO'}**; {len(primary)} recommended independent primary target(s) meet all preliminary rules.",
        "",
        "This report stops after Step 0. It does not authorize replay, hidden-state extraction, clustering, probes, associations, or hypothesis tests.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def discover_enriched_files(root: Path) -> list[dict[str, Any]]:
    patterns = (
        "final_multimodal_dataset_*.parquet",
        "participant_static_features.parquet",
        "participant_measurements_selected_long.*",
        "participant_medications_long.*",
        "clinical_measurement_unit_audit.csv",
        "clinical_measurement_coverage.csv",
        "medication_class_coverage.csv",
        "demographic_coverage.csv",
        "clinical_enrichment_metadata.json",
        "*segment*",
        "*cohort*",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found[str(path.resolve())] = path.resolve()
    return [file_record(path) for path in sorted(found.values())]


def paper_evidence() -> dict[str, Any]:
    candidates = [ROOT / "main.pdf", ROOT / "main.tex", ROOT / "aireadi_ssmcgm_report_overleaf.zip"]
    found = [file_record(p) for p in candidates if p.exists()]
    return {
        "files_inspected": found,
        "appendix_a13_exact_text_found": False,
        "note": (
            "local main.pdf/main.tex/report archive inspected; exact Appendix A.13 wording was not found, "
            "so mappings use the supplied study specification"
        ),
    }


def git_info() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], check=False)
    status = run_command(["git", "status", "--short"], check=False)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "dirty": bool(status.stdout.strip()),
        "status_short": status.stdout.splitlines(),
    }


def print_terminal_summary(
    output: Path,
    split_counts: Mapping[str, int],
    inventory: pd.DataFrame,
    context: pd.DataFrame,
    primary: pd.DataFrame,
) -> None:
    external = inventory[
        inventory["strict_test_eligible"].map(as_bool)
        & inventory["model_input_status"].eq("not_model_input")
    ].sort_values(["n_valid_numeric_test", "normalized_target_name"], ascending=[False, True])
    model_inputs = inventory[
        inventory["model_input_status"].isin(["static_input", "dynamic_input", "derived_input"])
    ]
    published = inventory[inventory["published_cca_status"].isin(["explicitly_named", "family_match"])]
    print("\nSTEP 0 COMPLETE")
    print("Files created:")
    for name in REQUIRED_OUTPUTS:
        print(f"  {output / name}")
    print(
        "Cohort/splits: "
        f"total={sum(split_counts.values())}, train={split_counts.get('train', 0)}, "
        f"validation={split_counts.get('validation', 0)}, test={split_counts.get('test', 0)}"
    )
    print(f"Distinct clinical targets inventoried: {len(inventory)}")
    print(
        "Top external targets by test coverage: "
        + ", ".join(external["normalized_target_name"].head(10))
    )
    print("Targets supplied to model: " + ", ".join(model_inputs["normalized_target_name"]))
    print("Published CCA explicit/family: " + ", ".join(published["normalized_target_name"]))
    for definition, label in (
        ("night_any", "validation/test with valid night data"),
        ("day_any", "validation/test with valid day data"),
        ("exercise_ge_1_clean", "validation/test with ≥1 clean exercise episode"),
        ("exercise_ge_3_clean", "validation/test with ≥3 clean exercise episodes"),
    ):
        row = aggregate_lookup(context, "validation_plus_test", definition)
        print(f"{label}: {int(row.get('n_participants_eligible', 0))}")
    print("Recommended primary targets: " + ", ".join(primary["normalized_target_name"]))
    print(
        "Blockers/manual review: absent concept.csv; exact Appendix A.13 source text unavailable; "
        "exercise wearable-missingness threshold and condition-negative semantics require confirmation."
    )


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    output, cache, run_id = setup_output(args)
    logger = setup_logging(output / "step0_run.log")
    logger.info("Starting Step 0 feasibility audit run_id=%s", run_id)
    validate_paths(args)
    clinical, gcs_meta = ensure_inputs(args, cache, logger)

    split_df, split_map = load_split(args.split_manifest)
    split_counts = split_df["split"].value_counts().to_dict()
    logger.info("Resolved exact saved split counts: %s", split_counts)
    model_schema, _, feature_status = load_model_schema(args.feature_schema)
    config = yaml.safe_load(args.model_config.read_text())
    detector_config = json.loads(args.exercise_detector_config.read_text())

    panel, panel_columns = read_panel_context(args, set(split_map), logger)
    panel_unique = set(panel["participant_id"])
    missing_panel = set(split_map) - panel_unique
    if missing_panel:
        raise ValueError(f"Canonical split participants absent from final panel: {sorted(missing_panel)[:20]}")
    cgm_bounds = get_cgm_bounds(panel)
    if set(split_map) - set(cgm_bounds.index):
        raise ValueError("Some canonical split participants have no valid CGM timestamp.")
    clean = prepare_clean_panel(panel, logger)
    clean_ids = set(clean["participant_id"])
    if clean_ids != set(split_map):
        raise ValueError(
            f"Canonical clean panel participant mismatch: missing={sorted(set(split_map)-clean_ids)[:20]}, "
            f"extra={sorted(clean_ids-set(split_map))[:20]}"
        )
    context_base, _ = context_participant_metrics(clean, split_map, cgm_bounds, logger)
    # Preserve declared timezone for episode local-time conversion.
    timezones = panel.groupby("participant_id")["timezone"].first()
    context_base["timezone"] = context_base.index.map(timezones)
    participants, episode_audit = exercise_metrics(
        clean,
        context_base,
        args.raw_exercise_episodes,
        args.clean_exercise_episodes,
        logger,
    )
    participants.index.name = "participant_id"
    context_audit = context_audit_rows(participants)

    tables = load_clinical_tables(clinical, logger)
    full_alignment = all_table_participant_alignment(clinical, split_map)
    inventory, alignment = inventory_targets(
        tables,
        split_map,
        cgm_bounds,
        set(panel_columns),
        feature_status,
        args.feature_schema,
        logger,
    )
    primary, backup = choose_recommendations(inventory)

    inventory.to_csv(output / "clinical_target_inventory.csv", index=False)
    context_audit.to_csv(output / "context_coverage_audit.csv", index=False)
    requested_first = [
        "split",
        "clinical_site",
        "study_group",
        "cgm_start",
        "cgm_end",
        "total_clean_hours",
        "n_segments",
        "median_segment_hours",
        "fraction_rows_near_reset",
        "night_valid_hours",
        "night_distinct_nights",
        "night_nights_ge_2h",
        "night_eligible_any",
        "night_eligible_2nights",
        "day_valid_hours",
        "day_distinct_days",
        "day_eligible_any",
        "day_eligible_2days",
        "exercise_raw_episodes",
        "exercise_clean_episodes",
        "exercise_total_clean_minutes",
        "exercise_eligible_ge_1",
        "exercise_eligible_ge_3",
    ]
    participant_out = participants.reset_index()
    participant_out = participant_out[
        ["participant_id"] + requested_first
        + [c for c in participant_out.columns if c not in {"participant_id", *requested_first}]
    ]
    participant_out.to_csv(output / "context_coverage_by_participant.csv", index=False)

    paper = paper_evidence()
    write_report(
        output / "step0_feasibility_report.md",
        args,
        run_id,
        split_counts,
        full_alignment,
        inventory,
        context_audit,
        participants,
        primary,
        backup,
        detector_config,
        paper,
    )

    saved_segments = pd.read_csv(args.segments_path, dtype={"participant_id": str})
    saved_segments["participant_id"] = saved_segments["participant_id"].map(normalize_id)
    saved_segments = saved_segments[saved_segments["participant_id"].isin(split_map)]
    computed_segments = (
        clean[["participant_id", "segment_id"]].drop_duplicates().groupby("participant_id").size()
    )
    saved_counts = saved_segments.groupby("participant_id")["segment_id"].nunique()
    segment_count_mismatches = (
        computed_segments.to_frame("computed")
        .join(saved_counts.rename("saved"), how="outer")
        .query("computed != saved")
    )
    manifest = {
        "run_id": run_id,
        "execution_date_utc": datetime.now(timezone.utc).isoformat(),
        "study_step": "Step 0 feasibility and coverage audit only",
        "explicitly_not_run": [
            "model training or modification",
            "checkpoint modification",
            "hidden-state extraction",
            "clustering",
            "PCA",
            "CCA",
            "probes",
            "associations",
            "hypothesis tests",
            "FDR correction",
        ],
        "repository": git_info(),
        "python_environment": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "resolved_arguments": {k: jsonable(v) for k, v in vars(args).items()},
        "gcs_root": args.gcs_root,
        "gcs_clinical_objects": gcs_meta,
        "clinical_directory_listing": {
            "present_required": [name for name in CLINICAL_FILES if (clinical / name).exists()],
            "absent_checked": [
                name for name in OPTIONAL_CLINICAL_FILES if not (clinical / name).exists()
            ],
            "present_not_downloaded": list(DISCOVERY_ONLY_FILES),
            "restriction": "immediate clinical_data tabular objects only; no imaging/waveform recursion",
        },
        "canonical_sources": {
            "model_config": file_record(args.model_config, hash_file=True),
            "split_manifest": file_record(args.split_manifest, hash_file=True),
            "checkpoint": file_record(args.checkpoint, hash_file=False),
            "feature_schema": file_record(args.feature_schema, hash_file=True),
            "enriched_panel": file_record(args.multimodal_parquet, hash_file=False),
            "static_features": file_record(args.static_parquet, hash_file=True),
            "cohort": file_record(args.cohort_path, hash_file=True),
            "segments": file_record(args.segments_path, hash_file=True),
            "exercise_detector_notebook": file_record(args.exercise_detector_notebook, hash_file=True),
            "exercise_detector_config": file_record(args.exercise_detector_config, hash_file=True),
            "raw_exercise_episodes": file_record(args.raw_exercise_episodes, hash_file=True),
            "detector_clean_exercise_episodes": file_record(
                args.clean_exercise_episodes, hash_file=True
            ),
        },
        "enriched_files_discovered_read_only": discover_enriched_files(DEFAULT_ENRICHED_ROOT),
        "canonical_model_configuration": config,
        "canonical_feature_schema": model_schema,
        "split_counts": split_counts,
        "participant_id_alignment": full_alignment,
        "inventory_source_participant_alignment": alignment,
        "cgm_and_segments": {
            "cgm_start_definition": "minimum valid timestamp_local with cgm_count>0 in canonical panel",
            "cgm_end_definition": "maximum valid timestamp_local with cgm_count>0 in canonical panel",
            "clean_segment_source": "ssmcgm/data/aireadi.py::prepare_aireadi_panel",
            "clean_min_segment_hours": 49,
            "core_gap_threshold_bins": {"cgm": 6, "hr": 12, "rr": 12, "activity": 12},
            "bin_minutes": BIN_MINUTES,
            "near_reset_minutes": RESET_NEAR_MINUTES,
            "computed_n_segments": int(
                clean[["participant_id", "segment_id"]].drop_duplicates().shape[0]
            ),
            "saved_n_segments": int(len(saved_segments)),
            "participant_segment_count_mismatches": int(len(segment_count_mismatches)),
            "mismatch_sample": segment_count_mismatches.reset_index().head(20).to_dict("records"),
        },
        "clinical_inventory": {
            "n_targets": len(inventory),
            "n_by_source": inventory["source_table"].value_counts().to_dict(),
            "n_by_role": inventory["target_role"].value_counts().to_dict(),
            "n_strict_test_eligible": int(inventory["strict_test_eligible"].map(as_bool).sum()),
            "n_combined_exploratory_eligible": int(
                inventory["combined_exploratory_eligible"].map(as_bool).sum()
            ),
            "observation_selection_rule": (
                "only source prefixes cestl (CESD-10 total), paidscore (PAID total), "
                "and dietscore; questionnaire items excluded"
            ),
            "condition_semantics": (
                "condition-occurrence rows are positive indicators; absence is treated as negative "
                "for preliminary feasibility counts and flagged for manual confirmation"
            ),
            "date_assumption": (
                "measurement/observation/condition datetime preferred; date-only fallback parsed "
                "at local midnight and compared with participant-local-naive CGM start"
            ),
            "raw_values_converted": False,
            "raw_rows_discarded_for_units": False,
        },
        "context_definitions": {
            "night_primary": "light>0 OR deep>0 OR REM>0; unknown excluded; model-valid CGM",
            "night_clock_sensitivity": "participant-local 00:00–06:00; model-valid CGM",
            "day_primary": "participant-local 08:00–20:00; not light/deep/REM sleep; model-valid CGM",
            "local_time_handling": (
                "stored timestamp instant converted using each row's declared participant timezone"
            ),
            "row_duration_minutes": BIN_MINUTES,
        },
        "exercise": {
            "detector_config": detector_config,
            "analysis_clean_extension": {
                "pre_context_minutes": EXERCISE_PRE_MINUTES,
                "recovery_audit_minutes": EXERCISE_RECOVERY_MINUTES,
                "minimum_hr_and_activity_availability_pre_and_active": EXERCISE_WEARABLE_MIN_FRACTION,
                "requires_complete_model_valid_cgm_pre_and_active": True,
                "requires_single_clean_segment": True,
                "overlap_allowed": False,
            },
            "detector_segment_label_note": "saved detector segment_id is a global label unrelated to canonical per-participant segment_id; episodes aligned by participant and timestamp containment",
            "n_detector_clean_episodes_audited": len(episode_audit),
            "n_analysis_clean_episodes": int(episode_audit["analysis_clean"].sum()),
            "analysis_exclusion_reasons": dict(
                Counter(
                    reason
                    for reasons in episode_audit["exclusion_reason"].fillna("")
                    for reason in str(reasons).split(";")
                    if reason
                )
            ),
        },
        "published_cca_evidence": paper,
        "recommended_primary_targets": primary["normalized_target_name"].tolist(),
        "backup_exploratory_targets": backup["normalized_target_name"].tolist(),
        "outputs": [str(output / name) for name in REQUIRED_OUTPUTS],
    }
    (output / "step0_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Step 0 outputs written successfully")
    print_terminal_summary(output, split_counts, inventory, context_audit, primary)


if __name__ == "__main__":
    main()
