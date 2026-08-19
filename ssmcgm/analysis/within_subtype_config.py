"""Shared constants for the within-subtype phenotype preservation study."""

from pathlib import Path

SEED = 42
N_INIT = 100
BOOTSTRAP_N = 1000
PHASE0_BOOTSTRAP_N = 200
PHASE0_PARALLEL_JOBS = 4
PERMUTATION_N = 1000

REPO = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
REPO_BRANCH = "aireadi-ssmcgm-stream-report"
STUDY1_ROOT = REPO / "outputs/static_phenotype_trajectory"
STUDY2_ROOT = REPO / "outputs/static_phenotype_trajectory_stratified_v2"
CHECKPOINT = REPO / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
DATASET = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "final_multimodal_dataset_20260515_184339.parquet"
)
SPLIT = "adapt6h_seed42"
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/"
    "split_participants.csv"
)

FIGURE_ROOT = STUDY2_ROOT / "figures"
TABLE_ROOT = STUDY2_ROOT / "tables"
LOG_ROOT = STUDY2_ROOT / "logs"
DECISION_ROOT = STUDY2_ROOT / "decisions"

STRATIFIER = "participants_study_group"
CANONICAL_STRATA = [
    "healthy",
    "pre_diabetes",
    "t2d_oral_non_insulin",
    "insulin_dependent",
]
RAW_STRATUM_MAP = {
    "healthy": "healthy",
    "pre_diabetes_lifestyle_controlled": "pre_diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "t2d_oral_non_insulin",
    "insulin_dependent": "insulin_dependent",
}

CORE_FACTORS = [
    "participants_age",
    "bmi_baseline",
    "hba1c_percent_baseline",
    "c_peptide_ngml_baseline",
    "tg_hdl_ratio",
]
SIXTH_FACTOR_CANDIDATES = [
    "waist_to_hip_ratio_baseline",
    "fasting_insulin_baseline",
    "systolic_blood_pressure_baseline",
]
FACTOR_COLUMN_MAP = {
    "participants_age": "participants_age",
    "bmi_baseline": "bmi_baseline",
    "hba1c_percent_baseline": "hba1c_percent_baseline",
    "c_peptide_ngml_baseline": "c_peptide_ngml_baseline",
    "tg_hdl_ratio": "tg_hdl_ratio",
    "waist_to_hip_ratio_baseline": "waist_to_hip_ratio_baseline",
    "fasting_insulin_baseline": "fasting_insulin_baseline",
    "systolic_blood_pressure_baseline": "clinical_systolic_bp_mmhg_baseline",
}
SOURCE_FACTOR_COLUMNS = {
    "tg_hdl_ratio": ["triglycerides_mgdl_baseline", "hdl_cholesterol_mgdl_baseline"]
}
FACTOR_UNITS = {
    "participants_age": "Years",
    "bmi_baseline": "kg/m^2",
    "hba1c_percent_baseline": "%",
    "c_peptide_ngml_baseline": "ng/mL",
    "triglycerides_mgdl_baseline": "mg/dL",
    "hdl_cholesterol_mgdl_baseline": "mg/dL",
    "tg_hdl_ratio": "Ratio from two mg/dL measurements",
    "waist_to_hip_ratio_baseline": "Ratio",
    "fasting_insulin_baseline": "Unresolved",
    "systolic_blood_pressure_baseline": "mmHg",
}
FACTOR_DOMAINS = {
    "participants_age": "Age or life stage",
    "bmi_baseline": "Adiposity",
    "hba1c_percent_baseline": "Glycemic control",
    "c_peptide_ngml_baseline": "Insulin secretion",
    "tg_hdl_ratio": "Insulin resistance",
    "waist_to_hip_ratio_baseline": "Body fat distribution",
    "fasting_insulin_baseline": "Insulin resistance",
    "systolic_blood_pressure_baseline": "Cardiovascular risk",
}
BASELINE_AUDIT_COLUMNS = {
    "bmi_baseline": ("bmi_n_records", "bmi_days_to_cgm_start"),
    "hba1c_percent_baseline": ("hba1c_percent_n_records", "hba1c_percent_days_to_cgm_start"),
    "c_peptide_ngml_baseline": ("c_peptide_ngml_n_records", "c_peptide_ngml_days_to_cgm_start"),
    "triglycerides_mgdl_baseline": (
        "triglycerides_mgdl_n_records",
        "triglycerides_mgdl_days_to_cgm_start",
    ),
    "hdl_cholesterol_mgdl_baseline": (
        "hdl_cholesterol_mgdl_n_records",
        "hdl_cholesterol_mgdl_days_to_cgm_start",
    ),
    "waist_to_hip_ratio_baseline": (
        "waist_to_hip_ratio_n_records",
        "waist_to_hip_ratio_days_to_cgm_start",
    ),
    "systolic_blood_pressure_baseline": (
        "clinical_systolic_bp_mmhg_n_records",
        "clinical_systolic_bp_mmhg_days_to_cgm_start",
    ),
}

TARGET_FACTOR_COUNT = 6
MIN_FACTOR_COUNT = 5
MAX_FACTOR_CORRELATION = 0.85
MIN_FACTOR_COVERAGE = 0.70
COMPLETE_CASE_MIN_RETAINED_FRACTION = 0.70
SKEW_LOG_THRESHOLD = 1.0

K_RANGE = [2, 3, 4]
MIN_CLUSTER_TRAIN_N = 20
MIN_CLUSTER_TRAIN_FRACTION = 0.08
MIN_BOOTSTRAP_ARI = 0.60
SILHOUETTE_TOLERANCE = 0.02
USE_GAP_ONE_STANDARD_ERROR_RULE = True

UNDERPOWERED_TEST_N = 30
UNDERPOWERED_CLUSTER_TEST_N = 10
KNN_K_FRACTION = 0.15
KNN_K_FLOOR = 8
KNN_K_CEILING = 30
PCA_VARIANCE_TARGET = 0.90

CLINICAL_METRIC = "euclidean"
LATENT_METRIC = "cosine"
MOVEMENT_METRIC = "cosine"
COHERENCE_RATIO_BAR = 0.30
BOOTSTRAP_CI_LEVEL = 0.95
TIME_RESOLVED_HOURS = [0, 6, 12, 24, 48]

# Named audit thresholds. These operationalize the prompt's "strongly associated"
# and impossible-value language and are fixed before viewing test latent states.
SITE_ASSOCIATION_P_THRESHOLD = 0.01
SITE_ASSOCIATION_CRAMERS_V_THRESHOLD = 0.30
SEVERITY_SMD_THRESHOLD = 0.50
MIN_SEVERITY_GROUP_N = 5
PARQUET_BATCH_SIZE = 65_536
ITERATIVE_IMPUTER_MAX_ITER = 20
MIN_BASELINE_AT_OR_BEFORE_CGM_FRACTION = 0.80
MIN_POSITIVE_HDL = 0.0
IMPOSSIBLE_VALUE_BOUNDS = {
    "participants_age": (18.0, 120.0),
    "bmi_baseline": (10.0, 100.0),
    "hba1c_percent_baseline": (2.0, 25.0),
    "c_peptide_ngml_baseline": (0.0, 100.0),
    "triglycerides_mgdl_baseline": (1.0, 3000.0),
    "hdl_cholesterol_mgdl_baseline": (1.0, 300.0),
    "tg_hdl_ratio": (0.0, 100.0),
    "waist_to_hip_ratio_baseline": (0.3, 3.0),
    "fasting_insulin_baseline": (0.0, 1000.0),
    "systolic_blood_pressure_baseline": (50.0, 300.0),
}

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
COLOR_POSITIVE = "#BA2828"
COLOR_REFERENCE = "#003366"
COLOR_ADJUSTED = "#5BBABA"
COLOR_EVENT = "#FF0000"
COLOR_NULL = "#888888"
COLOR_OBSERVED = "#000000"
COLOR_ANNOTATION = "#888888"
CLUSTER_COLORS = ["#003366", "#5BBABA", "#BA2828", "#888888"]
LINE_WIDTH_PRIMARY = 2.0
LINE_WIDTH_SECONDARY = 1.25
MARKER_SIZE = 4.0
EVENT_MARKER_SIZE = 8.0
BAND_ALPHA = 0.18
FILL_ALPHA = 0.22
FIGURE_DPI = 200
THUMBNAIL_DPI = 80

