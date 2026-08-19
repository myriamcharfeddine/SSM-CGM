"""Pre-registered positive-control probe: pooled h_t + PCA -> mean glucose.

Checks whether mean-pooling h_t per participant and projecting onto 5
train-fit PCA components recovers participant-level mean CGM glucose, as a
sanity gate before applying the same pipeline to clinical features
(C-peptide, TG/HDL, age, BMI). The frozen checkpoint is loaded read-only and
never retrained. Split: adapt6h_seed42 (experiment_c_split_adapt6h_seed42).

Stop condition: report the result and stop. Do not proceed to the clinical
battery until the user reviews whether the bootstrap CI excludes zero and R^2
is in a reasonable range.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssmcgm.data.aireadi import (  # noqa: E402
    AireadiFeatureSpec,
    AireadiPreprocessor,
    infer_or_validate_schema,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import (  # noqa: E402
    AireadiStreamModel,
    AireadiStreamModelConfig,
)

# ---------------------------------------------------------------------------
# Named constants (pre-registered)
# ---------------------------------------------------------------------------
POOL_METHOD = "mean"
N_COMPONENTS = 5
N_BOOTSTRAP = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
EXPECTED_VAL_PINBALL = 3.286316
SPLIT_PATH = Path("/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv")
DATASET = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet")
VECTOR_DIMENSION = 35072
ANCHOR_STRIDE_STEPS = 3  # matches ssmcgm.evaluation.aireadi_streaming.evaluate_aireadi_streams default
WARMUP_STEPS = 0

OUTPUT_DIR = ROOT / "outputs/positive_control_pooled_pca_glucose_probe"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_anchors(stream, horizon: int, stride: int, warmup_steps: int = 0):
    """Exact local copy of the canonical evaluation anchor rule
    (ssmcgm.evaluation.aireadi_streaming._valid_anchors). Pure positional
    NumPy/tensor indexing over stream.observed; no timestamps involved.
    """
    anchors = []
    last = -10 ** 9
    for t in range(0, stream.n_steps - horizon):
        if t + 1 < warmup_steps:
            continue
        if (t - last) < stride:
            continue
        if bool(stream.observed[t]) and bool(stream.observed[t + 1:t + 1 + horizon].all()):
            anchors.append(t)
            last = t
    return anchors


def load_model_from_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reported = ckpt["metrics"]["val_pinball_mgdl"]
    if abs(reported - EXPECTED_VAL_PINBALL) > 1e-4:
        raise SystemExit(f"STOP: checkpoint val_pinball_mgdl={reported} != expected {EXPECTED_VAL_PINBALL}")
    md = ckpt["metadata"]
    spec = AireadiFeatureSpec(**md["feature_spec"])
    pre = AireadiPreprocessor.from_jsonable(md["preprocessor"])
    mcfg = AireadiStreamModelConfig(**md["model_config"])
    model = AireadiStreamModel(spec, pre, mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spec, pre, ckpt


def flatten_state(state, batch_index: int = 0) -> np.ndarray:
    parts = [x[batch_index].detach().cpu().numpy().reshape(-1) for x in state.layer_states]
    parts += [x[batch_index].detach().cpu().numpy().reshape(-1) for x in (state.conv_states or []) if x is not None]
    return np.concatenate(parts)


def sha256_file(path: Path, block: int = 2 ** 20) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_ground_truth_lookup(participant_ids: set[str]) -> dict:
    """Raw ground-truth mg/dL glucose, gated exactly like
    ssmcgm.analysis.hba1c_positive_control.panel_glucose_range (cgm_count>0 &
    notna()), never taken from model output or from stream.target (which can
    contain short-gap linear interpolation under the clean-segment path).
    """
    panel = pd.read_parquet(
        DATASET, columns=["participant_id", "timestamp_local", "cgm_glucose_mean", "cgm_count"]
    )
    panel["participant_id"] = panel["participant_id"].astype(str)
    panel = panel[panel["participant_id"].isin(participant_ids)].copy()
    glucose = pd.to_numeric(panel["cgm_glucose_mean"], errors="coerce")
    valid = panel["cgm_count"].fillna(0).gt(0) & glucose.notna()
    panel = panel.loc[valid, ["participant_id", "timestamp_local", "cgm_glucose_mean"]]
    panel["timestamp_local"] = pd.to_datetime(panel["timestamp_local"])
    lookup = {
        (pid, ts): float(value)
        for pid, ts, value in zip(panel["participant_id"], panel["timestamp_local"], panel["cgm_glucose_mean"])
    }
    return lookup


def extract_pooled_states(
    model,
    streams_by_pid: dict,
    ground_truth_lookup: dict,
    split_map: dict,
) -> pd.DataFrame:
    horizon = model.feature_spec.horizon_steps
    rows = []
    started = time.time()
    with torch.no_grad():
        for index, pid in enumerate(sorted(streams_by_pid), 1):
            participant_streams = streams_by_pid[pid]
            first = participant_streams[0].to(DEVICE)
            context = model.encode_static(first.static_cat, first.static_cont)
            state_sum = np.zeros(VECTOR_DIMENSION, dtype=np.float64)
            glucose_sum = 0.0
            anchor_n = 0
            total_scoreable_anchors = 0
            for raw_stream in participant_streams:
                stream = raw_stream.to(DEVICE)
                anchors = valid_anchors(stream, horizon, ANCHOR_STRIDE_STEPS, WARMUP_STEPS)
                total_scoreable_anchors += len(anchors)
                if not anchors:
                    continue
                dynamic = stream.dynamic
                if dynamic.dim() == 2:
                    dynamic = dynamic.unsqueeze(0)
                timestamps = pd.to_datetime(stream.timestamps)
                state = model.init_stream(context)
                previous = 0
                for position in anchors:
                    chunk = dynamic[:, previous:position + 1, :]
                    state, _ = model.scan_chunk(chunk, context, state)
                    previous = position + 1
                    glucose = ground_truth_lookup.get((pid, pd.Timestamp(timestamps[position])))
                    if glucose is None:
                        continue
                    vector = flatten_state(state, 0).astype(np.float64)
                    state_sum += vector
                    glucose_sum += glucose
                    anchor_n += 1
            rows.append(
                {
                    "participant_id": pid,
                    "split": split_map[pid],
                    "scoreable_anchor_count": total_scoreable_anchors,
                    "pooled_anchor_count": anchor_n,
                    "mean_glucose_target": (glucose_sum / anchor_n) if anchor_n else np.nan,
                    "pooled_h_t": (state_sum / anchor_n).astype(np.float32) if anchor_n else None,
                }
            )
            if index % 100 == 0:
                print(f"  processed {index}/{len(streams_by_pid)} participants ({time.time() - started:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def clustered_bootstrap_r2(observed: np.ndarray, predicted: np.ndarray, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(observed)
    index = rng.integers(0, n, size=(n_bootstrap, n))
    boot_r2 = np.array([r2_score(observed[idx], predicted[idx]) for idx in index])
    ci_low, ci_high = np.percentile(boot_r2, [2.5, 97.5])
    return float(ci_low), float(ci_high), boot_r2


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 80, flush=True)
    print("POSITIVE CONTROL: pooled h_t + PCA -> mean glucose", flush=True)
    print("=" * 80, flush=True)

    checkpoint_hash = sha256_file(CHECKPOINT)
    model, spec, pre, checkpoint = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    print(f"Checkpoint verified (val_pinball_mgdl={checkpoint['metrics']['val_pinball_mgdl']}). Device: {DEVICE}", flush=True)

    split_frame = pd.read_csv(SPLIT_PATH)
    split_frame["participant_id"] = split_frame["participant_id"].astype(str)
    split_map_csv = dict(zip(split_frame["participant_id"], split_frame["split"]))
    train_ids_csv = {pid for pid, s in split_map_csv.items() if s == "train"}
    test_ids_csv = {pid for pid, s in split_map_csv.items() if s == "test"}
    print(f"Split file: {len(train_ids_csv)} train / {len(test_ids_csv)} test participants", flush=True)

    panel_started = time.time()
    data = pd.read_parquet(DATASET)
    data["participant_id"] = data.participant_id.astype(str)
    data = data[data.participant_id.isin(train_ids_csv | test_ids_csv)].copy()
    schema = infer_or_validate_schema(data)
    prepared = prepare_aireadi_panel(data, schema)
    split = make_aireadi_stream_splits(prepared, existing_split_path=SPLIT_PATH)
    streams = make_participant_streams(
        prepared, split, schema, feature_spec=spec, preprocessor=pre, splits=("train", "test")
    )
    print(f"Built {len(streams)} clean streams in {time.time() - panel_started:.1f}s", flush=True)

    streams_by_pid: dict = {}
    split_map: dict = {}
    for stream in streams:
        streams_by_pid.setdefault(stream.participant_id, []).append(stream)
        split_map[stream.participant_id] = stream.split

    ground_truth_lookup = build_ground_truth_lookup(set(streams_by_pid))
    print(f"Ground-truth glucose lookup: {len(ground_truth_lookup)} rows", flush=True)

    print("Extracting pooled h_t per participant (all scoreable anchors)...", flush=True)
    extraction_started = time.time()
    pooled = extract_pooled_states(model, streams_by_pid, ground_truth_lookup, split_map)
    print(f"Extraction complete in {time.time() - extraction_started:.1f}s", flush=True)

    usable = pooled[pooled["pooled_anchor_count"] > 0].copy()
    excluded = pooled[pooled["pooled_anchor_count"] == 0]
    if len(excluded):
        print(f"WARNING: {len(excluded)} participants excluded (zero pooled anchors with ground truth)", flush=True)

    train_rows = usable[usable["split"] == "train"].reset_index(drop=True)
    test_rows = usable[usable["split"] == "test"].reset_index(drop=True)

    # Persist pooled vectors (not intermediate per-anchor states) for reuse/audit.
    def save_pooled(frame: pd.DataFrame, path: Path) -> None:
        matrix = np.stack(frame["pooled_h_t"].to_numpy())
        out = pd.DataFrame(matrix)
        out.insert(0, "participant_id", frame["participant_id"].to_numpy())
        out.insert(1, "split", frame["split"].to_numpy())
        out.insert(2, "pooled_anchor_count", frame["pooled_anchor_count"].to_numpy())
        out.insert(3, "mean_glucose_target", frame["mean_glucose_target"].to_numpy())
        out.to_parquet(path, index=False)

    save_pooled(train_rows, OUTPUT_DIR / "pooled_train.parquet")
    save_pooled(test_rows, OUTPUT_DIR / "pooled_test.parquet")

    train_matrix = np.stack(train_rows["pooled_h_t"].to_numpy()).astype(np.float64)
    test_matrix = np.stack(test_rows["pooled_h_t"].to_numpy()).astype(np.float64)
    train_glucose = train_rows["mean_glucose_target"].to_numpy(dtype=np.float64)
    test_glucose = test_rows["mean_glucose_target"].to_numpy(dtype=np.float64)

    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    train_scores = pca.fit_transform(train_matrix)
    test_scores = pca.transform(test_matrix)

    ols = LinearRegression()
    ols.fit(train_scores, train_glucose)
    test_predicted = ols.predict(test_scores)
    held_out_r2 = float(r2_score(test_glucose, test_predicted))

    ci_low, ci_high, boot_r2 = clustered_bootstrap_r2(test_glucose, test_predicted, N_BOOTSTRAP, SEED)

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    report = {
        "created_at": now_iso(),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_val_pinball_mgdl": checkpoint["metrics"]["val_pinball_mgdl"],
        "split_path": str(SPLIT_PATH),
        "git_branch": branch,
        "git_commit": commit,
        "pool_method": POOL_METHOD,
        "n_components": N_COMPONENTS,
        "n_bootstrap": N_BOOTSTRAP,
        "anchor_stride_steps": ANCHOR_STRIDE_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "vector_dimension": VECTOR_DIMENSION,
        "n_train_participants_used": int(len(train_rows)),
        "n_test_participants_used": int(len(test_rows)),
        "n_train_participants_excluded": int((pooled["split"] == "train").sum() - len(train_rows)),
        "n_test_participants_excluded": int((pooled["split"] == "test").sum() - len(test_rows)),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "held_out_r2": held_out_r2,
        "held_out_r2_bootstrap_ci_low": ci_low,
        "held_out_r2_bootstrap_ci_high": ci_high,
        "bootstrap_method": "participant-clustered percentile bootstrap on frozen train-fit predictions, 2.5/97.5 percentiles (Study 2 ablation convention)",
        "total_seconds": time.time() - started,
    }
    with (OUTPUT_DIR / "positive_control_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    np.save(OUTPUT_DIR / "bootstrap_r2_values.npy", boot_r2)

    print("=" * 80, flush=True)
    print(f"N train used: {report['n_train_participants_used']} (excluded {report['n_train_participants_excluded']})", flush=True)
    print(f"N test used:  {report['n_test_participants_used']} (excluded {report['n_test_participants_excluded']})", flush=True)
    print(f"PCA explained variance ratio (5 comps): {pca.explained_variance_ratio_}", flush=True)
    print(f"Held-out R^2 (test): {held_out_r2:.4f}", flush=True)
    print(f"95% participant-clustered bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]", flush=True)
    print(f"Total time: {time.time() - started:.1f}s", flush=True)
    print("STOP: reporting result only. Not proceeding to clinical-feature battery.", flush=True)


if __name__ == "__main__":
    main()
