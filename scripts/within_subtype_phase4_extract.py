"""Phase 4 extraction for the within-subtype preservation study.

Runs the explicitly approved new forward pass through the frozen checkpoint and
saves matched literal internal SSM states at 6, 12, 24, and 48 elapsed hours.
Hour 0 reuses the immutable Study 1 h0 table. Factual and static-neutral states
are scanned chronologically within every clean segment and averaged across
segments per participant. The model is never retrained.
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SEED = 42
TIME_RESOLVED_HOURS = [0, 6, 12, 24, 48]
SNAPSHOT_HOURS = [6, 12, 24, 48]
MAX_TARGET_OFFSET_MINUTES = 5.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = ROOT / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
EXPECTED_VAL_PINBALL = 3.286316
DATASET = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet")
SPLIT_PATH = Path("/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv")
FROZEN_NEUTRAL_REFERENCE = ROOT / "outputs/hidden_state_phenotype/step1_static_neutralization/20260724T223612Z/static_reference_profile.json"
STUDY1_H0 = ROOT / "outputs/static_phenotype_trajectory/step2/h0_matrix.parquet"
PHASE_ROOT = ROOT / "outputs/static_phenotype_trajectory_stratified_v2/phase4_time_resolved_extension"
SNAPSHOT_ROOT = PHASE_ROOT / "snapshots"


def load_model_from_checkpoint(ckpt_path, device):
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


def flatten_state(state, batch_index=0):
    parts = [x[batch_index].detach().cpu().numpy().reshape(-1) for x in state.layer_states]
    parts += [x[batch_index].detach().cpu().numpy().reshape(-1) for x in (state.conv_states or []) if x is not None]
    return np.concatenate(parts)


def sha256_file(path: Path, block: int = 2 ** 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_matrix(path: Path, participant_ids, splits, counts, offsets, matrix) -> None:
    frame = pd.DataFrame(matrix)
    frame.insert(0, "participant_id", participant_ids)
    frame.insert(1, "split", splits)
    frame.insert(2, "qualifying_segment_count", counts)
    frame.insert(3, "mean_abs_target_offset_minutes", offsets)
    frame.to_parquet(path, index=False)


def main() -> None:
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 80, flush=True)
    print("PHASE 4 EXTRACTION: matched 0, 6, 12, 24, and 48 hour states", flush=True)
    print("=" * 80, flush=True)

    checkpoint_hash = sha256_file(CHECKPOINT)
    model, spec, pre, checkpoint = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    print(f"Checkpoint verified. Device: {DEVICE}", flush=True)
    with FROZEN_NEUTRAL_REFERENCE.open() as handle:
        neutral_ref = json.load(handle)
    if neutral_ref["checkpoint_identifier"] != checkpoint_hash:
        raise RuntimeError("Frozen neutral reference checkpoint hash mismatch")
    if neutral_ref["feature_names"]["static_reals"] != spec.static_reals or neutral_ref["feature_names"]["static_categoricals"] != spec.static_categoricals:
        raise RuntimeError("Frozen neutral reference feature schema mismatch")
    neutral_cont = torch.tensor(np.asarray(neutral_ref["transformed_static_cont"], dtype=np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    neutral_cat = torch.tensor(np.asarray(neutral_ref["transformed_static_cat"], dtype=np.int64), dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        neutral_context = model.encode_static(neutral_cat, neutral_cont)
        neutral_initial_state = model.init_stream(neutral_context)
    neutral_h0 = flatten_state(neutral_initial_state, 0).astype(np.float32)
    np.save(PHASE_ROOT / "neutral_h0_exact.npy", neutral_h0)

    h0_index = pd.read_parquet(STUDY1_H0, columns=["participant_id", "split"])
    h0_index["participant_id"] = h0_index.participant_id.astype(str)
    participant_order = h0_index.participant_id.tolist()
    participant_set = set(participant_order)
    split_map = dict(zip(h0_index.participant_id, h0_index.split))
    print(f"Frozen Study 1 cohort: {len(participant_order)} participants", flush=True)

    panel_started = time.time()
    data = pd.read_parquet(DATASET)
    data["participant_id"] = data.participant_id.astype(str)
    data = data[data.participant_id.isin(participant_set)].copy()
    schema = infer_or_validate_schema(data)
    prepared = prepare_aireadi_panel(data, schema)
    split = make_aireadi_stream_splits(prepared, existing_split_path=SPLIT_PATH)
    streams = make_participant_streams(prepared, split, schema, feature_spec=spec, preprocessor=pre, splits=("train", "validation", "test"))
    panel_seconds = time.time() - panel_started
    streams_by_pid = {}
    for stream in streams:
        streams_by_pid.setdefault(stream.participant_id, []).append(stream)
    if set(streams_by_pid) != participant_set:
        missing = sorted(participant_set - set(streams_by_pid))
        raise RuntimeError(f"Participant streams missing for {len(missing)} frozen participants")
    print(f"Built {len(streams)} clean streams for {len(streams_by_pid)} participants in {panel_seconds:.1f}s", flush=True)

    sums_full = {hour: {} for hour in SNAPSHOT_HOURS}
    sums_neutral = {hour: {} for hour in SNAPSHOT_HOURS}
    counts = {hour: {} for hour in SNAPSHOT_HOURS}
    offset_sums = {hour: {} for hour in SNAPSHOT_HOURS}
    offset_max = {hour: 0.0 for hour in SNAPSHOT_HOURS}
    factual_h0_qa = {}
    extraction_started = time.time()
    with torch.no_grad():
        for participant_index, pid in enumerate(participant_order, 1):
            participant_streams = streams_by_pid[pid]
            first = participant_streams[0].to(DEVICE)
            factual_context = model.encode_static(first.static_cat, first.static_cont)
            if len(factual_h0_qa) < 20:
                factual_h0_qa[pid] = flatten_state(model.init_stream(factual_context), 0).astype(np.float32)
            for raw_stream in participant_streams:
                stream = raw_stream.to(DEVICE)
                timestamps = pd.to_datetime(stream.timestamps)
                elapsed_minutes = np.asarray((timestamps - timestamps[0]).total_seconds() / 60.0, dtype=np.float64)
                positions = {}
                offsets = {}
                for hour in SNAPSHOT_HOURS:
                    target = hour * 60.0
                    position = int(np.argmin(np.abs(elapsed_minutes - target)))
                    offset = float(abs(elapsed_minutes[position] - target))
                    if offset > MAX_TARGET_OFFSET_MINUTES:
                        raise RuntimeError(f"{pid} segment {stream.segment_id}: hour {hour} offset {offset:.2f} minutes")
                    positions[hour] = position
                    offsets[hour] = offset
                dynamic = stream.dynamic
                if dynamic.dim() == 2:
                    dynamic = dynamic.unsqueeze(0)
                state_full = model.init_stream(factual_context)
                state_neutral = model.init_stream(neutral_context)
                previous = 0
                for hour in SNAPSHOT_HOURS:
                    position = positions[hour]
                    chunk = dynamic[:, previous:position + 1, :]
                    state_full, _ = model.scan_chunk(chunk, factual_context, state_full)
                    state_neutral, _ = model.scan_chunk(chunk, neutral_context, state_neutral)
                    vector_full = flatten_state(state_full, 0).astype(np.float32)
                    vector_neutral = flatten_state(state_neutral, 0).astype(np.float32)
                    if pid not in sums_full[hour]:
                        sums_full[hour][pid] = np.zeros_like(vector_full)
                        sums_neutral[hour][pid] = np.zeros_like(vector_neutral)
                        counts[hour][pid] = 0
                        offset_sums[hour][pid] = 0.0
                    sums_full[hour][pid] += vector_full
                    sums_neutral[hour][pid] += vector_neutral
                    counts[hour][pid] += 1
                    offset_sums[hour][pid] += offsets[hour]
                    offset_max[hour] = max(offset_max[hour], offsets[hour])
                    previous = position + 1
            if participant_index % 100 == 0:
                elapsed = time.time() - extraction_started
                print(f"Processed {participant_index}/{len(participant_order)} participants in {elapsed:.1f}s", flush=True)
    extraction_seconds = time.time() - extraction_started

    if any(set(counts[hour]) != participant_set for hour in SNAPSHOT_HOURS):
        raise RuntimeError("At least one target hour lacks a frozen-cohort participant")
    split_order = [split_map[pid] for pid in participant_order]
    output_paths = []
    coverage_rows = []
    for hour in SNAPSHOT_HOURS:
        count_order = np.asarray([counts[hour][pid] for pid in participant_order], dtype=np.int64)
        offset_order = np.asarray([offset_sums[hour][pid] / counts[hour][pid] for pid in participant_order], dtype=np.float64)
        full_matrix = np.stack([sums_full[hour][pid] / counts[hour][pid] for pid in participant_order]).astype(np.float32)
        neutral_matrix = np.stack([sums_neutral[hour][pid] / counts[hour][pid] for pid in participant_order]).astype(np.float32)
        full_path = SNAPSHOT_ROOT / f"h_t_full_hour{hour:02d}.parquet"
        neutral_path = SNAPSHOT_ROOT / f"h_t_neutral_hour{hour:02d}.parquet"
        save_matrix(full_path, participant_order, split_order, count_order, offset_order, full_matrix)
        save_matrix(neutral_path, participant_order, split_order, count_order, offset_order, neutral_matrix)
        output_paths.extend([full_path, neutral_path])
        coverage_rows.append({
            "hour": hour,
            "participant_n": len(participant_order),
            "segment_snapshot_n": int(count_order.sum()),
            "min_segments_per_participant": int(count_order.min()),
            "median_segments_per_participant": float(np.median(count_order)),
            "max_segments_per_participant": int(count_order.max()),
            "mean_abs_target_offset_minutes": float(offset_order.mean()),
            "max_abs_target_offset_minutes": float(offset_max[hour]),
        })
        print(f"Saved hour {hour}: {full_path.name}, {neutral_path.name}", flush=True)

    qa_ids = list(factual_h0_qa)
    existing_qa = pd.read_parquet(STUDY1_H0, filters=[("participant_id", "in", qa_ids)])
    existing_qa["participant_id"] = existing_qa.participant_id.astype(str)
    vector_columns = [column for column in existing_qa.columns if column not in ("participant_id", "split")]
    existing_map = {row.participant_id: existing_qa.loc[index, vector_columns].to_numpy(dtype=np.float32) for index, row in existing_qa.iterrows()}
    h0_max_abs_difference = max(float(np.max(np.abs(factual_h0_qa[pid] - existing_map[pid]))) for pid in qa_ids)
    with torch.no_grad():
        neutral_batch_context = model.encode_static(neutral_cat.repeat(5, 1), neutral_cont.repeat(5, 1))
        neutral_batch_state = model.init_stream(neutral_batch_context)
    neutral_batch = np.stack([flatten_state(neutral_batch_state, index) for index in range(5)])
    neutral_h0_batch_max_difference = float(np.max(np.abs(neutral_batch - neutral_batch[0])))

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(PHASE_ROOT / "snapshot_coverage.csv", index=False)
    report = {
        "created_at": now_iso(),
        "approved_new_forward_pass": True,
        "model_retrained": False,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "device": DEVICE,
        "time_resolved_hours": TIME_RESOLVED_HOURS,
        "snapshot_rule": "For each clean segment, scan chronologically and retain the state immediately after the observation nearest each elapsed target hour; average states across all qualifying segments for each participant.",
        "hour_zero_source": str(STUDY1_H0),
        "neutral_hour_zero_source": str(PHASE_ROOT / "neutral_h0_exact.npy"),
        "participant_count": len(participant_order),
        "stream_count": len(streams),
        "vector_dimension": int(neutral_h0.shape[0]),
        "panel_build_seconds": panel_seconds,
        "extraction_seconds": extraction_seconds,
        "total_seconds": time.time() - started,
        "factual_h0_qa_participant_count": len(qa_ids),
        "factual_h0_max_abs_difference_vs_study1": h0_max_abs_difference,
        "neutral_h0_batch_max_difference": neutral_h0_batch_max_difference,
        "coverage": coverage_rows,
        "output_files": [str(path.relative_to(PHASE_ROOT)) for path in output_paths],
    }
    with (PHASE_ROOT / "snapshot_extraction_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    hashes = {str(path.relative_to(PHASE_ROOT)): sha256_file(path) for path in [PHASE_ROOT / "neutral_h0_exact.npy", PHASE_ROOT / "snapshot_coverage.csv", *output_paths]}
    with (PHASE_ROOT / "snapshot_artifact_hashes.json").open("w") as handle:
        json.dump(hashes, handle, indent=2)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance = {"git_branch": branch, "git_commit": commit, "expected_branch": "aireadi-ssmcgm-stream-report", "new_forward_pass": True, "checkpoint_sha256": checkpoint_hash}
    with (PHASE_ROOT / "snapshot_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"Phase 4 snapshot extraction complete in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
