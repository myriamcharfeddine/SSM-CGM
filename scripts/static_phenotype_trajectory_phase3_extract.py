"""Phase 3, extraction: h_t summary per participant, full pass and static-neutralized
pass, over the pooled overnight anchor set. Also fits the glucose-residualization
regression and runs the PC1-glucose sanity checkpoint.

Static-only clinical phenotype partition project. See build prompt for full spec.

h0 = psi(g_phi(s_i)) is the literal flattened internal SSM state (layer_states +
conv_states), matching exactly what Phase 2 saved to step2/h0_matrix.parquet. For
"h_t minus h0" to be a valid vector subtraction, h_t must live in the SAME space:
this script extracts h_t as the literal internal SSM state at each anchor timestep
(not the 128-dim projected decoder output used elsewhere in this codebase), via
chunked incremental scan_chunk calls that carry state across anchor boundaries.
"""

import json
import sys
import time
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
from ssmcgm.evaluation.aireadi_streaming import _valid_anchors  # noqa: E402
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SEED = 42
KNN_K = 15
BOOTSTRAP_N = 1000
LATENT_METRIC = "cosine"
CLINICAL_METRIC = "euclidean"
COHERENCE_RATIO_BAR = 0.30

OVERNIGHT_START_HOUR = 0
OVERNIGHT_END_HOUR = 6
MIN_OVERNIGHT_ANCHORS = 3

ANCHOR_STRIDE_STEPS = 3      # matches ssmcgm.evaluation.aireadi_streaming's default forecast-anchor cadence
OVERNIGHT_STATE_THIN = 4     # further thin overnight anchors ~4x before literal-state extraction (compute cost)
SANITY_SAMPLE_RATE = 0.15    # probability an extracted anchor joins the PC1-glucose sanity sample
SANITY_SAMPLE_MAX = 8000     # cap on sanity sample size

PC1_GLUCOSE_CORR_EXPECTED = 0.82
PC1_GLUCOSE_CORR_MIN = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = f"{ROOT}/outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
EXPECTED_VAL_PINBALL = 3.286316
DATASET = "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet"
SPLIT_PATH = "/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv"
FROZEN_NEUTRAL_REFERENCE = (
    f"{ROOT}/outputs/hidden_state_phenotype/step1_static_neutralization/"
    "20260724T223612Z/static_reference_profile.json"
)

STUDY1_ROOT = f"{ROOT}/outputs/static_phenotype_trajectory"
STEP1_DIR = f"{STUDY1_ROOT}/step1"
STEP2_DIR = f"{STUDY1_ROOT}/step2"
STEP3_DIR = f"{STUDY1_ROOT}/step3"


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


def assert_phase0to2_artifacts():
    print("=" * 80)
    print("STEP 0: Assert Phase 0-2 artifacts exist and share one participant_id index")
    print("=" * 80)
    labels_df = pd.read_parquet(f"{STEP1_DIR}/participant_cluster_labels.parquet")
    z_df = pd.read_parquet(f"{STEP1_DIR}/zscored_factor_matrix.parquet")
    clin = np.load(f"{STEP1_DIR}/clinical_pairwise_distance.npz", allow_pickle=True)
    clinical_dist = clin["distance"].astype(np.float64)
    pid_order = clin["participant_id"].astype(str)

    labels_df["participant_id"] = labels_df["participant_id"].astype(str)
    z_df["participant_id"] = z_df["participant_id"].astype(str)
    assert set(labels_df["participant_id"]) == set(pid_order), "step1 labels/distance participant_id mismatch"
    assert set(z_df["participant_id"]) == set(pid_order), "step1 z-matrix participant_id mismatch"

    if not Path(f"{STEP2_DIR}/h0_matrix.parquet").exists():
        raise SystemExit("STOP: step2/h0_matrix.parquet missing")
    if not Path(f"{STEP2_DIR}/h0_vs_pca_preservation_metrics.json").exists():
        raise SystemExit("STOP: step2/h0_vs_pca_preservation_metrics.json missing")
    h0_df = pd.read_parquet(f"{STEP2_DIR}/h0_matrix.parquet", columns=["participant_id", "split"])
    h0_df["participant_id"] = h0_df["participant_id"].astype(str)
    assert set(h0_df["participant_id"]) == set(pid_order), "step2 h0 participant_id mismatch"
    print(f"  OK: {len(pid_order)} participants consistent across step1/step2 artifacts")
    return pid_order, clinical_dist, labels_df, z_df


def main():
    Path(STEP3_DIR).mkdir(parents=True, exist_ok=True)
    pid_order, clinical_dist, labels_df, z_df = assert_phase0to2_artifacts()

    print("\n" + "=" * 80)
    print("STEP 1: Load model, neutral static reference")
    print("=" * 80)
    model, spec, pre, ckpt = load_model_from_checkpoint(CHECKPOINT, DEVICE)
    print(f"  Checkpoint OK. Device: {DEVICE}")

    import hashlib

    def sha256_file(path, block=2 ** 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(block)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    ckpt_hash = sha256_file(CHECKPOINT)
    with open(FROZEN_NEUTRAL_REFERENCE) as f:
        neutral_ref = json.load(f)
    if neutral_ref["checkpoint_identifier"] != ckpt_hash:
        raise SystemExit(
            f"STOP: frozen neutral reference checkpoint hash {neutral_ref['checkpoint_identifier']} "
            f"!= current checkpoint hash {ckpt_hash}"
        )
    assert neutral_ref["feature_names"]["static_reals"] == spec.static_reals
    assert neutral_ref["feature_names"]["static_categoricals"] == spec.static_categoricals
    neutral_cont_vec = np.asarray(neutral_ref["transformed_static_cont"], dtype=np.float32)
    neutral_cat_vec = np.asarray(neutral_ref["transformed_static_cat"], dtype=np.int64)
    print(f"  Frozen neutral reference verified (checkpoint hash match, {neutral_ref['training_participant_count']} "
          f"training participants used to build it)")

    print("\n" + "=" * 80)
    print("STEP 2: Load panel, build participant streams for the complete-case cohort")
    print("=" * 80)
    t0 = time.time()
    df = pd.read_parquet(DATASET)
    df["participant_id"] = df["participant_id"].astype(str)
    df = df[df["participant_id"].isin(set(pid_order))].copy()
    schema = infer_or_validate_schema(df)
    prepared = prepare_aireadi_panel(df, schema)
    print(f"  prepare_aireadi_panel: {time.time() - t0:.1f}s, shape {prepared.shape}")

    split = make_aireadi_stream_splits(prepared, existing_split_path=SPLIT_PATH)
    t0 = time.time()
    streams = make_participant_streams(
        prepared, split, schema, feature_spec=spec, preprocessor=pre, splits=("train", "validation", "test")
    )
    print(f"  make_participant_streams: {time.time() - t0:.1f}s, n_streams={len(streams)}")

    streams_by_pid = {}
    for s in streams:
        streams_by_pid.setdefault(s.participant_id, []).append(s)
    print(f"  Participants with at least one stream: {len(streams_by_pid)} / {len(pid_order)}")

    print("\n" + "=" * 80)
    print("STEP 3: Neutral static context (computed once, identical for every participant)")
    print("=" * 80)
    neutral_cont_t = torch.tensor(neutral_cont_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    neutral_cat_t = torch.tensor(neutral_cat_vec, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        sctx_neutral_global = model.encode_static(neutral_cat_t, neutral_cont_t)
    # QA: batch this neutral input 5x and confirm h0 is numerically identical across the batch
    with torch.no_grad():
        qa_cont = neutral_cont_t.repeat(5, 1)
        qa_cat = neutral_cat_t.repeat(5, 1)
        qa_sctx = model.encode_static(qa_cat, qa_cont)
        qa_state = model.init_stream(qa_sctx)
    qa_vecs = np.stack([flatten_state(qa_state, i) for i in range(5)])
    qa_max_diff = float(np.abs(qa_vecs - qa_vecs[0]).max())
    print(f"  Neutral h0 batch-of-5 max pairwise difference: {qa_max_diff:.3e} (expect ~0.0)")

    print("\n" + "=" * 80)
    print("STEP 4: Extract h_t (factual and neutral) over pooled, thinned overnight anchors")
    print("=" * 80)
    H = spec.horizon_steps
    rng = np.random.default_rng(SEED)

    n_flat = None
    sum_h_factual = {}
    sum_glucose = {}
    count_anchors = {}
    sum_h_neutral = {}

    global_n = 0
    global_sum_g = 0.0
    global_sum_g2 = 0.0
    global_sum_h = None
    global_sum_gh = None

    sanity_glucose = []
    sanity_h = []

    t_extract0 = time.time()
    n_participants_done = 0
    with torch.no_grad():
        for pid, pid_streams in streams_by_pid.items():
            first = pid_streams[0].to(DEVICE)
            sctx_factual = model.encode_static(first.static_cat, first.static_cont)

            p_sum_h_f = None
            p_sum_h_n = None
            p_sum_g = 0.0
            p_count = 0

            for raw_stream in pid_streams:
                s = raw_stream.to(DEVICE)
                anchors = _valid_anchors(s, H, stride=ANCHOR_STRIDE_STEPS)
                if not anchors:
                    continue
                ts = s.timestamps
                hours = np.array([pd.Timestamp(ts[a]).hour for a in anchors])
                overnight = [a for a, hr in zip(anchors, hours) if OVERNIGHT_START_HOUR <= hr < OVERNIGHT_END_HOUR]
                keep = overnight[::OVERNIGHT_STATE_THIN]
                if not keep:
                    continue

                dynamic = s.dynamic
                if dynamic.dim() == 2:
                    dynamic = dynamic.unsqueeze(0)

                state_f = model.init_stream(sctx_factual)
                state_n = model.init_stream(sctx_neutral_global)
                prev = 0
                for pos in keep:
                    chunk = dynamic[:, prev:pos + 1, :]
                    state_f, _ = model.scan_chunk(chunk, sctx_factual, state_f)
                    state_n, _ = model.scan_chunk(chunk, sctx_neutral_global, state_n)
                    vec_f = flatten_state(state_f, 0)
                    vec_n = flatten_state(state_n, 0)
                    glucose = float(s.target[pos].item())

                    if n_flat is None:
                        n_flat = vec_f.shape[0]
                        global_sum_h = np.zeros(n_flat, dtype=np.float64)
                        global_sum_gh = np.zeros(n_flat, dtype=np.float64)

                    p_sum_h_f = vec_f.astype(np.float64) if p_sum_h_f is None else p_sum_h_f + vec_f
                    p_sum_h_n = vec_n.astype(np.float64) if p_sum_h_n is None else p_sum_h_n + vec_n
                    p_sum_g += glucose
                    p_count += 1

                    global_n += 1
                    global_sum_g += glucose
                    global_sum_g2 += glucose ** 2
                    global_sum_h += vec_f
                    global_sum_gh += glucose * vec_f

                    if len(sanity_glucose) < SANITY_SAMPLE_MAX and rng.random() < SANITY_SAMPLE_RATE:
                        sanity_glucose.append(glucose)
                        sanity_h.append(vec_f.copy())

                    prev = pos + 1

            if p_count > 0:
                sum_h_factual[pid] = p_sum_h_f
                sum_h_neutral[pid] = p_sum_h_n
                sum_glucose[pid] = p_sum_g
                count_anchors[pid] = p_count

            n_participants_done += 1
            if n_participants_done % 200 == 0:
                elapsed = time.time() - t_extract0
                print(f"  ... {n_participants_done}/{len(streams_by_pid)} participants, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t_extract0
    print(f"  Extraction complete: {elapsed:.0f}s for {n_participants_done} participants, "
          f"{global_n} total kept anchors, h_t dim={n_flat}")

    print("\n" + "=" * 80)
    print("STEP 5: Retention, averaging, glucose regression, residualization")
    print("=" * 80)
    retained_pids = [pid for pid in pid_order if count_anchors.get(pid, 0) >= MIN_OVERNIGHT_ANCHORS]
    dropped_pids = [pid for pid in pid_order if pid not in set(retained_pids)]
    print(f"  Retained: {len(retained_pids)} / {len(pid_order)}  (dropped {len(dropped_pids)})")

    split_map = dict(zip(labels_df["participant_id"], labels_df["split"]))
    retained_per_split = pd.Series([split_map[p] for p in retained_pids]).value_counts().to_dict()
    dropped_per_split = pd.Series([split_map[p] for p in dropped_pids]).value_counts().to_dict() if dropped_pids else {}
    print(f"  Retained per split: {retained_per_split}")
    print(f"  Dropped per split: {dropped_per_split}")

    beta = (global_sum_gh / global_n - (global_sum_g / global_n) * (global_sum_h / global_n)) / (
        global_sum_g2 / global_n - (global_sum_g / global_n) ** 2
    )
    intercept = global_sum_h / global_n - beta * (global_sum_g / global_n)
    print(f"  Glucose-regression fit on {global_n} anchors. |beta| mean={np.abs(beta).mean():.5f}, "
          f"|intercept| mean={np.abs(intercept).mean():.5f}")

    h_t_factual = np.stack([sum_h_factual[p] / count_anchors[p] for p in retained_pids]).astype(np.float32)
    h_t_neutral = np.stack([sum_h_neutral[p] / count_anchors[p] for p in retained_pids]).astype(np.float32)
    avg_glucose = np.array([sum_glucose[p] / count_anchors[p] for p in retained_pids], dtype=np.float64)
    n_anchors_arr = np.array([count_anchors[p] for p in retained_pids], dtype=np.int64)
    h_t_factual_resid = (h_t_factual.astype(np.float64) - np.outer(avg_glucose, beta) - intercept).astype(np.float32)

    print("\n" + "=" * 80)
    print("STEP 6: Soft sanity checkpoint, PC1 vs current glucose")
    print("=" * 80)
    sanity_h_arr = np.stack(sanity_h)
    sanity_glucose_arr = np.array(sanity_glucose)
    from sklearn.decomposition import PCA
    pca_sanity = PCA(n_components=1, random_state=SEED)
    pc1_sanity = pca_sanity.fit_transform(sanity_h_arr)[:, 0]
    corr = float(np.corrcoef(pc1_sanity, sanity_glucose_arr)[0, 1])
    corr = abs(corr)
    print(f"  Sanity sample size: {len(sanity_glucose)}")
    print(f"  |corr(h_t PC1, current glucose)| = {corr:.4f}  (expected ~{PC1_GLUCOSE_CORR_EXPECTED}, "
          f"hard floor {PC1_GLUCOSE_CORR_MIN})")
    print("  NOTE: this study's h_t is the literal internal SSM state (matching h0's representation), "
          "not the 128-dim post-update decoder output used in the prior exploratory pipeline, so the "
          "expected 0.82 reference value may not transfer exactly; only the floor is a hard stop.")
    if corr < PC1_GLUCOSE_CORR_MIN:
        raise SystemExit(f"STOP: PC1-glucose correlation {corr:.4f} below floor {PC1_GLUCOSE_CORR_MIN}")

    print("\n" + "=" * 80)
    print("STEP 7: Save outputs")
    print("=" * 80)
    def save_matrix(path, ids, splits_, matrix, extra_cols=None):
        d = pd.DataFrame(matrix)
        d.insert(0, "participant_id", ids)
        d.insert(1, "split", splits_)
        if extra_cols:
            for i, (name, vals) in enumerate(extra_cols.items()):
                d.insert(2 + i, name, vals)
        d.to_parquet(path, index=False)

    splits_retained = [split_map[p] for p in retained_pids]
    save_matrix(f"{STEP3_DIR}/h_t_full.parquet", retained_pids, splits_retained, h_t_factual,
                {"n_overnight_anchors": n_anchors_arr, "avg_glucose_mgdl": avg_glucose})
    save_matrix(f"{STEP3_DIR}/h_t_neutral.parquet", retained_pids, splits_retained, h_t_neutral,
                {"n_overnight_anchors": n_anchors_arr})
    save_matrix(f"{STEP3_DIR}/h_t_full_residualized.parquet", retained_pids, splits_retained, h_t_factual_resid,
                {"n_overnight_anchors": n_anchors_arr})

    np.savez_compressed(f"{STEP3_DIR}/glucose_regression_coefficients.npz", beta=beta, intercept=intercept, n=global_n)

    anchor_report = {
        "anchor_stride_steps": ANCHOR_STRIDE_STEPS,
        "overnight_state_thin": OVERNIGHT_STATE_THIN,
        "overnight_hour_window": [OVERNIGHT_START_HOUR, OVERNIGHT_END_HOUR],
        "min_overnight_anchors": MIN_OVERNIGHT_ANCHORS,
        "n_participants_with_streams": len(streams_by_pid),
        "n_retained": len(retained_pids),
        "n_dropped": len(dropped_pids),
        "dropped_participant_ids": dropped_pids,
        "retained_per_split": retained_per_split,
        "dropped_per_split": dropped_per_split,
        "total_kept_anchors": int(global_n),
        "mean_anchors_per_participant": float(n_anchors_arr.mean()),
        "median_anchors_per_participant": float(np.median(n_anchors_arr)),
        "min_anchors_per_participant": int(n_anchors_arr.min()),
        "max_anchors_per_participant": int(n_anchors_arr.max()),
        "neutral_h0_batch5_max_diff": qa_max_diff,
        "pc1_glucose_correlation": corr,
        "pc1_glucose_correlation_expected": PC1_GLUCOSE_CORR_EXPECTED,
        "pc1_glucose_correlation_min": PC1_GLUCOSE_CORR_MIN,
        "sanity_sample_size": len(sanity_glucose),
        "extraction_seconds": elapsed,
        "h_t_dim": int(n_flat),
        "checkpoint_hash": ckpt_hash,
        "neutral_reference_source": FROZEN_NEUTRAL_REFERENCE,
    }
    with open(f"{STEP3_DIR}/anchor_extraction_report.json", "w") as f:
        json.dump(anchor_report, f, indent=2)
    print(f"  Wrote {STEP3_DIR}/h_t_full.parquet, h_t_neutral.parquet, h_t_full_residualized.parquet")
    print(f"  Wrote {STEP3_DIR}/anchor_extraction_report.json")
    print("\nExtraction stage done.")


if __name__ == "__main__":
    main()
