#!/usr/bin/env python3
"""AI-READI interpretability v2, Phase 1: corrected global attribution.

Enriches v1's group_overall_lomo_full_test.csv / group_overall_rigorous.csv /
group_overall_compare.csv with per-horizon signed/raw effects, predictive
ablation deltas (MAE, pinball loss, 80% interval coverage), and participant-
clustered bootstrap CIs (1000 reps, seed 42) on every headline value. Writes
_v2-suffixed CSVs; does not touch or overwrite the v1 files.

Uses the model's own existing train-fitted preprocessor and the existing
participant_heldout seed-42 split (via load_test_streams). No new split, no
retraining, no checkpoint modification.

Same GPU cost as v1's LOMO + rigorous passes (381 streams x (1 baseline + 11
ablated + 1 rigorous scan) = 4953 scans), already proven tractable.

PAUSE: by default this only counts streams/anchors/batches and exits.

Usage:
  python scripts/interpret_airedi_global_v2.py                # count only
  python scripts/interpret_airedi_global_v2.py --execute       # run + write CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.interpret_airedi_export import (
    BIN_MIN,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    HORIZON,
    MAX_LAG_STEPS,
    anchors_for_stream,
    load_model,
    load_test_streams,
    scan_with_record,
)
from scripts.interpret_airedi_groups import GROUP_ORDER, build_group_members
from scripts.interpret_airedi_grouped_export import h_t_group_members
from ssmcgm.ops.mes_attribution import compute_hidden_attention

SEED = 42
N_BOOT = 1000
CI_LO_PCTILE = 2.5
CI_HI_PCTILE = 97.5
COVERAGE_INTERVAL_LO_Q = 0.1
COVERAGE_INTERVAL_HI_Q = 0.9
OUT_DIR = ROOT / "outputs/interpretability_v2"


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    diff = y - pred
    return np.maximum(q * diff, (q - 1.0) * diff)


def count_only(streams):
    total_anchors = sum(len(anchors_for_stream(s.n_steps)) for s in streams)
    n_participants = len(set(str(s.participant_id) for s in streams))
    n_scans = len(streams) * (1 + len(GROUP_ORDER) + 1)
    print(f"[global-v2] participants: {n_participants}  streams: {len(streams)}  "
          f"anchors: {total_anchors}")
    print(f"[global-v2] GPU batches (scans) required: {n_scans} "
          f"({len(streams)} streams x (1 baseline + {len(GROUP_ORDER)} ablated + 1 rigorous scan))")
    print("[global-v2] PAUSE: count-only dry run. Re-run with --execute to run the GPU pass.")
    return n_scans


@torch.no_grad()
def _clone_and_predict(model, stream, anchors, device, idxs, group, groups):
    dyn_idx, time_idx, scn_idx, scont_idx, scat_idx = idxs
    dynamic = stream.dynamic.to(device).clone()
    time_features = stream.time_features.to(device).clone()
    scenario_values = stream.scenario_values.to(device).clone()
    scenario_mask = torch.zeros_like(stream.scenario_mask.to(device))
    static_cont = stream.static_cont.to(device).clone()
    static_cat = stream.static_cat.to(device).clone()
    target = stream.target.to(device)
    if group is not None:
        for pathway, name in groups[group]:
            if pathway == "dynamic" and name in dyn_idx:
                dynamic[:, dyn_idx[name]] = 0.0
            elif pathway == "time" and name in time_idx:
                time_features[:, time_idx[name]] = 0.0
            elif pathway == "scenario" and name in scn_idx:
                j = scn_idx[name]
                scenario_values[:, j] = 0.0
                scenario_mask[:, j] = 0.0
            elif pathway == "static_cont" and name in scont_idx:
                static_cont[scont_idx[name]] = 0.0
            elif pathway == "static_cat" and name in scat_idx:
                static_cat[scat_idx[name]] = 0
    sctx = model.encode_static(static_cat, static_cont)
    state = model.init_stream(sctx)
    _state, out = model.scan_chunk(dynamic.unsqueeze(0), sctx, state)
    pos = torch.tensor(anchors, dtype=torch.long, device=device)
    fut = pos[:, None] + 1 + torch.arange(HORIZON, device=device)[None, :]
    fut = torch.clamp(fut, max=stream.n_steps - 1)
    raw = model.decode_horizon(out[0, pos], sctx, time_features[fut], scenario_values[fut], scenario_mask[fut])
    pred = target[pos].view(-1, 1, 1) + raw
    return pred.detach().cpu().numpy()          # (n_anchors, HORIZON, n_quantiles)


def run_global_v2(model, streams, device, groups, groups_dyn, out_dir: Path):
    spec = model.feature_spec
    idxs = (
        {n: i for i, n in enumerate(spec.dynamic_reals)},
        {n: i for i, n in enumerate(spec.time_reals)},
        {n: i for i, n in enumerate(spec.scenario_reals)},
        {n: i for i, n in enumerate(spec.static_reals)},
        {n: i for i, n in enumerate(spec.static_categoricals)},
    )
    quantiles = np.asarray(model.quantiles)
    q10i, q50i, q90i = int(np.argmin(quantiles)), int(np.argmin(np.abs(quantiles - 0.5))), int(np.argmax(quantiles))
    ht_group_names = list(groups_dyn)
    ngroups_ssm = model.temporal.blocks[0].ssm.ngroups

    n_g, n_h = len(GROUP_ORDER), HORIZON
    per_pid_lomo = {}          # pid -> dict of (n_g, n_h) sum arrays + n anchors
    per_pid_rig = {}           # pid -> dict of (n_ht_g,) sum arrays + n anchors
    n_streams_used = 0
    total_anchors = 0
    per_pid_posthoc = {}       # pid -> per-individual-dynamic-feature raw sums, same units as per_pid_rig
    dyn_feature_names = list(spec.dynamic_reals)

    for si, stream in enumerate(streams):
        anchors = anchors_for_stream(stream.n_steps)
        if len(anchors) == 0:
            continue
        pid = str(stream.participant_id)
        tgt = stream.target.numpy()
        fut_idx = np.clip(anchors[:, None] + 1 + np.arange(HORIZON)[None, :], 0, stream.n_steps - 1)
        observed = tgt[fut_idx]                                   # (n_anchors, HORIZON)

        base = _clone_and_predict(model, stream, anchors, device, idxs, None, groups)
        base_med, base_q10, base_q90 = base[:, :, q50i], base[:, :, q10i], base[:, :, q90i]
        base_abs_err = np.abs(base_med - observed)
        base_pinball = (pinball_loss(observed, base_q10, COVERAGE_INTERVAL_LO_Q)
                        + pinball_loss(observed, base_med, 0.5)
                        + pinball_loss(observed, base_q90, COVERAGE_INTERVAL_HI_Q)) / 3.0
        base_cov = ((observed >= base_q10) & (observed <= base_q90)).astype(float)

        rec = per_pid_lomo.setdefault(pid, {
            "abs_delta": np.zeros((n_g, n_h)), "signed_delta": np.zeros((n_g, n_h)),
            "delta_mae": np.zeros((n_g, n_h)), "delta_pinball": np.zeros((n_g, n_h)),
            "delta_coverage": np.zeros((n_g, n_h)), "n": 0,
        })
        for gi, g in enumerate(GROUP_ORDER):
            if not groups[g]:
                continue
            ablated = _clone_and_predict(model, stream, anchors, device, idxs, g, groups)
            abl_med, abl_q10, abl_q90 = ablated[:, :, q50i], ablated[:, :, q10i], ablated[:, :, q90i]
            abs_delta = np.abs(abl_med - base_med)
            signed_delta = abl_med - base_med
            abl_abs_err = np.abs(abl_med - observed)
            delta_mae = abl_abs_err - base_abs_err
            abl_pinball = (pinball_loss(observed, abl_q10, COVERAGE_INTERVAL_LO_Q)
                          + pinball_loss(observed, abl_med, 0.5)
                          + pinball_loss(observed, abl_q90, COVERAGE_INTERVAL_HI_Q)) / 3.0
            delta_pinball = abl_pinball - base_pinball
            abl_cov = ((observed >= abl_q10) & (observed <= abl_q90)).astype(float)
            delta_coverage = abl_cov - base_cov

            rec["abs_delta"][gi] += abs_delta.sum(axis=0)
            rec["signed_delta"][gi] += signed_delta.sum(axis=0)
            rec["delta_mae"][gi] += delta_mae.sum(axis=0)
            rec["delta_pinball"][gi] += delta_pinball.sum(axis=0)
            rec["delta_coverage"][gi] += delta_coverage.sum(axis=0)
        rec["n"] += len(anchors)

        caches, contribs = scan_with_record(model, stream, device)
        layer = len(caches) - 1
        rows_t = torch.tensor(anchors, dtype=torch.long, device=device)
        attn, _ = compute_hidden_attention(
            caches[layer], rows=rows_t, max_lags=MAX_LAG_STEPS,
            ngroups=ngroups_ssm, normalize="softmax", out_device="cpu",
        )
        attn_np = attn.float().numpy()[0].sum(axis=0)             # (R, K+1)
        lag_idx = np.arange(MAX_LAG_STEPS + 1, dtype=int)
        src = anchors[:, None] - lag_idx[None, :]
        valid = src >= 0
        src_clip = np.clip(src, 0, stream.n_steps - 1)
        rrec = per_pid_rig.setdefault(pid, {"raw": np.zeros(len(ht_group_names)), "n": 0})
        for gi, g in enumerate(ht_group_names):
            u_sum = sum(contribs[f] for f in groups_dyn[g])
            u_norm = u_sum.detach().cpu().norm(dim=-1).numpy()[0]
            vals = np.where(valid, u_norm[src_clip], 0.0)
            rrec["raw"][gi] += (attn_np * vals).sum()
        rrec["n"] += len(anchors)

        # same accumulation convention, applied to INDIVIDUAL dynamic features, so the
        # post-hoc sum-of-norms is computed in units directly comparable to the rigorous
        # joint norm above (both from this same pass, both attn*u_norm summed per anchor).
        frec = per_pid_posthoc.setdefault(pid, {"raw": np.zeros(len(dyn_feature_names)), "n": 0})
        for fi, fname in enumerate(dyn_feature_names):
            u_norm_f = contribs[fname].detach().cpu().norm(dim=-1).numpy()[0]
            vals_f = np.where(valid, u_norm_f[src_clip], 0.0)
            frec["raw"][fi] += (attn_np * vals_f).sum()
        frec["n"] += len(anchors)

        n_streams_used += 1
        total_anchors += len(anchors)
        if (si + 1) % 50 == 0:
            print(f"[global-v2] processed {si + 1}/{len(streams)} streams")

    print(f"[global-v2] pass complete: {n_streams_used} streams, {total_anchors} anchors, "
          f"{len(per_pid_lomo)} participants")

    rng = np.random.default_rng(SEED)
    lomo_rows, horizon_rows, predictive_rows = [], [], []
    pids = sorted(per_pid_lomo)

    def boot_ci_1d(get_metric_fn):
        """get_metric_fn(pid_list) -> (n_g,) array. Returns point, lo, hi (n_g,) each."""
        point = get_metric_fn(pids)
        boot = np.zeros((N_BOOT, n_g))
        for b in range(N_BOOT):
            samp = rng.choice(pids, size=len(pids), replace=True)
            boot[b] = get_metric_fn(samp)
        lo = np.percentile(boot, CI_LO_PCTILE, axis=0)
        hi = np.percentile(boot, CI_HI_PCTILE, axis=0)
        return point, lo, hi

    def mean_abs_delta(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid_lomo[p]
            num += r["abs_delta"].sum(axis=1)      # sum over horizons
            den += r["n"]
        return num / (den * n_h) if den > 0 else num

    def mean_signed_delta(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid_lomo[p]
            num += r["signed_delta"].sum(axis=1)
            den += r["n"]
        return num / (den * n_h) if den > 0 else num

    def mean_delta_mae(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid_lomo[p]
            num += r["delta_mae"].sum(axis=1)
            den += r["n"]
        return num / (den * n_h) if den > 0 else num

    def mean_delta_pinball(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid_lomo[p]
            num += r["delta_pinball"].sum(axis=1)
            den += r["n"]
        return num / (den * n_h) if den > 0 else num

    def mean_delta_coverage(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid_lomo[p]
            num += r["delta_coverage"].sum(axis=1)
            den += r["n"]
        return num / (den * n_h) if den > 0 else num

    abs_pt, abs_lo, abs_hi = boot_ci_1d(mean_abs_delta)
    signed_pt, signed_lo, signed_hi = boot_ci_1d(mean_signed_delta)
    mae_pt, mae_lo, mae_hi = boot_ci_1d(mean_delta_mae)
    pinball_pt, pinball_lo, pinball_hi = boot_ci_1d(mean_delta_pinball)
    cov_pt, cov_lo, cov_hi = boot_ci_1d(mean_delta_coverage)

    total_abs = abs_pt.sum()
    n_participants = len(pids)
    n_anchors_lomo = sum(per_pid_lomo[p]["n"] for p in pids)
    for gi, g in enumerate(GROUP_ORDER):
        lomo_rows.append({
            "group": g,
            "raw_mean_abs_delta_mgdl": abs_pt[gi], "raw_mean_abs_delta_ci_low": abs_lo[gi],
            "raw_mean_abs_delta_ci_high": abs_hi[gi],
            "normalized_share": abs_pt[gi] / total_abs if total_abs > 0 else 0.0,
            "signed_mean_delta_mgdl": signed_pt[gi], "signed_mean_delta_ci_low": signed_lo[gi],
            "signed_mean_delta_ci_high": signed_hi[gi],
            "delta_mae_mgdl": mae_pt[gi], "delta_mae_ci_low": mae_lo[gi], "delta_mae_ci_high": mae_hi[gi],
            "delta_pinball": pinball_pt[gi], "delta_pinball_ci_low": pinball_lo[gi],
            "delta_pinball_ci_high": pinball_hi[gi],
            "delta_coverage80": cov_pt[gi], "delta_coverage80_ci_low": cov_lo[gi],
            "delta_coverage80_ci_high": cov_hi[gi],
            "n_participants": n_participants, "n_streams": n_streams_used, "n_anchors": n_anchors_lomo,
        })
        for h in range(n_h):
            raw_h = np.mean([per_pid_lomo[p]["abs_delta"][gi, h] / per_pid_lomo[p]["n"]
                             for p in pids if per_pid_lomo[p]["n"] > 0])
            signed_h = np.mean([per_pid_lomo[p]["signed_delta"][gi, h] / per_pid_lomo[p]["n"]
                                for p in pids if per_pid_lomo[p]["n"] > 0])
            horizon_rows.append({"group": g, "horizon_step": h + 1, "horizon_minutes": (h + 1) * BIN_MIN,
                                "raw_abs_delta_mgdl": raw_h, "signed_delta_mgdl": signed_h})
        predictive_rows.append({
            "group": g, "delta_mae_mgdl": mae_pt[gi], "delta_mae_ci_low": mae_lo[gi],
            "delta_mae_ci_high": mae_hi[gi], "delta_pinball": pinball_pt[gi],
            "delta_pinball_ci_low": pinball_lo[gi], "delta_pinball_ci_high": pinball_hi[gi],
            "delta_coverage80": cov_pt[gi], "delta_coverage80_ci_low": cov_lo[gi],
            "delta_coverage80_ci_high": cov_hi[gi],
        })

    lomo_df = pd.DataFrame(lomo_rows).sort_values("normalized_share", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    lomo_df.to_csv(out_dir / "group_overall_lomo_v2.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(out_dir / "group_horizon_lomo_v2.csv", index=False)
    pd.DataFrame(predictive_rows).to_csv(out_dir / "group_predictive_ablation_v2.csv", index=False)
    print(f"[global-v2] wrote group_overall_lomo_v2.csv, group_horizon_lomo_v2.csv, "
          f"group_predictive_ablation_v2.csv")

    # -- rigorous h_t --
    rig_pids = sorted(per_pid_rig)
    n_ht = len(ht_group_names)

    def rigorous_raw(pid_list):
        num = np.zeros(n_ht)
        den = 0
        for p in pid_list:
            r = per_pid_rig[p]
            num += r["raw"]
            den += r["n"]
        return num / den if den > 0 else num

    # separate bootstrap loop: the rigorous pass has its own participant set
    rig_point = rigorous_raw(rig_pids)
    rig_boot = np.zeros((N_BOOT, n_ht))
    for b in range(N_BOOT):
        samp = rng.choice(rig_pids, size=len(rig_pids), replace=True)
        rig_boot[b] = rigorous_raw(samp)
    rig_lo = np.percentile(rig_boot, CI_LO_PCTILE, axis=0)
    rig_hi = np.percentile(rig_boot, CI_HI_PCTILE, axis=0)
    rig_total = rig_point.sum()

    n_anchors_rig = sum(per_pid_rig[p]["n"] for p in rig_pids)
    rig_rows = []
    for gi, g in enumerate(ht_group_names):
        rig_rows.append({
            "group": g, "raw_joint_norm": rig_point[gi], "raw_joint_norm_ci_low": rig_lo[gi],
            "raw_joint_norm_ci_high": rig_hi[gi],
            "normalized_share": rig_point[gi] / rig_total if rig_total > 0 else 0.0,
            "n_participants": len(rig_pids), "n_streams": n_streams_used, "n_anchors": n_anchors_rig,
        })
    rig_df = pd.DataFrame(rig_rows).sort_values("normalized_share", ascending=False)
    rig_df.to_csv(out_dir / "group_overall_rigorous_v2.csv", index=False)
    print(f"[global-v2] wrote group_overall_rigorous_v2.csv")

    build_compare_v2(rig_df, ht_group_names, groups_dyn, per_pid_posthoc, dyn_feature_names, rig_pids, rng, out_dir)
    return lomo_df, rig_df


def build_compare_v2(rig_df: pd.DataFrame, ht_group_names, groups_dyn, per_pid_posthoc: dict,
                     dyn_feature_names: list, rig_pids: list, rng: np.random.Generator, out_dir: Path):
    """group_overall_compare_v2.csv: rigorous vs post-hoc, RAW ratio kept separate from
    normalized-share ratio, per the v2 instruction not to conflate the two.

    The post-hoc sum is computed HERE, within this same pass, per individual dynamic
    feature, using the identical attn*u_norm-summed-per-anchor convention as the rigorous
    joint norm above, so the two raw quantities are in comparable units (v1's
    attribution_feature_overall.csv used a different internal accumulation scale and is
    NOT reused for this ratio to avoid a spurious unit mismatch).
    """
    def posthoc_raw_per_feature(pid_list):
        num = np.zeros(len(dyn_feature_names))
        den = 0
        for p in pid_list:
            r = per_pid_posthoc[p]
            num += r["raw"]
            den += r["n"]
        return num / den if den > 0 else num

    per_feature_point = posthoc_raw_per_feature(rig_pids)
    per_feature_series = pd.Series(per_feature_point, index=dyn_feature_names)

    rows = []
    rig_by_group = rig_df.set_index("group")
    for g in ht_group_names:
        members = groups_dyn[g]
        posthoc_raw = float(per_feature_series.reindex(members).fillna(0.0).sum())
        rigorous_raw = float(rig_by_group.loc[g, "raw_joint_norm"])
        rows.append({"group": g, "posthoc_raw_sum_of_norms": posthoc_raw,
                    "rigorous_raw_joint_norm": rigorous_raw,
                    "raw_norm_ratio_posthoc_over_rigorous": (posthoc_raw / rigorous_raw
                                                              if rigorous_raw > 0 else float("nan"))})
    cmp_df = pd.DataFrame(rows)
    posthoc_total = cmp_df["posthoc_raw_sum_of_norms"].sum()
    rigorous_total = cmp_df["rigorous_raw_joint_norm"].sum()
    cmp_df["posthoc_normalized_share"] = cmp_df["posthoc_raw_sum_of_norms"] / posthoc_total if posthoc_total > 0 else 0.0
    cmp_df["rigorous_normalized_share"] = cmp_df["rigorous_raw_joint_norm"] / rigorous_total if rigorous_total > 0 else 0.0
    cmp_df["normalized_share_ratio_posthoc_over_rigorous"] = (
        cmp_df["posthoc_normalized_share"] / cmp_df["rigorous_normalized_share"].clip(lower=1e-12)
    )
    cmp_df = cmp_df.sort_values("rigorous_normalized_share", ascending=False)
    cmp_df.to_csv(out_dir / "group_overall_compare_v2.csv", index=False)
    print(f"[global-v2] wrote group_overall_compare_v2.csv")
    print("[global-v2] NOTE: raw_norm_ratio_posthoc_over_rigorous is the correct field for "
          "discussing overcounting. normalized_share_ratio_posthoc_over_rigorous also reflects "
          "denominator effects from other groups and must not be described as a direct "
          "overcounting factor on its own.")


def main():
    ap = argparse.ArgumentParser(description="AI-READI v2 Phase 1: corrected global attribution")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model, spec, pre = load_model(CHECKPOINT_PATH, device)
    groups = build_group_members({
        "dynamic_reals": spec.dynamic_reals, "time_reals": spec.time_reals,
        "scenario_reals": spec.scenario_reals, "static_reals": spec.static_reals,
        "static_categoricals": spec.static_categoricals,
    })
    groups_dyn = h_t_group_members(groups)
    streams = load_test_streams(cfg, spec, pre)
    count_only(streams)
    if not args.execute:
        return
    run_global_v2(model, streams, device, groups, groups_dyn, out_dir)


if __name__ == "__main__":
    main()
