#!/usr/bin/env python3
"""AI-READI RIGOROUS 7-group h_t attribution for the environment-augmented model.

Additive copy of the *rigorous joint-group-norm* method used in
scripts/interpret_airedi_grouped_export.py::run_rigorous and
scripts/interpret_airedi_global_v2.py::run_global_v2 (rigorous section only),
pointed at the newly trained environment checkpoint and extended to 7 groups
(the original 6 h_t-reachable groups + "environmental exposure").

Per the task, this does NOT rerun LOMO / probe / residual-head / counterfactual
/ feature-permutation experiments -- only the rigorous joint-group norm
``J_g = ||sum_{f in g} u_f||`` with the same participant-clustered bootstrap
(seed 42, 1000 reps, 2.5/97.5 percentiles) used everywhere else in this
pipeline, normalized over the 7-group denominator G7:

    S_g^h = 100 * J_g / sum_{k in G7} J_k

Methodology is otherwise byte-for-byte identical to the existing pipeline --
same anchor grid (WARMUP_STEPS=12, ANCHOR_STRIDE_MIN=15), same attention-banded
accumulation, same test-split loader convention (only the schema call differs,
to include env_* dynamic_reals).

Usage:
  python scripts/interpret_airedi_grouped_export_environment.py                # count only
  python scripts/interpret_airedi_grouped_export_environment.py --execute      # run + write CSVs
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
import yaml

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec,
    AireadiPreprocessor,
    load_aireadi_panel,
    make_aireadi_stream_splits,
    make_participant_streams,
    prepare_aireadi_panel,
)
from ssmcgm.data.aireadi_environment import infer_or_validate_schema_with_environment
from ssmcgm.ops.mes_attribution import compute_hidden_attention

from scripts.interpret_airedi_export import (
    BIN_MIN,
    HORIZON,
    MAX_LAG_STEPS,
    WARMUP_STEPS,
    anchors_for_stream,
    load_model,
    scan_with_record,
)
from scripts.interpret_airedi_groups_environment import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    G7_ORDER,
    build_group_members,
)

SEED = 42
N_BOOT = 1000
CI_LO_PCTILE = 2.5
CI_HI_PCTILE = 97.5
DEFAULT_OUT_DIR = ROOT / "outputs/environment_ht"


def load_test_streams_environment(cfg: dict, spec: AireadiFeatureSpec, pre: AireadiPreprocessor):
    """Same as interpret_airedi_export.load_test_streams, except the schema call
    includes the environmental dynamic_reals (the only difference)."""
    data_cfg = cfg["data"]
    df = load_aireadi_panel(
        data_cfg["panel_path"],
        static_path=data_cfg.get("static_path"),
        cohort_path=data_cfg.get("cohort_path"),
    )
    schema = infer_or_validate_schema_with_environment(df, data_cfg.get("schema"))
    df = prepare_aireadi_panel(
        df, schema,
        bin_minutes=cfg["dataset"].get("bin_minutes", BIN_MIN),
        clean_min_segment_hours=cfg["dataset"].get("clean_min_segment_hours", 49),
    )
    split_cfg = cfg["split"]
    split = make_aireadi_stream_splits(
        df,
        split_mode=split_cfg.get("mode", "participant_heldout"),
        train=split_cfg.get("train", 0.70), val=split_cfg.get("val", 0.15),
        test=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_col=split_cfg.get("stratify_col"),
        existing_split_path=split_cfg.get("existing_split_path"),
    )
    streams = make_participant_streams(
        df, split, schema, feature_spec=spec, preprocessor=pre,
        splits=["test"], min_steps=spec.horizon_steps + WARMUP_STEPS + 2,
    )
    return streams


def h_t_group_members(groups) -> dict:
    out = {}
    for g, members in groups.items():
        dyn = [n for p, n in members if p == "dynamic"]
        if dyn:
            out[g] = dyn
    return out


@torch.no_grad()
def run_rigorous_7group(model, streams, device, groups_dyn: dict, out_dir: Path):
    group_names = [g for g in G7_ORDER if g in groups_dyn]
    assert group_names == G7_ORDER, (
        f"expected all 7 G7 groups to have dynamic members, got {group_names} vs {G7_ORDER}"
    )

    per_pid = {}  # pid -> {"raw": (7,) array, "n": int}
    n_streams_used = 0
    total_anchors = 0

    for si, stream in enumerate(streams):
        anchors = anchors_for_stream(stream.n_steps)
        if len(anchors) == 0:
            continue
        pid = str(stream.participant_id)
        caches, contribs = scan_with_record(model, stream, device)
        layer = len(caches) - 1
        rows_t = torch.tensor(anchors, dtype=torch.long, device=device)
        attn, _ = compute_hidden_attention(
            caches[layer], rows=rows_t, max_lags=MAX_LAG_STEPS,
            ngroups=model.temporal.blocks[0].ssm.ngroups, normalize="softmax", out_device="cpu",
        )
        attn_np = attn.float().numpy()[0].sum(axis=0)          # (R, K+1)

        lag_idx = np.arange(MAX_LAG_STEPS + 1, dtype=int)
        src = anchors[:, None] - lag_idx[None, :]
        valid = src >= 0
        src_clip = np.clip(src, 0, stream.n_steps - 1)

        rec = per_pid.setdefault(pid, {"raw": np.zeros(len(group_names)), "n": 0})
        for gi, g in enumerate(group_names):
            members = groups_dyn[g]
            u_sum = sum(contribs[f] for f in members)             # (1, L, hidden)
            u_norm = u_sum.detach().cpu().norm(dim=-1).numpy()[0]  # (L,)
            vals = np.where(valid, u_norm[src_clip], 0.0)
            rec["raw"][gi] += (attn_np * vals).sum()
        rec["n"] += len(anchors)

        n_streams_used += 1
        total_anchors += len(anchors)
        if (si + 1) % 50 == 0:
            print(f"[grouped-export-env] processed {si + 1}/{len(streams)} streams")

    pids = sorted(per_pid)
    n_g = len(group_names)

    def raw_mean(pid_list):
        num = np.zeros(n_g)
        den = 0
        for p in pid_list:
            r = per_pid[p]
            num += r["raw"]
            den += r["n"]
        return num / den if den > 0 else num

    rng = np.random.default_rng(SEED)
    point = raw_mean(pids)
    boot = np.zeros((N_BOOT, n_g))
    for b in range(N_BOOT):
        samp = rng.choice(pids, size=len(pids), replace=True)
        boot[b] = raw_mean(samp)
    lo = np.percentile(boot, CI_LO_PCTILE, axis=0)
    hi = np.percentile(boot, CI_HI_PCTILE, axis=0)
    total = point.sum()

    n_anchors = sum(per_pid[p]["n"] for p in pids)
    rows = []
    for gi, g in enumerate(group_names):
        rows.append({
            "group": g,
            "raw_joint_norm": point[gi],
            "raw_joint_norm_ci_low": lo[gi],
            "raw_joint_norm_ci_high": hi[gi],
            "normalized_share_over_seven_groups": 100.0 * point[gi] / total if total > 0 else 0.0,
            "n_participants": len(pids),
            "n_streams": n_streams_used,
            "n_anchors": n_anchors,
        })
    df = pd.DataFrame(rows).sort_values("normalized_share_over_seven_groups", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "environment_ht_group_attribution.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[grouped-export-env] RIGOROUS 7-group h_t shares, {n_anchors} anchors / "
          f"{n_streams_used} streams / {len(pids)} participants:")
    for _, r in df.iterrows():
        print(f"  {r['group']:32s} {r['normalized_share_over_seven_groups']:6.2f}%  "
              f"[{100*r['raw_joint_norm_ci_low']/total if total>0 else 0:5.2f}, "
              f"{100*r['raw_joint_norm_ci_high']/total if total>0 else 0:5.2f}]")
    print(f"[grouped-export-env] wrote {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description="AI-READI environment-model rigorous 7-group h_t attribution")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model, spec, pre = load_model(CHECKPOINT_PATH, device)
    groups = build_group_members({
        "dynamic_reals": spec.dynamic_reals, "time_reals": spec.time_reals,
        "scenario_reals": spec.scenario_reals, "static_reals": spec.static_reals,
        "static_categoricals": spec.static_categoricals,
    })
    groups_dyn = h_t_group_members(groups)
    print(f"[grouped-export-env] h_t-reachable groups: {list(groups_dyn)}")

    streams = load_test_streams_environment(cfg, spec, pre)
    n_anchors = sum(len(anchors_for_stream(s.n_steps)) for s in streams)
    print(f"[grouped-export-env] test streams: {len(streams)}  total anchors: {n_anchors}")
    if not args.execute:
        print("[grouped-export-env] PAUSE: count-only dry run. Re-run with --execute to run the GPU pass.")
        return

    run_rigorous_7group(model, streams, device, groups_dyn, out_dir)


if __name__ == "__main__":
    main()
