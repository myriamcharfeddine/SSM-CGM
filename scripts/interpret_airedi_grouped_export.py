#!/usr/bin/env python3
"""AI-READI RIGOROUS group attribution export + full-test-split LOMO (Step 2, grouped pass).

Adapts T1DEXI's ``interpret_grouped_export.py``. Per scripts/interpret_airedi_groups.py,
only 6 of the 11 requested modality groups have a "dynamic" pathway member (i.e. reach
the SSM hidden state h_t via GroupedLinearFusion's per-feature contributions u_f):
glucose history, heart rate / physiology, wearable activity / exercise, stress, sleep,
data quality / missingness. Those 6 get the RIGOROUS joint-group norm
``||sum_{f in g} u_f||`` -- summing the raw per-feature contribution TENSORS before
taking the norm, so collinear members are not overcounted (``||sum u|| <= sum ||u||``).

The other 5 groups (time / calendar, static clinical, medication metadata, site /
cohort, meal proxy) never touch h_t in this architecture -- there is no u_f for them
to sum. For those, and as a same-method, bigger-sample successor to Table 36 for ALL
11 groups, this script also reruns the existing leave-one-modality-out (LOMO) output
ablation (scripts/export_study2_interpretability.py::compute_ablation_attribution)
over the FULL test split's dense anchor grid instead of the original 58 triggered
anchors.

Writes, into ``--out-dir`` (default <checkpoint run>/interpret):
  * group_overall_rigorous.csv   -- joint-norm share, the 6 h_t-reachable groups only
  * group_temporal_rigorous.csv  -- joint-norm share x lag bin, same 6 groups
  * group_overall_compare.csv    -- rigorous vs post-hoc sum-of-norms, same 6 groups
  * group_overall_lomo_full_test.csv -- LOMO ablation-on-output, all 11 groups, full split

Usage:
  python scripts/interpret_airedi_grouped_export.py                # count only
  python scripts/interpret_airedi_grouped_export.py --execute      # run + write CSVs
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
    ANCHOR_STRIDE_MIN,
    BIN_MIN,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    DEFAULT_OUT_DIR,
    FINE_LAG_EDGES,
    HORIZON,
    MAX_LAG_STEPS,
    anchors_for_stream,
    load_model,
    load_test_streams,
    scan_with_record,
)
from scripts.interpret_airedi_groups import GROUP_ORDER, H_T_PATHWAYS, build_group_members
from ssmcgm.ops.mes_attribution import compute_hidden_attention

H_T_GROUP_ORDER = [g for g in GROUP_ORDER]   # filled in main() to only the 6 with dynamic members


def h_t_group_members(groups) -> dict:
    """{group: [dynamic feature name, ...]} for groups with >=1 dynamic-pathway member."""
    out = {}
    for g, members in groups.items():
        dyn = [n for p, n in members if p in H_T_PATHWAYS]
        if dyn:
            out[g] = dyn
    return out


def stream_key(stream):
    return str(stream.participant_id), int(stream.segment_id)


# ---------------------------------------------------------------------------
# Rigorous joint-group-norm pass (h_t-reachable groups only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_rigorous(model, streams, device, groups_dyn: dict, out_dir: Path):
    group_names = list(groups_dyn)
    group_acc = {g: np.zeros(MAX_LAG_STEPS + 1, dtype="float64") for g in group_names}
    feature_group = {f: g for g, fs in groups_dyn.items() for f in fs}
    n_windows = 0
    n_streams_used = 0

    for stream in streams:
        anchors = anchors_for_stream(stream.n_steps)
        if len(anchors) == 0:
            continue
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

        for g, members in groups_dyn.items():
            # RIGOROUS: sum the raw per-feature contribution TENSORS first, norm second.
            u_sum = sum(contribs[f] for f in members)            # (1, L, hidden)
            u_norm = u_sum.detach().cpu().norm(dim=-1).numpy()[0]  # (L,)
            vals = np.where(valid, u_norm[src_clip], 0.0)
            group_acc[g] += (attn_np * vals).sum(axis=0)

        n_windows += len(anchors)
        n_streams_used += 1

    total = sum(float(v.sum()) for v in group_acc.values())
    rig = pd.Series({g: (v.sum() / total if total > 0 else 0.0) for g, v in group_acc.items()},
                     name="frac_influence").sort_values(ascending=False)
    rig_path = out_dir / "group_overall_rigorous.csv"
    rig.reset_index().rename(columns={"index": "group"}).to_csv(rig_path, index=False)

    lags_min = np.arange(MAX_LAG_STEPS + 1) * BIN_MIN
    bin_labels = pd.cut(lags_min, FINE_LAG_EDGES, right=False, include_lowest=True)
    temporal_rows = []
    for g, arr in group_acc.items():
        s = pd.Series(arr, index=bin_labels)
        binned = s.groupby(level=0, observed=True).sum()
        denom = arr.sum() if arr.sum() > 0 else 1.0
        for lag_bin, val in binned.items():
            temporal_rows.append({"group": g, "lag_bin": str(lag_bin),
                                  "raw_influence": float(val), "frac_influence": float(val) / denom})
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(out_dir / "group_temporal_rigorous.csv", index=False)

    print(f"\n[grouped-export] RIGOROUS joint-group-norm shares, {n_windows} windows / "
          f"{n_streams_used} streams (h_t-reachable groups only):")
    for g, v in rig.items():
        print(f"  {g:32s} {100*v:6.1f}%")
    print(f"[grouped-export] wrote {rig_path}")
    return rig, n_windows, n_streams_used


def compare_rigorous_vs_posthoc(rig: pd.Series, groups_dyn: dict, feature_overall_path: Path, out_dir: Path):
    if not feature_overall_path.exists():
        print(f"[grouped-export] {feature_overall_path} not found -- run interpret_airedi_export.py "
              "--execute first for the post-hoc comparison; skipping group_overall_compare.csv")
        return None
    ov = pd.read_csv(feature_overall_path).set_index("feature")["frac_influence"]
    post = pd.Series({g: float(ov.reindex(members).fillna(0.0).sum()) for g, members in groups_dyn.items()})
    post = post / post.sum() if post.sum() > 0 else post
    cmp = pd.DataFrame({"rigorous_pct": 100 * rig, "posthoc_pct": 100 * post})
    cmp["delta_pp"] = cmp["rigorous_pct"] - cmp["posthoc_pct"]
    cmp["overcount_x"] = cmp["posthoc_pct"] / cmp["rigorous_pct"].clip(lower=1e-9)
    cmp = cmp.sort_values("rigorous_pct", ascending=False)
    cmp.to_csv(out_dir / "group_overall_compare.csv")
    print(f"\n[grouped-export] rigorous vs post-hoc sum-of-norms (collinearity overcount check):")
    print(f"{'group':32s} {'rigorous':>9s} {'post-hoc':>9s} {'delta_pp':>9s} {'overcount':>10s}")
    for g, r in cmp.iterrows():
        print(f"{g:32s} {r['rigorous_pct']:8.1f}% {r['posthoc_pct']:8.1f}% "
              f"{r['delta_pp']:+8.1f} {r['overcount_x']:9.2f}x")
    return cmp


# ---------------------------------------------------------------------------
# Full-test-split LOMO ablation (all 11 groups; same method as Table 36, bigger sample)
# ---------------------------------------------------------------------------
@torch.no_grad()
def clone_stream_tensors(stream, device: str, group=None, groups=None):
    dynamic = stream.dynamic.to(device).clone()
    time_features = stream.time_features.to(device).clone()
    scenario_values = stream.scenario_values.to(device).clone()
    scenario_mask = torch.zeros_like(stream.scenario_mask.to(device))
    static_cont = stream.static_cont.to(device).clone()
    static_cat = stream.static_cat.to(device).clone()
    target = stream.target.to(device)

    if group is not None and groups is not None:
        spec = stream.feature_spec if hasattr(stream, "feature_spec") else None
        for pathway, name in groups[group]:
            if pathway == "dynamic" and name in DYN_INDEX:
                dynamic[:, DYN_INDEX[name]] = 0.0
            elif pathway == "time" and name in TIME_INDEX:
                time_features[:, TIME_INDEX[name]] = 0.0
            elif pathway == "scenario" and name in SCN_INDEX:
                j = SCN_INDEX[name]
                scenario_values[:, j] = 0.0
                scenario_mask[:, j] = 0.0
            elif pathway == "static_cont" and name in SCONT_INDEX:
                static_cont[SCONT_INDEX[name]] = 0.0
            elif pathway == "static_cat" and name in SCAT_INDEX:
                static_cat[SCAT_INDEX[name]] = 0
    return dynamic, time_features, scenario_values, scenario_mask, static_cont, static_cat, target


@torch.no_grad()
def predict_stream_q(model, stream, anchors: np.ndarray, device: str, group=None, groups=None) -> np.ndarray:
    dynamic, time_features, scenario_values, scenario_mask, static_cont, static_cat, target = \
        clone_stream_tensors(stream, device, group=group, groups=groups)
    sctx = model.encode_static(static_cat, static_cont)
    state = model.init_stream(sctx)
    state, out = model.scan_chunk(dynamic.unsqueeze(0), sctx, state)
    pos = torch.tensor(anchors, dtype=torch.long, device=device)
    fut = pos[:, None] + 1 + torch.arange(HORIZON, device=device)[None, :]
    raw = model.decode_horizon(out[0, pos], sctx, time_features[fut], scenario_values[fut], scenario_mask[fut])
    pred = target[pos].view(-1, 1, 1) + raw
    qidx = int(np.argmin(np.abs(np.asarray(model.quantiles) - 0.5)))
    return pred.detach().cpu().numpy()[:, :, qidx]


@torch.no_grad()
def run_lomo_full_test(model, streams, device, groups: dict, out_dir: Path):
    global DYN_INDEX, TIME_INDEX, SCN_INDEX, SCONT_INDEX, SCAT_INDEX
    spec = model.feature_spec
    DYN_INDEX = {n: i for i, n in enumerate(spec.dynamic_reals)}
    TIME_INDEX = {n: i for i, n in enumerate(spec.time_reals)}
    SCN_INDEX = {n: i for i, n in enumerate(spec.scenario_reals)}
    SCONT_INDEX = {n: i for i, n in enumerate(spec.static_reals)}
    SCAT_INDEX = {n: i for i, n in enumerate(spec.static_categoricals)}

    abs_delta_sum = {g: 0.0 for g in GROUP_ORDER}     # sum over all anchors of |delta|, per group
    n_anchor_rows = {g: 0 for g in GROUP_ORDER}
    n_windows = 0
    n_streams_used = 0
    for stream in streams:
        anchors = anchors_for_stream(stream.n_steps)
        if len(anchors) == 0:
            continue
        base = predict_stream_q(model, stream, anchors, device)
        for group in GROUP_ORDER:
            if not groups[group]:
                delta = np.zeros_like(base)
            else:
                ablated = predict_stream_q(model, stream, anchors, device, group=group, groups=groups)
                delta = base - ablated
            abs_delta_sum[group] += float(np.abs(delta).sum())
            n_anchor_rows[group] += delta.size
        n_windows += len(anchors)
        n_streams_used += 1

    glob = pd.DataFrame({
        "group": GROUP_ORDER,
        "mean_abs_delta_mgdl": [abs_delta_sum[g] / n_anchor_rows[g] if n_anchor_rows[g] else 0.0
                                 for g in GROUP_ORDER],
        "n_anchor_group_rows": [n_anchor_rows[g] for g in GROUP_ORDER],
    })
    total = glob["mean_abs_delta_mgdl"].sum()
    glob["frac_influence"] = glob["mean_abs_delta_mgdl"] / total if total > 0 else 0.0
    glob["group"] = pd.Categorical(glob["group"], GROUP_ORDER, ordered=True)
    glob = glob.sort_values("frac_influence", ascending=False)
    glob.to_csv(out_dir / "group_overall_lomo_full_test.csv", index=False)

    print(f"\n[grouped-export] full-test-split LOMO ablation (all 11 groups, {n_windows} windows / "
          f"{n_streams_used} streams; same method as Table 36, bigger sample):")
    for _, r in glob.iterrows():
        print(f"  {str(r['group']):32s} {100*r['frac_influence']:6.1f}%")
    return glob, n_windows


def main():
    ap = argparse.ArgumentParser(description="AI-READI rigorous group attribution + full-test LOMO")
    ap.add_argument("--execute", action="store_true", help="run the GPU passes (default: count only)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

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
    print(f"[grouped-export] h_t-reachable groups ({len(groups_dyn)}/11): {list(groups_dyn)}")

    streams = load_test_streams(cfg, spec, pre)
    n_anchors = sum(len(anchors_for_stream(s.n_steps)) for s in streams)
    print(f"[grouped-export] test streams: {len(streams)}  total anchors: {n_anchors}")
    print(f"[grouped-export] GPU passes required: {len(streams)} scans for the rigorous pass + "
          f"{len(streams)} x (1 baseline + 11 ablated) = {len(streams) * 12} scans for the full-test LOMO pass")
    if not args.execute:
        print("[grouped-export] PAUSE: count-only dry run. Re-run with --execute to run the GPU passes.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rig, _, _ = run_rigorous(model, streams, device, groups_dyn, out_dir)
    compare_rigorous_vs_posthoc(rig, groups_dyn, out_dir / "attribution_feature_overall.csv", out_dir)
    run_lomo_full_test(model, streams, device, groups, out_dir)


if __name__ == "__main__":
    main()
