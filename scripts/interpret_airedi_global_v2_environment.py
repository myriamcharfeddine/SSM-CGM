#!/usr/bin/env python3
"""AI-READI interpretability v2, environment variant: global attribution incl.
the 12th "environmental exposure" group, run on the environment-trained checkpoint.

Additive copy of scripts/interpret_airedi_global_v2.py (never edited), with
exactly two substitutions: CHECKPOINT_PATH/CONFIG_PATH point at the
environment-trained model, and GROUP_ORDER/build_group_members come from
scripts/interpret_airedi_groups_environment.py (12 groups: the original 11 plus
"environmental exposure") instead of the 11-group scripts/interpret_airedi_groups.py.
Everything else -- the ablation mechanism, bootstrap CIs, output schema -- is
byte-for-byte the same method as the original figure, just evaluated on the
environment-trained checkpoint so all 12 groups are measured under one
consistent model/run instead of splicing two different checkpoints' numbers
together.

Usage:
  python scripts/interpret_airedi_global_v2_environment.py                # count only
  python scripts/interpret_airedi_global_v2_environment.py --execute      # run + write CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.interpret_airedi_export import (
    anchors_for_stream,
    load_model,
    load_test_streams,
)
from scripts.interpret_airedi_groups_environment import CHECKPOINT_PATH, CONFIG_PATH, GROUP_ORDER, build_group_members
from scripts.interpret_airedi_grouped_export import h_t_group_members
import scripts.interpret_airedi_global_v2 as global_v2

# run_global_v2/count_only in interpret_airedi_global_v2.py read the module-level
# GROUP_ORDER as a free variable at call time (not a parameter), so it must be
# monkeypatched to the 12-group environment list before either is invoked --
# otherwise they'd silently run against the original 11-group order.
global_v2.GROUP_ORDER = GROUP_ORDER
count_only = global_v2.count_only
run_global_v2 = global_v2.run_global_v2

OUT_DIR = ROOT / "outputs/interpretability_v2_environment"


def main():
    ap = argparse.ArgumentParser(description="AI-READI v2 environment variant: 12-group global attribution")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print(f"[global-v2-env] checkpoint={CHECKPOINT_PATH}")
    print(f"[global-v2-env] config={CONFIG_PATH}")

    model, spec, pre = load_model(CHECKPOINT_PATH, device)
    groups = build_group_members({
        "dynamic_reals": spec.dynamic_reals, "time_reals": spec.time_reals,
        "scenario_reals": spec.scenario_reals, "static_reals": spec.static_reals,
        "static_categoricals": spec.static_categoricals,
    })
    groups_dyn = h_t_group_members(groups)
    print(f"[global-v2-env] GROUP_ORDER ({len(GROUP_ORDER)} groups): {GROUP_ORDER}")
    print(f"[global-v2-env] h_t-reachable groups ({len(groups_dyn)}): {list(groups_dyn)}")

    streams = load_test_streams(cfg, spec, pre)
    count_only(streams)
    if not args.execute:
        return
    run_global_v2(model, streams, device, groups, groups_dyn, out_dir)


if __name__ == "__main__":
    main()
