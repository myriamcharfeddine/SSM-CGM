"""AI-READI modality group definitions for the environment-augmented model.

Additive copy of scripts/interpret_airedi_groups.py (never edited) pointed at the
newly trained environment checkpoint, with exactly one methodological addition:
a 7th group, "environmental exposure", built from the checkpoint's own
feature_spec (env_* dynamic_reals), auto-discovered exactly the way every other
group already is -- not a hardcoded per-feature list. Everything else (group
definitions, H_T_PATHWAYS, validation logic) is unchanged.

Run:
  python scripts/interpret_airedi_groups_environment.py
"""
import json
import os
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

# Training (Phase C) completed; see
# outputs/environment_model_trained/environment_training_manifest.csv.
# NOTE: the run directory was renamed from outputs/environment_ht/... to
# outputs/environment_model_trained/... after this constant was first written.
CHECKPOINT_PATH = os.path.join(
    _ROOT, "outputs/environment_model_trained/aireadi_stream_mamba_stateful_environment/checkpoints/best_model_checkpoint.pt")
CONFIG_PATH = os.path.join(
    _ROOT, "outputs/environment_model_trained/aireadi_stream_mamba_stateful_environment/config_resolved.yaml")

GROUP_ORDER = [
    "glucose history",
    "time / calendar",
    "wearable activity / exercise",
    "heart rate / physiology",
    "stress",
    "sleep",
    "environmental exposure",
    "static clinical",
    "medication metadata",
    "site / cohort",
    "data quality / missingness",
    "meal proxy (diagnostic only)",
]

H_T_PATHWAYS = {"dynamic"}

# The 6 original h_t-reachable groups, in the task's specified order, plus the
# new 7th group appended at the end -- this is G7 for the rigorous 7-group share.
G7_ORDER = [
    "glucose history",
    "heart rate / physiology",
    "data quality / missingness",
    "sleep",
    "wearable activity / exercise",
    "stress",
    "environmental exposure",
]


def _quality_dynamic(dynamic_reals):
    return [
        n for n in dynamic_reals
        if n.endswith("_count") or n.endswith("_device_availability") or n == "sleep_stage_unknown"
    ]


def build_group_members(feature_spec: dict) -> "OrderedDict[str, list[tuple[str, str]]]":
    dyn = list(feature_spec["dynamic_reals"])
    tim = list(feature_spec["time_reals"])
    scn = list(feature_spec["scenario_reals"])
    scont = list(feature_spec["static_reals"])
    scat = list(feature_spec["static_categoricals"])

    quality = set(_quality_dynamic(dyn))

    def dyn_scn(prefixes, extra_exact=()):
        d = [n for n in dyn if (n.startswith(prefixes) or n in extra_exact) and n not in quality]
        s = [n for n in scn if (n.startswith(prefixes) or n in extra_exact)]
        return [("dynamic", n) for n in d] + [("scenario", n) for n in s]

    med = [n for n in scont if n.startswith("med_")]
    clinical_cont = [n for n in scont if n not in med]
    site_cat = [n for n in scat if n in ("participants_clinical_site", "participants_study_group")]
    clinical_cat = [n for n in scat if n not in site_cat]

    groups: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict((g, []) for g in GROUP_ORDER)
    groups["glucose history"] = [("dynamic", n) for n in dyn if n == "cgm_glucose_mean"]
    groups["time / calendar"] = [("time", n) for n in tim]
    groups["wearable activity / exercise"] = dyn_scn(
        ("activity_",), extra_exact=("calories_total", "calories_per_min"))
    groups["heart rate / physiology"] = dyn_scn(("heart_rate_", "respiratory_rate_", "oxygen_saturation_"))
    groups["stress"] = dyn_scn(("stress_level_",))
    groups["sleep"] = dyn_scn(("sleep_",))
    groups["environmental exposure"] = dyn_scn(("env_",))
    groups["static clinical"] = [("static_cont", n) for n in clinical_cont] + \
        [("static_cat", n) for n in clinical_cat]
    groups["medication metadata"] = [("static_cont", n) for n in med]
    groups["site / cohort"] = [("static_cat", n) for n in site_cat]
    groups["data quality / missingness"] = [("dynamic", n) for n in dyn if n in quality]
    groups["meal proxy (diagnostic only)"] = [("scenario", n) for n in scn if n == "predmeal_flag"]
    return groups


def _all_model_keys(feature_spec: dict) -> set:
    keys = set()
    for n in feature_spec["dynamic_reals"]:
        keys.add(("dynamic", n))
    for n in feature_spec["time_reals"]:
        keys.add(("time", n))
    for n in feature_spec["scenario_reals"]:
        keys.add(("scenario", n))
    for n in feature_spec["static_reals"]:
        keys.add(("static_cont", n))
    for n in feature_spec["static_categoricals"]:
        keys.add(("static_cat", n))
    return keys


def validate_group_members(groups, feature_spec: dict):
    all_keys = _all_model_keys(feature_spec)
    assigned_to = {}
    for g, members in groups.items():
        for key in members:
            assigned_to.setdefault(key, []).append(g)

    dup = {k: gs for k, gs in assigned_to.items() if len(gs) > 1}
    assert not dup, f"feature assigned to more than one group: {dup}"

    unknown = [k for k in assigned_to if k not in all_keys]
    if unknown:
        raise SystemExit(f"[groups-env] group membership references non-existent model columns: {unknown}")

    unassigned = sorted(all_keys - set(assigned_to))
    return unassigned


def print_report(groups, feature_spec):
    print(f"[groups-env] checkpoint feature_spec: "
          f"{len(feature_spec['dynamic_reals'])} dynamic, {len(feature_spec['time_reals'])} time, "
          f"{len(feature_spec['scenario_reals'])} scenario, {len(feature_spec['static_reals'])} static_cont, "
          f"{len(feature_spec['static_categoricals'])} static_cat")

    env_members = groups["environmental exposure"]
    print(f"[groups-env] environmental exposure group: {len(env_members)} columns -> "
          f"{[n for _, n in env_members]}")

    print()
    for g in GROUP_ORDER:
        members = groups[g]
        n_dyn = sum(1 for p, _ in members if p in H_T_PATHWAYS)
        print(f"[groups-env] {g:32s} {len(members):3d} cols  h_t-attributable: {n_dyn}")

    unassigned = validate_group_members(groups, feature_spec)
    print()
    if unassigned:
        print(f"[groups-env] UNASSIGNED model columns: {unassigned}")
    else:
        print("[groups-env] OK: every model input feature is assigned to exactly one group.")

    n_ht_groups = sum(1 for g in G7_ORDER if any(p in H_T_PATHWAYS for p, _ in groups[g]))
    print(f"\n[groups-env] G7 (7 h_t-reachable groups): {G7_ORDER}")
    print(f"[groups-env] {n_ht_groups}/7 groups have at least one dynamic-pathway member.")


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    feature_spec = ckpt["metadata"]["feature_spec"]
    groups = build_group_members(feature_spec)
    print_report(groups, feature_spec)


if __name__ == "__main__":
    main()
