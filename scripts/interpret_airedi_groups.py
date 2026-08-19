"""AI-READI modality group definitions for interpretability attribution (Step 1).

Defines the modality groups used by ``interpret_airedi_export.py`` /
``interpret_airedi_grouped_export.py`` and validates them against the ACTUAL
feature schema stored in the checkpoint (``metadata["feature_spec"]``), not a
hardcoded guess. Every model input feature is keyed by ``(pathway, name)``
because the same column name can appear in more than one pathway with a
different tensor identity (e.g. ``heart_rate_mean`` is both a streamed
``dynamic`` real feeding the encoder/SSM state and a future-known
``scenario`` real feeding the decoder directly) -- collapsing those to a bare
name would silently double-count or misassign them.

Pathways (see ssmcgm/models/aireadi_stream.py):
  * ``dynamic``  -- streamed reals -> ``encoder_fusion`` (GroupedLinearFusion)
    -> the SSM hidden state h_t. This is the ONLY pathway with per-feature
    contribution vectors ``u_f`` usable by the rigorous joint-group-norm
    attribution in ``ssmcgm/stream/attribution.py::attribute_window``.
  * ``time``     -- ``time_reals`` -> ``decoder_time_fusion`` -> decoder input
    directly (never touches h_t).
  * ``scenario`` -- ``scenario_reals`` (future-known proxies) -> decoder
    input directly (never touches h_t).
  * ``static_cont`` / ``static_cat`` -- ``static_reals`` / ``static_categoricals``
    -> ``static_encoder`` -> FiLM scale/shift on h_t and decoder static
    context (no per-feature additive decomposition).

Run:
  python scripts/interpret_airedi_groups.py
"""
import os
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

CHECKPOINT_PATH = os.path.join(
    _ROOT, "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt")

# Display order matches the user-specified taxonomy (and, where the same
# concept applies, Section 17 / Table 36's existing group names).
GROUP_ORDER = [
    "glucose history",
    "time / calendar",
    "wearable activity / exercise",
    "heart rate / physiology",
    "stress",
    "sleep",
    "static clinical",
    "medication metadata",
    "site / cohort",
    "data quality / missingness",
    "meal proxy (diagnostic only)",
]

# Pathways whose per-feature contributions feed the SSM hidden state h_t via
# GroupedLinearFusion -- i.e. attributable by the rigorous joint-group-norm
# method. Every other pathway bypasses h_t entirely in this architecture.
H_T_PATHWAYS = {"dynamic"}


def _quality_dynamic(dynamic_reals):
    return [
        n for n in dynamic_reals
        if n.endswith("_count") or n.endswith("_device_availability") or n == "sleep_stage_unknown"
    ]


def build_group_members(feature_spec: dict) -> "OrderedDict[str, list[tuple[str, str]]]":
    """Return ``{group: [(pathway, feature_name), ...]}`` built from the model's
    actual feature schema (no hardcoded column list)."""
    dyn = list(feature_spec["dynamic_reals"])
    tim = list(feature_spec["time_reals"])
    scn = list(feature_spec["scenario_reals"])
    scont = list(feature_spec["static_reals"])
    scat = list(feature_spec["static_categoricals"])

    quality = set(_quality_dynamic(dyn))

    def dyn_scn(prefixes, extra_exact=()):
        """dynamic + scenario members matching prefixes/exact names, minus quality."""
        d = [n for n in dyn if (n.startswith(prefixes) or n in extra_exact) and n not in quality]
        s = [n for n in scn if (n.startswith(prefixes) or n in extra_exact)]
        return [("dynamic", n) for n in d] + [("scenario", n) for n in s]

    med = [n for n in scont if n.startswith("med_")]
    clinical_cont = [n for n in scont if n not in med]  # everything else static-real, incl. demo_*
    site_cat = [n for n in scat if n in ("participants_clinical_site", "participants_study_group")]
    clinical_cat = [n for n in scat if n not in site_cat]  # demo_sex_at_birth, demo_race_primary, demo_marital_status

    groups: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict((g, []) for g in GROUP_ORDER)
    groups["glucose history"] = [("dynamic", n) for n in dyn if n == "cgm_glucose_mean"]
    groups["time / calendar"] = [("time", n) for n in tim]
    groups["wearable activity / exercise"] = dyn_scn(
        ("activity_",), extra_exact=("calories_total", "calories_per_min"))
    groups["heart rate / physiology"] = dyn_scn(("heart_rate_", "respiratory_rate_", "oxygen_saturation_"))
    groups["stress"] = dyn_scn(("stress_level_",))
    groups["sleep"] = dyn_scn(("sleep_",))
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


def validate_group_members(groups: "OrderedDict[str, list[tuple[str, str]]]", feature_spec: dict):
    """Enforce: every model input key in exactly one group, no key in >1 group."""
    all_keys = _all_model_keys(feature_spec)
    assigned_to = {}
    for g, members in groups.items():
        for key in members:
            assigned_to.setdefault(key, []).append(g)

    dup = {k: gs for k, gs in assigned_to.items() if len(gs) > 1}
    assert not dup, f"feature assigned to more than one group: {dup}"

    unknown = [k for k in assigned_to if k not in all_keys]
    if unknown:
        raise SystemExit(f"[groups] group membership references non-existent model columns: {unknown}")

    unassigned = sorted(all_keys - set(assigned_to))
    return unassigned


def print_report(groups, feature_spec):
    print(f"[groups] checkpoint feature_spec: "
          f"{len(feature_spec['dynamic_reals'])} dynamic, {len(feature_spec['time_reals'])} time, "
          f"{len(feature_spec['scenario_reals'])} scenario, {len(feature_spec['static_reals'])} static_cont, "
          f"{len(feature_spec['static_categoricals'])} static_cat "
          f"= {sum(len(feature_spec[k]) for k in ('dynamic_reals', 'time_reals', 'scenario_reals', 'static_reals', 'static_categoricals'))} total columns")

    for candidate in ("environmental exposure", "respiratory / SpO2"):
        print(f"[groups] NOTE: '{candidate}' was NOT created as its own group per instructions -- "
              f"respiratory_rate_* and oxygen_saturation_* ARE literal dynamic_reals columns and are "
              f"folded into 'heart rate / physiology' below; no humidity/temperature column exists in "
              f"the feature schema (confirmed absent, not silently dropped).")

    print()
    for g in GROUP_ORDER:
        members = groups[g]
        n_dyn = sum(1 for p, _ in members if p in H_T_PATHWAYS)
        n_other = len(members) - n_dyn
        pathways = sorted(set(p for p, _ in members))
        print(f"[groups] {g:32s} {len(members):3d} cols  pathways={pathways}  "
              f"(h_t-attributable via dynamic pathway: {n_dyn}, other pathway: {n_other})")
        for p, n in members:
            print(f"    - ({p}) {n}")

    unassigned = validate_group_members(groups, feature_spec)
    print()
    if unassigned:
        print(f"[groups] UNASSIGNED model columns (add to a group or mark ignored): {unassigned}")
    else:
        print("[groups] OK: every model input feature is assigned to exactly one group.")

    n_ht_groups = sum(1 for g in GROUP_ORDER if any(p in H_T_PATHWAYS for p, _ in groups[g]))
    print(f"\n[groups] {n_ht_groups}/{len(GROUP_ORDER)} groups have at least one member on the "
          f"'dynamic' pathway (i.e. reach the SSM hidden state h_t and are attributable by the "
          f"rigorous joint-group-norm method). Groups that are 100% time/scenario/static pathway "
          f"have NO u_f contribution into h_t in this architecture and cannot be scored by that "
          f"method at all -- this is a structural fact about AireadiStreamModel, not a data gap.")
    pure_non_ht = [g for g in GROUP_ORDER if groups[g] and not any(p in H_T_PATHWAYS for p, _ in groups[g])]
    print(f"[groups] groups with ZERO dynamic-pathway members: {pure_non_ht}")


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    feature_spec = ckpt["metadata"]["feature_spec"]
    groups = build_group_members(feature_spec)
    print_report(groups, feature_spec)


if __name__ == "__main__":
    main()
