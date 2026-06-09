"""Intervention variable taxonomy for the scenario/counterfactual system (docs/causal.md).

Every covariate that can appear in future/scenario mode is assigned one of five classes,
which controls how the scenario engine may use it:

* ``known_future``           — truly known at forecast time (calendar). Always usable; not
  an intervention.
* ``planned_action``         — user-plannable future action (carbs, insulin dose/timing,
  exercise plan). Editable as a **causal** intervention; editing it also drives its
  derived "on-board"/absorption features (see :meth:`derived_features`) so the edit is
  *coherent* rather than contradictory.
* ``physiological_proxy``    — future physiological signal observed retrospectively but not
  directly controllable (HR, RR, SpO2, sleep stage). Allowed in factual replay / scenario
  **simulation** with uncertainty — **not** a clean causal action.
* ``static_profile``         — stable baseline (diabetes status, meds, age, site, embedding).
  Edits are **sensitivity analyses**, not causal interventions.
* ``unsupported_intervention``— everything else (derived absorption/IOB features, raw CGM,
  validity/missingness, timing-from-history). Refused as a direct causal edit unless the
  caller explicitly enables it (they are normally set *as a consequence* of a
  ``planned_action`` via its action curve, never edited directly).

The default mapping is the T1DEXI schema; pass a custom ``mapping`` to override.
``dynamic_insulin`` gates insulin: when only **static** medication status exists (no
time-varying dose), insulin is demoted to ``static_profile`` so it can only be a
sensitivity edit, never a dynamic causal action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

CLASSES = ("known_future", "planned_action", "physiological_proxy",
           "static_profile", "unsupported_intervention")

# strength of the known causal direction per planned action (+1 raises glucose, -1 lowers,
# 0 = context-dependent / no global direction). Drives which actions get a *hard* ranking
# loss vs a context-gated one (exercise) vs none.
DEFAULT_DIRECTION = {
    "carbs_g_at_t": +1,            # carbohydrate intake raises glucose (clean, acts < 60 min)
    "bolus_units_at_t": -1,        # bolus insulin lowers glucose (delayed; peak may exceed 60 min)
    "basal_units_at_t": -1,        # basal insulin lowers glucose (slow)
    "exercise_minutes_at_t": 0,    # context-dependent (can raise or lower) -> context-gated only
    "exercise_active_flag": 0,
}

# a planned action's *coherent* footprint: the always-on derived features it should drive
# when edited (so we never set "carbs high" while "carbs-on-board = 0").
DEFAULT_ACTION_DERIVED = {
    "carbs_g_at_t": ["carbs_absorption_fast", "carbs_absorption_slow",
                     "fat_protein_delayed_absorption", "meal_absorption_total",
                     "meal_event_count_last_60min"],
    "bolus_units_at_t": ["bolus_iob_units_fast", "bolus_iob_units_slow",
                         "bolus_iob_units_total", "insulin_activity_curve"],
    "basal_units_at_t": ["basal_iob_proxy", "basal_delivered_units_5min",
                         "basal_rate_u_per_hr_current"],
    "exercise_minutes_at_t": ["exercise_active_flag", "recent_exercise_flag_2h",
                              "exercise_intensity_current_or_recent"],
}

DEFAULT_T1DEXI_TAXONOMY: Dict[str, List[str]] = {
    "known_future": [
        "minute_of_day", "hour_of_day", "day_of_week", "is_weekend",
        "sin_hour_of_day", "cos_hour_of_day", "sin_day_of_week", "cos_day_of_week",
        "tod_sin", "tod_cos", "relative_time_idx",
    ],
    "planned_action": [
        "carbs_g_at_t", "bolus_units_at_t", "basal_units_at_t",
        "exercise_minutes_at_t", "exercise_active_flag",
    ],
    "physiological_proxy": [
        "heart_rate_value_for_model", "steps_value_for_model",
        "recent_sleep_flag_12h", "poor_sleep_flag",
        "previous_sleep_total_sleep_time_min", "previous_sleep_rem_duration_min",
        "previous_sleep_efficiency", "sleep_feature_age_hours",
        "recent_exercise_flag_2h", "exercise_intensity_current_or_recent",
        "competitive_exercise_flag_current_or_recent",
        "snack_before_exercise_flag_current_or_recent",
    ],
    "static_profile": [
        "sex", "race", "ethnicity", "randomized_arm", "insulin_modality",
        "insulin_delivery_group", "lifetime_hypoglycemia_category", "lifetime_dka_category",
        "education_level", "income_level", "health_insurance",
        "age_years", "height_in", "weight_lb", "bmi_kg_m2", "baseline_hba1c_value",
        "age_at_diabetes_onset_years", "diabetes_duration_years",
    ],
    # everything not listed above defaults to unsupported_intervention (derived/validity:
    # carbs_absorption_*, *_iob_*, insulin_activity_curve, glucose_*, missing_*, *_is_valid,
    # time_since_last_*, cgm_gap_*, basal_rate_*, etc.).
}


@dataclass
class EditDecision:
    """Outcome of :meth:`InterventionTaxonomy.check_edit`."""

    variable: str
    var_class: str
    allowed: bool
    level: str            # "causal_action" | "sensitivity_analysis" | "scenario_simulation" | "known_future" | "refused"
    direction: int = 0    # known causal sign for actions (+1/-1/0)
    warning: str = ""


class InterventionTaxonomy:
    """Resolve a covariate's class and how it may be edited."""

    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None, *,
                 direction: Optional[Dict[str, int]] = None,
                 action_derived: Optional[Dict[str, List[str]]] = None,
                 dynamic_insulin: bool = True,
                 enabled_unsupported: Optional[List[str]] = None):
        self.mapping = {k: list(v) for k, v in (mapping or DEFAULT_T1DEXI_TAXONOMY).items()}
        self.direction = dict(direction or DEFAULT_DIRECTION)
        self.action_derived = {k: list(v) for k, v in (action_derived or DEFAULT_ACTION_DERIVED).items()}
        self.dynamic_insulin = bool(dynamic_insulin)
        self.enabled_unsupported = set(enabled_unsupported or [])
        self._insulin_vars = {"bolus_units_at_t", "basal_units_at_t"}
        self._lookup: Dict[str, str] = {}
        for cls, names in self.mapping.items():
            for n in names:
                self._lookup[n] = cls

    def class_of(self, var: str) -> str:
        cls = self._lookup.get(var, "unsupported_intervention")
        # demote dynamic insulin to static sensitivity when only static med status exists
        if not self.dynamic_insulin and var in self._insulin_vars:
            return "static_profile"
        return cls

    def variables_in_class(self, cls: str) -> List[str]:
        return [v for v, c in self._lookup.items() if self.class_of(v) == cls]

    def is_planned_action(self, var: str) -> bool:
        return self.class_of(var) == "planned_action"

    def is_editable_as_intervention(self, var: str) -> bool:
        """True if ``var`` may be set as a (causal) planned action."""
        return self.is_planned_action(var) or var in self.enabled_unsupported

    def direction_of(self, var: str) -> int:
        """Known global causal sign (+1/-1) or 0 if context-dependent/unknown."""
        return self.direction.get(var, 0) if self.is_planned_action(var) else 0

    def has_known_direction(self, var: str) -> bool:
        return self.is_planned_action(var) and self.direction_of(var) != 0

    def derived_features(self, action: str) -> List[str]:
        """Always-on derived features an action drives (for coherent edits)."""
        return list(self.action_derived.get(action, []))

    def check_edit(self, var: str, *, as_causal: bool = True) -> EditDecision:
        """Decide whether/how ``var`` may be edited, with the right interpretation label.

        Refuses unsupported variables as direct causal edits; labels static edits as
        sensitivity analyses and physiological proxies as scenario simulations.
        """
        cls = self.class_of(var)
        if cls == "planned_action":
            return EditDecision(var, cls, True, "causal_action", self.direction_of(var))
        if cls == "known_future":
            return EditDecision(var, cls, True, "known_future")
        if cls == "static_profile":
            return EditDecision(var, cls, True, "sensitivity_analysis", 0,
                                f"{var!r} is a static-profile feature; edits are a sensitivity "
                                "analysis, NOT a causal intervention.")
        if cls == "physiological_proxy":
            return EditDecision(var, cls, True, "scenario_simulation", 0,
                                f"{var!r} is a physiological proxy (not directly controllable); "
                                "treat edits as scenario simulation with uncertainty, not a causal action.")
        # unsupported
        if var in self.enabled_unsupported:
            return EditDecision(var, cls, True, "scenario_simulation", 0,
                                f"{var!r} is an explicitly-enabled unsupported variable; not a clean causal action.")
        return EditDecision(var, cls, False, "refused", 0,
                            f"{var!r} is not an editable intervention (class=unsupported_intervention); "
                            "it is normally set as a consequence of a planned_action via its action curve.")
