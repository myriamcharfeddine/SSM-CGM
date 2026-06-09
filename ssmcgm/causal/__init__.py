"""Scientifically-defensible scenario / counterfactual layer for SSM-CGM-Stream.

An **optional** extension (does not change the baseline model or training path). It
organizes future/scenario covariates into interpretable *intervention classes* with the
right inductive biases — causal-ranking regularization where domain direction is strong
(carbs, insulin), context-gated priors where it is ambiguous (exercise), uncertainty for
physiological proxies (HR/RR), and static-profile *sensitivity* (not causal) — plus
support/positivity checks and personalization hooks. See ``docs/causal.md``.

Modules:
  * :mod:`ssmcgm.causal.taxonomy`     — intervention variable classes + edit/refusal rules.
  * :mod:`ssmcgm.causal.interventions`— intervention-set generator (coherent action edits).
  * :mod:`ssmcgm.causal.support`      — positivity / support checks for an edited scenario.
  * :mod:`ssmcgm.causal.losses`       — causal-coherence loss modules (ranking/slope/shape).
  * :mod:`ssmcgm.causal.personalize`  — per-user scenario-effect adapters (calibrate Δŷ only).
  * :mod:`ssmcgm.causal.evaluation`   — causal-ranking eval (5th mode), diagnostics, α-sweep.
"""
from .taxonomy import (  # noqa: F401
    CLASSES,
    DEFAULT_T1DEXI_TAXONOMY,
    InterventionTaxonomy,
    EditDecision,
)
from .interventions import (  # noqa: F401
    InterventionLibrary,
    InterventionVariant,
    make_intervention_set,
)
from .support import SupportChecker, SupportResult, glycemic_state  # noqa: F401
from .losses import (  # noqa: F401
    score_glucose,
    intervention_ranking_loss,
    slope_loss,
    mask_consistency_loss,
    effect_shape_loss,
)
from .personalize import (  # noqa: F401
    ScenarioEffectAdapter,
    calibrate_least_squares,
    apply_feedback,
)
from .evaluation import (  # noqa: F401
    causal_ranking_eval,
    describe_eval_modes,
    alpha_sweep_table,
    EVAL_MODES,
)
