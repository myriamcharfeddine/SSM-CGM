"""Scenario helper methods shared by the streamable model.

These mirror the small helpers on :class:`ssmcgm.models.ssmcgm.SSMCGM` (index
lookups, ``make_*_batch`` wrappers, scenario-mask resolution) but live in a mixin
so the existing model is **not modified**. They delegate to the same free functions
in :mod:`ssmcgm.scenario`, so a single scenario/counterfactual implementation backs
both models.
"""

from __future__ import annotations

import torch


class ScenarioStreamMixin:
    """Mix into a ``BaseModelWithCovariates`` subclass that defines the scenario
    attributes (``scenario_reals``, ``scenario_categoricals``, ``scenario_vars``,
    ``n_scenario_vars``, ``scenario_masker``, ``_eval_scenario_mode``, and the
    ``scenario_dropout_*`` / ``scenario_train_mode`` / ``scenario_lambda`` config)."""

    # ----- covariate index lookups -----
    def _dec_real_index(self, name):
        return self.hparams.x_reals.index(name)

    def _dec_cat_index(self, name):
        return self.hparams.x_categoricals.index(name)

    # ----- eval-default scenario mode -----
    def set_scenario_mode(self, mode: str):
        assert mode in ("forecast_only", "factual")
        self._eval_scenario_mode = mode

    def _resolve_scenario_mask(self, x, B, H, device):
        if "scenario_mask" in x and x["scenario_mask"] is not None:
            return x["scenario_mask"].to(device=device, dtype=self.dtype)
        if self.training:
            from ..scenario import sample_scenario_dropout_mask
            return sample_scenario_dropout_mask(
                B, H, self.n_scenario_vars, self.hparams.scenario_dropout_mode,
                self.hparams.scenario_dropout_p, device).to(self.dtype)
        fill = 1.0 if self._eval_scenario_mode == "factual" else 0.0
        return torch.full((B, H, self.n_scenario_vars), fill, device=device, dtype=self.dtype)

    # ----- batch builders (thin wrappers over ssmcgm.scenario) -----
    def make_forecast_only_batch(self, x):
        from ..scenario import make_forecast_only_batch
        return make_forecast_only_batch(self, x)

    def make_factual_scenario_batch(self, x):
        from ..scenario import make_factual_scenario_batch
        return make_factual_scenario_batch(self, x)

    def make_planned_scenario_batch(self, x, scenario_spec):
        from ..scenario import make_planned_scenario_batch
        return make_planned_scenario_batch(self, x, scenario_spec)

    def set_scenario_variable(self, x, name, values, mask=1, steps=None):
        from ..scenario import set_scenario_variable
        return set_scenario_variable(self, x, name, values, mask=mask, steps=steps)

    def zero_scenario_variables(self, x):
        from ..scenario import zero_scenario_variables
        return zero_scenario_variables(self, x)

    def set_static_variable(self, x, name, value, **kw):
        from ..scenario import set_static_variable
        return set_static_variable(self, x, name, value, **kw)

    def make_static_profile_batch(self, x, static_spec):
        from ..scenario import make_static_profile_batch
        return make_static_profile_batch(self, x, static_spec)

    def make_mixed_counterfactual_batch(self, x, static_spec=None, scenario_spec=None,
                                        baseline="forecast_only"):
        from ..scenario import make_mixed_counterfactual_batch
        return make_mixed_counterfactual_batch(self, x, static_spec, scenario_spec, baseline)

    # ----- training-mode resolution (back-compat with SSMCGM) -----
    def _resolved_scenario_train_mode(self) -> str:
        if getattr(self, "scenario_two_pass", False):
            return "two_pass"
        return getattr(self, "scenario_train_mode", "mixed")
