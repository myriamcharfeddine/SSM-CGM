"""Streamable counterfactuals for SSM-CGM-Stream.

The unified, batch-dict counterfactual surface (``counterfactual_effect``,
``StaticEdit`` / ``ScenarioEdit`` / ``MixedEdit``) in
:mod:`ssmcgm.counterfactual.scenario_cf` works on this model unchanged. This module
adds the **streamability distinctions** the spec calls out (§8):

* **scenario edits** keep the history state ``h_t`` fixed and edit only the future
  scenario path — *streamable*, constant-time given ``h_t``.
* **static-profile edits** can change ``h0`` / FiLM and therefore the whole state
  trajectory. Two modes:
    * ``decoder_only_sensitivity`` — keep the original ``h_t``, change only the
      decoder's static context. Fast; a *sensitivity* analysis, not a coherent
      counterfactual.
    * ``state_replay`` — re-encode history under the edited static context. Coherent
      but not constant-time.

``counterfactual_delta`` returns the baseline / counterfactual forecasts, their
delta, the two history outputs ``h_t`` (so callers can verify whether history was
replayed), and metadata labeling streamability + the interpretation level.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..scenario import baseline_batch, make_mixed_counterfactual_batch, make_planned_scenario_batch
from ..counterfactual.engine import extract_prediction


def _encode_state(model, batch):
    """Encode static + history for a batch → ``(static_context, stream_state)``."""
    sctx = model.encode_static(batch)
    enc_feats = model._features(batch["encoder_cat"], batch["encoder_cont"], model.encoder_variables)
    state = model.encode_history(enc_feats, sctx)
    return sctx, state


def _decode(model, batch, state, sctx, quantile_idx):
    dec_cat, dec_cont = batch["decoder_cat"], batch["decoder_cont"]
    B, H = dec_cont.shape[0], dec_cont.shape[1]
    T, A, M = model._decoder_inputs(batch, dec_cat, dec_cont, B, H, dec_cont.device)
    pred = model.decode_horizon(state, sctx, T, A, M)
    pred = model.transform_output(pred, target_scale=batch["target_scale"])
    return extract_prediction(pred)[..., quantile_idx]


@torch.inference_mode()
def forecast_baseline(model, x: Dict, *, baseline="forecast_only", quantile_idx: int = 1):
    """Baseline forecast (median quantile) under ``baseline`` scenario treatment."""
    model.eval()
    b = baseline_batch(model, x, baseline)
    sctx, state = _encode_state(model, b)
    return _decode(model, b, state, sctx, quantile_idx)


@torch.inference_mode()
def forecast_scenario(model, x: Dict, scenario_spec: Dict, *, baseline="forecast_only",
                      quantile_idx: int = 1):
    """Planned-scenario forecast: keep ``h_t`` fixed, edit the future scenario path."""
    model.eval()
    cf = make_planned_scenario_batch(model, x, scenario_spec) if baseline == "forecast_only" \
        else make_mixed_counterfactual_batch(model, x, None, scenario_spec, baseline)
    sctx, state = _encode_state(model, cf)
    return _decode(model, cf, state, sctx, quantile_idx)


@torch.inference_mode()
def counterfactual_delta(
    model, x: Dict, *, scenario_spec: Optional[Dict] = None, static_spec: Optional[Dict] = None,
    static_edit_mode: str = "state_replay", baseline="forecast_only", quantile_idx: int = 1,
) -> Dict:
    """Streamable counterfactual delta (cf − baseline) with streamability metadata.

    ``static_edit_mode`` (only relevant when ``static_spec`` is set):
    ``"decoder_only_sensitivity"`` reuses the baseline ``h_t`` (no replay);
    ``"state_replay"`` re-encodes history under the edited static context.
    """
    if static_edit_mode not in ("decoder_only_sensitivity", "state_replay"):
        raise ValueError("static_edit_mode must be 'decoder_only_sensitivity' or 'state_replay'")
    model.eval()
    has_s, has_a = bool(static_spec), bool(scenario_spec)

    base_batch = baseline_batch(model, x, baseline)
    cf_batch = make_mixed_counterfactual_batch(model, x, static_spec, scenario_spec, baseline)

    sctx_b, state_b = _encode_state(model, base_batch)

    history_replayed = False
    if has_s and static_edit_mode == "state_replay":
        sctx_c, state_c = _encode_state(model, cf_batch)   # re-encode under edited static
        history_replayed = True
    elif has_s:                                            # decoder_only_sensitivity
        sctx_c, state_c = model.encode_static(cf_batch), state_b   # reuse h_t, edit e_s only
    else:                                                 # scenario-only: h_t fixed
        sctx_c, state_c = sctx_b, state_b

    base = _decode(model, base_batch, state_b, sctx_b, quantile_idx)
    cf = _decode(model, cf_batch, state_c, sctx_c, quantile_idx)

    if not has_s and not has_a:
        level = "forecast"
    elif has_s and has_a:
        level = "heterogeneity_analysis"
    elif has_a:
        level = "planned_scenario_simulation"
    else:
        level = "static_profile_sensitivity"
    meta = {
        "intervention_type": ("mixed_edit" if has_s and has_a else
                              "static_profile_edit" if has_s else
                              "scenario_edit" if has_a else "none"),
        "streamable": (not has_s) or (static_edit_mode == "decoder_only_sensitivity"),
        "history_replayed": history_replayed,
        "static_edit_mode": static_edit_mode if has_s else None,
        "causal_interpretation_level": level,
        "warning": ("Static edits change h0/FiLM; decoder_only_sensitivity is a "
                    "sensitivity analysis (h_t held fixed), not a coherent counterfactual."
                    if has_s and static_edit_mode == "decoder_only_sensitivity" else None),
    }
    return {"baseline": base, "counterfactual": cf, "delta": cf - base,
            "baseline_h": state_b.last_output, "cf_h": state_c.last_output, "metadata": meta}
