"""Scenario-conditioned interface for SSM-CGM.

The forecasting problem is

    y_hat[t+1:t+H] = f_theta(H_t, T[t+1:t+H], A[t+1:t+H], M[t+1:t+H], S)

where future *scenario* variables ``A`` (wearables, meals, sleep, HR/RR, steps,
medication plan) are **not** known at deployment time. They are supplied
optionally, with a binary availability mask ``M``. The mask matters because a
zero is ambiguous (unknown vs. baseline vs. a real zero) — the model must tell an
*unknown* scenario variable apart from a *supplied* one.

This module provides:
  * :class:`ScenarioMasker` — gates decoder scenario inputs with a **learned
    "unknown" token** (not zero) when masked-out, and adds an explicit
    availability context, so the VSN / MES temporal processor is preserved.
  * :func:`sample_scenario_dropout_mask` — training-time scenario dropout
    (forecast-only / factual / partial), so one checkpoint supports all modes.
  * batch helpers (``make_forecast_only_batch`` etc.) that operate on a model +
    batch dict and set ``x["scenario_mask"]`` (B, H, n_scenario).

Three evaluation modes from one trained checkpoint:
  forecast_only  — all scenario masks 0 (honest, deployable).
  factual        — observed future scenario values, masks 1 (retrospective upper bound).
  planned        — user-specified scenario values/masks (counterfactual / control).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# masking module
# ---------------------------------------------------------------------------
class ScenarioMasker(nn.Module):
    """Gate decoder scenario inputs by availability, with learned unknown tokens.

    ``scenario_reals`` and ``scenario_cat_sizes`` (name -> embedding dim) define
    the scenario variables, in that order; the mask tensor columns follow the
    same order ``scenario_reals + list(scenario_cat_sizes)``.
    """

    def __init__(self, scenario_reals: List[str], scenario_cat_sizes: Dict[str, int],
                 hidden_size: int):
        super().__init__()
        self.scenario_reals = list(scenario_reals)
        self.scenario_cats = list(scenario_cat_sizes)
        self.vars = self.scenario_reals + self.scenario_cats
        self.n_vars = len(self.vars)

        # learned "unknown" token for each scenario variable (raw-value scalar
        # for reals; an embedding-sized vector for categoricals)
        if self.scenario_reals:
            self.unknown_real = nn.Parameter(torch.zeros(len(self.scenario_reals)))
        self.unknown_cat = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(size)) for name, size in scenario_cat_sizes.items()
        })
        # explicit availability context added to the decoder embeddings
        self.mask_proj = nn.Linear(self.n_vars, hidden_size) if self.n_vars else None

    def gate(self, emb_dec: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Replace masked-out scenario inputs with their learned unknown token.

        ``emb_dec``: decoder variable dict (reals as ``(B,H,1)``, cats as
        ``(B,H,emb)``). ``mask``: ``(B, H, n_vars)`` in ``{0,1}``.
        """
        for i, name in enumerate(self.scenario_reals):
            if name not in emb_dec:
                continue
            m = mask[..., i:i + 1]                                  # (B,H,1)
            emb_dec[name] = m * emb_dec[name] + (1.0 - m) * self.unknown_real[i]
        for j, name in enumerate(self.scenario_cats):
            if name not in emb_dec:
                continue
            m = mask[..., len(self.scenario_reals) + j: len(self.scenario_reals) + j + 1]
            emb_dec[name] = m * emb_dec[name] + (1.0 - m) * self.unknown_cat[name]
        return emb_dec

    def context(self, mask: torch.Tensor) -> torch.Tensor:
        """Explicit availability context ``(B, H, hidden)`` from the mask."""
        return self.mask_proj(mask)


# ---------------------------------------------------------------------------
# training-time scenario dropout
# ---------------------------------------------------------------------------
def sample_scenario_dropout_mask(B: int, H: int, n_vars: int, mode: str, p: float,
                                 device) -> torch.Tensor:
    """Sample a ``(B, H, n_vars)`` availability mask for training.

    Modes (a scenario var is available for the whole horizon or not):
      * ``"all_or_none"`` — per sample: forecast-only (all 0) w.p. ``p``, else factual (all 1).
      * ``"per_variable"`` — per (sample, var): available w.p. ``1-p``.
      * ``"mixed"`` (default) — per sample: forecast-only w.p. ``p``; otherwise
        half factual (all 1), half partial (per-var Bernoulli 0.5).
    """
    if n_vars == 0:
        return torch.zeros(B, H, 0, device=device)
    if mode == "all_or_none":
        keep = (torch.rand(B, 1, 1, device=device) >= p).float()
        return keep.expand(B, H, n_vars).contiguous()
    if mode == "per_variable":
        keep = (torch.rand(B, 1, n_vars, device=device) >= p).float()
        return keep.expand(B, H, n_vars).contiguous()
    if mode == "mixed":
        u = torch.rand(B, 1, 1, device=device)
        forecast = (u < p).float()                                   # all-0
        partial = (torch.rand(B, 1, n_vars, device=device) < 0.5).float()
        factual = (u >= p) & (torch.rand(B, 1, 1, device=device) < 0.5)
        m = torch.where(factual, torch.ones(B, 1, n_vars, device=device), partial)
        m = m * (1.0 - forecast)                                     # forecast-only -> all 0
        return m.expand(B, H, n_vars).contiguous()
    raise ValueError(f"unknown scenario_dropout_mode {mode!r}")


# ---------------------------------------------------------------------------
# batch helpers (operate on a model + batch dict, set x["scenario_mask"])
# ---------------------------------------------------------------------------
def _empty_mask(model, x) -> torch.Tensor:
    B = x["decoder_cont"].shape[0]
    H = x["decoder_cont"].shape[1]
    return torch.zeros(B, H, model.n_scenario_vars, device=x["decoder_cont"].device)


def _copy_batch(x: Dict) -> Dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}


def make_forecast_only_batch(model, x: Dict) -> Dict:
    """All scenario masks 0 — only deployment-time information is used."""
    x = _copy_batch(x)
    x["scenario_mask"] = _empty_mask(model, x)
    return x


def make_factual_scenario_batch(model, x: Dict) -> Dict:
    """All scenario masks 1 — supply the observed future scenario trajectory."""
    x = _copy_batch(x)
    x["scenario_mask"] = _empty_mask(model, x) + 1.0
    return x


def zero_scenario_variables(model, x: Dict) -> Dict:
    """Set scenario decoder values to 0 and masks to 0 (explicit no-info)."""
    x = make_forecast_only_batch(model, x)
    for name in model.scenario_reals:
        idx = model._dec_real_index(name)
        x["decoder_cont"][..., idx] = 0.0
    return x


def set_scenario_variable(model, x: Dict, name: str, values, mask=1, steps=None) -> Dict:
    """Set one scenario variable's decoder trajectory + its availability mask.

    ``values``: scalar or ``(B,H)``/``(H,)`` tensor in the model's input space
    (z-scored for reals, encoded code for categoricals). ``steps``: optional
    ``slice``/index range restricting the intervention to part of the horizon.
    """
    x = _copy_batch(x)
    if "scenario_mask" not in x:
        x["scenario_mask"] = _empty_mask(model, x)
    else:
        x["scenario_mask"] = x["scenario_mask"].clone()
    B, H = x["decoder_cont"].shape[0], x["decoder_cont"].shape[1]
    sl = steps if steps is not None else slice(0, H)
    var_idx = model.scenario_vars.index(name)

    if name in model.scenario_reals:
        idx = model._dec_real_index(name)
        v = torch.as_tensor(values, dtype=x["decoder_cont"].dtype,
                            device=x["decoder_cont"].device)
        x["decoder_cont"][:, sl, idx] = v if v.ndim else v.expand(B, _slice_len(sl, H))
    else:
        idx = model._dec_cat_index(name)
        v = torch.as_tensor(values, dtype=x["decoder_cat"].dtype,
                            device=x["decoder_cat"].device)
        x["decoder_cat"][:, sl, idx] = v if v.ndim else v.expand(B, _slice_len(sl, H))

    x["scenario_mask"][:, sl, var_idx] = float(mask)
    return x


def make_planned_scenario_batch(model, x: Dict, scenario_spec: Dict) -> Dict:
    """Build a planned-scenario batch from forecast-only + a spec.

    ``scenario_spec``: ``{name: value}`` or ``{name: {"values":.., "steps":.., "mask":1}}``.
    Only the named scenario variables are activated; everything else stays masked.
    """
    x = make_forecast_only_batch(model, x)
    for name, spec in scenario_spec.items():
        if isinstance(spec, dict):
            x = set_scenario_variable(model, x, name, spec.get("values", 0.0),
                                      mask=spec.get("mask", 1), steps=spec.get("steps"))
        else:
            x = set_scenario_variable(model, x, name, spec, mask=1)
    return x


def _slice_len(sl, H):
    if isinstance(sl, slice):
        return len(range(*sl.indices(H)))
    return H


# ---------------------------------------------------------------------------
# generic static editing (S-edits) + unified batch builders
# ---------------------------------------------------------------------------
def baseline_batch(model, x: Dict, baseline="forecast_only") -> Dict:
    """Build the baseline batch for a counterfactual comparison.

    ``baseline``: ``"forecast_only"`` (no future scenario info — honest default),
    ``"factual"`` (observed future scenario), or an explicit planned-scenario dict.
    """
    if baseline == "forecast_only":
        return make_forecast_only_batch(model, x)
    if baseline == "factual":
        return make_factual_scenario_batch(model, x)
    if isinstance(baseline, dict):
        return make_planned_scenario_batch(model, x, baseline)
    raise ValueError(f"unknown baseline {baseline!r}")


def _identifier_vars(model) -> List[str]:
    """Best-effort list of participant/group identifier variables (not normally edited)."""
    dp = getattr(model, "dataset_parameters", None)
    if isinstance(dp, dict):
        for key in ("group_ids", "group_id"):
            if dp.get(key):
                return list(dp[key])
    return []


def _broadcast_static(value, ref: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast a scalar or per-sample ``(B,)`` value to ``ref[..., idx]`` shape ``(B, T)``."""
    B, T = ref.shape[0], ref.shape[1]
    v = torch.as_tensor(value, device=ref.device)
    if v.ndim == 0:
        return v.to(ref.dtype)
    if v.ndim == 1 and v.shape[0] == B:
        return v.to(ref.dtype).view(B, 1).expand(B, T)
    raise ValueError(f"static edit for {name!r}: value must be scalar or shape (B={B},), "
                     f"got shape {tuple(v.shape)}")


def set_static_variable(model, x: Dict, name: str, value, *, allow_identifier: bool = False) -> Dict:
    """Edit a static variable across the whole window (encoder + decoder tensors).

    Handles static continuous (``static_reals``) and static categorical
    (``static_categoricals``) variables. Categorical edits require an **encoded
    integer code** (not a label). Participant/group identifiers are refused
    unless ``allow_identifier=True``.
    """
    x = _copy_batch(x)
    static_reals = list(model.hparams.static_reals)
    static_cats = list(model.hparams.static_categoricals)

    if name in _identifier_vars(model) and not allow_identifier:
        raise ValueError(
            f"{name!r} is a participant/group identifier; pass allow_identifier=True to edit it.")

    if name in static_reals:
        idx = model.hparams.x_reals.index(name)
        for key in ("encoder_cont", "decoder_cont"):
            x[key][..., idx] = _broadcast_static(value, x[key], name)
    elif name in static_cats:
        if name not in model.hparams.x_categoricals:
            raise KeyError(f"static categorical {name!r} not in x_categoricals "
                           f"(grouped categoricals are not directly editable)")
        idx = model.hparams.x_categoricals.index(name)
        code = torch.as_tensor(value)
        if code.dtype.is_floating_point and not torch.equal(code, code.round()):
            raise ValueError(
                f"categorical edit for {name!r} needs an encoded integer code, got {value!r}. "
                f"Use the dataset's categorical_encoders[{name!r}].transform([...]) to encode a label.")
        for key in ("encoder_cat", "decoder_cat"):
            x[key][..., idx] = _broadcast_static(value, x[key], name).to(x[key].dtype)
    else:
        raise KeyError(
            f"static variable {name!r} not found; static_reals={static_reals}, "
            f"static_categoricals={static_cats}")
    return x


def make_static_profile_batch(model, x: Dict, static_spec: Dict) -> Dict:
    """Apply static-profile edits (S-edits). ``static_spec``: ``{name: value}``
    (encoded code for categoricals; scalar or per-sample ``(B,)`` for reals)."""
    for name, value in static_spec.items():
        x = set_static_variable(model, x, name, value)
    return x


def _apply_scenario_edits(model, x: Dict, scenario_spec: Dict) -> Dict:
    """Apply scenario (A) edits on top of the current batch (keeps other vars' masks)."""
    for name, spec in scenario_spec.items():
        if isinstance(spec, dict):
            x = set_scenario_variable(model, x, name, spec.get("values", 0.0),
                                      mask=spec.get("mask", 1), steps=spec.get("steps"))
        else:
            x = set_scenario_variable(model, x, name, spec, mask=1)
    return x


def make_mixed_counterfactual_batch(model, x: Dict, static_spec: Optional[Dict] = None,
                                    scenario_spec: Optional[Dict] = None,
                                    baseline="forecast_only") -> Dict:
    """Build a batch with optional static (S) and scenario (A) edits over a baseline.

    Scenario edits are applied on top of the baseline scenario treatment (so
    non-edited scenario vars keep the baseline's masks); static edits then edit S.
    """
    x = baseline_batch(model, x, baseline)
    if scenario_spec:
        x = _apply_scenario_edits(model, x, scenario_spec)
    if static_spec:
        x = make_static_profile_batch(model, x, static_spec)
    return x


def partial_scenario_batch(model, x: Dict, name: str) -> Dict:
    """Forecast-only batch with a single scenario variable made available
    (observed values, mask 1) — isolates that variable's contribution."""
    x = make_forecast_only_batch(model, x)
    x["scenario_mask"][:, :, model.scenario_vars.index(name)] = 1.0
    return x


# ---------------------------------------------------------------------------
# evaluation: one checkpoint, multiple scenario modes
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_scenarios(model, dataloader, *, modes=("forecast_only", "factual", "partial"),
                       planned_spec: Optional[Dict] = None, device: Optional[str] = None) -> Dict:
    """Evaluate one trained checkpoint under several scenario modes.

    Reports MAE / RMSE of the median forecast vs. the observed future CGM for:
      * ``forecast_only`` — deployable benchmark (primary result),
      * ``factual``       — retrospective upper bound (observed future scenario),
      * ``partial``       — each scenario variable available one at a time,
      * ``planned``       — a user ``planned_spec`` (if given).

    Returns ``{mode: {"mae":.., "rmse":.., "n":..}}`` (``partial`` -> per variable).
    """
    from .counterfactual.engine import extract_prediction
    device = device or next(model.parameters()).device
    model = model.to(device).eval()

    def _batch_for(mode, x, var=None):
        if mode == "forecast_only":
            return make_forecast_only_batch(model, x)
        if mode == "factual":
            return make_factual_scenario_batch(model, x)
        if mode == "partial":
            return partial_scenario_batch(model, x, var)
        if mode == "planned":
            return make_planned_scenario_batch(model, x, planned_spec)
        raise ValueError(mode)

    # accumulators: name -> [sum_abs, sum_sq, n]
    acc: Dict[str, list] = {}

    def _accumulate(key, pred_med, target):
        e = (pred_med - target)
        a = acc.setdefault(key, [0.0, 0.0, 0])
        a[0] += e.abs().sum().item()
        a[1] += e.pow(2).sum().item()
        a[2] += e.numel()

    for batch in dataloader:
        x, y = batch if isinstance(batch, (list, tuple)) else (batch, None)
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        target = (y[0] if isinstance(y, (list, tuple)) else y).to(device)
        for mode in modes:
            if mode == "partial":
                for var in model.scenario_vars:
                    pred = extract_prediction(model(_batch_for("partial", x, var)))[..., 1]
                    _accumulate(f"partial:{var}", pred, target)
            else:
                pred = extract_prediction(model(_batch_for(mode, x)))[..., 1]
                _accumulate(mode, pred, target)

    out = {}
    for key, (sa, ss, n) in acc.items():
        out[key] = {"mae": sa / max(n, 1), "rmse": (ss / max(n, 1)) ** 0.5, "n": n}
    return out
