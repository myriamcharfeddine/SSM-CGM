"""Intervention-set generator: K coherent scenario variants for one history (docs/causal.md).

A planned action (carbs, insulin, exercise) is not just its instantaneous intake — its
forward glucose effect is carried by the derived "on-board"/absorption features. So a
*coherent* edit sets the whole feature group consistently. We learn an
:class:`InterventionLibrary` of **dose-level footprints** empirically from training
windows (the scaled trajectory of an action's intake + its derived features following a
real event of that magnitude), which is scale-correct and faithful — then apply them to
any history to build an intervention set:

    {none, low, high} carbs ;  {0U, low, high} bolus ;  {none, light, vigorous} exercise

Each variant carries: the edited batch, an ordinal **ranking label** (oriented by the
action's known causal direction), a **context-gate weight** (1 for hard-direction
actions, context-dependent for exercise), and a **support flag**. The hard ordinal is
only attached where the direction is globally known (carbs +, insulin −); ambiguous
actions (exercise) get context-gated weights instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .taxonomy import InterventionTaxonomy


@dataclass
class InterventionVariant:
    """One scenario variant for the intervention set."""

    x: Dict[str, torch.Tensor]          # edited batch dict (forecast-only mask + this action revealed)
    action: str
    level: str                          # "none" | "low" | "high" | ...
    dose_rank: int                      # 0..K-1 by intake magnitude
    direction: int                      # action's known causal sign (+1/-1/0)
    intake_scaled: float
    context_gate: float = 1.0           # ranking weight (1 hard-direction; context-dependent for exercise)
    support: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class InterventionLibrary:
    """Dose-level coherent footprints for plannable actions, fit on train windows."""

    taxonomy: InterventionTaxonomy
    x_reals: List[str]
    actions: List[str]
    levels: Dict[str, List[str]]                       # action -> level labels (rank order)
    footprint: Dict[str, Dict[str, torch.Tensor]]      # action -> level -> (H, n_derived) scaled trajectory
    intake_scaled: Dict[str, Dict[str, float]]         # action -> level -> scaled intake value
    derived_idx: Dict[str, List[int]]                  # action -> x_reals indices of derived feats
    intake_idx: Dict[str, int]
    onset_step: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, dataloader, taxonomy: InterventionTaxonomy, spec, *,
            actions: Sequence[str] = ("carbs_g_at_t", "bolus_units_at_t"),
            n_dose_levels: int = 2, max_batches: Optional[int] = None,
            level_names: Optional[Sequence[str]] = None, device: str = "cpu") -> "InterventionLibrary":
        """Learn per-action dose-level footprints from windows (no model needed).

        ``n_dose_levels`` positive dose bins (by intake quantile) plus a ``none`` level.
        Footprints are the mean scaled trajectory of the action's intake + derived
        features over the horizon for events in that dose bin.
        """
        xr = list(spec.x_reals)
        actions = [a for a in actions if a in xr and taxonomy.is_planned_action(a)]
        derived_idx = {a: [xr.index(f) for f in taxonomy.derived_features(a) if f in xr] for a in actions}
        intake_idx = {a: xr.index(a) for a in actions}
        names = list(level_names) if level_names else (["none"] + [f"dose{i+1}" for i in range(n_dose_levels)])

        intake_vals = {a: [] for a in actions}
        traj = {a: [] for a in actions}     # per window: (H, 1+n_derived) scaled [intake, *derived]
        for bi, (x, _y) in enumerate(dataloader):
            if max_batches and bi >= max_batches:
                break
            dc = x["decoder_cont"]
            for a in actions:
                cols = [intake_idx[a]] + derived_idx[a]
                intake_vals[a].append(dc[:, 0, intake_idx[a]].reshape(-1).cpu())   # intake at onset
                traj[a].append(dc[:, :, cols].cpu())                          # (B,H,1+nd)
        return cls._from_collected(taxonomy, xr, actions, derived_idx, intake_idx, names,
                                   n_dose_levels, intake_vals, traj)

    # ------------------------------------------------------------------
    @classmethod
    def fit_from_streams(cls, streams, taxonomy: InterventionTaxonomy, spec, *,
                         actions: Sequence[str] = ("carbs_g_at_t", "bolus_units_at_t"),
                         n_dose_levels: int = 2, stride: int = 3,
                         max_windows_per_stream: Optional[int] = 2000,
                         level_names: Optional[Sequence[str]] = None) -> "InterventionLibrary":
        """Learn per-action dose-level footprints directly from :class:`ParticipantStream`
        objects (the BPTT trainer's representation), with no windowed dataloader.

        Mirrors :meth:`fit`'s binning exactly, but slides a length-``H`` window over each
        stream's scaled ``full_cont`` with ``stride``: the window-start row gives the onset
        intake and ``full_cont[t:t+H, [intake, *derived]]`` gives the footprint trajectory.
        """
        xr = list(spec.x_reals)
        H = spec.horizon
        actions = [a for a in actions if a in xr and taxonomy.is_planned_action(a)]
        derived_idx = {a: [xr.index(f) for f in taxonomy.derived_features(a) if f in xr] for a in actions}
        intake_idx = {a: xr.index(a) for a in actions}
        names = list(level_names) if level_names else (["none"] + [f"dose{i+1}" for i in range(n_dose_levels)])
        cols = {a: [intake_idx[a]] + derived_idx[a] for a in actions}

        intake_vals = {a: [] for a in actions}
        traj = {a: [] for a in actions}     # per window: (H, 1+n_derived) scaled [intake, *derived]
        for s in streams:
            fc = s.full_cont                                   # (T, n_cont) scaled
            T = int(fc.shape[0])
            if T <= H:
                continue
            starts = list(range(0, T - H, max(1, stride)))
            if max_windows_per_stream and len(starts) > max_windows_per_stream:
                starts = starts[:max_windows_per_stream]
            for a in actions:
                ic = intake_idx[a]
                intake_vals[a].append(torch.stack([fc[t, ic] for t in starts]).reshape(-1).cpu())
                traj[a].append(torch.stack([fc[t:t + H, cols[a]] for t in starts], dim=0).cpu())  # (W,H,1+nd)
        return cls._from_collected(taxonomy, xr, actions, derived_idx, intake_idx, names,
                                   n_dose_levels, intake_vals, traj)

    # ------------------------------------------------------------------
    @classmethod
    def _from_collected(cls, taxonomy, xr, actions, derived_idx, intake_idx, names,
                        n_dose_levels, intake_vals, traj) -> "InterventionLibrary":
        """Bin collected (intake, trajectory) samples into dose-level footprints.

        ``intake_vals[a]`` is a list of ``(n,)`` intake tensors and ``traj[a]`` a list of
        ``(n, H, 1+n_derived)`` scaled-trajectory tensors; level 0 (``none``) is the
        low-intake (<=20th pct) baseline and the positive mass is split into
        ``n_dose_levels`` intake-quantile bins. Shared by :meth:`fit`/:meth:`fit_from_streams`."""
        footprint, intake_scaled, levels = {}, {}, {}
        for a in actions:
            iv = torch.cat(intake_vals[a]); tj = torch.cat(traj[a], dim=0)     # (N,) , (N,H,1+nd)
            lo = float(iv.quantile(0.20))                                       # "none" baseline intake
            pos = iv[iv > lo]
            edges = ([float(pos.quantile(q)) for q in np.linspace(0, 1, n_dose_levels + 1)]
                     if len(pos) else [lo] * (n_dose_levels + 1))
            footprint[a], intake_scaled[a], levels[a] = {}, {}, list(names)
            # level 0 = none: low-intake windows
            none_mask = iv <= lo
            base_traj = tj[none_mask].mean(0) if none_mask.any() else tj.mean(0)   # (H,1+nd)
            footprint[a]["none"] = base_traj[:, 1:].clone()
            intake_scaled[a]["none"] = float(base_traj[0, 0])
            for i in range(n_dose_levels):
                name = names[i + 1]
                m = (iv > edges[i]) & (iv <= edges[i + 1] + 1e-9) & (iv > lo)
                t = tj[m].mean(0) if m.any() else base_traj
                footprint[a][name] = t[:, 1:].clone()                          # (H, n_derived)
                intake_scaled[a][name] = float(t[0, 0])
        return cls(taxonomy=taxonomy, x_reals=xr, actions=actions, levels=levels,
                   footprint=footprint, intake_scaled=intake_scaled,
                   derived_idx=derived_idx, intake_idx=intake_idx)

    # ------------------------------------------------------------------
    def apply(self, x: Dict[str, torch.Tensor], action: str, level: str) -> Dict[str, torch.Tensor]:
        """Return a copy of ``x`` with ``action`` set to ``level``: intake + its coherent
        derived-feature footprint written into the decoder horizon (from ``onset_step``)."""
        xk = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        dc = xk["decoder_cont"]
        H = dc.shape[1]
        fp = self.footprint[action][level].to(dc.device, dc.dtype)             # (H, n_derived)
        dc[:, self.onset_step:, self.intake_idx[action]] = self.intake_scaled[action][level]
        for j, idx in enumerate(self.derived_idx[action]):
            dc[:, :, idx] = fp[:, j]
        return xk


def make_intervention_set(model, x: Dict[str, torch.Tensor], action: str,
                          library: InterventionLibrary, *, context_gate: float = 1.0,
                          support_flags: Optional[Dict[str, bool]] = None) -> List[InterventionVariant]:
    """Build the K coherent variants for ``action`` over history ``x`` (forecast-only base
    with this action revealed). Ordinal ``dose_rank`` is by intake magnitude; the known
    causal ``direction`` orients the expected outcome ranking. ``context_gate`` < 1 down-
    weights ambiguous actions (exercise); ``support_flags`` marks out-of-support levels."""
    from ..scenario import make_forecast_only_batch, set_scenario_variable
    direction = library.taxonomy.direction_of(action)
    levels = library.levels[action]
    variants = []
    for rank, lvl in enumerate(levels):
        xk = library.apply(make_forecast_only_batch(model, x), action, lvl)
        # reveal this action in the mask (the derived features are always-on decoder vars)
        if action in getattr(model, "scenario_vars", []):
            xk = set_scenario_variable(model, xk, action, library.intake_scaled[action][lvl], mask=1)
        sup = True if support_flags is None else bool(support_flags.get(lvl, True))
        variants.append(InterventionVariant(
            x=xk, action=action, level=lvl, dose_rank=rank, direction=direction,
            intake_scaled=library.intake_scaled[action][lvl], context_gate=float(context_gate),
            support=sup, metadata={"n_levels": len(levels)}))
    return variants
