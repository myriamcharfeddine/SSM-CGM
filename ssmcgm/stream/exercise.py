"""Structured, cohort-wide exercise sensitivity head.

The response magnitude is imposed from a labeled prior. AI-READI contributes
future HR timing, response shape, and the pointwise glucose-state gate. Outputs
are planning responses, not causal estimates.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ExerciseSensitivityHead(nn.Module):
    """Apply Delta_y = -A * R * S to a base residual forecast.

    A is a dead-zone HR intensity passed through a causal short FIR.
    R is relu(y_base - g_floor), evaluated from current glucose plus the
    scenario-zero base median residual at every horizon.
    S is one cohort-wide positive scalar.
    """

    def __init__(
        self,
        *,
        horizon_steps: int,
        bin_minutes: int,
        quantiles: tuple[float, ...],
        gain_target: float,
        hr_deadzone_bpm: float = 15.0,
        lag_support_min: int = 60,
        g_floor_mgdl: float = 105.0,
        bout_median_min: int = 30,
        rise_to_peak_min: int = 12,
        decay_min: int = 45,
    ):
        super().__init__()
        if gain_target <= 0:
            raise ValueError("gain_target must be positive")
        if horizon_steps <= 0 or bin_minutes <= 0:
            raise ValueError("horizon_steps and bin_minutes must be positive")
        self.horizon_steps = int(horizon_steps)
        self.bin_minutes = int(bin_minutes)
        self.hr_deadzone_bpm = float(hr_deadzone_bpm)
        self.g_floor_mgdl = float(g_floor_mgdl)
        self.bout_median_min = int(bout_median_min)
        self.rise_to_peak_min = int(rise_to_peak_min)
        self.decay_min = int(decay_min)
        self.median_index = min(
            range(len(quantiles)),
            key=lambda i: abs(float(quantiles[i]) - 0.5),
        )

        inverse_softplus = math.log(math.expm1(float(gain_target)))
        self.a_dir = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))

        support_steps = max(
            1,
            min(
                self.horizon_steps,
                int(math.ceil(float(lag_support_min) / float(bin_minutes))),
            ),
        )
        lag_minutes = torch.arange(support_steps, dtype=torch.float32) * float(bin_minutes)
        lag_kernel = torch.zeros_like(lag_minutes)
        positive = lag_minutes > 0
        scaled = lag_minutes[positive] / max(float(rise_to_peak_min), 1.0)
        lag_kernel[positive] = scaled * torch.exp(1.0 - scaled)
        if not bool((lag_kernel > 0).any()):
            lag_kernel[0] = 1.0
        lag_kernel = lag_kernel / lag_kernel.sum()
        self.register_buffer("lag_kernel", lag_kernel, persistent=False)

        reference_delta = self._reference_hr_delta(self.horizon_steps)
        reference_effective = F.relu(reference_delta - self.hr_deadzone_bpm)
        reference_response = self._causal_fir(reference_effective.unsqueeze(0))[0]
        reference_norm = reference_response.max().clamp_min(1e-6)
        self.register_buffer("reference_norm", reference_norm, persistent=False)

    @property
    def sensitivity(self) -> torch.Tensor:
        return F.softplus(self.a_dir)

    def _reference_hr_delta(self, horizon_steps: int) -> torch.Tensor:
        minutes = (
            torch.arange(1, horizon_steps + 1, dtype=torch.float32)
            * float(self.bin_minutes)
        )
        peak_delta_bpm = 39.0
        rising = peak_delta_bpm * (
            minutes / max(float(self.rise_to_peak_min), 1.0)
        ).clamp(max=1.0)
        decay = peak_delta_bpm * torch.exp(
            -(minutes - float(self.bout_median_min)).clamp_min(0.0)
            / max(float(self.decay_min), 1.0)
        )
        return torch.where(minutes <= float(self.bout_median_min), rising, decay)

    def _causal_fir(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() != 2:
            raise ValueError("FIR input must have shape (batch, horizon)")
        kernel = self.lag_kernel.to(device=values.device, dtype=values.dtype)
        padded = F.pad(values.unsqueeze(1), (kernel.numel() - 1, 0))
        return F.conv1d(padded, kernel.flip(0).view(1, 1, -1)).squeeze(1)

    def effect(
        self,
        base_residual: torch.Tensor,
        current_glucose_mgdl: torch.Tensor,
        future_hr_delta_bpm: torch.Tensor,
        future_hr_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if base_residual.dim() != 3:
            raise ValueError("base_residual must have shape (batch, horizon, quantiles)")
        batch, horizon, _ = base_residual.shape
        if horizon != self.horizon_steps:
            raise ValueError(
                f"Expected horizon {self.horizon_steps}, received {horizon}"
            )

        current = torch.as_tensor(
            current_glucose_mgdl,
            device=base_residual.device,
            dtype=base_residual.dtype,
        )
        if current.dim() == 0:
            current = current.expand(batch)
        current = current.reshape(batch, -1)
        if current.shape[1] != 1:
            raise ValueError("current_glucose_mgdl must be scalar per batch item")

        hr_delta = torch.as_tensor(
            future_hr_delta_bpm,
            device=base_residual.device,
            dtype=base_residual.dtype,
        )
        if hr_delta.dim() == 1:
            hr_delta = hr_delta.unsqueeze(0)
        if hr_delta.shape != (batch, horizon):
            raise ValueError(
                f"future_hr_delta_bpm must have shape {(batch, horizon)}, "
                f"received {tuple(hr_delta.shape)}"
            )
        if future_hr_mask is not None:
            mask = torch.as_tensor(
                future_hr_mask,
                device=base_residual.device,
                dtype=base_residual.dtype,
            )
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != hr_delta.shape:
                raise ValueError("future_hr_mask must match future_hr_delta_bpm")
            hr_delta = hr_delta * mask

        effective_hr = F.relu(hr_delta - self.hr_deadzone_bpm)
        activation = self._causal_fir(effective_hr) / self.reference_norm.to(
            device=base_residual.device,
            dtype=base_residual.dtype,
        )
        y_base = current + base_residual[..., self.median_index]
        ramp = F.relu(y_base - self.g_floor_mgdl)
        sensitivity = self.sensitivity.to(
            device=base_residual.device,
            dtype=base_residual.dtype,
        )
        delta = -activation * ramp * sensitivity
        components = {
            "activation": activation,
            "ramp": ramp,
            "sensitivity": sensitivity,
            "y_base_median_mgdl": y_base,
            "effective_hr_bpm": effective_hr,
        }
        return delta, components

    def forward(
        self,
        base_residual: torch.Tensor,
        current_glucose_mgdl: torch.Tensor,
        future_hr_delta_bpm: torch.Tensor,
        future_hr_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        delta, components = self.effect(
            base_residual,
            current_glucose_mgdl,
            future_hr_delta_bpm,
            future_hr_mask,
        )
        final = base_residual + delta.unsqueeze(-1)
        return final, delta, components
