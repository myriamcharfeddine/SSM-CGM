"""Native AI-READI stream model built from the SSMCGM-Stream primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from ..data.aireadi import AireadiFeatureSpec, AireadiPreprocessor
from ..stream.exercise import ExerciseSensitivityHead
from ..stream.decoder import ScenarioHorizonDecoder
from ..stream.fusion import GroupedLinearFusion
from ..stream.ssm import StreamingMESStack
from ..stream.state import StaticContext, StreamState
from ..stream.static import StaticEncoder, StaticFiLM, StaticStateInitializer


@dataclass
class AireadiStreamModelConfig:
    hidden_size: int = 128
    dropout: float = 0.10
    mamba_depth: int = 2
    mamba_style: str = "mes"
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    ngroups: int = 1
    scan_mode: str = "chunked"
    chunk_len: int = 64
    use_static_film: bool = True
    film_mode: str = "scale_shift"
    fusion_mode: str = "grouped_sum"
    scenario_decompose: bool = True
    hard_gate_scenario_effect: bool = False
    hr_exercise: bool = False
    hr_exercise_gain_target: float = 0.30363636363636365
    hr_exercise_deadzone_bpm: float = 15.0
    hr_exercise_lag_support_min: int = 60
    hr_exercise_g_floor_mgdl: float = 105.0
    hr_exercise_bout_median_min: int = 30
    hr_exercise_rise_to_peak_min: int = 12
    route_future_hr_via_exercise_head: bool = True
    hr_exercise_decay_min: int = 45
    horizon_emb_dim: int = 16
    decoder_hidden_size: Optional[int] = None
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    target_transform: str = "residual_current"


def _embedding_dim(cardinality: int) -> int:
    return int(min(16, max(2, round(cardinality ** 0.25 * 4))))


class AireadiStreamModel(nn.Module):
    """Stateful stream forecaster for AI-READI participant segments.

    The model consumes only observed dynamic features in the streaming state. Future
    scenario values enter the short-horizon decoder with explicit masks, so unknown
    future values differ from known zeros.
    """

    def __init__(
        self,
        feature_spec: AireadiFeatureSpec,
        preprocessor: AireadiPreprocessor,
        config: Optional[AireadiStreamModelConfig] = None,
    ):
        super().__init__()
        self.feature_spec = feature_spec
        self.preprocessor = preprocessor
        self.config = config or AireadiStreamModelConfig()
        cfg = self.config
        if cfg.target_transform != "residual_current":
            raise ValueError("AI-READI stream model currently supports residual_current target transform only.")

        hs = int(cfg.hidden_size)
        cat_cards = [
            max(1, len(preprocessor.static_category_maps.get(col, {"__unknown__": 0})))
            for col in feature_spec.static_categoricals
        ]
        cat_dims = [_embedding_dim(c) for c in cat_cards]
        self.static_encoder = StaticEncoder(
            cat_cardinalities=cat_cards,
            cat_emb_dims=cat_dims,
            n_continuous=len(feature_spec.static_reals),
            hidden_size=hs,
            dropout=cfg.dropout,
        )
        self.encoder_fusion = GroupedLinearFusion(
            {name: 1 for name in feature_spec.dynamic_reals},
            hs,
            fusion_mode=cfg.fusion_mode,
            dropout=cfg.dropout,
        )
        self.temporal = StreamingMESStack(
            d_model=hs,
            depth=cfg.mamba_depth,
            dropout=cfg.dropout,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            mamba_style=cfg.mamba_style,
            x_share_mode="mean",
            scan_mode=cfg.scan_mode,
            chunk_len=cfg.chunk_len,
            static_timescale_mode="none",
        )
        self.state_initializer = StaticStateInitializer(
            e_s_dim=hs,
            depth=cfg.mamba_depth,
            nheads=self.temporal.nheads,
            d_state=self.temporal.d_state,
            state_init_mode="patient_static",
        )
        self.film = StaticFiLM(hs, hs, film_mode=cfg.film_mode if cfg.use_static_film else "none")
        self.decoder_time_fusion = (
            GroupedLinearFusion(
                {name: 1 for name in feature_spec.time_reals},
                hs,
                fusion_mode=cfg.fusion_mode,
                dropout=cfg.dropout,
            )
            if feature_spec.time_reals
            else None
        )
        n_time = hs if self.decoder_time_fusion is not None else 0
        self.decoder = ScenarioHorizonDecoder(
            d_model=hs,
            e_s_dim=hs,
            n_time_features=n_time,
            n_scenario=len(feature_spec.scenario_reals),
            horizon=feature_spec.horizon_steps,
            output_size=len(cfg.quantiles),
            hidden_size=cfg.decoder_hidden_size or hs,
            decoder_mode="shared_mlp_with_horizon_embedding",
            horizon_emb_dim=cfg.horizon_emb_dim,
            dropout=cfg.dropout,
            scenario_decompose=cfg.scenario_decompose,
        )
        self.exercise_head = (
            ExerciseSensitivityHead(
                horizon_steps=feature_spec.horizon_steps,
                bin_minutes=feature_spec.bin_minutes,
                quantiles=tuple(float(q) for q in cfg.quantiles),
                gain_target=cfg.hr_exercise_gain_target,
                hr_deadzone_bpm=cfg.hr_exercise_deadzone_bpm,
                lag_support_min=cfg.hr_exercise_lag_support_min,
                g_floor_mgdl=cfg.hr_exercise_g_floor_mgdl,
                bout_median_min=cfg.hr_exercise_bout_median_min,
                rise_to_peak_min=cfg.hr_exercise_rise_to_peak_min,
                decay_min=cfg.hr_exercise_decay_min,
            )
            if cfg.hr_exercise else None
        )

    @property
    def quantiles(self) -> List[float]:
        return [float(q) for q in self.config.quantiles]

    @property
    def n_scenario_vars(self) -> int:
        return len(self.feature_spec.scenario_reals)

    def encode_static(self, static_cat: torch.Tensor, static_cont: torch.Tensor) -> StaticContext:
        if static_cat.dim() == 1:
            static_cat = static_cat.unsqueeze(0)
        if static_cont.dim() == 1:
            static_cont = static_cont.unsqueeze(0)
        e_s = self.static_encoder(static_cat.long(), static_cont.float())
        return StaticContext(embedding=e_s, raw_static_cat=static_cat, raw_static_cont=static_cont)

    def init_stream(self, static_context: StaticContext) -> StreamState:
        e_s = static_context.embedding
        batch, device, dtype = e_s.shape[0], e_s.device, e_s.dtype
        reduced = self.state_initializer(e_s)
        layer_states = [
            self.temporal.blocks[i].ssm.expand_state(reduced[i])
            for i in range(self.temporal.depth)
        ]
        conv_states = [
            self.temporal.blocks[i].ssm.zero_conv(batch, device, dtype)
            for i in range(self.temporal.depth)
        ]
        return StreamState(
            layer_states=layer_states,
            conv_states=conv_states,
            last_output=None,
            static_context=static_context,
            step=0,
        )

    def _dynamic_features(self, dynamic: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            name: dynamic[..., i:i + 1]
            for i, name in enumerate(self.feature_spec.dynamic_reals)
        }

    def _time_features(self, time_features: torch.Tensor) -> torch.Tensor:
        if self.decoder_time_fusion is None:
            return time_features.new_zeros(time_features.shape[0], time_features.shape[1], 0)
        feats = {
            name: time_features[..., i:i + 1]
            for i, name in enumerate(self.feature_spec.time_reals)
        }
        fused, _ = self.decoder_time_fusion(feats, return_contributions=False)
        return fused

    def fuse_history(self, dynamic: torch.Tensor, static_embedding: torch.Tensor) -> torch.Tensor:
        fused, _ = self.encoder_fusion(self._dynamic_features(dynamic), return_contributions=False)
        return self.film(fused, static_embedding)

    def scan_chunk(self, dynamic: torch.Tensor, static_context: StaticContext,
                   state: StreamState) -> StreamState:
        if dynamic.dim() == 2:
            dynamic = dynamic.unsqueeze(0)
        u = self.fuse_history(dynamic, static_context.embedding)
        out, layer_states, conv_states, _ = self.temporal.scan(
            u,
            state.layer_states,
            state.conv_states,
            record=False,
            static_embedding=static_context.embedding,
        )
        return StreamState(
            layer_states=layer_states,
            conv_states=conv_states,
            last_output=out[:, -1],
            static_context=static_context,
            step=state.step + int(dynamic.shape[1]),
        ), out

    def update_stream(self, state: StreamState, dynamic_t: torch.Tensor) -> StreamState:
        if dynamic_t.dim() == 1:
            dynamic_t = dynamic_t.unsqueeze(0)
        feats = self._dynamic_features(dynamic_t)
        fused, _ = self.encoder_fusion(feats, return_contributions=False)
        fused = self.film(fused, state.static_context.embedding)
        out_t, layer_states, conv_states = self.temporal.step(
            fused,
            state.layer_states,
            state.conv_states,
            static_embedding=state.static_context.embedding,
        )
        return StreamState(
            layer_states=layer_states,
            conv_states=conv_states,
            last_output=out_t,
            static_context=state.static_context,
            step=state.step + 1,
        )

    def _exercise_hr_inputs(
        self,
        scenario_values: torch.Tensor,
        scenario_mask: torch.Tensor,
        *,
        resting_hr_bpm: torch.Tensor | None,
        future_hr_delta_bpm: torch.Tensor | None,
        future_hr_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if "heart_rate_mean" not in self.feature_spec.scenario_reals:
            raise ValueError(
                "hr_exercise requires heart_rate_mean in scenario_reals"
            )
        hr_index = self.feature_spec.scenario_reals.index("heart_rate_mean")
        batch, horizon = scenario_values.shape[:2]

        if future_hr_delta_bpm is not None:
            hr_delta = torch.as_tensor(
                future_hr_delta_bpm,
                device=scenario_values.device,
                dtype=scenario_values.dtype,
            )
            if hr_delta.dim() == 1:
                hr_delta = hr_delta.unsqueeze(0)
            if hr_delta.shape != (batch, horizon):
                raise ValueError(
                    f"future_hr_delta_bpm must have shape {(batch, horizon)}"
                )
            if future_hr_mask is None:
                hr_mask = torch.ones_like(hr_delta)
            else:
                hr_mask = torch.as_tensor(
                    future_hr_mask,
                    device=scenario_values.device,
                    dtype=scenario_values.dtype,
                )
                if hr_mask.dim() == 1:
                    hr_mask = hr_mask.unsqueeze(0)
            return torch.relu(hr_delta), hr_mask, hr_index

        if resting_hr_bpm is None:
            raise ValueError(
                "hr_exercise requires resting_hr_bpm or future_hr_delta_bpm"
            )
        stats = self.preprocessor.continuous_stats["heart_rate_mean"]
        future_hr_bpm = (
            scenario_values[..., hr_index] * float(stats["std"])
            + float(stats["mean"])
        )
        resting = torch.as_tensor(
            resting_hr_bpm,
            device=scenario_values.device,
            dtype=scenario_values.dtype,
        )
        if resting.dim() == 0:
            resting = resting.expand(batch)
        resting = resting.reshape(batch, -1)
        if resting.shape[1] != 1:
            raise ValueError("resting_hr_bpm must be scalar per batch item")
        hr_delta = torch.relu(future_hr_bpm - resting)
        hr_mask = scenario_mask[..., hr_index]
        return hr_delta, hr_mask, hr_index

    def decode_horizon(
        self,
        h_t: torch.Tensor,
        static_context: StaticContext,
        time_features: torch.Tensor,
        scenario_values: torch.Tensor,
        scenario_mask: torch.Tensor,
        *,
        current_glucose_mgdl: torch.Tensor | None = None,
        resting_hr_bpm: torch.Tensor | None = None,
        future_hr_delta_bpm: torch.Tensor | None = None,
        future_hr_mask: torch.Tensor | None = None,
        return_decomposition: bool = False,
        return_exercise_components: bool = False,
        return_components: bool = False,
    ):
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        time_fused = self._time_features(time_features)
        e_s = static_context.embedding
        if e_s.shape[0] == 1 and h_t.shape[0] > 1:
            e_s = e_s.expand(h_t.shape[0], -1)

        hard_gate = bool(self.config.hard_gate_scenario_effect)
        if self.exercise_head is None:
            if not return_components and not hard_gate:
                return self.decoder(
                    h_t,
                    e_s,
                    time_fused,
                    scenario_values,
                    scenario_mask,
                    return_decomposition=return_decomposition,
                )
            decoder_components = self.decoder(
                h_t,
                e_s,
                time_fused,
                scenario_values,
                scenario_mask,
                return_components=True,
            )
            if hard_gate:
                gate = decoder_components["scenario_availability"].to(
                    dtype=decoder_components["scenario_effect"].dtype
                )
                decoder_components["scenario_effect"] = (
                    decoder_components["scenario_effect"] * gate
                )
                decoder_components["final"] = (
                    decoder_components["base"]
                    + decoder_components["scenario_effect"]
                )
            if return_components:
                decoder_components.update(
                    {
                        "historical_state": h_t,
                        "static_embedding": e_s,
                        "time_latent": time_fused,
                        "scenario_values": scenario_values,
                        "scenario_mask": scenario_mask,
                    }
                )
                return decoder_components
            if return_decomposition:
                return (
                    decoder_components["final"],
                    decoder_components["base"],
                    decoder_components["scenario_effect"],
                )
            return decoder_components["final"]
        if current_glucose_mgdl is None:
            raise ValueError(
                "hr_exercise requires current_glucose_mgdl so R uses y_base"
            )

        hr_delta, hr_mask, hr_index = self._exercise_hr_inputs(
            scenario_values,
            scenario_mask,
            resting_hr_bpm=resting_hr_bpm,
            future_hr_delta_bpm=future_hr_delta_bpm,
            future_hr_mask=future_hr_mask,
        )
        decoder_mask = scenario_mask
        if self.config.route_future_hr_via_exercise_head:
            decoder_mask = scenario_mask.clone()
            decoder_mask[..., hr_index] = 0.0

        decoder_components = self.decoder(
            h_t,
            e_s,
            time_fused,
            scenario_values,
            decoder_mask,
            return_components=True,
        )
        decoder_final = decoder_components["final"]
        decoder_base = decoder_components["base"]
        decoder_effect = decoder_components["scenario_effect"]
        exercise_delta, components = self.exercise_head.effect(
            decoder_base,
            current_glucose_mgdl,
            hr_delta,
            hr_mask,
        )
        structured_delta = exercise_delta.unsqueeze(-1)
        final = decoder_final + structured_delta
        effect = decoder_effect + structured_delta
        if hard_gate:
            gate = scenario_mask.bool().any(dim=-1, keepdim=True).to(
                dtype=effect.dtype
            )
            effect = effect * gate
            final = decoder_base + effect
        if return_components:
            decoder_components.update(
                {
                    "final": final,
                    "base": decoder_base,
                    "scenario_effect": effect,
                    "historical_state": h_t,
                    "static_embedding": e_s,
                    "time_latent": time_fused,
                    "scenario_values": scenario_values,
                    "scenario_mask": scenario_mask,
                    "scenario_availability": scenario_mask.bool().any(
                        dim=-1, keepdim=True
                    ),
                    "exercise_components": components,
                }
            )
            return decoder_components
        if return_exercise_components:
            return final, decoder_base, effect, components
        if return_decomposition:
            return final, decoder_base, effect
        return final

    def configure_exercise_head_training(self) -> List[str]:
        """Freeze the base and expose only the cohort exercise gain to training."""
        if self.exercise_head is None:
            raise RuntimeError("hr_exercise must be enabled before configuring training")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.exercise_head.parameters():
            parameter.requires_grad_(True)
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def checkpoint_metadata(self) -> dict:
        return {
            "feature_spec": asdict(self.feature_spec),
            "preprocessor": self.preprocessor.to_jsonable(),
            "model_config": asdict(self.config),
        }
