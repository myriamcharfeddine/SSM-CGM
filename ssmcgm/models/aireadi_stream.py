"""Native AI-READI stream model built from the SSMCGM-Stream primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from ..data.aireadi import AireadiFeatureSpec, AireadiPreprocessor
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

    def decode_horizon(self, h_t: torch.Tensor, static_context: StaticContext,
                       time_features: torch.Tensor, scenario_values: torch.Tensor,
                       scenario_mask: torch.Tensor, *, return_decomposition: bool = False):
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        time_fused = self._time_features(time_features)
        e_s = static_context.embedding
        if e_s.shape[0] == 1 and h_t.shape[0] > 1:
            e_s = e_s.expand(h_t.shape[0], -1)
        return self.decoder(
            h_t,
            e_s,
            time_fused,
            scenario_values,
            scenario_mask,
            return_decomposition=return_decomposition,
        )

    def checkpoint_metadata(self) -> dict:
        return {
            "feature_spec": asdict(self.feature_spec),
            "preprocessor": self.preprocessor.to_jsonable(),
            "model_config": asdict(self.config),
        }
