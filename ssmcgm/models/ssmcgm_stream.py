"""SSM-CGM-Stream — a streamable, personalized state-space forecasting model.

A second, **opt-in** architecture alongside :class:`ssmcgm.models.ssmcgm.SSMCGM`
(which is left untouched). The data flow is

    static covariates  → patient-specific h0 + static FiLM
    observed history   → grouped fusion → streaming MES state update (h_t)
    future scenario    → scenario-aware horizon decoder   (does NOT touch h_t)

with the key rule: **anything observed up to t updates the SSM state; anything
planned after t enters only the horizon decoder.** History (encoder) features never
see future scenario values, so there is no look-ahead leakage.

It subclasses ``BaseModelWithCovariates`` (same base as TFT / ``SSMCGM``) so its
``from_dataset`` / covariate plumbing / quantile training loop come for free, and so
the existing scenario (:mod:`ssmcgm.scenario`) and counterfactual
(:mod:`ssmcgm.counterfactual`) machinery — which operate on the batch dict and
``out.prediction`` — work unchanged.

Conceptual streaming API (spec §1):

    encode_static(x)                     -> StaticContext
    init_stream(static_context)          -> StreamState
    encode_history(enc_feats, sctx, s0)  -> StreamState
    update_stream(state, obs_features)   -> StreamState
    decode_horizon(state, sctx, T, A, M) -> prediction (B, H, Q)
    forward(x)                           -> network output

The streaming SSM is the pure-PyTorch :class:`ssmcgm.stream.ssm.StreamingMESStack`
(differentiable ``h0``, real ``step``); a fused ``mamba_ssm`` backend is a
documented follow-up.
"""

from __future__ import annotations

from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

try:
    import numpy as np  # noqa: F401
    from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
    from pytorch_forecasting.models.base_model import BaseModelWithCovariates
    from pytorch_forecasting.models.nn import MultiEmbedding
    _HAS_PTF = True
except Exception:  # pragma: no cover
    BaseModelWithCovariates = object  # type: ignore
    _HAS_PTF = False

from ..stream.state import StaticContext, StreamState
from ..stream.static import StaticEncoder, StaticStateInitializer, StaticFiLM
from ..stream.fusion import GroupedLinearFusion
from ..stream.ssm import StreamingMESStack
from ..stream.decoder import ScenarioHorizonDecoder
from ..stream.attribution import attribute_window
from ..stream.scenario_mixin import ScenarioStreamMixin


if _HAS_PTF:

    class SSMCGMStream(ScenarioStreamMixin, BaseModelWithCovariates):
        """Streamable, personalized MES state-space glucose forecaster."""

        def __init__(
            self,
            hidden_size: int = 128,
            dropout: float = 0.1,
            output_size: Union[int, List[int]] = 7,
            loss: "MultiHorizonMetric" = None,
            # --- streaming MES temporal core ---
            mamba_depth: int = 2,
            mamba_style: str = "mes",          # "mes" | "standard"
            x_share_mode: str = "mean",
            use_light_mamba: bool = False,     # the old extra light Mamba (off by default)
            mamba_block_config: dict = None,
            # --- optional static-conditioned timescale (experimental ablation) ---
            static_timescale_mode: str = "none",   # none|additive
            delta_min: float = 1e-4,
            delta_max: float = 1.0,
            # --- history scan backend (deployment `step` is always sequential) ---
            scan_mode: str = "sequential",         # sequential | chunked (fast, h0-differentiable)
            chunk_len: int = 64,
            # --- static personalization ---
            state_init_mode: str = "patient_static",   # zero|learned_global|patient_static
            use_static_film: bool = True,
            film_mode: str = "scale_shift",            # scale_shift|scale_only|none
            # --- feature fusion ---
            fusion_mode: str = "grouped_sum",          # grouped_sum|dense_linear|grouped_concat_project|cheap_gated
            # --- horizon decoder ---
            decoder_mode: str = "shared_mlp_with_horizon_embedding",
            decoder_hidden_size: int = None,
            horizon_emb_dim: int = 16,
            # --- scenario conditioning ---
            scenario_reals: List[str] = [],
            scenario_categoricals: List[str] = [],
            scenario_dropout_p: float = 0.5,
            scenario_dropout_mode: str = "mixed",
            scenario_train_mode: str = "mixed",        # mixed|mixed+factual|two_pass
            scenario_two_pass: bool = False,
            scenario_lambda: float = 1.0,
            scenario_decompose: bool = False,          # ŷ = ŷ_base + Δŷ_scenario (docs/causal.md)
            # --- covariate plumbing (filled by from_dataset, mirrors TFT) ---
            max_encoder_length: int = 10,
            max_prediction_length: int = 1,
            static_categoricals: List[str] = [],
            static_reals: List[str] = [],
            time_varying_categoricals_encoder: List[str] = [],
            time_varying_categoricals_decoder: List[str] = [],
            categorical_groups: Dict[str, List[str]] = {},
            time_varying_reals_encoder: List[str] = [],
            time_varying_reals_decoder: List[str] = [],
            x_reals: List[str] = [],
            x_categoricals: List[str] = [],
            hidden_continuous_size: int = 8,
            hidden_continuous_sizes: Dict[str, int] = {},
            embedding_sizes: Dict[str, Tuple[int, int]] = {},
            embedding_paddings: List[str] = [],
            embedding_labels: Dict[str, "np.ndarray"] = {},
            learning_rate: float = 1e-3,
            log_interval: Union[int, float] = -1,
            log_val_interval: Union[int, float] = None,
            log_gradient_flow: bool = False,
            logging_metrics: nn.ModuleList = None,
            **kwargs,
        ):
            if logging_metrics is None:
                logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
            if loss is None:
                loss = QuantileLoss()
            self.save_hyperparameters(ignore=["loss", "logging_metrics"])
            super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

            mb = dict(mamba_block_config or {})
            mb.setdefault("d_state", 64)
            mb.setdefault("d_conv", 4)
            mb.setdefault("expand", 2)
            mb.setdefault("headdim", 32)
            mb.setdefault("ngroups", 1)

            hs = self.hparams.hidden_size

            # ---------- shared categorical embeddings + real prescalers --------
            self.input_embeddings = MultiEmbedding(
                embedding_sizes=self.hparams.embedding_sizes,
                categorical_groups=self.hparams.categorical_groups,
                embedding_paddings=self.hparams.embedding_paddings,
                x_categoricals=self.hparams.x_categoricals,
                max_embedding_size=hs,
            )
            self.prescalers = nn.ModuleDict({
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals
            })

            # ---------- static personalization --------------------------------
            static_cats = list(self.hparams.static_categoricals)
            static_realnames = list(self.hparams.static_reals)
            self.static_encoder = StaticEncoder(
                cat_cardinalities=[self.input_embeddings.embedding_sizes[c][0] for c in static_cats],
                cat_emb_dims=[self.input_embeddings.output_size[c] for c in static_cats],
                n_continuous=len(static_realnames),
                hidden_size=hs, dropout=self.hparams.dropout,
            )
            self._static_cat_idx = [self.hparams.x_categoricals.index(c) for c in static_cats]
            self._static_real_idx = [self.hparams.x_reals.index(r) for r in static_realnames]

            # ---------- streaming MES temporal core ---------------------------
            self.temporal = StreamingMESStack(
                d_model=hs, depth=mamba_depth, dropout=self.hparams.dropout,
                d_state=mb["d_state"], d_conv=mb["d_conv"], expand=mb["expand"],
                headdim=mb["headdim"], ngroups=mb["ngroups"], mamba_style=mamba_style,
                x_share_mode=x_share_mode,
                static_timescale_mode=static_timescale_mode, e_s_dim=hs,
                delta_min=delta_min, delta_max=delta_max,
                scan_mode=scan_mode, chunk_len=chunk_len,
            )
            self.state_initializer = StaticStateInitializer(
                e_s_dim=hs, depth=mamba_depth, nheads=self.temporal.nheads,
                d_state=self.temporal.d_state, state_init_mode=state_init_mode,
            )
            self.film = StaticFiLM(hs, hs, film_mode=film_mode if use_static_film else "none")

            # ---------- grouped feature fusion --------------------------------
            enc_sizes = self._variable_sizes(self.encoder_variables)
            self.encoder_fusion = GroupedLinearFusion(
                enc_sizes, hs, fusion_mode=fusion_mode, dropout=self.hparams.dropout)

            self.scenario_reals = list(scenario_reals)
            self.scenario_categoricals = list(scenario_categoricals)
            self.scenario_vars = self.scenario_reals + self.scenario_categoricals
            self.n_scenario_vars = len(self.scenario_vars)
            self._time_decoder_vars = [v for v in self.decoder_variables if v not in self.scenario_vars]
            time_sizes = self._variable_sizes(self._time_decoder_vars)
            self.n_time_features = hs if time_sizes else 0
            self.decoder_time_fusion = (
                GroupedLinearFusion(time_sizes, hs, fusion_mode=fusion_mode,
                                    dropout=self.hparams.dropout) if time_sizes else None)

            # ---------- scenario-aware horizon decoder ------------------------
            self.decoder = ScenarioHorizonDecoder(
                d_model=hs, e_s_dim=hs, n_time_features=self.n_time_features,
                n_scenario=self.n_scenario_vars, horizon=self.hparams.max_prediction_length,
                output_size=self.hparams.output_size, hidden_size=decoder_hidden_size or hs,
                decoder_mode=decoder_mode, horizon_emb_dim=horizon_emb_dim, dropout=self.hparams.dropout,
                scenario_decompose=scenario_decompose,
            )

            # ---------- scenario config + attribution recording ---------------
            self._eval_scenario_mode = "forecast_only"
            self.scenario_two_pass = bool(scenario_two_pass)
            self.scenario_lambda = float(scenario_lambda)
            self.scenario_train_mode = str(scenario_train_mode)
            if self.scenario_train_mode not in ("mixed", "mixed+factual", "two_pass"):
                raise ValueError("scenario_train_mode must be 'mixed', 'mixed+factual', or 'two_pass'")
            if self.n_scenario_vars:
                for s in self.scenario_vars:
                    if s not in self.decoder_variables:
                        raise ValueError(
                            f"scenario variable {s!r} must be a decoder covariate; "
                            f"decoder has {self.decoder_variables}")
            self._record_stream = False
            self._stream_caches = None
            self._stream_contribs = None
            self.effect_adapter = None        # optional per-user Δŷ_scenario adapter (docs/causal.md)

        # ==================================================================
        # construction helpers
        # ==================================================================
        def _variable_sizes(self, names) -> Dict[str, int]:
            """Per-variable feature width: embedding size for cats, prescaler width
            for reals — the input widths for :class:`GroupedLinearFusion`."""
            sizes = {}
            for n in names:
                if n in self.hparams.x_categoricals:
                    sizes[n] = self.input_embeddings.output_size[n]
                else:
                    sizes[n] = self.hparams.hidden_continuous_sizes.get(
                        n, self.hparams.hidden_continuous_size)
            return sizes

        @classmethod
        def from_dataset(cls, dataset, allowed_encoder_known_variable_names=None, **kwargs):
            new_kwargs = copy(kwargs)
            new_kwargs["max_encoder_length"] = dataset.max_encoder_length
            new_kwargs["max_prediction_length"] = dataset.max_prediction_length
            new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))
            return super().from_dataset(
                dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
                **new_kwargs)

        # ==================================================================
        # feature builders (embeddings + prescalers)
        # ==================================================================
        def _features(self, cat: torch.Tensor, cont: torch.Tensor, names) -> Dict[str, torch.Tensor]:
            """Per-variable feature reps ``{name: (..., size)}`` for ``names``."""
            emb = self.input_embeddings(cat)
            feats = {}
            for n in names:
                if n in self.hparams.x_categoricals:
                    feats[n] = emb[n]
                else:
                    idx = self.hparams.x_reals.index(n)
                    feats[n] = self.prescalers[n](cont[..., idx:idx + 1])
            return feats

        def _scenario_values(self, dec_cat: torch.Tensor, dec_cont: torch.Tensor) -> torch.Tensor:
            """``(B, H, n_scenario)`` numeric scenario path ``A`` (z-scored reals;
            categorical codes as floats), ordered ``scenario_reals + scenario_cats``."""
            B, H = dec_cont.shape[0], dec_cont.shape[1]
            cols = []
            for n in self.scenario_reals:
                cols.append(dec_cont[..., self.hparams.x_reals.index(n)])
            for n in self.scenario_categoricals:
                cols.append(dec_cat[..., self.hparams.x_categoricals.index(n)].to(dec_cont.dtype))
            if not cols:
                return dec_cont.new_zeros(B, H, 0)
            return torch.stack(cols, dim=-1)

        # ==================================================================
        # conceptual streaming API
        # ==================================================================
        def encode_static(self, x: Dict[str, torch.Tensor]) -> StaticContext:
            """Encode static covariates once → :class:`StaticContext` (cached ``e_s``)."""
            enc_cat, enc_cont = x["encoder_cat"], x["encoder_cont"]
            static_cat = (enc_cat[:, 0][:, self._static_cat_idx]
                          if self._static_cat_idx else enc_cat[:, 0, :0].long())
            static_cont = (enc_cont[:, 0][:, self._static_real_idx]
                           if self._static_real_idx else enc_cont[:, 0, :0])
            e_s = self.static_encoder(static_cat, static_cont)
            return StaticContext(embedding=e_s, raw_static_cat=static_cat, raw_static_cont=static_cont)

        def init_stream(self, static_context: StaticContext) -> StreamState:
            """Patient-specific initial :class:`StreamState` (``h0`` per layer)."""
            e_s = static_context.embedding
            B, device, dtype = e_s.shape[0], e_s.device, e_s.dtype
            reduced = self.state_initializer(e_s)
            layer_states = [self.temporal.blocks[i].ssm.expand_state(reduced[i])
                            for i in range(self.temporal.depth)]
            conv_states = [self.temporal.blocks[i].ssm.zero_conv(B, device, dtype)
                           for i in range(self.temporal.depth)]
            return StreamState(layer_states=layer_states, conv_states=conv_states,
                               last_output=None, static_context=static_context, step=0)

        def _fuse_history(self, encoder_features, e_s, record=False):
            fused, contribs = self.encoder_fusion(encoder_features, return_contributions=record)
            fused = self.film(fused, e_s)
            return fused, contribs

        def encode_history(self, encoder_features, static_context: StaticContext,
                           initial_state: Optional[StreamState] = None,
                           record: bool = False) -> StreamState:
            """Scan observed history into the SSM state. ``encoder_features`` is the
            ``{name: (B, L, size)}`` dict from :meth:`_features`."""
            state = initial_state or self.init_stream(static_context)
            u, contribs = self._fuse_history(encoder_features, static_context.embedding, record)
            out, layer_states, conv_states, caches = self.temporal.scan(
                u, state.layer_states, state.conv_states, record=record,
                static_embedding=static_context.embedding)
            new = StreamState(layer_states=layer_states, conv_states=conv_states,
                              last_output=out[:, -1], static_context=static_context,
                              step=state.step + u.shape[1])
            if record:
                self._stream_caches = caches
                self._stream_contribs = contribs
            return new

        def update_stream(self, state: StreamState, obs_features: Dict[str, torch.Tensor]
                          ) -> StreamState:
            """Advance one streaming timestep from a per-step feature dict
            ``{name: (B, size)}`` (observed history only)."""
            z, _ = self.encoder_fusion(obs_features, return_contributions=False)
            z = self.film(z, state.static_context.embedding)
            out_t, layer_states, conv_states = self.temporal.step(
                z, state.layer_states, state.conv_states,
                static_embedding=state.static_context.embedding)
            return StreamState(layer_states=layer_states, conv_states=conv_states,
                               last_output=out_t, static_context=state.static_context,
                               step=state.step + 1)

        def _decoder_inputs(self, x, dec_cat, dec_cont, B, H, device):
            if self.decoder_time_fusion is not None:
                tfeats = self._features(dec_cat, dec_cont, self._time_decoder_vars)
                T, _ = self.decoder_time_fusion(tfeats, return_contributions=False)
            else:
                T = dec_cont.new_zeros(B, H, 0)
            A = self._scenario_values(dec_cat, dec_cont)
            M = self._resolve_scenario_mask(x, B, H, device)
            return T, A, M

        def set_effect_adapter(self, adapter) -> None:
            """Attach (or clear with ``None``) a per-user :class:`ScenarioEffectAdapter`.

            The adapter calibrates ONLY Δŷ_scenario, so it requires a ``scenario_decompose``
            decoder. It is not part of the global optimizer unless explicitly fine-tuned —
            one user's calibration never mutates the shared base model."""
            if adapter is not None and not self.decoder.scenario_decompose:
                raise ValueError("effect adapter requires scenario_decompose=True "
                                 "(it modifies only Δŷ_scenario)")
            self.effect_adapter = adapter

        def decode_horizon(self, state: StreamState, static_context: StaticContext,
                           time_features, scenario_values, scenario_mask, *,
                           return_decomposition: bool = False):
            """Decode the forecast horizon from ``h_t`` + future known/scenario inputs.

            With ``return_decomposition`` (and a ``scenario_decompose`` decoder) returns
            ``(final, base, effect)`` raw-output triples for interpretability/eval. When a
            per-user effect adapter is attached, Δŷ_scenario is calibrated before recombining."""
            adapter = getattr(self, "effect_adapter", None)
            if adapter is None and not return_decomposition:
                return self.decoder(state.last_output, static_context.embedding,
                                    time_features, scenario_values, scenario_mask)
            final, base, effect = self.decoder(
                state.last_output, static_context.embedding,
                time_features, scenario_values, scenario_mask, return_decomposition=True)
            if adapter is not None:
                if isinstance(effect, (list, tuple)):
                    raise NotImplementedError("effect adapter supports single-target decoders only")
                effect = adapter(effect)
                final = base + effect
            return (final, base, effect) if return_decomposition else final

        # ==================================================================
        # forward (windowed training / inference)
        # ==================================================================
        def _prediction_output(self, pred, x, encoder_lengths, decoder_lengths):
            pred = self.transform_output(pred, target_scale=x["target_scale"])
            return self.to_network_output(prediction=pred, encoder_lengths=encoder_lengths,
                                          decoder_lengths=decoder_lengths)

        def forward(self, x: Dict[str, torch.Tensor], mode: str = "forecast"):
            enc_cat, dec_cat = x["encoder_cat"], x["decoder_cat"]
            enc_cont, dec_cont = x["encoder_cont"], x["decoder_cont"]
            B, H = dec_cont.shape[0], dec_cont.shape[1]
            device = enc_cont.device
            record = self._record_stream

            sctx = self.encode_static(x)
            enc_feats = self._features(enc_cat, enc_cont, self.encoder_variables)
            state = self.encode_history(enc_feats, sctx, record=record)
            if record:
                self._stream_history_len = enc_cont.shape[1]

            T, A, M = self._decoder_inputs(x, dec_cat, dec_cont, B, H, device)
            pred = self.decode_horizon(state, sctx, T, A, M)
            return self._prediction_output(pred, x, x["encoder_lengths"], x["decoder_lengths"])

        @torch.no_grad()
        def forward_streaming_equivalent(self, x: Dict[str, torch.Tensor]):
            """Same forecast as :meth:`forward`, but the history is consumed one step
            at a time via :meth:`update_stream` — the windowed-vs-streaming
            consistency check."""
            enc_cat, dec_cat = x["encoder_cat"], x["decoder_cat"]
            enc_cont, dec_cont = x["encoder_cont"], x["decoder_cont"]
            B, H, L = dec_cont.shape[0], dec_cont.shape[1], enc_cont.shape[1]
            device = enc_cont.device
            sctx = self.encode_static(x)
            state = self.init_stream(sctx)
            enc_feats = self._features(enc_cat, enc_cont, self.encoder_variables)
            for t in range(L):
                obs_t = {n: enc_feats[n][:, t] for n in self.encoder_variables}
                state = self.update_stream(state, obs_t)
            T, A, M = self._decoder_inputs(x, dec_cat, dec_cont, B, H, device)
            pred = self.decode_horizon(state, sctx, T, A, M)
            return self._prediction_output(pred, x, x["encoder_lengths"], x["decoder_lengths"])

        # ==================================================================
        # scenario-aware training (reuses the encoded history state)
        # ==================================================================
        def training_step(self, batch, batch_idx):
            mode = self._resolved_scenario_train_mode()
            if mode == "mixed" or not self.n_scenario_vars:
                return super().training_step(batch, batch_idx)
            x, y = batch
            target = y[0] if isinstance(y, (list, tuple)) else y

            # encode history ONCE, then decode under multiple scenario masks
            enc_cat, dec_cat = x["encoder_cat"], x["decoder_cat"]
            enc_cont, dec_cont = x["encoder_cont"], x["decoder_cont"]
            B, H, device = dec_cont.shape[0], dec_cont.shape[1], enc_cont.device
            sctx = self.encode_static(x)
            enc_feats = self._features(enc_cat, enc_cont, self.encoder_variables)
            state = self.encode_history(enc_feats, sctx)
            T, A, _ = self._decoder_inputs(x, dec_cat, dec_cont, B, H, device)

            def _decode(M):
                pred = self.decode_horizon(state, sctx, T, A, M)
                return self.transform_output(pred, target_scale=x["target_scale"])

            zeros = torch.zeros(B, H, self.n_scenario_vars, device=device, dtype=self.dtype)
            ones = zeros + 1.0
            if mode == "two_pass":
                loss = (self.loss(_decode(zeros), target)
                        + self.scenario_lambda * self.loss(_decode(ones), target))
            elif mode == "mixed+factual":
                from ..scenario import sample_scenario_dropout_mask
                M = sample_scenario_dropout_mask(B, H, self.n_scenario_vars,
                                                 self.hparams.scenario_dropout_mode,
                                                 self.hparams.scenario_dropout_p, device).to(self.dtype)
                loss = (self.loss(_decode(M), target)
                        + self.scenario_lambda * self.loss(_decode(ones), target))
            else:  # pragma: no cover
                raise ValueError(mode)
            if getattr(self, "_trainer", None) is not None:
                self.log("train_loss", loss, prog_bar=True, batch_size=target.size(0))
            return loss

        # ==================================================================
        # interpretability + attribution
        # ==================================================================
        def enable_attention_recording(self, on: bool = True):
            self._record_stream = bool(on)
            if not on:
                self._stream_caches = None
                self._stream_contribs = None

        @torch.no_grad()
        def attribute_window(self, **kwargs):
            """Offline MES feature-time attribution from the last recorded forward.

            Run a forward after ``enable_attention_recording(True)``; this consumes
            the cached per-layer scan parameters + grouped-fusion contributions.
            """
            if self._stream_caches is None or self._stream_contribs is None:
                raise RuntimeError(
                    "no recorded stream cache; call enable_attention_recording(True) then a forward.")
            return attribute_window(self._stream_caches, self._stream_contribs,
                                    ngroups=self.temporal.blocks[0].ssm.ngroups, **kwargs)

        @torch.no_grad()
        def interpret_output(self, out, reduction: str = "none", **kwargs):
            """Variable importance from grouped-fusion contribution magnitudes
            (requires a recorded forward; falls back to empty if none)."""
            if self._stream_contribs is None:
                return {"encoder_variables": None, "encoder_variable_names": self.encoder_variables}
            imp = {n: c.norm(dim=-1).mean(dim=(0, 1)) for n, c in self._stream_contribs.items()}
            return {"encoder_variables": imp, "encoder_variable_names": list(imp)}

    SSMCGMStreamModel = SSMCGMStream

else:  # pragma: no cover

    class SSMCGMStream:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("SSMCGMStream requires pytorch_forecasting.")
