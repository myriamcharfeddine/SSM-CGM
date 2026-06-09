"""State containers for the streamable SSM-CGM model.

Two small dataclasses carry everything the streaming API needs between calls:

* :class:`StaticContext` — the once-per-participant static encoding ``e_s`` plus
  the raw static tensors (so a static-profile *replay* can re-encode), computed by
  :meth:`SSMCGMStream.encode_static`.
* :class:`StreamState` — the per-layer recurrent SSM state and causal-conv state,
  the last top-layer output ``h_t`` (the decoder reads this), and the step
  counter. Produced by :meth:`init_stream` and advanced by ``encode_history`` /
  ``update_stream``.

The shapes deliberately match the pure-PyTorch streaming scan in
:mod:`ssmcgm.stream.ssm` so the *same* state object flows through both the
batched ``scan`` (training) and the single-step ``step`` (deployment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class StaticContext:
    """Per-participant static encoding, computed once and reused over all steps.

    Attributes:
        embedding: ``e_s`` static embedding, shape ``(B, hidden)``.
        raw_static_cat: the static categorical slice ``(B, n_static_cat)`` (kept
            so a static-profile *state replay* can re-encode under an edit).
        raw_static_cont: the static continuous slice ``(B, n_static_cont)``.
        metadata: free-form dict for provenance (e.g. edited variables).
    """

    embedding: torch.Tensor
    raw_static_cat: Optional[torch.Tensor] = None
    raw_static_cont: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.embedding.shape[0])

    def to(self, device) -> "StaticContext":
        return StaticContext(
            embedding=self.embedding.to(device),
            raw_static_cat=None if self.raw_static_cat is None else self.raw_static_cat.to(device),
            raw_static_cont=None if self.raw_static_cont is None else self.raw_static_cont.to(device),
            metadata=dict(self.metadata),
        )


@dataclass
class StreamState:
    """Recurrent state of the streaming MES stack.

    ``layer_states[ℓ]`` is the SSM hidden state of layer ``ℓ`` with shape
    ``(B, H, P, N)`` (heads × head-dim × state). ``conv_states[ℓ]`` is the causal
    depthwise-conv rolling buffer ``(B, C_conv, d_conv-1)`` (or ``None`` if the
    block has no conv). ``last_output`` is the top-layer output at the current
    step, ``(B, d_model)`` — this is ``h_t``, the only thing the horizon decoder
    consumes from history.
    """

    layer_states: List[torch.Tensor]
    conv_states: Optional[List[Optional[torch.Tensor]]] = None
    last_output: Optional[torch.Tensor] = None
    static_context: Optional[StaticContext] = None
    step: int = 0

    @property
    def depth(self) -> int:
        return len(self.layer_states)

    def detach(self) -> "StreamState":
        """Detach every tensor (for streaming inference loops, to free graph)."""
        return StreamState(
            layer_states=[s.detach() for s in self.layer_states],
            conv_states=None if self.conv_states is None
            else [None if c is None else c.detach() for c in self.conv_states],
            last_output=None if self.last_output is None else self.last_output.detach(),
            static_context=self.static_context,
            step=self.step,
        )

    def clone(self) -> "StreamState":
        """Deep-copy the tensors (so a streaming counterfactual can branch)."""
        return StreamState(
            layer_states=[s.clone() for s in self.layer_states],
            conv_states=None if self.conv_states is None
            else [None if c is None else c.clone() for c in self.conv_states],
            last_output=None if self.last_output is None else self.last_output.clone(),
            static_context=self.static_context,
            step=self.step,
        )
