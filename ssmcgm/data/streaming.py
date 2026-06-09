"""Production-style chronological participant streams for SSM-CGM-Stream.

This module is the *deployment* data path. Where :mod:`ssmcgm.data.t1dexi` produces
overlapping fixed-length :class:`~pytorch_forecasting.TimeSeriesDataSet` **windows**
(good for windowed training/debug), here we produce, for each participant, **one
chronologically-ordered stream** of the model's already-scaled inputs that the
streaming API (:meth:`SSMCGMStream.init_stream` / ``update_stream`` /
``decode_horizon``) consumes step-by-step — never rebuilding a 24h/48h window at an
anchor (spec §4, §10, §21).

How the scaled stream is built (leakage-safe by construction)
-------------------------------------------------------------
The model consumes *pytorch-forecasting-scaled* covariates. To get the full scaled
sequence for a participant **without re-fitting any scaler**, we reuse the
*train-fitted* :class:`TimeSeriesDataSet` and ask it for a single **maximal window**
whose encoder spans ``[0 .. T-H-1]`` and decoder spans ``[T-H .. T-1]``. Concatenating
``encoder_cont``/``decoder_cont`` (and the cats) along time reconstructs the whole
scaled timeline ``(T, n_cont)`` / ``(T, n_cat)``. Because the fitted dataset's
``scalers`` / ``categorical_encoders`` / ``target_normalizer`` are reused, a
held-out participant is scaled with **train-only** statistics (GroupNormalizer falls
back to its global parameters for an unseen group) — spec §6.6/§6.7.

``relative_time_idx`` handling
------------------------------
``relative_time_idx`` is the only *window-relative* time-varying covariate (the
``encoder_length`` / ``*_center`` / ``*_scale`` reals are static constants per
participant). Its scaled values are window-anchor-relative, so we read off the
**canonical** scaled encodings once: the decoder slice ``full_cont[T-H:T, ridx]`` is
the scaled ``[1..H]`` horizon ramp (identical at every anchor), and
``full_cont[T-H-1, ridx]`` is the scaled "now" value. At stream time we overwrite the
``relative_time_idx`` column with these canonical values so every observed step looks
like "the present" (``rel_now``) and every decoded horizon uses the same ``[1..H]``
ramp the decoder was trained on (``rel_policy="anchor_now"``, default). Pass
``rel_policy="as_is"`` to keep the maximal-window values (used by the windowed
consistency diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .t1dexi import GROUP_ID, TARGET, TIME_IDX, _fill_missing, get_t1dexi_feature_schema

SPLITS = ("train", "validation", "test")


# ===========================================================================
# feature spec — names + the relative_time_idx column index + cadence
# ===========================================================================
@dataclass
class StreamFeatureSpec:
    """Static, model/dataset-level description needed to drive a stream.

    Captures the covariate column ordering (so a ``(T, n_cont)`` / ``(T, n_cat)``
    tensor lines up with ``model.hparams.x_reals`` / ``x_categoricals``), the forecast
    horizon, the sampling cadence (minutes per step), and the index of
    ``relative_time_idx`` in ``x_reals`` (``None`` if the dataset has none).
    """

    x_reals: List[str]
    x_categoricals: List[str]
    horizon: int
    bin_minutes: int = 5
    rel_time_idx_col: Optional[int] = None
    target_name: str = TARGET

    @property
    def n_reals(self) -> int:
        return len(self.x_reals)

    @property
    def n_categoricals(self) -> int:
        return len(self.x_categoricals)

    def horizon_minutes(self) -> List[int]:
        return [self.bin_minutes * (h + 1) for h in range(self.horizon)]


def build_stream_feature_spec(
    *,
    dataset=None,
    model=None,
    horizon: Optional[int] = None,
    bin_minutes: int = 5,
    target_name: str = TARGET,
) -> StreamFeatureSpec:
    """Build a :class:`StreamFeatureSpec` from a fitted ``TimeSeriesDataSet`` or model.

    Either ``dataset`` (a ``TimeSeriesDataSet``) or ``model`` (an ``SSMCGMStream``)
    must be given; both expose the same ``x_reals`` / ``x_categoricals`` ordering.
    """
    if dataset is not None:
        x_reals = list(dataset.reals)
        x_categoricals = list(dataset.flat_categoricals)
        hor = horizon if horizon is not None else int(dataset.max_prediction_length)
    elif model is not None:
        x_reals = list(model.hparams.x_reals)
        x_categoricals = list(model.hparams.x_categoricals)
        hor = horizon if horizon is not None else int(model.hparams.max_prediction_length)
    else:
        raise ValueError("build_stream_feature_spec needs `dataset` or `model`")
    rel_col = x_reals.index("relative_time_idx") if "relative_time_idx" in x_reals else None
    return StreamFeatureSpec(
        x_reals=x_reals, x_categoricals=x_categoricals, horizon=int(hor),
        bin_minutes=int(bin_minutes), rel_time_idx_col=rel_col, target_name=target_name,
    )


# ===========================================================================
# participant stream container
# ===========================================================================
@dataclass
class ParticipantStream:
    """One participant's chronologically-ordered, model-ready stream.

    ``full_cont`` / ``full_cat`` are the *scaled / encoded* covariate sequences
    ``(T, n_cont)`` / ``(T, n_cat)`` aligned with ``spec.x_reals`` / ``x_categoricals``.
    ``target`` is the **raw** glucose (mg/dL) per step; ``target_observed`` flags
    steps with a genuine reading (forecasts are scored only against observed future
    targets). ``target_scale`` is the per-participant ``(center, scale)`` used to map
    model output back to mg/dL.

    ``anchor_split`` (optional, length ``T``) tags each step's split for the
    chronological-by-participant mode; when ``None`` the whole stream uses ``split``.
    ``rel_decoder`` / ``rel_now`` are the canonical scaled ``relative_time_idx``
    encodings (see module docstring).
    """

    participant_id: str
    split: str
    full_cont: torch.Tensor        # (T, n_cont) scaled
    full_cat: torch.Tensor         # (T, n_cat)  encoded
    target: torch.Tensor           # (T,) raw mg/dL
    target_observed: torch.Tensor  # (T,) bool
    target_scale: torch.Tensor     # (2,)
    time_idx: torch.Tensor         # (T,) int
    rel_decoder: Optional[torch.Tensor] = None   # (H,) scaled [1..H]
    rel_now: Optional[float] = None              # scaled "now"
    anchor: Optional[torch.Tensor] = None        # (T,) current KNOWN glucose mg/dL (causal)
    timestamps: Optional[np.ndarray] = None
    segment_id: Optional[torch.Tensor] = None    # (T,) int or None
    anchor_split: Optional[np.ndarray] = None     # (T,) str or None
    metadata: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.full_cont.shape[0])

    def to(self, device) -> "ParticipantStream":
        mv = lambda t: None if t is None else t.to(device)
        return ParticipantStream(
            participant_id=self.participant_id, split=self.split,
            full_cont=self.full_cont.to(device), full_cat=self.full_cat.to(device),
            target=self.target.to(device), target_observed=self.target_observed.to(device),
            target_scale=self.target_scale.to(device), time_idx=self.time_idx.to(device),
            rel_decoder=mv(self.rel_decoder), rel_now=self.rel_now, anchor=mv(self.anchor),
            timestamps=self.timestamps, segment_id=mv(self.segment_id),
            anchor_split=self.anchor_split, metadata=dict(self.metadata),
        )

    def split_at(self, t: int) -> str:
        """Split label governing the *anchor* at step ``t``."""
        if self.anchor_split is not None:
            return str(self.anchor_split[t])
        return self.split


# ===========================================================================
# splits
# ===========================================================================
@dataclass
class StreamSplit:
    """Resolved split plan for a set of participants.

    ``participant_split`` maps participant id -> ``train|validation|test`` (used by
    ``participant_heldout`` and ``existing_panel_split`` when subject-held-out).
    ``mode`` records how it was produced; ``classification`` records, for an existing
    panel split, whether participants overlap across splits (``chronological``) or not
    (``participant_heldout``).
    """

    mode: str
    participant_split: Dict[str, str] = field(default_factory=dict)
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42
    classification: str = ""

    def participants(self, split: str) -> List[str]:
        return sorted(p for p, s in self.participant_split.items() if s == split)


def _stratify_lookup(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    return df.groupby(GROUP_ID)[col].first().astype(str)


def make_stream_splits(
    df: pd.DataFrame,
    *,
    split_mode: str = "participant_heldout",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> StreamSplit:
    """Assign whole participants to train/val/test at the **stream** level.

    * ``participant_heldout`` (default, the primary production claim): each
      participant goes to exactly one split, optionally stratified by ``stratify_col``.
    * ``chronological_by_participant``: every participant is in all three splits but
      sliced by time — assignment happens at stream-build time, so this returns an
      empty ``participant_split`` (mode is recorded for the builder).
    * ``existing_panel_split``: read the panel's ``split`` column; classify whether it
      is subject-held-out or chronological and, if subject-held-out, adopt its
      participant -> split mapping.
    """
    fr = (float(train_fraction), float(val_fraction), float(test_fraction))
    if split_mode == "chronological_by_participant":
        return StreamSplit(mode=split_mode, fractions=fr, seed=seed)

    if split_mode == "existing_panel_split":
        if "split" not in df.columns:
            raise ValueError("existing_panel_split requested but the panel has no `split` column")
        per_part = df.groupby(GROUP_ID)["split"].agg(lambda s: set(s.dropna().unique()))
        overlap = any(len(s & {"train", "validation", "test"}) > 1 for s in per_part)
        classification = "chronological" if overlap else "participant_heldout"
        # adopt the dominant split per participant (works for either; for chronological
        # this collapses to the participant's most frequent split and is logged).
        dominant = (df[df["split"].isin(SPLITS)]
                    .groupby(GROUP_ID)["split"].agg(lambda s: s.value_counts().idxmax()))
        mapping = {str(k): str(v) for k, v in dominant.items()}
        return StreamSplit(mode=split_mode, participant_split=mapping, fractions=fr,
                           seed=seed, classification=classification)

    if split_mode != "participant_heldout":
        raise ValueError(f"unknown split_mode {split_mode!r}")

    rng = np.random.default_rng(seed)
    strata = _stratify_lookup(df, stratify_col)
    # if the panel marks quality-excluded rows (split == "excluded"), only assign
    # participants that have at least one usable row — never score deliberately-excluded
    # data (mirrors existing_panel_split).
    if "split" in df.columns:
        eligible = df[df["split"].isin(SPLITS)][GROUP_ID].astype(str).unique()
    else:
        eligible = df[GROUP_ID].astype(str).unique()
    participants = sorted({str(p) for p in eligible})
    if strata is None:
        groups = [(None, participants)]
    else:
        strata = strata.astype(str)
        groups = [(name, [p for p in participants if strata.get(p, "__unknown__") == name])
                  for name in sorted(set(strata.values))]
    mapping: Dict[str, str] = {}
    for _name, ids in groups:
        ids = list(rng.permutation(np.array(ids, dtype=object)))
        n = len(ids)
        if n == 0:
            continue
        n_test = max(1, round(n * fr[2])) if n >= 3 else 0
        n_val = max(1, round(n * fr[1])) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = max(0, n - 2)
            n_val = max(0, n - 1 - n_test)
        for p in ids[:n_test]:
            mapping[str(p)] = "test"
        for p in ids[n_test:n_test + n_val]:
            mapping[str(p)] = "validation"
        for p in ids[n_test + n_val:]:
            mapping[str(p)] = "train"
    return StreamSplit(mode=split_mode, participant_split=mapping, fractions=fr, seed=seed)


def split_assignments_frame(split: StreamSplit) -> pd.DataFrame:
    """``participant_id, split`` table (spec §7.1 ``split_assignments.csv``)."""
    rows = [{"participant_id": p, "split": s} for p, s in sorted(split.participant_split.items())]
    return pd.DataFrame(rows, columns=["participant_id", "split"])


def assert_split_integrity(split: StreamSplit) -> None:
    """Raise if any participant appears in more than one split (spec §17.1)."""
    seen: Dict[str, str] = {}
    for pid, s in split.participant_split.items():
        if pid in seen and seen[pid] != s:
            raise AssertionError(f"participant {pid} in two splits: {seen[pid]} and {s}")
        seen[pid] = s


def _chronological_anchor_split(
    n: int, fractions: Tuple[float, float, float]
) -> np.ndarray:
    """Per-step split labels: earliest ``train``, middle ``validation``, latest ``test``."""
    tr, va, _te = fractions
    i_tr = int(round(n * tr))
    i_va = int(round(n * (tr + va)))
    labels = np.empty(n, dtype=object)
    labels[:i_tr] = "train"
    labels[i_tr:i_va] = "validation"
    labels[i_va:] = "test"
    return labels


# ===========================================================================
# building scaled streams from the fitted dataset
# ===========================================================================
def _maximal_window_tensors(training, pdf: pd.DataFrame, horizon: int):
    """Encode one participant's whole timeline as a single maximal scaled window.

    Returns ``(full_cont (T,n_cont), full_cat (T,n_cat), target_scale (2,))`` or
    ``None`` if the participant is too short (``len <= horizon``).
    """
    from pytorch_forecasting import TimeSeriesDataSet

    n = len(pdf)
    if n <= horizon:
        return None
    # the maximal-window trick needs a contiguous time_idx (the training dataset was
    # fitted with allow_missing_timesteps=False); skip gappy participants gracefully
    # rather than aborting the whole build.
    if int(pd.to_numeric(pdf[TIME_IDX]).diff().fillna(1).max()) > 1:
        return None
    enc_len = n - horizon
    try:
        one = TimeSeriesDataSet.from_dataset(
            training, pdf, predict=True, stop_randomization=True,
            max_encoder_length=enc_len, min_encoder_length=enc_len,
            max_prediction_length=horizon, min_prediction_length=horizon,
        )
        dl = one.to_dataloader(train=False, batch_size=1, num_workers=0)
        x, _ = next(iter(dl))
    except (AssertionError, ValueError):
        return None
    full_cont = torch.cat([x["encoder_cont"][0], x["decoder_cont"][0]], dim=0)
    full_cat = torch.cat([x["encoder_cat"][0], x["decoder_cat"][0]], dim=0)
    target_scale = x["target_scale"][0].detach().clone()
    if full_cont.shape[0] != n:  # defensive: gaps / missing timesteps broke contiguity
        return None
    return full_cont.detach().clone(), full_cat.detach().clone(), target_scale


def _build_one_stream(
    training, pdf: pd.DataFrame, spec: StreamFeatureSpec, split: str,
    *, observed_col: Optional[str], segment_col: Optional[str],
    timestamp_col: Optional[str], anchor_col: Optional[str], anchor_split: Optional[np.ndarray],
) -> Optional[ParticipantStream]:
    pdf = pdf.sort_values(TIME_IDX).reset_index(drop=True)
    pid = str(pdf[GROUP_ID].iloc[0])
    enc = _maximal_window_tensors(training, pdf, spec.horizon)
    if enc is None:
        return None
    full_cont, full_cat, target_scale = enc
    n = full_cont.shape[0]

    target = torch.tensor(pdf[spec.target_name].to_numpy(dtype="float32"))
    if observed_col and observed_col in pdf.columns:
        # explicit >0.5 (not .bool(), which maps NaN/fractional -> True) and AND with a
        # finite target, so a step is "observed" only with a real reading.
        flag = np.nan_to_num(pdf[observed_col].to_numpy(dtype="float64"), nan=0.0) > 0.5
        obs = torch.tensor(flag) & torch.isfinite(target)
    else:
        obs = torch.isfinite(target)
    # anchor = current KNOWN glucose at each step (causal: the model-input CGM, which is
    # ffill/interp-imputed up to t), used by residual_current target transform. Falls back
    # to a forward-filled target. This is available at forecast time -> leakage-safe.
    if anchor_col and anchor_col in pdf.columns:
        anchor_np = pd.to_numeric(pdf[anchor_col], errors="coerce").ffill().bfill().to_numpy(dtype="float32")
    else:
        anchor_np = pd.Series(target.numpy()).ffill().bfill().to_numpy(dtype="float32")
    anchor = torch.tensor(anchor_np)
    time_idx = torch.tensor(pdf[TIME_IDX].to_numpy(dtype="int64"))
    seg = (torch.tensor(pd.to_numeric(pdf[segment_col], errors="coerce")
                        .fillna(-1).to_numpy(dtype="int64"))
           if segment_col and segment_col in pdf.columns else None)
    ts = pdf[timestamp_col].to_numpy() if timestamp_col and timestamp_col in pdf.columns else None

    rel_decoder = rel_now = None
    if spec.rel_time_idx_col is not None:
        ridx = spec.rel_time_idx_col
        rel_decoder = full_cont[n - spec.horizon:n, ridx].detach().clone()  # scaled [1..H]
        rel_now = float(full_cont[n - spec.horizon - 1, ridx])              # scaled "now"

    return ParticipantStream(
        participant_id=pid, split=split, full_cont=full_cont, full_cat=full_cat,
        target=target, target_observed=obs, target_scale=target_scale, time_idx=time_idx,
        rel_decoder=rel_decoder, rel_now=rel_now, anchor=anchor, timestamps=ts, segment_id=seg,
        anchor_split=anchor_split,
        metadata={"n_observed_target": int(obs.sum())},
    )


def make_participant_streams(
    df: pd.DataFrame,
    training,
    split: StreamSplit,
    *,
    spec: Optional[StreamFeatureSpec] = None,
    splits: Sequence[str] = SPLITS,
    observed_col: str = "glucose_model_input_is_observed",
    segment_col: str = "cgm_segment_id",
    timestamp_col: str = "timestamp",
    anchor_col: str = "glucose_model_input_mg_dl",
    fill_missing: bool = True,
    max_participants: Optional[int] = None,
) -> List[ParticipantStream]:
    """Build one (or, for chronological mode, one full-timeline) stream per participant.

    ``df`` is the loaded panel (see :func:`ssmcgm.data.t1dexi.load_t1dexi_data`);
    ``training`` is the **train-fitted** ``TimeSeriesDataSet`` whose scalers are reused.
    ``split`` selects which participants/streams to build. Streams are returned only
    for participants assigned to one of ``splits``.
    """
    if spec is None:
        spec = build_stream_feature_spec(dataset=training)
    # fill columns from the fitted dataset's *raw* covariate roles (works for the
    # T1DEXI schema and for any small synthetic schema); PTF-derived reals such as
    # relative_time_idx / encoder_length / *_scale are absent from ``df`` and are
    # skipped by ``_fill_missing``'s presence guard.
    _role = lambda name: list(getattr(training, name, None) or [])
    real_cols = (_role("static_reals") + _role("time_varying_known_reals")
                 + _role("time_varying_unknown_reals"))
    cat_cols = (_role("time_varying_known_categoricals")
                + _role("time_varying_unknown_categoricals"))
    if not real_cols and not cat_cols:  # fallback to the T1DEXI schema
        (_sc, static_reals, tv_known_reals, tv_known_cats,
         _tuc, tv_unk_reals) = get_t1dexi_feature_schema()
        real_cols = static_reals + tv_known_reals + tv_unk_reals
        cat_cols = tv_known_cats

    streams: List[ParticipantStream] = []
    n_built = 0
    for pid, pdf in df.groupby(GROUP_ID, sort=True):
        pid = str(pid)
        if fill_missing:
            pdf = pdf.copy()
            _fill_missing(pdf, real_cols, cat_cols)

        if split.mode == "chronological_by_participant":
            anchor = _chronological_anchor_split(len(pdf), split.fractions)
            if not (set(anchor) & set(splits)):
                continue
            s = _build_one_stream(training, pdf, spec, split="multi",
                                  observed_col=observed_col, segment_col=segment_col,
                                  timestamp_col=timestamp_col, anchor_col=anchor_col, anchor_split=anchor)
        else:
            assigned = split.participant_split.get(pid)
            if assigned not in splits:
                continue
            s = _build_one_stream(training, pdf, spec, split=assigned,
                                  observed_col=observed_col, segment_col=segment_col,
                                  timestamp_col=timestamp_col, anchor_col=anchor_col, anchor_split=None)
        if s is not None:
            streams.append(s)
            n_built += 1
            if max_participants is not None and n_built >= max_participants:
                break
    return streams


def iter_participant_streams(streams: Sequence[ParticipantStream], split: Optional[str] = None
                             ) -> Iterator[ParticipantStream]:
    """Iterate streams, optionally filtering to a single participant-level ``split``."""
    for s in streams:
        if split is None or s.split == split or s.anchor_split is not None:
            yield s
