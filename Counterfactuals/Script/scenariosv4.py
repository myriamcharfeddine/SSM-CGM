from typing import Dict, List, Tuple, Literal, Optional, Iterable, Set
from dataclasses import dataclass, asdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet
# Add tqdm progress bar with a safe fallback
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback if tqdm not installed
    def tqdm(iterable=None, **kwargs):
        return iterable

# -----------------------
# Default Config (used by main())
# -----------------------
# Participant & continuous intervention
target_id: str = "1032"                 # participant to analyze
covariate_name: Optional[str] = "respiration_rate"  # CONTINUOUS covariate to intervene on (decoder_cont, z-scored)
counterfact_std: Optional[float] = 2.0   # how many standard deviations to shift (z on the individual's scale)

# Categorical intervention (decoder_cat) — e.g., sleep stage
apply_categorical_cf: bool = True        # set False to skip categorical counterfactuals
cat_cov_name: Optional[str] = "sleep_stage"   # name as used in your dataset categoricals
cat_force_label: Optional[str] = "deep"       # category label to force over decoder horizon (e.g., "deep")

# Plot/aggregation knobs
k_to_plot: int = 0                       # which window index to plot in the per-window figure
aggregation: Literal['mean', 'end_horizon', 'auc'] = 'mean'  # aggregation for Δ series
LOW_TIR: float = 70.0                    # mg/dL lower bound for TIR
HIGH_TIR: float = 180.0                  # mg/dL upper bound for TIR
delta_t_minutes: int = 5                 # minutes per forecast step (dx for AUC and x-axes)
use_raw_value: bool = False
counterfact_value_raw: Optional[float] = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _extract_prediction_tensor(out) -> torch.Tensor:
    """Return a [B, H, Q] float tensor from various TFT output shapes.
    Handles:
      - out with attribute `.prediction` ([B,H,Q])
      - tuple where first element is the prediction
      - raw tensor already shaped [B,H,Q]
    """
    if hasattr(out, "prediction"):
        pred = out.prediction
    elif isinstance(out, (tuple, list)) and len(out) > 0:
        pred = out[0]
    else:
        pred = out
    return pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)


def compute_stats(train_df, participant_id: str, covar: str, z: float) -> Tuple[float, float, float, float, float]:
    """Compute individual & population μ, σ on RAW data, and population +zσ raw value.
    Returns (mu_i, sigma_i, mu_p, sigma_p, raw_p).
    """
    sub = train_df[train_df["participant_id"] == participant_id]
    mu_i = sub[covar].mean()
    sigma_i = sub[covar].std()
    mu_p = train_df[covar].mean()
    sigma_p = train_df[covar].std()
    raw_p = mu_p + z * sigma_p
    return float(mu_i), float(sigma_i), float(mu_p), float(sigma_p), float(raw_p)


@dataclass
class BaselineSpec:
    mode: Literal[
        "planned",          # use planned future covariates (default)
        "cont_fixed_z",     # set decoder_cont[..., cont_idx] = z_value
        "cont_fixed_raw",   # set decoder_cont to raw value mapped to individual's z
        "cat_force_label",  # set decoder_cat[..., cat_idx] = encoded label
    ] = "planned"
    # continuous options
    z_value: Optional[float] = None
    raw_value: Optional[float] = None
    # categorical options
    cat_label: Optional[str] = None


def _make_baseline_variant(
    x: Dict[str, torch.Tensor],
    *,
    spec: Optional[BaselineSpec],
    cont_idx: Optional[int],
    cat_idx: Optional[int],
    cat_encoder,
    mu_i: float,
    sigma_i: float,
):
    """Return x modified according to the baseline spec. If spec is None or planned, return x."""
    if spec is None or spec.mode == "planned":
        return x

    x2 = x.copy()

    if spec.mode == "cont_fixed_z":
        if cont_idx is None or spec.z_value is None:
            raise ValueError("cont_fixed_z baseline needs cont_idx and z_value")
        t = x2["decoder_cont"].clone()
        t[..., cont_idx] = float(spec.z_value)
        x2["decoder_cont"] = t
        return x2

    if spec.mode == "cont_fixed_raw":
        if cont_idx is None or spec.raw_value is None:
            raise ValueError("cont_fixed_raw baseline needs cont_idx and raw_value")
        if not np.isfinite(sigma_i) or sigma_i == 0:
            raise ValueError("sigma_i must be finite and non-zero to map raw to z")
        z_indiv = (float(spec.raw_value) - float(mu_i)) / float(sigma_i)
        t = x2["decoder_cont"].clone()
        t[..., cont_idx] = float(z_indiv)
        x2["decoder_cont"] = t
        return x2

    if spec.mode == "cat_force_label":
        if cat_idx is None or cat_encoder is None or spec.cat_label is None:
            raise ValueError("cat_force_label baseline needs cat_idx, encoder, and cat_label")
        code = int(cat_encoder.transform([spec.cat_label])[0])
        t = x2["decoder_cat"].clone()
        t[..., cat_idx] = code
        x2["decoder_cat"] = t
        return x2

    raise ValueError(f"Unknown baseline mode: {spec.mode}")


def build_full_dataloader(
    training_ds: TimeSeriesDataSet,
    sub_df,
    *,
    covariate_name: Optional[str],
    apply_categorical_cf: bool,
    cat_cov_name: Optional[str],
    batch_size: int = 128, #32,           # ↑ push this until you hit OOM
    num_workers: int = 4,           # 4–8 is a good start
    pin_memory: bool = True, #False,       # set True when device is CUDA
    prefetch_factor: int = 2,       # batches prefetched per worker
    persistent_workers: Optional[bool] = None,  # keep workers alive
):
    """
    Build a non-randomized dataset/dataloader that iterates all windows for this participant.
    Returns: (full_ds, dataloader, cont_idx, cat_idx, cat_encoder)
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # Deterministic, full traversal of windows for this participant
    full_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        sub_df,
        stop_randomization=True,
        predict=False,
    )

    # NOTE: prefetch_factor is only used when num_workers > 0
    dl = full_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        # shuffle=False is implied by train=False in pytorch-forecasting
        # drop_last=False keeps all windows
    )

    # --- Indices/encoders (fail fast with clear errors) ---
    cont_idx = None
    if covariate_name is not None:
        try:
            cont_idx = full_ds.reals.index(covariate_name)
        except ValueError:
            raise KeyError(
                f"Continuous covariate '{covariate_name}' not found. Available: {full_ds.reals}"
            )

    cat_idx = None
    cat_encoder = None
    if apply_categorical_cf and cat_cov_name is not None:
        try:
            cat_idx = full_ds.categoricals.index(cat_cov_name)
        except ValueError:
            raise KeyError(
                f"Categorical covariate '{cat_cov_name}' not found. Available: {full_ds.categoricals}"
            )
        cat_encoder = full_ds.categorical_encoders[cat_cov_name]

    return full_ds, dl, cont_idx, cat_idx, cat_encoder


import contextlib, math, torch
from typing import Dict, List, Optional

def _autocast_ctx(device, prefer_bf16=True, enable_amp=True):
    if device.type != "cuda" or not enable_amp:
        return contextlib.nullcontext()
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.float16)

def _forward_all(tft, x_cat, amp_ctx):
    # one forward under AMP
    with amp_ctx:
        out = tft(x_cat)
        pred = _extract_prediction_tensor(out)
    # NaN guard: retry without AMP
    if not torch.isfinite(pred).all():
        with torch.cuda.amp.autocast(enabled=False):
            out = tft(x_cat)
            pred = _extract_prediction_tensor(out)
    return pred

def _clone_with_modified(x, key, col_idx, value):
    x2 = x.copy()
    t = x2[key].clone()
    t[..., col_idx] = value
    x2[key] = t
    return x2

def _concat_variants(variants):
    keys = variants[0].keys()
    return {k: torch.cat([v[k] for v in variants], dim=0) for k in keys}

def _median_slice(t: torch.Tensor) -> torch.Tensor:
    # Use Q=1 or Q>=2 median consistently
    return t[..., 1] if (t.ndim >= 3 and t.size(-1) > 1) else t

def run_counterfactual_windows(
    tft,
    train_dl,
    cont_idx: Optional[int],
    z_indiv: float,
    z_pop_to_indiv: Optional[float],
    device: torch.device,
    cat_idx: Optional[int] = None,
    cat_code: Optional[int] = None,
    show_progress: bool = True,
    keep_windows: bool = False, #True, Not sure if xs is needed for future computations?
    *,
    baseline_spec: Optional[BaselineSpec] = None,
    baseline_stats: Optional[Tuple[float, float]] = None,  # (mu_i, sigma_i)
    cat_encoder=None,
    timeidx_to_minofday: Optional[Dict[int, int]] = None,
    valid_minutes: Optional[set] = None,
    print_drop_summary: bool = True,
):
    tft.eval()

    xs: List[Dict[str, torch.Tensor]] = []
    base_preds: List[torch.Tensor] = []
    cf_i_preds: List[torch.Tensor] = []
    cf_pi_preds: List[torch.Tensor] = []
    cf_cat_preds: List[torch.Tensor] = []

    iterator = train_dl
    if show_progress:
        try:
            total = len(train_dl)
        except Exception:
            total = None
        iterator = tqdm(iterator, total=total, desc="Computing counterfactual windows", leave=False)

    amp_ctx = _autocast_ctx(device, prefer_bf16=True, enable_amp=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    need_cf_i  = cont_idx is not None
    need_cf_pi = (cont_idx is not None) and (z_pop_to_indiv is not None and math.isfinite(z_pop_to_indiv))
    need_cf_cat = (cat_idx is not None) and (cat_code is not None)

    mu_i_b, sigma_i_b = (baseline_stats or (0.0, 1.0))

    # Initialize counters
    dropped_windows = 0
    total_windows = 0

    with torch.inference_mode():
        for x, _y in iterator:
            # robust batch size
            x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
            if "encoder_lengths" in x:
                B = x["encoder_lengths"].shape[0]
            else:
                B = next(v.shape[0] for v in x.values() if v.ndim > 0)

            total_windows += B 
            # ===================== FILTER BY minute_of_day via decoder_time_idx ======================
            # knobs
            MATCH_MODE = "all"   # "all" (strict), "any" (lenient), or "first" (current behavior)

            if (valid_minutes is not None) and (timeidx_to_minofday is not None) and ("decoder_time_idx" in x):
                #print("decoder_time_idx sample:", x["decoder_time_idx"][:3, :5].detach().cpu().numpy())
                #sample_ti = int(x["decoder_time_idx"][0,0].item())
                #print("mapped minute_of_day:", timeidx_to_minofday.get(sample_ti, None))

                # x["decoder_time_idx"]: [B, H]
                dec_ti = x["decoder_time_idx"].detach().cpu().numpy().astype(int)  # shape [B, H]

                # map each time_idx -> minute_of_day; unknown -> -1
                # result: mod_mat shape [B, H]
                mod_mat = np.vectorize(lambda ti: timeidx_to_minofday.get(int(ti), -1))(dec_ti)

                # (Optional) sanity checks
                # assert np.all((mod_mat == -1) | ((mod_mat % 5) == 0)), "minute_of_day not on 5-min grid?"

                # membership in valid set
                in_set = np.vectorize(lambda m: (m >= 0) and (m in valid_minutes))(mod_mat)  # [B, H] bool

                if MATCH_MODE == "all":
                    keep_mask = in_set.all(axis=1)
                elif MATCH_MODE == "any":
                    keep_mask = in_set.any(axis=1)
                else:  # "first"
                    keep_mask = in_set[:, 0]

                # tiny debug (first batch only)
                if dropped_windows == 0:  # crude way to print once early
                    tot = in_set.shape[0]
                    # print(f"[filter] mode={MATCH_MODE}, B={tot}, "
                    #     f"rows all-in={keep_mask.sum()}, any-in={(in_set.any(axis=1)).sum()}, "
                    #     f"first-in={(in_set[:,0]).sum()}")

                if not keep_mask.any():
                    dropped_windows += B
                    continue  # drop entire minibatch

                if not keep_mask.all():
                    # count the rows you're about to drop
                    num_dropped = int((~keep_mask).sum())
                    dropped_windows += num_dropped
                    # filter all tensors in x
                    for k, v in x.items():
                        if torch.is_tensor(v) and (v.size(0) == B):
                            x[k] = v[keep_mask]
                    B = x["decoder_time_idx"].shape[0]  # recompute after filtering
            # ================================================================================


            # build baseline variant first
            x_base = _make_baseline_variant(
                x,
                spec=baseline_spec,
                cont_idx=cont_idx,
                cat_idx=cat_idx,
                cat_encoder=cat_encoder,
                mu_i=float(mu_i_b),
                sigma_i=float(sigma_i_b),
            )

            # build variants (baseline first)
            variants = [x_base]
            if need_cf_i:
                variants.append(_clone_with_modified(x, "decoder_cont", cont_idx, float(z_indiv)))
                if need_cf_pi:
                    variants.append(_clone_with_modified(x, "decoder_cont", cont_idx, float(z_pop_to_indiv)))
            if need_cf_cat:
                variants.append(_clone_with_modified(x, "decoder_cat",  cat_idx,  int(cat_code)))

            # ONE forward for all variants -> [V*B, ...]
            x_cat = _concat_variants(variants)
            pred = _forward_all(tft, x_cat, amp_ctx)
            V = len(variants)
            assert pred.size(0) == V * B, f"Bad split: total={pred.size(0)} vs V*B={V*B}"

            # split back into [V] chunks, each [B, H, Q] (or [B, H])
            chunks = list(pred.split(B, dim=0))
            base_preds.append(chunks[0])
            i = 1
            if need_cf_i:
                cf_i_preds.append(chunks[i]); i += 1
                if need_cf_pi:
                    cf_pi_preds.append(chunks[i]); i += 1
            if need_cf_cat:
                cf_cat_preds.append(chunks[i]); i += 1

            # save ONE x per sample so len(xs) == N windows
            if keep_windows:
                keys_keep = [k for k in ("decoder_time_idx","encoder_lengths","encoder_target","decoder_target") if (k in x and x[k] is not None)]
                for b in range(B):
                    xs.append({k: x[k][b:b+1].detach().cpu() for k in keys_keep})

    # Assemble outputs
    if not base_preds:
        raise RuntimeError("No windows produced – check dataset/filtering.")

    base_tensor = torch.cat(base_preds, dim=0)                 # [N, H, Q] or [N, H]
    out = {"base_tensor": base_tensor, "base_med": _median_slice(base_tensor)}
    if keep_windows:
        assert len(xs) == base_tensor.size(0), f"xs ({len(xs)}) != preds ({base_tensor.size(0)})"
        out["xs"] = xs

    if cf_i_preds:
        cf_i_tensor = torch.cat(cf_i_preds, dim=0)
        out["cf_i_tensor"] = cf_i_tensor
        out["cf_i_med"] = _median_slice(cf_i_tensor)

    if cf_pi_preds:
        cf_pi_tensor = torch.cat(cf_pi_preds, dim=0)
        out["cf_pi_tensor"] = cf_pi_tensor
        out["cf_pi_med"] = _median_slice(cf_pi_tensor)

    if cf_cat_preds:
        cf_cat_tensor = torch.cat(cf_cat_preds, dim=0)
        out["cf_cat_tensor"] = cf_cat_tensor
        out["cf_cat_med"] = _median_slice(cf_cat_tensor)
    
    out["dropped_windows"] = dropped_windows   # ✅ optional
    out["total_windows"] = total_windows
    if print_drop_summary:
         print(f"Dropped {dropped_windows}/{total_windows} windows "
              f"({dropped_windows/total_windows:.2%}) due to minute_of_day filtering.")

    return out


def aggregate_deltas(base_med: torch.Tensor, cf_med: torch.Tensor, method: str = 'mean', delta_t: int = 5) -> np.ndarray:
    delta = cf_med - base_med  # [N, H]
    if method == 'end_horizon':
        agg_gpu = delta[:, -1]
    elif method == 'mean':
        agg_gpu = delta.mean(dim=1)
    elif method == 'auc':
        agg_gpu = torch.trapz(delta, dx=delta_t, dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    # 👇 cast to fp32 before NumPy (bf16 is unsupported by NumPy)
    return agg_gpu.to(torch.float32).cpu().numpy()


def compute_delta_TIR_series(base_med: torch.Tensor, cf_med: torch.Tensor, low: float, high: float) -> np.ndarray:
    in_base = ((base_med >= low) & (base_med <= high)).float()
    in_cf = ((cf_med >= low) & (cf_med <= high)).float()
    delta_tir = (in_cf - in_base).mean(dim=1) * 100.0
    return delta_tir.to(torch.float32).cpu().numpy()   # 👈 cast here too


@dataclass
class Scenario:
    target_id: str
    # continuous CF
    covariate_name: Optional[str] = None
    counterfact_std: Optional[float] = None
    use_raw_value: bool = False
    counterfact_value_raw: Optional[float] = None
    # categorical CF
    apply_categorical_cf: bool = False
    cat_cov_name: Optional[str] = None
    cat_force_label: Optional[str] = None
    # aggregation & thresholds
    aggregation: str = 'mean'
    delta_t_minutes: int = 5
    LOW_TIR: float = 70.0
    HIGH_TIR: float = 180.0
    # support filter (minute_of_day set like {0,5,...,1435})
    valid_minutes: Optional[Set[int]] = None
    # plot helper (stored in meta for convenience)
    k_to_plot: int = 0
    # baseline controls
    baseline_mode: Literal["planned","cont_fixed_z","cont_fixed_raw","cat_force_label"] = "planned"
    baseline_z_value: Optional[float] = None
    baseline_raw_value: Optional[float] = None
    baseline_cat_label: Optional[str] = None

    def tag(self) -> str:
        parts = [f"id={self.target_id}"]
        if self.covariate_name is not None:
            if self.use_raw_value and (self.counterfact_value_raw is not None):
                parts.append(f"cont={self.covariate_name}:raw={self.counterfact_value_raw}")
            elif self.counterfact_std is not None:
                parts.append(f"cont={self.covariate_name}:z={self.counterfact_std}")
        if self.apply_categorical_cf and self.cat_cov_name and self.cat_force_label:
            parts.append(f"cat={self.cat_cov_name}:{self.cat_force_label}")
        parts.append(self.aggregation)
        # include baseline short tag
        parts.append(f"base={self.baseline_mode}")
        return "|".join(parts)


def compute_counterfactuals(train, training: TimeSeriesDataSet, tft, scen: Scenario, *, verbose: bool = False, print_drop_summary: Optional[bool] = None):
    """Compute counterfactual predictions and all metrics. No plotting.

    Returns a dict with keys: meta, stats, windows, medians, metrics.
    """
    # Stats & z mapping
    label_i = "indiv"
    label_pi: Optional[str] = None

    if scen.covariate_name is not None:
        # use counterfact_std only when provided
        cf_std = float(scen.counterfact_std or 0.0)
        mu_i, sigma_i, mu_p, sigma_p, raw_p = compute_stats(train, scen.target_id, scen.covariate_name, cf_std)

        if scen.use_raw_value and (scen.counterfact_value_raw is not None):
            # Only one meaningful continuous path when using a fixed raw value
            z_indiv = (float(scen.counterfact_value_raw) - mu_i) / sigma_i
            z_p_to_i: Optional[float] = None
            label_i = f"raw={scen.counterfact_value_raw}"
        else:
            # z-shift scenarios
            z_indiv = cf_std
            z_p_to_i = ((mu_p + cf_std * sigma_p) - mu_i) / sigma_i if (scen.counterfact_std is not None) else None
            if z_p_to_i is not None:
                label_pi = "pop→indiv"
    else:
        # No continuous covariate → skip all continuous paths
        mu_i = sigma_i = mu_p = sigma_p = raw_p = 0.0
        z_indiv = 0.0
        z_p_to_i = None

    if verbose and scen.covariate_name is not None:
        print(f"Individual   μ = {mu_i:.3f}, σ = {sigma_i:.3f}")
        print(f"Population   μ = {mu_p:.3f}, σ = {sigma_p:.3f}")
        if scen.use_raw_value and scen.counterfact_value_raw is not None:
            print(f"Using RAW value {scen.counterfact_value_raw} mapped to individual z = {z_indiv:.3f}")
        elif scen.counterfact_std is not None:
            print(f"raw_p (μₚ+{scen.counterfact_std}σₚ) = {raw_p:.3f} → z_p→i = {0.0 if z_p_to_i is None else z_p_to_i:.3f}")

    # Build dataset/loader
    sub = train[train["participant_id"] == scen.target_id]
    _full_ds, train_dl, cont_idx, cat_idx, cat_encoder = build_full_dataloader(
        training,
        sub,
        covariate_name=scen.covariate_name,
        apply_categorical_cf=scen.apply_categorical_cf,
        cat_cov_name=scen.cat_cov_name,
    )

    timeidx_to_minofday = (
                                sub[["ds", "minute_of_day"]]
                                .drop_duplicates("ds")
                                .set_index("ds")["minute_of_day"]
                                .to_dict()
                            )



    # ADD PLAUSIBLE/POSITIVITY/SUPPORT WINDOW FOR COUNTERFACTUALS!!!!
    # If not provided in the Scenario, default to 9:00 to 21:00 on a 5-min grid.
    VALID_MINUTES = scen.valid_minutes if scen.valid_minutes is not None else set(range(9*60, 21*60 + 1, 5))


    # Determine categorical code
    cat_code = None
    cat_label_pretty = None
    if scen.apply_categorical_cf and (cat_idx is not None) and (cat_encoder is not None):
        if scen.cat_force_label is None:
            raise ValueError("apply_categorical_cf=True but cat_force_label is None")
        cat_code = int(cat_encoder.transform([scen.cat_force_label])[0])
        cat_label_pretty = f"force {scen.cat_cov_name}={scen.cat_force_label}"

    # Build baseline spec
    baseline_spec = BaselineSpec(
        mode=scen.baseline_mode,
        z_value=scen.baseline_z_value,
        raw_value=scen.baseline_raw_value,
        cat_label=scen.baseline_cat_label,
    )

    # Run inference
    device = next(tft.parameters()).device
    outs = run_counterfactual_windows(
        tft=tft,
        train_dl=train_dl,
        cont_idx=cont_idx,
        z_indiv=float(z_indiv),
        z_pop_to_indiv=z_p_to_i,
        device=device,
        cat_idx=cat_idx,
        cat_code=cat_code,
        baseline_spec=baseline_spec,
        baseline_stats=(float(mu_i), float(sigma_i)),
        cat_encoder=cat_encoder,
        # 👇 use mapping approach; stop relying on decoder_cont[:, :, minute_of_day]
        timeidx_to_minofday=timeidx_to_minofday,
        valid_minutes=VALID_MINUTES,
        print_drop_summary=(True if print_drop_summary is None else bool(print_drop_summary)),
    )

    base_med = outs["base_med"]
    cf_i_med = outs.get("cf_i_med")
    cf_pi_med = outs.get("cf_pi_med")
    cf_cat_med = outs.get("cf_cat_med")

    # Aggregations & deltas
    agg_i = agg_pi = agg_cat = None
    deltas_i = deltas_pi = delta_cat = None
    if cf_i_med is not None:
        agg_i = aggregate_deltas(base_med, cf_i_med, method=scen.aggregation, delta_t=scen.delta_t_minutes)
        deltas_i = (cf_i_med - base_med)
    if cf_pi_med is not None:
        agg_pi = aggregate_deltas(base_med, cf_pi_med, method=scen.aggregation, delta_t=scen.delta_t_minutes)
        deltas_pi = (cf_pi_med - base_med)
    if cf_cat_med is not None:
        agg_cat = aggregate_deltas(base_med, cf_cat_med, method=scen.aggregation, delta_t=scen.delta_t_minutes)
        delta_cat = (cf_cat_med - base_med)

    # ΔTIR
    tir_i = tir_pi = tir_cat = None
    if cf_i_med is not None:
        tir_i = compute_delta_TIR_series(base_med, cf_i_med, scen.LOW_TIR, scen.HIGH_TIR)
    if cf_pi_med is not None:
        tir_pi = compute_delta_TIR_series(base_med, cf_pi_med, scen.LOW_TIR, scen.HIGH_TIR)
    if cf_cat_med is not None:
        tir_cat = compute_delta_TIR_series(base_med, cf_cat_med, scen.LOW_TIR, scen.HIGH_TIR)

    results = {
        "meta": {
            "target_id": scen.target_id,
            "covariate_name": scen.covariate_name,
            "counterfact_std": scen.counterfact_std,
            "use_raw_value": scen.use_raw_value,
            "counterfact_value_raw": scen.counterfact_value_raw,
            "apply_categorical_cf": scen.apply_categorical_cf,
            "cat_cov_name": scen.cat_cov_name,
            "cat_force_label": scen.cat_force_label,
            "aggregation": scen.aggregation,
            "delta_t_minutes": scen.delta_t_minutes,
            "LOW_TIR": scen.LOW_TIR,
            "HIGH_TIR": scen.HIGH_TIR,
            "k_to_plot": scen.k_to_plot,
            "label_i": label_i,
            **({"label_pi": label_pi} if label_pi is not None else {}),
            "label_cat": cat_label_pretty,
            # baseline meta
            "baseline_mode": scen.baseline_mode,
            "baseline_z_value": scen.baseline_z_value,
            "baseline_raw_value": scen.baseline_raw_value,
            "baseline_cat_label": scen.baseline_cat_label,
        },
        "stats": {
            "mu_i": float(mu_i),
            "sigma_i": float(sigma_i),
            "mu_p": float(mu_p),
            "sigma_p": float(sigma_p),
            "raw_p": float(raw_p),
            "z_indiv": float(z_indiv),
            "z_p_to_i": (None if ("cf_pi_med" not in outs) else float(z_p_to_i) if z_p_to_i is not None else None),
        },
        "windows": {k: outs[k] for k in ("xs", "base_tensor", "cf_i_tensor", "cf_pi_tensor", "cf_cat_tensor") if k in outs},
        "medians": {k: outs[k] for k in ("base_med", "cf_i_med", "cf_pi_med", "cf_cat_med") if k in outs},
        "metrics": {
            "agg": {"i": agg_i, "pi": agg_pi, "cat": agg_cat},
            "deltas": {"i": deltas_i, "pi": deltas_pi, "cat": delta_cat},
            "tir": {"i": tir_i, "pi": tir_pi, "cat": tir_cat},
        },
    }
    return results


def main(train, training: TimeSeriesDataSet, tft) -> Dict:
    """Compute-only main using the default config variables at top of file."""
    scen = Scenario(
        target_id=target_id,
        covariate_name=covariate_name,
        counterfact_std=counterfact_std,
        use_raw_value=use_raw_value,
        counterfact_value_raw=counterfact_value_raw,
        apply_categorical_cf=apply_categorical_cf,
        cat_cov_name=cat_cov_name,
        cat_force_label=cat_force_label,
        aggregation=aggregation,
        delta_t_minutes=delta_t_minutes,
        LOW_TIR=LOW_TIR,
        HIGH_TIR=HIGH_TIR,
        k_to_plot=k_to_plot,
    )
    return compute_counterfactuals(train, training, tft, scen)


def main_nb(
    train,
    training: TimeSeriesDataSet,
    tft,
    *,
    target_id_in: str,
    covariate_name_in: Optional[str] = None,
    counterfact_std_in: Optional[float] = None,
    use_raw_value_in: bool = False,
    counterfact_value_raw_in: Optional[float] = None,
    apply_categorical_cf_in: bool = False,
    cat_cov_name_in: Optional[str] = None,
    cat_force_label_in: Optional[str] = None,
    aggregation_in: str = 'mean',
    delta_t_minutes_in: int = 5,
    LOW_TIR_in: float = 70.0,
    HIGH_TIR_in: float = 180.0,
    k_to_plot_in: int = 0,
    # baseline controls
    baseline_mode_in: Literal["planned","cont_fixed_z","cont_fixed_raw","cat_force_label"] = "planned",
    baseline_z_value_in: Optional[float] = None,
    baseline_raw_value_in: Optional[float] = None,
    baseline_cat_label_in: Optional[str] = None,
    # support filter
    valid_minutes_in: Optional[Set[int]] = None,
) -> Dict:
    """Notebook-friendly compute-only wrapper. Returns a dict of results. No plotting."""
    scen = Scenario(
        target_id=target_id_in,
        covariate_name=covariate_name_in,
        counterfact_std=counterfact_std_in,
        use_raw_value=use_raw_value_in,
        counterfact_value_raw=counterfact_value_raw_in,
        apply_categorical_cf=apply_categorical_cf_in,
        cat_cov_name=cat_cov_name_in,
        cat_force_label=cat_force_label_in,
        aggregation=aggregation_in,
        delta_t_minutes=delta_t_minutes_in,
        LOW_TIR=LOW_TIR_in,
        HIGH_TIR=HIGH_TIR_in,
        k_to_plot=k_to_plot_in,
        baseline_mode=baseline_mode_in,
        baseline_z_value=baseline_z_value_in,
        baseline_raw_value=baseline_raw_value_in,
        baseline_cat_label=baseline_cat_label_in,
        valid_minutes=valid_minutes_in,
    )
    return compute_counterfactuals(train, training, tft, scen)


def run_batch_nb(train, training: TimeSeriesDataSet, tft, scenarios, show_progress: bool = True):
    """Run many scenarios (participants & interventions). Returns dict[tag] -> results."""
    out = {}

    first_only = True

    if show_progress:
        try:
            total = len(scenarios)  # works if scenarios is a list/tuple
        except TypeError:
            total = None           # still fine for any Iterable

        bar = tqdm(scenarios, total=total, desc="Scenarios", leave=False)
        for sc in bar:
            # show the current participant in the bar
            bar.set_postfix_str(f"id={sc.target_id}")
            res = compute_counterfactuals(train, training, tft, sc, print_drop_summary=first_only)
            out[sc.tag()] = res
            if first_only:
                first_only = False
    else:
        for sc in scenarios:
            res = compute_counterfactuals(train, training, tft, sc, print_drop_summary=first_only)
            out[sc.tag()] = res
            if first_only:
                first_only = False

    return out