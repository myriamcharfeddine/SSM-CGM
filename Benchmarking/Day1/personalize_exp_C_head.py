"""
personalize_exp_C_head.py
=========================
Participant-level head adaptation of the global Experiment C model.

For each unseen val/test participant:
  1. Load the trained global Exp C checkpoint.
  2. Freeze the full Mamba/TFT backbone.
  3. Fine-tune only the final prediction head on that participant's adaptation windows.
  4. Evaluate on the participant's evaluation windows (never seen during adaptation).
  5. Compare global (C0) vs personalized (C1) metrics on the same evaluation windows.

Unfreeze modes:
  output_layer  → only the linear output projection
  head          → output_layer + pos_wise_ff + pre_output_gate_norm

Usage:
  # Smoke test (5 participants, 1 adapt epoch)
  python personalize_exp_C_head.py --split test --smoke

  # Full test-set personalization
  python personalize_exp_C_head.py --split test --adapt-epochs 3 --unfreeze-mode output_layer

  # Val-set with broader head unfreeze
  python personalize_exp_C_head.py --split val --adapt-epochs 3 --unfreeze-mode head

Outputs (per split):
  personalization_metrics_{split}.csv       — one row per participant
  personalization_summary_{split}.csv       — aggregate stats (C0 vs C1)
  skipped_participants_{split}.csv          — skipped with reason
  predictions_global_{split}.parquet        — C0 predictions on eval windows
  predictions_personalized_{split}.parquet  — C1 predictions on eval windows
  personalization_config.json               — full run config + git hash
"""

import gc
import json
import sys
import copy
import argparse
import datetime
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# ── Reuse model + utilities from Experiment C training script ──────────────────
_BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(_BENCH_DIR))
from mamba288_static_participant_split_local import (
    MambaTFT,
    param_24h,
    log_memory,
    load_static_feature_list,
)

# ── Constants (must match Exp C training config) ───────────────────────────────
CONTEXT_LENGTH = int(param_24h["windows"]["context_length"])  # 288 bins = 24 h
HORIZON        = int(param_24h["windows"]["horizon"])          # 12 bins  =  1 h
MIN_ADAPT_WINS = 3
MIN_EVAL_WINS  = 1

_DATA_ROOT    = Path("/home/myriamcharfeddine/CGM/Data")
_FEATHER_DIR  = _DATA_ROOT / "ssmcgm_ready_exp_C"
_SPLIT_DIR    = _DATA_ROOT / "experiment_c_split_adapt48h_seed42"
_CKPT_FULL    = _DATA_ROOT / "results/exp_C_full/checkpoints"
_CKPT_SMOKE   = _DATA_ROOT / "results/exp_C_smoke/checkpoints"
_OUT_FULL     = _DATA_ROOT / "results/exp_C_personalization"
_OUT_SMOKE    = _DATA_ROOT / "results/exp_C_personalization_smoke"


# ── Utilities ──────────────────────────────────────────────────────────────────

def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_BENCH_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def find_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Train Experiment C first with mamba288_static_participant_split_local.py"
        )
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(ckpt_dir.glob("mamba288-static-c-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return ckpts[-1]


# ── Rebuild training TSDS (needed for from_dataset on OOV participants) ────────

def build_training_tsds(
    train_df: pd.DataFrame,
    param: dict,
    extra_static_reals: List[str],
    include_pid_embed: bool = False,
) -> TimeSeriesDataSet:
    """
    Reconstruct the Exp C training TimeSeriesDataSet so that
    TimeSeriesDataSet.from_dataset() can build per-participant dataloaders
    with the correct encoders and normalizers.
    """
    horizon        = int(param["windows"]["horizon"])
    context_length = int(param["windows"]["context_length"])

    static_categoricals = ["clinical_site", "study_group"]
    if include_pid_embed:
        static_categoricals = ["participant_id", "clinical_site", "study_group"]

    static_reals = ["age"] + (extra_static_reals or [])

    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals = [
        "ds", "minute_of_day", "tod_sin", "tod_cos",
        "activity_steps", "calories_value", "heartrate",
        "oxygen_saturation", "respiration_rate", "stress_level", "predmeal_flag",
    ]
    time_varying_unknown_reals = [
        "cgm_glucose",
        "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
        "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
        "cgm_lagdiff_1_3", "cgm_lagdiff_3_6",
        "cgm_rolling_mean", "cgm_rolling_std",
    ]

    cut_off = train_df["ds"].max() - horizon
    categorical_encoders = {
        "participant_id": NaNLabelEncoder(add_nan=True),
        "clinical_site":  NaNLabelEncoder(add_nan=True),
        "study_group":    NaNLabelEncoder(add_nan=True),
    }

    training = TimeSeriesDataSet(
        train_df[train_df["ds"] < cut_off],
        time_idx="ds",
        target="cgm_glucose",
        group_ids=["participant_id"],
        max_encoder_length=context_length,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["participant_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        categorical_encoders=categorical_encoders,
    )
    return training


# ── Freezing ───────────────────────────────────────────────────────────────────

def _should_unfreeze(name: str, mode: str) -> bool:
    if mode == "output_layer":
        return "output_layer" in name
    if mode == "head":
        return any(m in name for m in ("output_layer", "pos_wise_ff", "pre_output_gate_norm"))
    raise ValueError(f"Unknown unfreeze_mode: {mode!r}")


# ── Per-participant DataLoaders ─────────────────────────────────────────────────

BINS_PER_HOUR = 12  # 5-min CGM bins


def build_participant_dls(
    pid: str,
    feather_df: pd.DataFrame,
    training: TimeSeriesDataSet,
    pers_info: pd.Series,
    batch_size: int = 16,
    adapt_h_cap: Optional[float] = None,
) -> Tuple[object, object, int, int]:
    """
    Build adaptation and evaluation DataLoaders for one participant.

    Adaptation: rows ds ∈ [0, adaptation_end_bin]
      Includes context phase rows so early adaptation windows have proper lookback.
      adapt_h_cap (float): if set, caps adaptation to first N hours beyond context.

    Evaluation: rows ds ∈ [evaluation_start_bin - CONTEXT_LENGTH, evaluation_end_bin]
      Includes adaptation-phase rows as context for the first evaluation windows.

    Returns (adapt_dl, eval_dl, n_adapt_windows, n_eval_windows).
    Raises ValueError if the participant has too few windows.
    """
    pid_rows = feather_df[feather_df["participant_id"] == pid].copy()
    if len(pid_rows) == 0:
        raise ValueError(f"No rows in feather for participant {pid}")

    context_end = int(pers_info["context_end_bin"])
    adapt_end   = int(pers_info["adaptation_end_bin"])
    eval_start  = int(pers_info["evaluation_start_bin"])
    eval_end    = int(pers_info["evaluation_end_bin"])

    # Cap adaptation data to first adapt_h_cap hours beyond context
    if adapt_h_cap is not None:
        adapt_end = min(adapt_end, context_end + int(adapt_h_cap * BINS_PER_HOUR))

    adapt_rows = pid_rows[pid_rows["ds"] <= adapt_end].copy()
    eval_rows  = pid_rows[
        (pid_rows["ds"] >= max(0, eval_start - CONTEXT_LENGTH)) &
        (pid_rows["ds"] <= eval_end)
    ].copy()

    if adapt_h_cap is not None:
        print(f"      [adapt_h_cap={adapt_h_cap}h] adapt_end capped to bin {adapt_end}")

    min_rows = CONTEXT_LENGTH + HORIZON
    if len(adapt_rows) < min_rows:
        raise ValueError(f"Too few adaptation rows: {len(adapt_rows)} < {min_rows}")
    if len(eval_rows) < min_rows:
        raise ValueError(f"Too few evaluation rows: {len(eval_rows)} < {min_rows}")

    try:
        adapt_tsds = TimeSeriesDataSet.from_dataset(
            training, adapt_rows, predict=False, stop_randomization=True
        )
        eval_tsds = TimeSeriesDataSet.from_dataset(
            training, eval_rows, predict=False, stop_randomization=True
        )
    except Exception as e:
        raise ValueError(f"TimeSeriesDataSet.from_dataset failed: {e}") from e

    n_adapt = len(adapt_tsds)
    n_eval  = len(eval_tsds)

    if n_adapt < MIN_ADAPT_WINS:
        raise ValueError(f"Only {n_adapt} adaptation windows (min {MIN_ADAPT_WINS})")
    if n_eval < MIN_EVAL_WINS:
        raise ValueError(f"Only {n_eval} evaluation windows (min {MIN_EVAL_WINS})")

    adapt_dl = adapt_tsds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, pin_memory=False
    )
    eval_dl = eval_tsds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, pin_memory=False
    )
    return adapt_dl, eval_dl, n_adapt, n_eval


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def fine_tune(
    global_tft: MambaTFT,
    adapt_dl,
    adapt_epochs: int,
    adapt_lr: float,
    unfreeze_mode: str,
) -> MambaTFT:
    """
    Deep-copy the global model, freeze backbone, fine-tune selected head layers
    using a raw PyTorch training loop.

    We bypass Lightning's training_step deliberately: pytorch-forecasting's
    training_step accesses out["interpretation"] for logging, which is not
    present in MambaTFT's to_network_output() and raises KeyError.

    Loss is computed in original glucose scale (mg/dL): both out.prediction
    (already denormalized by transform_output) and y_true (denormalized via
    target_scale from the batch) are in mg/dL.
    """
    # load_from_checkpoint outside a Trainer maps weights to CPU by default;
    # explicitly target CUDA so Mamba's CUDA kernel receives GPU tensors.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tft = copy.deepcopy(global_tft).to(device)

    for p in tft.parameters():
        p.requires_grad_(False)

    unfrozen_params = []
    unfrozen_names  = []
    for name, p in tft.named_parameters():
        if _should_unfreeze(name, unfreeze_mode):
            p.requires_grad_(True)
            unfrozen_params.append(p)
            unfrozen_names.append(name)

    if not unfrozen_params:
        raise ValueError(
            f"No parameters matched unfreeze_mode={unfreeze_mode!r}. "
            "Check layer names in MambaTFT."
        )

    n_unfrozen = sum(p.numel() for p in unfrozen_params)
    print(f"      Unfreezing {n_unfrozen:,} params | {len(unfrozen_names)} tensors "
          f"| mode={unfreeze_mode}")

    optimizer = torch.optim.AdamW(
        unfrozen_params, lr=adapt_lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    quantiles = list(tft.loss.quantiles)  # e.g. [0.1, 0.5, 0.9]

    tft.train()
    for epoch in range(adapt_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in adapt_dl:
            x, (y, *_) = batch  # y: (batch, horizon) normalized; weight ignored

            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in x.items()}
            y = y.to(device)

            optimizer.zero_grad()

            # out.prediction: (batch, horizon, n_quantiles) — denormalized (mg/dL)
            out = tft(x)

            # Denormalize y to match prediction scale.
            # GroupNormalizer z-score: y_norm = (y - center) / scale
            # Inverse:                 y_orig  = y_norm * scale + center
            ts    = x["target_scale"]              # (batch, 2): [:, 0]=center, [:, 1]=scale
            y_mg  = y * ts[:, 1:2] + ts[:, 0:1]   # (batch, horizon) in mg/dL

            # Quantile (pinball) loss — same objective as global training
            y_pred = out.prediction                # (batch, horizon, n_quantiles)
            loss   = torch.zeros(1, device=device)
            for i, q in enumerate(quantiles):
                err  = y_mg - y_pred[..., i]
                loss = loss + torch.mean(torch.max((q - 1) * err, q * err))
            loss = loss / len(quantiles)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unfrozen_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        print(f"      Adapt epoch {epoch+1}/{adapt_epochs}  loss={avg:.4f} mg/dL")

    tft.eval()
    return tft


# ── Evaluation ─────────────────────────────────────────────────────────────────

def predict_on_dl(tft: MambaTFT, dl) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on dl and return (y_true_flat, y_pred_flat) in original glucose scale.
    Uses tft.predict() which handles denormalization via GroupNormalizer internally.
    """
    device_str = "gpu" if torch.cuda.is_available() else "cpu"
    preds = tft.predict(
        dl,
        mode="prediction",
        return_y=True,
        trainer_kwargs={
            "accelerator": device_str,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    y_pred = preds.output.cpu().numpy()  # (n_windows, horizon) — P50, denormalized
    y_true = preds.y[0].cpu().numpy()   # (n_windows, horizon) — denormalized
    return y_true.flatten(), y_pred.flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae":  float(np.mean(np.abs(y_pred - y_true))),
        "tir":  float(np.mean((y_pred >= 70) & (y_pred <= 180)) * 100),
    }


# ── Main personalization loop ──────────────────────────────────────────────────

def personalize_split(
    args,
    global_tft: MambaTFT,
    training: TimeSeriesDataSet,
    feather_df: pd.DataFrame,
    pers_df: pd.DataFrame,
    out_dir: Path,
    stratum_map: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each participant in pers_df:
      1. Build adaptation and evaluation DataLoaders.
      2. Evaluate global model on evaluation windows (C0 baseline).
      3. Fine-tune head on adaptation windows.
      4. Evaluate personalized model on evaluation windows (C1).
      5. Record per-participant metrics.

    Returns (results_df, pred_global_df, pred_pers_df, skipped_df).
    """
    split = args.split
    participants = pers_df["participant_id"].astype(str).tolist()
    if args.max_participants:
        participants = participants[: args.max_participants]

    n_total = len(participants)
    print(f"\n[Personalization] Processing {n_total} {split} participants")
    print(f"  adapt_epochs={args.adapt_epochs}  adapt_lr={args.adapt_lr}"
          f"  unfreeze_mode={args.unfreeze_mode}")

    results     = []
    pred_global = []
    pred_pers   = []
    skipped     = []

    for i, pid in enumerate(participants):
        print(f"\n  [{i+1}/{n_total}] pid={pid}")
        pid_info = pers_df[pers_df["participant_id"].astype(str) == pid].iloc[0]

        # ── Build per-participant DataLoaders ──────────────────────────────────
        try:
            adapt_dl, eval_dl, n_adapt, n_eval = build_participant_dls(
                pid, feather_df, training, pid_info,
                batch_size=args.batch_size,
                adapt_h_cap=args.adapt_h_cap,
            )
        except ValueError as exc:
            msg = str(exc)
            print(f"    SKIP (build): {msg}")
            skipped.append({"participant_id": pid, "split": split, "reason": msg})
            continue

        print(f"    adapt_windows={n_adapt}  eval_windows={n_eval}")

        # ── C0 — global model on evaluation windows ────────────────────────────
        try:
            y_true_g, y_pred_g = predict_on_dl(global_tft, eval_dl)
        except Exception as exc:
            msg = f"Global eval failed: {exc}"
            print(f"    SKIP: {msg}")
            skipped.append({"participant_id": pid, "split": split, "reason": msg})
            continue

        g_metrics = compute_metrics(y_true_g, y_pred_g)
        n_win_g   = len(y_true_g) // HORIZON

        pred_global.append(pd.DataFrame({
            "participant_id": pid,
            "split":          split,
            "window_idx":     np.repeat(np.arange(n_win_g), HORIZON),
            "step":           np.tile(np.arange(HORIZON), n_win_g),
            "y_true":         y_true_g,
            "y_pred":         y_pred_g,
        }))

        # ── C1 — fine-tune head on adaptation windows ──────────────────────────
        if args.adapt_epochs == 0:
            # 0h baseline: no fine-tuning, personalized == global
            pers_tft = global_tft
        else:
            try:
                pers_tft = fine_tune(
                    global_tft, adapt_dl,
                    adapt_epochs=args.adapt_epochs,
                    adapt_lr=args.adapt_lr,
                    unfreeze_mode=args.unfreeze_mode,
                )
            except Exception as exc:
                msg = f"Fine-tune failed: {exc}"
                print(f"    SKIP: {msg}")
                skipped.append({"participant_id": pid, "split": split, "reason": msg})
                continue

        try:
            y_true_p, y_pred_p = predict_on_dl(pers_tft, eval_dl)
        except Exception as exc:
            msg = f"Personalized eval failed: {exc}"
            print(f"    SKIP: {msg}")
            skipped.append({"participant_id": pid, "split": split, "reason": msg})
            del pers_tft
            continue

        p_metrics  = compute_metrics(y_true_p, y_pred_p)
        n_win_p    = len(y_true_p) // HORIZON

        pred_pers.append(pd.DataFrame({
            "participant_id": pid,
            "split":          split,
            "window_idx":     np.repeat(np.arange(n_win_p), HORIZON),
            "step":           np.tile(np.arange(HORIZON), n_win_p),
            "y_true":         y_true_p,
            "y_pred":         y_pred_p,
        }))

        mae_imp  = g_metrics["mae"]  - p_metrics["mae"]   # positive = improved
        rmse_imp = g_metrics["rmse"] - p_metrics["rmse"]

        results.append({
            "participant_id":         pid,
            "split":                  split,
            "stratum":                (stratum_map or {}).get(pid, "unknown"),
            "n_adaptation_windows":   n_adapt,
            "n_evaluation_windows":   n_eval,
            "global_rmse":            g_metrics["rmse"],
            "personalized_rmse":      p_metrics["rmse"],
            "global_mae":             g_metrics["mae"],
            "personalized_mae":       p_metrics["mae"],
            "global_tir":             g_metrics["tir"],
            "personalized_tir":       p_metrics["tir"],
            "mae_improvement":        mae_imp,
            "rmse_improvement":       rmse_imp,
            "improved_mae":           bool(mae_imp > 0),
            "adapt_epochs":           args.adapt_epochs,
            "adapt_h_cap":            args.adapt_h_cap,
            "adapt_lr":               args.adapt_lr,
            "unfreeze_mode":          args.unfreeze_mode,
        })

        tag = "IMPROVED" if mae_imp > 0 else "degraded"
        print(f"    C0 (global)  RMSE={g_metrics['rmse']:.3f}  "
              f"MAE={g_metrics['mae']:.3f}  TIR={g_metrics['tir']:.1f}%")
        print(f"    C1 (person)  RMSE={p_metrics['rmse']:.3f}  "
              f"MAE={p_metrics['mae']:.3f}  TIR={p_metrics['tir']:.1f}%")
        print(f"    MAE Δ = {mae_imp:+.3f}  ({tag})")

        # Flush partial results after each participant so progress is visible
        _partial = pd.DataFrame(results)
        _partial_path = out_dir / f"personalization_metrics_{split}_partial.csv"
        _partial.to_csv(_partial_path, index=False)

        if args.adapt_epochs > 0:
            del pers_tft
        gc.collect()

    results_df = pd.DataFrame(results)
    pred_gdf   = pd.concat(pred_global, ignore_index=True) if pred_global else pd.DataFrame()
    pred_pdf   = pd.concat(pred_pers,   ignore_index=True) if pred_pers   else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)
    return results_df, pred_gdf, pred_pdf, skipped_df


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame, split: str) -> None:
    if len(results_df) == 0:
        print(f"\n[Summary] No completed participants for split={split}")
        return

    n_total = len(results_df)
    n_imp   = int(results_df["improved_mae"].sum())
    pct_imp = n_imp / n_total * 100

    print(f"\n{'='*62}")
    print(f"  PERSONALIZATION SUMMARY — {split.upper()}")
    print(f"{'='*62}")
    print(f"  Participants processed     : {n_total}")
    print(f"  Improved (MAE)             : {n_imp} / {n_total}  ({pct_imp:.1f}%)")
    print()
    for key, label in [("rmse", "RMSE"), ("mae", "MAE"), ("tir", "TIR")]:
        g = results_df[f"global_{key}"].mean()
        p = results_df[f"personalized_{key}"].mean()
        d = p - g
        sign = "+" if d > 0 else ""
        print(f"  Global    {label}  : {g:.4f}")
        print(f"  Personal  {label}  : {p:.4f}  ({sign}{d:.4f})")
        if key == "mae":
            med = results_df["mae_improvement"].median()
            print(f"  MAE improvement (median) : {med:.4f}")
        print()

    # ── Per-stratum breakdown ──────────────────────────────────────────────────
    if "stratum" in results_df.columns and results_df["stratum"].nunique() > 1:
        print(f"  {'─'*58}")
        print(f"  PER-STRATUM BREAKDOWN")
        print(f"  {'─'*58}")
        print(f"  {'Stratum':<18} {'n':>4}  {'%imp':>5}  "
              f"{'gMAE':>7}  {'pMAE':>7}  {'ΔMAE':>7}  "
              f"{'gRMSE':>7}  {'pRMSE':>7}")
        for stratum, grp in results_df.groupby("stratum"):
            n_s    = len(grp)
            imp_s  = grp["improved_mae"].mean() * 100
            g_mae  = grp["global_mae"].mean()
            p_mae  = grp["personalized_mae"].mean()
            d_mae  = grp["mae_improvement"].mean()
            g_rmse = grp["global_rmse"].mean()
            p_rmse = grp["personalized_rmse"].mean()
            sign   = "+" if d_mae > 0 else ""
            print(f"  {stratum:<18} {n_s:>4}  {imp_s:>4.0f}%  "
                  f"{g_mae:>7.3f}  {p_mae:>7.3f}  {sign}{d_mae:>6.3f}  "
                  f"{g_rmse:>7.3f}  {p_rmse:>7.3f}")
        print()
    print(f"{'='*62}\n")


# ── Config ─────────────────────────────────────────────────────────────────────

def save_config(args, ckpt_path: Path, git_hash: str, out_dir: Path) -> None:
    config = {
        "split":            args.split,
        "checkpoint":       str(ckpt_path),
        "git_hash":         git_hash,
        "adapt_epochs":     args.adapt_epochs,
        "adapt_lr":         args.adapt_lr,
        "batch_size":       args.batch_size,
        "unfreeze_mode":    args.unfreeze_mode,
        "smoke":            args.smoke,
        "max_participants": args.max_participants,
        "train_feather":    str(args.train),
        "feather_dir":      str(args.feather_dir),
        "split_dir":        str(args.split_dir),
        "run_timestamp":    datetime.datetime.now().isoformat(),
    }
    path = out_dir / "personalization_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp C head personalization (C0 global → C1 participant-adapted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=["val", "test"], default="test")

    # Paths
    p.add_argument("--checkpoint",  type=Path, default=None,
                   help="Checkpoint file or directory. Default: auto-detect from exp_C_full/smoke")
    p.add_argument("--train",       type=Path, default=_FEATHER_DIR / "train_timeseries_static.feather",
                   help="Train feather (needed to rebuild training TSDS)")
    p.add_argument("--feather-dir", type=Path, default=_FEATHER_DIR,
                   help="Dir containing {split}_timeseries_static.feather")
    p.add_argument("--split-dir",   type=Path, default=_SPLIT_DIR,
                   help="Dir with {split}_personalization_windows.csv")
    p.add_argument("--static-feature-list", type=Path,
                   default=_FEATHER_DIR / "static_feature_list.json")
    p.add_argument("--output-dir",  type=Path, default=None,
                   help="Output root (default: auto from --smoke)")

    # Adaptation hyperparams
    p.add_argument("--adapt-epochs",  type=int,   default=3)
    p.add_argument("--adapt-lr",      type=float, default=1e-4)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--unfreeze-mode", choices=["output_layer", "head"],
                   default="output_layer",
                   help="'output_layer': final linear only. "
                        "'head': output_layer + pos_wise_ff + pre_output_gate_norm")
    p.add_argument("--adapt-h-cap",   type=float, default=None,
                   help="Cap adaptation data to first N hours beyond context window. "
                        "Use 0 (with --adapt-epochs 0) for pure C0 global baseline. "
                        "Default: use all adaptation windows.")

    # Participant cap
    p.add_argument("--max-participants", type=int, default=None)

    # Mode flags
    p.add_argument("--smoke",     action="store_true",
                   help="5 participants, 1 adapt epoch, output to exp_C_personalization_smoke/")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results in output directory")

    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] {name}")
    else:
        print("[GPU] CUDA not available — running on CPU (slow)")

    # PyTorch 2.6 changed weights_only default to True in torch.load, which
    # blocks pytorch-forecasting/pandas classes stored in Lightning checkpoints.
    # Patch torch.load to keep weights_only=False for checkpoints from this
    # trusted environment (same machine that created them).
    # Also force map_location="cpu" on CPU-only machines: Lightning derives the
    # target device from the first tensor in the state_dict, so loading to CPU
    # makes it call model.to("cpu") instead of model.to("cuda:0").
    import functools as _functools
    _orig_load = torch.load
    @_functools.wraps(_orig_load)
    def _patched_load(f, *_a, **_kw):
        _kw["weights_only"] = False
        if not torch.cuda.is_available():
            _kw["map_location"] = "cpu"
        return _orig_load(f, *_a, **_kw)
    torch.load = _patched_load

    # torchmetrics Metric._apply creates a dummy tensor on self._device (still
    # "cuda:0" from the checkpoint) to update its device attribute, before the
    # device has actually been changed — crashes on CPU-only machines.
    # Reset _device to CPU first so the dummy tensor is safely created on CPU.
    if not torch.cuda.is_available():
        from torchmetrics import Metric as _TorchMetric
        _orig_tm_apply = _TorchMetric._apply
        def _safe_tm_apply(self, fn):
            self._device = torch.device("cpu")
            return _orig_tm_apply(self, fn)
        _TorchMetric._apply = _safe_tm_apply

    args = parse_args()

    # ── Smoke overrides ───────────────────────────────────────────────────────
    if args.smoke:
        print("\n[SMOKE] 5 participants, 1 adapt epoch")
        if args.max_participants is None:
            args.max_participants = 5
        if args.adapt_epochs == 3:   # only override default, not explicit --adapt-epochs
            args.adapt_epochs = 1

    # ── Output directory ──────────────────────────────────────────────────────
    out_root = args.output_dir or (_OUT_SMOKE if args.smoke else _OUT_FULL)
    out_dir  = out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Overwrite guard ───────────────────────────────────────────────────────
    results_path = out_dir / f"personalization_metrics_{args.split}.csv"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Results already exist: {results_path}\n"
            "Pass --overwrite to replace."
        )

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if args.checkpoint is None:
        for ckpt_root in [_CKPT_FULL, _CKPT_SMOKE]:
            try:
                ckpt_path = find_checkpoint(ckpt_root)
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(
                "No Exp C checkpoint found.\n"
                "Train with mamba288_static_participant_split_local.py first,\n"
                "or pass --checkpoint explicitly."
            )
    elif args.checkpoint.is_dir():
        ckpt_path = find_checkpoint(args.checkpoint)
    else:
        ckpt_path = args.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    git_hash = _get_git_hash()
    print(f"\n[Checkpoint] {ckpt_path}")
    print(f"[Git]        {git_hash}")

    # ── Validate input files ──────────────────────────────────────────────────
    feather_path = args.feather_dir / f"{args.split}_timeseries_static.feather"
    pers_csv     = args.split_dir  / f"{args.split}_personalization_windows.csv"

    for fpath, label in [
        (args.train,                "train feather"),
        (feather_path,              f"{args.split} feather"),
        (args.static_feature_list,  "static_feature_list.json"),
        (pers_csv,                  f"{args.split}_personalization_windows.csv"),
    ]:
        if not fpath.exists():
            raise FileNotFoundError(f"[ERROR] {label} not found: {fpath}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[Loading] train feather ...")
    train_df   = pd.read_feather(args.train)
    train_df["participant_id"] = train_df["participant_id"].astype(str)
    print(f"  {train_df['participant_id'].nunique():,} pids | {len(train_df):,} rows")

    print(f"[Loading] {args.split} feather ...")
    feather_df = pd.read_feather(feather_path)
    feather_df["participant_id"] = feather_df["participant_id"].astype(str)
    print(f"  {feather_df['participant_id'].nunique():,} pids | {len(feather_df):,} rows")

    extra_static_reals = load_static_feature_list(args.static_feature_list)

    pers_df = pd.read_csv(pers_csv)
    pers_df["participant_id"] = pers_df["participant_id"].astype(str)
    print(f"[Pers windows] {len(pers_df)} participants from {pers_csv.name}")

    # ── Load stratum map ──────────────────────────────────────────────────────
    stratum_map: dict = {}
    split_participants_csv = args.split_dir / "split_participants.csv"
    if split_participants_csv.exists():
        sp = pd.read_csv(split_participants_csv)
        sp["participant_id"] = sp["participant_id"].astype(str)
        stratum_map = dict(zip(sp["participant_id"], sp["stratum"]))
        strata_present = sp[sp["split"] == args.split]["stratum"].value_counts().to_dict()
        print(f"[Stratum map]  {len(stratum_map)} total | {args.split}: {strata_present}")
    else:
        print(f"[Stratum map]  split_participants.csv not found — stratum will be 'unknown'")

    # ── Rebuild training TSDS ─────────────────────────────────────────────────
    print("\n[Training TSDS] Rebuilding from train feather ...")
    training = build_training_tsds(train_df, param_24h, extra_static_reals)
    print(f"  {len(training):,} training windows")
    del train_df
    gc.collect()

    # ── Load global model ─────────────────────────────────────────────────────
    print(f"\n[Global model] Loading checkpoint ...")
    # weights_only is NOT a valid load_from_checkpoint parameter in Lightning ≤2.5.x
    # (it was added in v2.6). Passing it there causes it to leak into the model
    # __init__ kwargs and crash with TypeError. The torch.load patch above already
    # forces weights_only=False at the lower level, so we must NOT pass it here.
    # map_location IS a valid explicit parameter in all Lightning versions.
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    global_tft = MambaTFT.load_from_checkpoint(str(ckpt_path), map_location=map_loc)
    global_tft.eval()
    n_params = sum(p.numel() for p in global_tft.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters")
    log_memory("Global model ready")

    save_config(args, ckpt_path, git_hash, out_dir)

    # ── Run personalization ───────────────────────────────────────────────────
    results_df, pred_gdf, pred_pdf, skipped_df = personalize_split(
        args, global_tft, training, feather_df, pers_df, out_dir,
        stratum_map=stratum_map,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    split = args.split
    print("\n[Saving]")

    if len(results_df) > 0:
        results_df.to_csv(results_path, index=False)
        print(f"  Metrics      → {results_path}")

        n_total = len(results_df)
        n_imp   = int(results_df["improved_mae"].sum())
        summary = pd.DataFrame([{
            "split":                  split,
            "n_participants":         n_total,
            "n_improved_mae":         n_imp,
            "pct_improved_mae":       n_imp / n_total * 100 if n_total else 0,
            "mean_global_mae":        results_df["global_mae"].mean(),
            "mean_personalized_mae":  results_df["personalized_mae"].mean(),
            "median_mae_improvement": results_df["mae_improvement"].median(),
            "mean_global_rmse":       results_df["global_rmse"].mean(),
            "mean_personalized_rmse": results_df["personalized_rmse"].mean(),
            "mean_global_tir":        results_df["global_tir"].mean(),
            "mean_personalized_tir":  results_df["personalized_tir"].mean(),
            "adapt_epochs":           args.adapt_epochs,
            "adapt_lr":               args.adapt_lr,
            "unfreeze_mode":          args.unfreeze_mode,
        }])
        summary_path = out_dir / f"personalization_summary_{split}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Summary      → {summary_path}")

        # ── Per-stratum summary ───────────────────────────────────────────────
        if "stratum" in results_df.columns and results_df["stratum"].nunique() > 1:
            stratum_rows = []
            for stratum, grp in results_df.groupby("stratum"):
                n_s = len(grp)
                n_imp_s = int(grp["improved_mae"].sum())
                stratum_rows.append({
                    "split":                  split,
                    "stratum":                stratum,
                    "n_participants":         n_s,
                    "n_improved_mae":         n_imp_s,
                    "pct_improved_mae":       round(n_imp_s / n_s * 100, 2) if n_s else 0,
                    "mean_global_mae":        round(grp["global_mae"].mean(), 4),
                    "mean_personalized_mae":  round(grp["personalized_mae"].mean(), 4),
                    "mean_mae_improvement":   round(grp["mae_improvement"].mean(), 4),
                    "median_mae_improvement": round(grp["mae_improvement"].median(), 4),
                    "mean_global_rmse":       round(grp["global_rmse"].mean(), 4),
                    "mean_personalized_rmse": round(grp["personalized_rmse"].mean(), 4),
                    "mean_global_tir":        round(grp["global_tir"].mean(), 4),
                    "mean_personalized_tir":  round(grp["personalized_tir"].mean(), 4),
                    "adapt_epochs":           args.adapt_epochs,
                    "adapt_h_cap":            args.adapt_h_cap,
                    "unfreeze_mode":          args.unfreeze_mode,
                })
            stratum_df = pd.DataFrame(stratum_rows)
            stratum_path = out_dir / f"personalization_stratum_{split}.csv"
            stratum_df.to_csv(stratum_path, index=False)
            print(f"  Stratum      → {stratum_path}")

        if len(pred_gdf) > 0:
            gp = out_dir / f"predictions_global_{split}.parquet"
            pred_gdf.to_parquet(gp, index=False)
            print(f"  Global preds → {gp}")

        if len(pred_pdf) > 0:
            pp = out_dir / f"predictions_personalized_{split}.parquet"
            pred_pdf.to_parquet(pp, index=False)
            print(f"  Pers preds   → {pp}")

        print_summary(results_df, split)
    else:
        print("[WARNING] No participants completed successfully.")

    if len(skipped_df) > 0:
        sp = out_dir / f"skipped_participants_{split}.csv"
        skipped_df.to_csv(sp, index=False)
        print(f"  Skipped ({len(skipped_df)}) → {sp}")

    print("\n[Done]")
