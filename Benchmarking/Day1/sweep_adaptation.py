"""
sweep_adaptation.py
===================
Adaptation-length sweep for Exp C personalization.

For each adaptation cap in [0h, 6h, 12h, 24h, 48h]:
  - Runs personalize_exp_C_head.py with --adapt-h-cap N --adapt-epochs 3
  - 0h uses --adapt-epochs 0 (pure global model, C0 baseline)
  - Evaluation windows are always the same (fixed by the 48h split)

Saves per-sweep metrics to:
  Data/results/exp_C_personalization/sweep_adapt{N}h/
Saves aggregate summary to:
  Data/results/exp_C_personalization/adaptation_sweep_summary.csv

Usage:
  python sweep_adaptation.py
  python sweep_adaptation.py --split val --adapt-epochs 5
  python sweep_adaptation.py --checkpoint /path/to/best.ckpt
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_CGM_ROOT       = Path("/home/myriamcharfeddine/CGM")
_DATA_ROOT      = _CGM_ROOT / "Data"
_PERSONALIZE    = _CGM_ROOT / "SSM-CGM/Benchmarking/Day1/personalize_exp_C_head.py"
_PYTHON         = "/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"

ADAPT_HOURS     = [0, 6, 12, 24, 48]


def parse_args():
    p = argparse.ArgumentParser(description="Adaptation-length sweep for Exp C")
    p.add_argument("--split",         choices=["val", "test"], default="test")
    p.add_argument("--checkpoint",    type=Path, default=None,
                   help="Checkpoint file/dir (default: auto-detect)")
    p.add_argument("--adapt-epochs",  type=int,  default=3,
                   help="Fine-tune epochs per participant (0h always uses 0)")
    p.add_argument("--unfreeze-mode", choices=["output_layer", "head"],
                   default="output_layer")
    p.add_argument("--smoke",         action="store_true",
                   help="Smoke: 5 participants, 1 adapt epoch, output to sweep_smoke dirs")
    p.add_argument("--overwrite",     action="store_true")
    # Cloud / custom paths
    p.add_argument("--python",              type=str, default=_PYTHON,
                   help="Python interpreter to use when calling personalize script")
    p.add_argument("--personalize-script",  type=Path, default=_PERSONALIZE,
                   help="Path to personalize_exp_C_head.py")
    p.add_argument("--sweep-output-root",   type=Path, default=None,
                   help="Root dir for sweep outputs (default: Data/results/exp_C_personalization)")
    # Data paths forwarded to personalize_exp_C_head.py
    p.add_argument("--train",               type=Path, default=None,
                   help="Train feather (forwarded to personalize script)")
    p.add_argument("--feather-dir",         type=Path, default=None,
                   help="Feather dir (forwarded to personalize script)")
    p.add_argument("--split-dir",           type=Path, default=None,
                   help="Split dir with personalization_windows.csv (forwarded)")
    p.add_argument("--static-feature-list", type=Path, default=None,
                   help="static_feature_list.json (forwarded to personalize script)")
    return p.parse_args()


def run_one(adapt_h: int, args) -> Path:
    smoke_suffix = "_smoke" if args.smoke else ""
    sweep_root = args.sweep_output_root or (_DATA_ROOT / "results/exp_C_personalization")
    out_dir = sweep_root / f"sweep_adapt{adapt_h}h{smoke_suffix}"
    # personalize_exp_C_head.py appends /{split}/ to --output-dir before saving
    metrics_path = out_dir / args.split / f"personalization_metrics_{args.split}.csv"

    if metrics_path.exists() and not args.overwrite:
        print(f"\n[sweep] adapt_h={adapt_h}h — already done, skipping ({metrics_path})")
        return metrics_path

    epochs = 0 if adapt_h == 0 else (1 if args.smoke else args.adapt_epochs)

    python_exe = str(getattr(args, "python", _PYTHON) or _PYTHON)
    personalize = str(getattr(args, "personalize_script", _PERSONALIZE) or _PERSONALIZE)

    cmd = [
        python_exe, personalize,
        "--split",         args.split,
        "--adapt-epochs",  str(epochs),
        "--adapt-h-cap",   str(float(adapt_h)),
        "--unfreeze-mode", args.unfreeze_mode,
        "--output-dir",    str(out_dir),
        "--overwrite",
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.smoke:
        cmd += ["--smoke"]
    if getattr(args, "train", None):
        cmd += ["--train", str(args.train)]
    if getattr(args, "feather_dir", None):
        cmd += ["--feather-dir", str(args.feather_dir)]
    if getattr(args, "split_dir", None):
        cmd += ["--split-dir", str(args.split_dir)]
    if getattr(args, "static_feature_list", None):
        cmd += ["--static-feature-list", str(args.static_feature_list)]

    label = "C0 global (no fine-tune)" if adapt_h == 0 else f"{adapt_h}h adaptation"
    print(f"\n{'='*62}")
    print(f"  Sweep: {label}  ({epochs} epochs)")
    print(f"{'='*62}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] adapt_h={adapt_h}h exited with code {result.returncode}")

    return metrics_path


def load_summary_row(adapt_h: int, metrics_path: Path, split: str) -> dict:
    if not metrics_path.exists():
        print(f"  [WARN] metrics not found: {metrics_path}")
        return {"adapt_h": adapt_h, "error": "metrics file missing"}

    df = pd.read_csv(metrics_path)
    if len(df) == 0:
        return {"adapt_h": adapt_h, "error": "empty metrics"}

    return {
        "adapt_h":               adapt_h,
        "split":                 split,
        "n_participants":         len(df),
        "n_improved_mae":         int(df["improved_mae"].sum()),
        "pct_improved_mae":       round(df["improved_mae"].mean() * 100, 2),
        "mean_global_mae":        round(df["global_mae"].mean(), 4),
        "mean_personalized_mae":  round(df["personalized_mae"].mean(), 4),
        "mean_mae_improvement":   round(df["mae_improvement"].mean(), 4),
        "median_mae_improvement": round(df["mae_improvement"].median(), 4),
        "mean_global_rmse":       round(df["global_rmse"].mean(), 4),
        "mean_personalized_rmse": round(df["personalized_rmse"].mean(), 4),
        "mean_global_tir":        round(df["global_tir"].mean(), 4),
        "mean_personalized_tir":  round(df["personalized_tir"].mean(), 4),
    }


def main():
    args = parse_args()

    if args.smoke:
        args.adapt_epochs = 1

    mode = "SMOKE (5 participants, 1 epoch)" if args.smoke else "FULL"
    print(f"\n{'='*62}")
    print(f"  Adaptation sweep [{mode}]")
    print(f"  split={args.split}  unfreeze={args.unfreeze_mode}  epochs={args.adapt_epochs}")
    print(f"  Hours: {ADAPT_HOURS}")
    print(f"{'='*62}")

    metrics_paths = {}
    for adapt_h in ADAPT_HOURS:
        metrics_paths[adapt_h] = run_one(adapt_h, args)

    # Aggregate summary
    rows = []
    for adapt_h in ADAPT_HOURS:
        row = load_summary_row(adapt_h, metrics_paths[adapt_h], args.split)
        rows.append(row)
        if "error" not in row:
            sign = "+" if row["mean_mae_improvement"] > 0 else ""
            print(f"  {adapt_h:>2}h → global MAE {row['mean_global_mae']:.2f} → "
                  f"personal MAE {row['mean_personalized_mae']:.2f}  "
                  f"({sign}{row['mean_mae_improvement']:.2f})  "
                  f"{row['pct_improved_mae']:.1f}% improved")

    sweep_df = pd.DataFrame(rows)
    suffix = "_smoke" if args.smoke else ""
    sweep_root = args.sweep_output_root or (_DATA_ROOT / "results/exp_C_personalization")
    out_path = sweep_root / f"adaptation_sweep_summary{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_path, index=False)
    print(f"\n[sweep] Summary saved → {out_path}")


if __name__ == "__main__":
    main()
