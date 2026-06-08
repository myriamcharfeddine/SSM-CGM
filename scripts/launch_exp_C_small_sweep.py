#!/usr/bin/env python3
"""Launch Experiment C tuning jobs via Cloud Batch."""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SEARCH_SPACE = {
    "learning_rate": [0.001, 0.0005, 0.0002],
    "dropout": [0.1, 0.2],
    "weight_decay": [0.0, 0.0001],
    "batch_size": [8, 16],
    "max_val_windows": [5000, 10000],
}

DEFAULT_GCS_OUTPUT_BASE = os.environ.get("EXP_C_TUNING_GCS_OUTPUT_BASE", "").rstrip("/")


def gcs_output_root(name: str) -> str | None:
    if not DEFAULT_GCS_OUTPUT_BASE:
        return None
    return f"{DEFAULT_GCS_OUTPUT_BASE}/{name}/"


FULLROLLING_SHMSIZE_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_shmsize")
FULLROLLING_SHMSIZE_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_shmsize_results/"
SAFEVAL_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_safeval")
SAFEVAL_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_safeval_results/"
SAFEVAL_EXPERIMENT_NAME = "exp_C_fullrolling_safeval_5e4"
NOLEAK_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_safeval_noleak")
NOLEAK_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_safeval_noleak_results/"
NOLEAK_EXPERIMENT_NAME = "exp_C_fullrolling_safeval_5e4_noleak"
MEMFIX_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_memfix")
MEMFIX_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_memfix_results/"
MEMFIX_EXPERIMENT_NAME = "exp_C_fullrolling_memfix_5e4"
SURVIVAL_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_survival")
SURVIVAL_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_survival_results/"
SURVIVAL_EXPERIMENT_NAME = "exp_C_fullrolling_survival_5e4"
DDPFORK_PROBE_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_ddpfork_probe")
DDPFORK_PROBE_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_ddpfork_probe_results/"
DDPFORK_PROBE_EXPERIMENT_NAME = "exp_C_ddpfork_probe_5e4"
DDPFORK_TRAIN_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_ddpfork_train")
DDPFORK_TRAIN_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_ddpfork_train_results/"
DDPFORK_TRAIN_EXPERIMENT_NAME = "exp_C_fullrolling_ddpfork_train_5e4"
HIGHRAM_4GPU_OUTPUT_ROOT = gcs_output_root("exp_C_tuning_fullrolling_4gpu_highram")
HIGHRAM_4GPU_LOCAL_OUTPUT_ROOT = "/tmp/exp_C_tuning_fullrolling_4gpu_highram_results/"
HIGHRAM_4GPU_EXPERIMENT_NAME = "exp_C_fullrolling_4gpu_highram_5e4"
JOB_ID_RE = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-runs", type=int, default=10, help="Number of jobs to submit, max 12.")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--dry-run", action="store_true", help="Print generated Batch JSON without submitting jobs.")
    parser.add_argument("--submit-script", type=Path, default=Path("scripts/submit_exp_C_tuning_batch.sh"))
    parser.add_argument("--full-rolling-shmsize-configs", action="store_true",
                        help="Use the full rolling-window shared-memory Experiment C configs.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--machine-type", default=None)
    parser.add_argument("--num-sanity-val-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Backward-compatible alias for --train-num-workers.")
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--limit-train-batches", default=None,
                        help="Lightning limit_train_batches value, e.g. 1.0, 30000, or 36000.")
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--shm-size-mib", type=int, default=None)
    parser.add_argument("--use-wandb", choices=("true", "false"), default=None)
    parser.add_argument("--enable-memory-monitor", choices=("true", "false"), default=None)
    parser.add_argument("--scalar-only-training-validation", choices=("true", "false"), default=None)
    parser.add_argument("--disable-training-prediction-storage", choices=("true", "false"), default=None)
    parser.add_argument("--load-test-during-training", choices=("true", "false"), default=None)
    parser.add_argument("--skip-final-eval", choices=("true", "false"), default=None)
    parser.add_argument("--final-eval-return-x", choices=("true", "false"), default=None)
    parser.add_argument("--final-eval-return-y", choices=("true", "false"), default=None)
    parser.add_argument("--output-root", default=None,
                        help="GCS output root for submitted jobs. Required for full-rolling configs unless EXP_C_TUNING_GCS_OUTPUT_BASE is set.")
    return parser.parse_args()


def all_configs() -> list[dict[str, object]]:
    keys = list(SEARCH_SPACE)
    values = [SEARCH_SPACE[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def is_safeval_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_safeval" in output_root


def is_noleak_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_safeval_noleak" in output_root


def is_memfix_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_memfix" in output_root


def is_survival_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_survival" in output_root


def is_ddpfork_probe_output(output_root: str) -> bool:
    return "exp_C_tuning_ddpfork_probe" in output_root


def is_ddpfork_train_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_ddpfork_train" in output_root


def is_highram_4gpu_output(output_root: str) -> bool:
    return "exp_C_tuning_fullrolling_4gpu_highram" in output_root


def fullrolling_shmsize_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    output_root = args.output_root or FULLROLLING_SHMSIZE_OUTPUT_ROOT
    if not output_root:
        raise SystemExit(
            "Set --output-root or EXP_C_TUNING_GCS_OUTPUT_BASE before using "
            "--full-rolling-shmsize-configs."
        )
    highram_4gpu = is_highram_4gpu_output(output_root)
    ddpfork_probe = is_ddpfork_probe_output(output_root)
    ddpfork_train = is_ddpfork_train_output(output_root)
    ddpfork = ddpfork_probe or ddpfork_train
    survival = is_survival_output(output_root)
    memfix = is_memfix_output(output_root)
    noleak = is_noleak_output(output_root)
    safeval = highram_4gpu or ddpfork or survival or memfix or noleak or is_safeval_output(output_root)
    learning_rates = [args.learning_rate] if args.learning_rate is not None else [0.0005, 0.0002]
    train_num_workers = (
        args.train_num_workers if args.train_num_workers is not None
        else args.num_workers if args.num_workers is not None
        else 0 if (highram_4gpu or ddpfork)
        else 1 if survival
        else 4
    )
    configs: list[dict[str, object]] = []
    for lr in learning_rates:
        configs.append({
            "learning_rate": lr,
            "dropout": 0.2,
            "weight_decay": 0.0005,
            "batch_size": args.batch_size if args.batch_size is not None else 16,
            "epochs": args.epochs if args.epochs is not None else (1 if ddpfork_probe else None),
            "devices": args.devices if args.devices is not None else (4 if (highram_4gpu or ddpfork) else None),
            "strategy": args.strategy if args.strategy is not None else ("ddp" if highram_4gpu else "ddp_fork" if ddpfork else None),
            "machine_type": args.machine_type if args.machine_type is not None else ("a2-highgpu-8g" if highram_4gpu else None),
            "num_sanity_val_steps": args.num_sanity_val_steps if args.num_sanity_val_steps is not None else (0 if (highram_4gpu or ddpfork) else None),
            "train_num_workers": train_num_workers,
            "val_num_workers": args.val_num_workers if args.val_num_workers is not None else (0 if safeval else train_num_workers),
            "val_batch_size": args.val_batch_size if args.val_batch_size is not None else (1 if (highram_4gpu or ddpfork or survival) else 2 if safeval else 4),
            "limit_train_batches": args.limit_train_batches or ("200" if ddpfork_probe else "1.0"),
            "max_val_windows": args.max_val_windows if args.max_val_windows is not None else (10 if ddpfork_probe else 1000 if (highram_4gpu or ddpfork_train or survival) else 5000 if memfix else 10000 if safeval else 20000),
            "shm_size_mib": args.shm_size_mib if args.shm_size_mib is not None else 20480,
            "use_wandb": args.use_wandb or "false",
            "enable_memory_monitor": args.enable_memory_monitor or ("true" if safeval else "false"),
            "scalar_only_training_validation": args.scalar_only_training_validation or ("true" if (highram_4gpu or survival or memfix or noleak) else "false"),
            "disable_training_prediction_storage": args.disable_training_prediction_storage or ("true" if (highram_4gpu or survival or memfix) else "false"),
            "load_test_during_training": args.load_test_during_training or ("false" if (highram_4gpu or ddpfork or survival) else "true"),
            "skip_final_eval": args.skip_final_eval or ("true" if (highram_4gpu or ddpfork) else "false"),
            "final_eval_return_x": args.final_eval_return_x or ("false" if highram_4gpu else "true"),
            "final_eval_return_y": args.final_eval_return_y or ("false" if highram_4gpu else "true"),
            "gcs_output_root": output_root,
            "local_output_root": HIGHRAM_4GPU_LOCAL_OUTPUT_ROOT if highram_4gpu else DDPFORK_PROBE_LOCAL_OUTPUT_ROOT if ddpfork_probe else DDPFORK_TRAIN_LOCAL_OUTPUT_ROOT if ddpfork_train else SURVIVAL_LOCAL_OUTPUT_ROOT if survival else MEMFIX_LOCAL_OUTPUT_ROOT if memfix else NOLEAK_LOCAL_OUTPUT_ROOT if noleak else SAFEVAL_LOCAL_OUTPUT_ROOT if safeval else FULLROLLING_SHMSIZE_LOCAL_OUTPUT_ROOT,
            "experiment_name": HIGHRAM_4GPU_EXPERIMENT_NAME if highram_4gpu else DDPFORK_PROBE_EXPERIMENT_NAME if ddpfork_probe else DDPFORK_TRAIN_EXPERIMENT_NAME if ddpfork_train else SURVIVAL_EXPERIMENT_NAME if survival else MEMFIX_EXPERIMENT_NAME if memfix else NOLEAK_EXPERIMENT_NAME if noleak else SAFEVAL_EXPERIMENT_NAME if safeval else "exp_C_tuning_fullrolling_shmsize",
        })
    return configs


def sanitize_job_id(raw: str, max_len: int = 63) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", raw.lower())
    value = re.sub(r"-+", "-", value).strip("-")
    if not value or not value[0].isalpha():
        value = f"exp-c-tune-{value}" if value else "exp-c-tune"
    value = value[:max_len].rstrip("-")
    if not JOB_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid Batch job ID after sanitizing: {value}")
    return value


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "batch_job_id",
        "run_name",
        "learning_rate",
        "batch_size",
        "epochs",
        "devices",
        "strategy",
        "machine_type",
        "num_sanity_val_steps",
        "train_num_workers",
        "val_num_workers",
        "val_batch_size",
        "limit_train_batches",
        "max_val_windows",
        "shm_size_mib",
        "enable_memory_monitor",
        "scalar_only_training_validation",
        "disable_training_prediction_storage",
        "load_test_during_training",
        "skip_final_eval",
        "final_eval_return_x",
        "final_eval_return_y",
        "use_wandb",
        "job_status",
    ]
    widths = {h: max(len(h), *(len(str(row.get(h, ""))) for row in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def build_submit_command(submit_script: Path, cfg: dict[str, object], run_name: str, batch_job_id: str,
                         dry_run: bool) -> list[str]:
    cmd = [
        str(submit_script),
        "--learning-rate", str(cfg["learning_rate"]),
        "--dropout", str(cfg.get("dropout", 0.2)),
        "--weight-decay", str(cfg.get("weight_decay", 0.0005)),
        "--batch-size", str(cfg["batch_size"]),
        "--max-val-windows", str(cfg["max_val_windows"]),
        "--run-name", run_name,
        "--job-id", batch_job_id,
    ]
    optional_flags = [
        ("--epochs", "epochs"),
        ("--devices", "devices"),
        ("--strategy", "strategy"),
        ("--machine-type", "machine_type"),
        ("--num-sanity-val-steps", "num_sanity_val_steps"),
        ("--train-num-workers", "train_num_workers"),
        ("--val-num-workers", "val_num_workers"),
        ("--val-batch-size", "val_batch_size"),
        ("--limit-train-batches", "limit_train_batches"),
        ("--shm-size-mib", "shm_size_mib"),
        ("--use-wandb", "use_wandb"),
        ("--enable-memory-monitor", "enable_memory_monitor"),
        ("--scalar-only-training-validation", "scalar_only_training_validation"),
        ("--disable-training-prediction-storage", "disable_training_prediction_storage"),
        ("--load-test-during-training", "load_test_during_training"),
        ("--skip-final-eval", "skip_final_eval"),
        ("--final-eval-return-x", "final_eval_return_x"),
        ("--final-eval-return-y", "final_eval_return_y"),
        ("--gcs-output-root", "gcs_output_root"),
        ("--local-output-root", "local_output_root"),
        ("--experiment-name", "experiment_name"),
    ]
    for flag, key in optional_flags:
        if key in cfg and cfg[key] is not None:
            cmd.extend([flag, str(cfg[key])])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    args = parse_args()
    if args.num_runs < 1 or args.num_runs > 12:
        raise SystemExit("--num-runs must be between 1 and 12")

    repo_root = Path(__file__).resolve().parents[1]
    submit_script = args.submit_script
    if not submit_script.is_absolute():
        submit_script = repo_root / submit_script
    if not submit_script.exists():
        raise SystemExit(f"Submit script not found: {submit_script}")

    rng = random.Random(args.seed)
    if args.full_rolling_shmsize_configs:
        candidates = fullrolling_shmsize_configs(args)
        if args.num_runs > len(candidates):
            raise SystemExit(f"--num-runs={args.num_runs} exceeds available full-rolling shmsize configs ({len(candidates)})")
        selected = candidates[:args.num_runs]
    else:
        selected = rng.sample(all_configs(), k=args.num_runs)
        for cfg in selected:
            cfg.setdefault("use_wandb", args.use_wandb or "true")
            if args.output_root:
                cfg["gcs_output_root"] = args.output_root
        if not args.dry_run and any(str(cfg.get("use_wandb", "true")) == "true" for cfg in selected) and not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("WANDB_API_KEY is not set. Export it before launching W&B-enabled sweep jobs.")

    now = datetime.now(timezone.utc)
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    job_timestamp = now.strftime("%Y%m%d-%H%M%S")

    rows: list[dict[str, object]] = []
    for idx, cfg in enumerate(selected, start=1):
        if args.full_rolling_shmsize_configs:
            output_root = str(cfg.get("gcs_output_root", ""))
            highram_4gpu = is_highram_4gpu_output(output_root)
            ddpfork_probe = is_ddpfork_probe_output(output_root)
            ddpfork_train = is_ddpfork_train_output(output_root)
            ddpfork = ddpfork_probe or ddpfork_train
            survival = is_survival_output(output_root)
            memfix = is_memfix_output(output_root)
            noleak = is_noleak_output(output_root)
            safeval = highram_4gpu or ddpfork or survival or memfix or noleak or is_safeval_output(output_root)
            prefix = "expC_fullrolling_4gpu_highram" if highram_4gpu else "expC_ddpfork_probe" if ddpfork_probe else "expC_fullrolling_ddpfork_train" if ddpfork_train else "expC_fullrolling_survival" if survival else "expC_fullrolling_memfix" if memfix else "expC_fullrolling_safeval_noleak" if noleak else "expC_fullrolling_safeval" if safeval else "expC_fullrolling_shmsize"
            job_prefix = "exp-c-fr-highram" if highram_4gpu else "exp-c-ddpfork-probe" if ddpfork_probe else "exp-c-ddpfork-train" if ddpfork_train else "exp-c-fr-survival" if survival else "exp-c-fr-memfix" if memfix else "exp-c-fr-noleak" if noleak else "exp-c-fr-safeval" if safeval else "exp-c-fr-shm"
            run_name = (
                f"{prefix}_{run_timestamp}_{idx:02d}"
                f"_lr{cfg['learning_rate']}_bs{cfg['batch_size']}"
                f"_trnw{cfg['train_num_workers']}_valnw{cfg['val_num_workers']}"
                f"_val{cfg['max_val_windows']}"
            ).replace(".", "p")
            batch_job_id = sanitize_job_id(f"{job_prefix}-{job_timestamp}-{idx:02d}")
        else:
            run_name = (
                f"expC_tune_{run_timestamp}_{idx:02d}"
                f"_lr{cfg['learning_rate']}_do{cfg['dropout']}"
                f"_wd{cfg['weight_decay']}_bs{cfg['batch_size']}"
                f"_val{cfg['max_val_windows']}"
            ).replace(".", "p")
            batch_job_id = sanitize_job_id(f"exp-c-tune-{job_timestamp}-{idx:02d}")

        row = {
            "batch_job_id": batch_job_id,
            "run_name": run_name,
            **cfg,
            "dataloader_probe_only": False,
            "job_status": "dry_run" if args.dry_run else "pending",
        }
        cmd = build_submit_command(submit_script, cfg, run_name, batch_job_id, args.dry_run)
        row["submit_command"] = " ".join(cmd)
        subprocess.run(cmd, cwd=repo_root, check=True)
        if not args.dry_run:
            row["job_status"] = "submitted"
        rows.append(row)

    out_dir = repo_root / "sweep_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp_C_small_sweep_{run_timestamp}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"\nSaved sweep config: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
