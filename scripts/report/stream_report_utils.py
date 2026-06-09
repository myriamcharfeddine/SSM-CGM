#!/usr/bin/env python3
"""Shared helpers for AI-READI stream report automation."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_COLUMNS = [
    "source_path",
    "file_type",
    "file_size_mb",
    "detected_content",
    "suggested_section",
    "priority",
    "destination_path",
    "latex_label",
    "caption_or_table_title",
    "status",
    "notes",
]

FIGURE_EXTS = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
TABLE_EXTS = {".csv", ".json", ".yaml", ".yml"}
CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt", ".safetensors"}
RAW_EXTS = {".parquet", ".feather", ".pkl", ".pickle", ".h5", ".hdf5"}
CACHE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
MAX_COPY_MB = 20.0

MAIN_CONTENT = {
    "overall_forecast_metrics",
    "horizon_metrics",
    "mae_by_horizon_figure",
    "terminal_60min_metrics",
    "persistence_baseline",
    "persistence_by_horizon",
    "participant_level_metrics",
    "participant_level_summary",
    "personalization_sweep",
    "personalization_sweep_figure",
    "matched_anchor_personalization",
    "matched_anchor_personalization_figure",
    "bias_diagnostic",
    "scenario_metrics",
    "scenario_pathway_audit",
    "scenario_prediction_deltas",
    "scenario_delta_figure",
    "event_detection_quantile_alarms",
    "hypo_alarm_tradeoff_figure",
    "subgroup_metrics",
    "subgroup_participant_level_metrics",
    "subgroup_plot_study_group",
    "subgroup_plot_hba1c",
    "subgroup_plot_med_insulin",
    "subgroup_plot_site",
    "runtime_hardware_metrics",
    "training_summary",
}

APPENDIX_CONTENT = {
    "subgroup_plot_bmi",
    "subgroup_plot_med_any_diabetes_drug",
    "full_training_history",
    "config_summary",
    "runtime_memory_detail",
    "diagnostics_summary",
    "clinical_safety_metrics",
    "hardware_stream_state_memory",
}

CONTENT_SECTION = {
    "overall_forecast_metrics": "Forecasting Results",
    "horizon_metrics": "Forecasting Results",
    "mae_by_horizon_figure": "Forecasting Results",
    "terminal_60min_metrics": "Forecasting Results",
    "persistence_baseline": "Forecasting Results",
    "persistence_by_horizon": "Forecasting Results",
    "participant_level_metrics": "Forecasting Results",
    "participant_level_summary": "Forecasting Results",
    "personalization_sweep": "Personalization",
    "personalization_sweep_figure": "Personalization",
    "matched_anchor_personalization": "Personalization",
    "matched_anchor_personalization_figure": "Personalization",
    "bias_diagnostic": "Personalization",
    "scenario_metrics": "Proxy Scenario Evaluation",
    "scenario_pathway_audit": "Proxy Scenario Evaluation",
    "scenario_prediction_deltas": "Proxy Scenario Evaluation",
    "scenario_delta_figure": "Proxy Scenario Evaluation",
    "event_detection_quantile_alarms": "Clinical Safety",
    "hypo_alarm_tradeoff_figure": "Clinical Safety",
    "clinical_safety_metrics": "Clinical Safety",
    "subgroup_metrics": "Subgroup Analysis",
    "subgroup_participant_level_metrics": "Subgroup Analysis",
    "subgroup_plot_study_group": "Subgroup Analysis",
    "subgroup_plot_hba1c": "Subgroup Analysis",
    "subgroup_plot_med_insulin": "Subgroup Analysis",
    "subgroup_plot_site": "Subgroup Analysis",
    "subgroup_plot_bmi": "Appendix",
    "subgroup_plot_med_any_diabetes_drug": "Appendix",
    "runtime_hardware_metrics": "Ablation and Runtime",
    "runtime_memory_detail": "Appendix",
    "training_summary": "Ablation and Runtime",
    "full_training_history": "Appendix",
    "config_summary": "Appendix",
    "diagnostics_summary": "Appendix",
    "checkpoint_model": "Ignored",
    "raw_predictions": "Ignored",
    "unknown_output": "Needs Review",
}

CONTENT_PRIORITY = {
    "overall_forecast_metrics": "high",
    "horizon_metrics": "high",
    "mae_by_horizon_figure": "high",
    "persistence_baseline": "high",
    "participant_level_summary": "high",
    "matched_anchor_personalization": "high",
    "matched_anchor_personalization_figure": "high",
    "scenario_prediction_deltas": "high",
    "scenario_delta_figure": "high",
    "event_detection_quantile_alarms": "high",
    "hypo_alarm_tradeoff_figure": "high",
    "subgroup_plot_study_group": "medium",
    "subgroup_plot_hba1c": "medium",
    "subgroup_plot_med_insulin": "medium",
    "subgroup_plot_site": "medium",
    "runtime_hardware_metrics": "medium",
    "training_summary": "medium",
}

FIGURE_DEST = {
    "mae_by_horizon_figure": "horizon_mae.png",
    "personalization_sweep_figure": "personalization_warmup.png",
    "matched_anchor_personalization_figure": "personalization_matched_anchor.png",
    "scenario_delta_figure": "scenario_delta_by_horizon.png",
    "hypo_alarm_tradeoff_figure": "hypo_alarm_tradeoff.png",
    "subgroup_plot_study_group": "subgroup_mae_study_group.png",
    "subgroup_plot_hba1c": "subgroup_mae_hba1c.png",
    "subgroup_plot_med_insulin": "subgroup_mae_med_insulin.png",
    "subgroup_plot_site": "subgroup_mae_site.png",
    "subgroup_plot_bmi": "subgroup_mae_bmi.png",
    "subgroup_plot_med_any_diabetes_drug": "subgroup_mae_med_any_diabetes_drug.png",
}

TABLE_DEST = {
    "overall_forecast_metrics": "overall_metrics.json",
    "horizon_metrics": "horizon_metrics.csv",
    "persistence_baseline": "persistence_baseline.json",
    "persistence_by_horizon": "persistence_by_horizon.csv",
    "participant_level_metrics": "participant_level_metrics.csv",
    "participant_level_summary": "participant_level_summary.json",
    "personalization_sweep": "personalization_sweep.csv",
    "matched_anchor_personalization": "personalization_matched_anchor.csv",
    "bias_diagnostic": "bias_diagnostic.csv",
    "scenario_metrics": "scenario_metrics.csv",
    "scenario_pathway_audit": "scenario_pathway_audit.csv",
    "scenario_prediction_deltas": "scenario_prediction_deltas.csv",
    "event_detection_quantile_alarms": "event_detection_quantile_alarms.csv",
    "subgroup_metrics": "subgroup_metrics.csv",
    "subgroup_participant_level_metrics": "subgroup_participant_level_metrics.csv",
    "runtime_hardware_metrics": "runtime_metrics.json",
    "training_summary": "training_summary.json",
    "full_training_history": "training_history.csv",
    "config_summary": "config_resolved.yaml",
    "clinical_safety_metrics": "clinical_safety.json",
}

FIGURE_LABELS = {
    "mae_by_horizon_figure": "fig:horizon_mae",
    "personalization_sweep_figure": "fig:personalization_warmup",
    "matched_anchor_personalization_figure": "fig:personalization_matched_anchor",
    "scenario_delta_figure": "fig:scenario_delta",
    "hypo_alarm_tradeoff_figure": "fig:hypo_alarm_tradeoff",
    "subgroup_plot_study_group": "fig:subgroup_study_group",
    "subgroup_plot_hba1c": "fig:subgroup_hba1c",
    "subgroup_plot_med_insulin": "fig:subgroup_med_insulin",
    "subgroup_plot_site": "fig:subgroup_site",
    "subgroup_plot_bmi": "fig:subgroup_bmi",
    "subgroup_plot_med_any_diabetes_drug": "fig:subgroup_med_any_diabetes_drug",
}

TABLE_LABELS = {
    "overall_forecast_metrics": "tab:overall_metrics_generated",
    "horizon_metrics": "tab:horizon_metrics_generated",
    "persistence_baseline": "tab:persistence_baseline_generated",
    "persistence_by_horizon": "tab:persistence_by_horizon_generated",
    "participant_level_summary": "tab:participant_level_generated",
    "personalization_sweep": "tab:personalization_sweep_generated",
    "matched_anchor_personalization": "tab:personalization_matched_generated",
    "bias_diagnostic": "tab:bias_diagnostic_generated",
    "scenario_metrics": "tab:scenario_metrics_generated",
    "scenario_pathway_audit": "tab:scenario_pathway_audit_generated",
    "scenario_prediction_deltas": "tab:scenario_delta_generated",
    "event_detection_quantile_alarms": "tab:event_detection_quantile_generated",
    "subgroup_participant_level_metrics": "tab:subgroup_participant_metrics_generated",
    "runtime_hardware_metrics": "tab:runtime_metrics_generated",
    "training_summary": "tab:training_summary_generated",
}

TITLE_MAP = {
    "overall_forecast_metrics": "Overall forecast-only metrics on the held-out test split.",
    "horizon_metrics": "Forecast error by horizon on the held-out test split.",
    "persistence_baseline": "Model performance compared with persistence on matched anchors.",
    "participant_level_summary": "Participant-level averaged metrics with anonymized participant labels.",
    "personalization_sweep": "Warm-up sweep; anchor counts differ by warm-up length.",
    "matched_anchor_personalization": "Matched-anchor personalization comparison using the same eligible anchors.",
    "bias_diagnostic": "Bias and MAE under offset and personalization diagnostics.",
    "scenario_metrics": "Forecast metrics by proxy scenario mode.",
    "scenario_pathway_audit": "Scenario mask and value availability audit.",
    "scenario_prediction_deltas": "Prediction deltas relative to forecast-only mode.",
    "event_detection_quantile_alarms": "Median and risk-quantile hypo/hyper alarm metrics.",
    "subgroup_participant_level_metrics": "Participant-level subgroup metrics.",
    "runtime_hardware_metrics": "Single-GPU runtime and memory metrics.",
    "training_summary": "Training convergence summary.",
}

REQUIRED_CONTENT = [
    "overall_forecast_metrics",
    "horizon_metrics",
    "mae_by_horizon_figure",
    "persistence_baseline",
    "participant_level_summary",
    "personalization_sweep",
    "personalization_sweep_figure",
    "matched_anchor_personalization",
    "matched_anchor_personalization_figure",
    "bias_diagnostic",
    "scenario_metrics",
    "scenario_pathway_audit",
    "scenario_prediction_deltas",
    "scenario_delta_figure",
    "event_detection_quantile_alarms",
    "hypo_alarm_tradeoff_figure",
    "subgroup_participant_level_metrics",
    "subgroup_plot_study_group",
    "subgroup_plot_hba1c",
    "subgroup_plot_med_insulin",
    "subgroup_plot_site",
    "runtime_hardware_metrics",
    "training_summary",
]


def as_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def rel_to_cwd(path: Path) -> str:
    try:
        return as_posix(path.relative_to(Path.cwd()))
    except ValueError:
        return as_posix(path)


def latest_eval_test(outputs_root: Path) -> Path | None:
    candidates = [p for p in outputs_root.iterdir() if p.is_dir() and p.name.endswith("eval_test")]
    if not candidates:
        return None
    def score(p: Path):
        # Prefer the explicit 10epoch run if present, then mtime.
        bonus = 10_000_000 if "10epoch" in p.name else 0
        return bonus + p.stat().st_mtime
    return sorted(candidates, key=score, reverse=True)[0]


def normalize_run(outputs_root: Path, run: str | None) -> Path | None:
    if run:
        p = Path(run)
        return p if p.is_absolute() else outputs_root / p
    return latest_eval_test(outputs_root)


def infer_training_run_for_eval(eval_run: Path | None, outputs_root: Path) -> Path | None:
    # The 10-epoch continuation reused the 5epoch directory, so prefer the checkpoint path
    # recorded by diagnostics if available. Fallback: choose training summary with most epochs.
    if eval_run:
        summary = eval_run / "diagnostics" / "diagnostics_summary.json"
        if summary.exists():
            data = load_json(summary)
            ckpt = data.get("checkpoint")
            if ckpt:
                ckpt_path = Path(ckpt)
                if not ckpt_path.is_absolute():
                    ckpt_path = Path.cwd() / ckpt_path
                for parent in ckpt_path.parents:
                    if parent.parent == outputs_root or parent.name.startswith("aireadi_stream"):
                        if (parent / "metrics" / "training_summary.json").exists():
                            return parent
        direct = outputs_root / eval_run.name.replace("_eval_test", "").replace("_eval_validation", "")
        if (direct / "metrics" / "training_summary.json").exists():
            return direct
    best = None
    best_epochs = -1
    for path in outputs_root.glob("*/metrics/training_summary.json"):
        data = load_json(path)
        hist = data.get("history") or []
        if len(hist) > best_epochs and "smoke" not in str(path):
            best_epochs = len(hist)
            best = path.parents[1]
    return best


def file_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in FIGURE_EXTS:
        return "figure"
    if ext in CHECKPOINT_EXTS:
        return "model_checkpoint"
    if ext in RAW_EXTS:
        return "raw_data"
    if ext in TABLE_EXTS:
        name = path.name.lower()
        if name.endswith(".yaml") or name.endswith(".yml") or name == "config_resolved.yaml":
            return "config"
        if "training" in name or "summary" in name or "metric" in name or "diagnostic" in name or path.parent.name in {"metrics", "diagnostics", "hardware", "tables", "logs"}:
            return "table_metric"
        return "table_metric"
    if path.suffix.lower() in {".log", ".txt", ".out"}:
        return "log"
    return "unknown"


def detect_content(path: Path) -> str:
    name = path.name.lower()
    stem = path.stem.lower()
    parts = {p.lower() for p in path.parts}
    ext = path.suffix.lower()
    if ext in CHECKPOINT_EXTS:
        return "checkpoint_model"
    if name == "predictions.parquet" or (ext in RAW_EXTS and "prediction" in name):
        return "raw_predictions"
    if any(part in CACHE_PARTS for part in parts):
        return "cache_temp"
    if name == "overall_metrics.json":
        return "overall_forecast_metrics"
    if name == "clinical_safety.json":
        return "clinical_safety_metrics"
    if name == "horizon_metrics.csv":
        return "horizon_metrics"
    if name == "bias_diagnostic.csv":
        return "bias_diagnostic"
    if name == "personalization_sweep.csv":
        return "personalization_sweep"
    if name == "personalization_matched_anchor.csv":
        return "matched_anchor_personalization"
    if name == "scenario_metrics.csv":
        return "scenario_metrics"
    if name == "scenario_pathway_audit.csv":
        return "scenario_pathway_audit"
    if name == "scenario_prediction_deltas.csv":
        return "scenario_prediction_deltas"
    if name == "subgroup_metrics.csv":
        return "subgroup_metrics"
    if name == "subgroup_participant_level_metrics.csv":
        return "subgroup_participant_level_metrics"
    if name == "participant_level_metrics.csv":
        return "participant_level_metrics"
    if name == "participant_level_summary.json":
        return "participant_level_summary"
    if name == "event_detection_quantile_alarms.csv":
        return "event_detection_quantile_alarms"
    if name in {"event_detection_metrics.csv", "event_detection_metrics.json"}:
        return "event_detection_metrics"
    if name == "persistence_baseline.json" or name == "persistence_baseline.csv":
        return "persistence_baseline"
    if name == "persistence_by_horizon.csv":
        return "persistence_by_horizon"
    if name in {"hardware_metrics.json", "runtime_metrics.json", "runtime_metrics.csv", "hardware_metrics.csv", "benchmark_hardware_metrics.json"}:
        return "runtime_hardware_metrics"
    if name == "stream_state_memory.csv" or name == "benchmark_stream_state_memory.csv":
        return "hardware_stream_state_memory"
    if name == "training_history.csv":
        return "full_training_history"
    if name == "training_summary.json":
        return "training_summary"
    if name == "config_resolved.yaml":
        return "config_summary"
    if name in {"preprocessor.json", "schema_mapping.json"}:
        return "schema_or_preprocessor"
    if name == "diagnostics_summary.json":
        return "diagnostics_summary"
    if name == "mae_by_horizon.png":
        return "mae_by_horizon_figure"
    if name == "personalization_sweep.png":
        return "personalization_sweep_figure"
    if name == "personalization_matched_anchor.png":
        return "matched_anchor_personalization_figure"
    if name == "scenario_delta_by_horizon.png":
        return "scenario_delta_figure"
    if name == "hypo_alarm_tradeoff.png":
        return "hypo_alarm_tradeoff_figure"
    if name == "subgroup_mae_study_group.png" or name == "subgroup_study_group.png":
        return "subgroup_plot_study_group"
    if name == "subgroup_mae_hba1c.png" or name == "subgroup_hba1c.png":
        return "subgroup_plot_hba1c"
    if name == "subgroup_mae_med_insulin.png":
        return "subgroup_plot_med_insulin"
    if name == "subgroup_mae_site.png" or name == "subgroup_site.png":
        return "subgroup_plot_site"
    if name == "subgroup_mae_bmi.png" or name == "subgroup_bmi.png":
        return "subgroup_plot_bmi"
    if name == "subgroup_mae_med_any_diabetes_drug.png":
        return "subgroup_plot_med_any_diabetes_drug"
    if name in {"runtime_backend.png", "backend_runtime.png"}:
        return "runtime_backend_figure"
    if ext in FIGURE_EXTS:
        return "figure"
    return "unknown_output"


def in_dir(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def classify(path: Path, outputs_root: Path, selected_run: Path | None = None, training_run: Path | None = None) -> dict[str, str]:
    content = detect_content(path)
    ftype = file_type_for(path)
    section = CONTENT_SECTION.get(content, "Needs Review")
    priority = CONTENT_PRIORITY.get(content, "low")
    status = "needs_review"
    notes = ""

    size_mb = path.stat().st_size / (1024 ** 2) if path.exists() else 0.0
    selected = in_dir(path, selected_run)
    training_selected = in_dir(path, training_run)
    validation_sibling = selected_run is not None and selected_run.name.endswith("eval_test") and path.parts and selected_run.name.replace("eval_test", "eval_validation") in path.parts
    ablation = "ablations" in path.parts or "ablation" in str(path).lower()

    if content == "checkpoint_model":
        status = "ignored"
        notes = "checkpoint/model file is unsafe and not report content"
    elif content == "raw_predictions":
        status = "ignored"
        notes = "raw prediction parquet is too large and not report-safe"
    elif content == "cache_temp":
        status = "ignored"
        notes = "cache/temp file"
    elif content == "hardware_stream_state_memory":
        status = "ignored"
        notes = "stream memory detail contains raw participant/segment identifiers; summarized hardware JSON is used instead"
    elif size_mb > MAX_COPY_MB and ftype != "figure":
        status = "ignored"
        notes = f"file larger than {MAX_COPY_MB:.0f} MB; not copied to report"
    elif content == "schema_or_preprocessor":
        status = "needs_review"
        notes = "schema/preprocessor artifact; keep out of manuscript unless manually summarized"
    elif content == "unknown_output" or content == "figure":
        status = "needs_review"
        notes = "unrecognized generated output"
    elif selected and content in MAIN_CONTENT:
        status = "main_report"
        notes = "selected test run"
    elif training_selected and content in {"training_summary", "full_training_history", "config_summary"}:
        status = "main_report" if content == "training_summary" else "appendix"
        notes = "training run associated with selected evaluation"
    elif validation_sibling and content in (MAIN_CONTENT | APPENDIX_CONTENT):
        status = "appendix"
        notes = "validation split result; used for robustness check"
    elif ablation and content in (MAIN_CONTENT | APPENDIX_CONTENT):
        status = "appendix"
        notes = "ablation/probe output"
    elif content in APPENDIX_CONTENT and (selected or training_selected):
        status = "appendix"
        notes = "secondary detail for appendix"
    else:
        status = "needs_review"
        notes = "known output from non-selected run; not inserted automatically"

    return {
        "source_path": rel_to_cwd(path),
        "file_type": ftype,
        "file_size_mb": f"{size_mb:.4f}",
        "detected_content": content,
        "suggested_section": section,
        "priority": priority,
        "destination_path": "",
        "latex_label": FIGURE_LABELS.get(content) or TABLE_LABELS.get(content, ""),
        "caption_or_table_title": TITLE_MAP.get(content, ""),
        "status": status,
        "notes": notes,
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
            writer.writerow(out)


def discover_outputs(outputs_root: Path) -> list[Path]:
    paths = []
    for p in outputs_root.rglob("*"):
        if p.is_file():
            paths.append(p)
    return sorted(paths)


def destination_for(row: dict[str, str], report_dir: Path, selected_run: Path | None = None) -> Path | None:
    content = row.get("detected_content", "")
    src = row.get("source_path", "")
    if not src:
        return None
    source = Path(src)
    if row.get("file_type") == "figure" and content in FIGURE_DEST:
        name = FIGURE_DEST[content]
        if row.get("status") == "appendix" and selected_run and not in_dir(source, selected_run):
            run_name = next((part for part in source.parts if part.startswith("aireadi_stream") or part == "ablations"), "appendix")
            name = f"appendix_{run_name}_{name}"
        return report_dir / "figures" / "generated" / name
    if row.get("file_type") in {"table_metric", "config"} and content in TABLE_DEST:
        name = TABLE_DEST[content]
        if row.get("status") == "appendix" and selected_run and not in_dir(source, selected_run):
            run_name = next((part for part in source.parts if part.startswith("aireadi_stream") or part == "ablations"), "appendix")
            name = f"appendix_{run_name}_{name}"
        return report_dir / "tables" / "generated" / name
    return None


def copy_selected(rows: list[dict[str, str]], report_dir: Path, selected_run: Path | None = None) -> list[dict[str, str]]:
    for row in rows:
        if row.get("status") not in {"main_report", "appendix"}:
            continue
        src_s = row.get("source_path")
        if not src_s:
            continue
        src = Path(src_s)
        if not src.exists():
            continue
        dest = destination_for(row, report_dir, selected_run)
        if dest is None:
            continue
        if row.get("detected_content") == "participant_level_metrics":
            # Keep the full anonymized CSV discoverable, but do not copy per-participant rows into report by default.
            row["notes"] = (row.get("notes", "") + "; not copied: full participant-level CSV stays in outputs").strip("; ")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        row["destination_path"] = rel_to_cwd(dest)
    return rows


def add_missing_required(rows: list[dict[str, str]], selected_run: Path | None) -> list[dict[str, str]]:
    present = {r.get("detected_content") for r in rows if r.get("status") in {"main_report", "appendix"}}
    for content in REQUIRED_CONTENT:
        if content in present:
            continue
        rows.append({
            "source_path": "",
            "file_type": "missing",
            "file_size_mb": "0.0000",
            "detected_content": content,
            "suggested_section": CONTENT_SECTION.get(content, "Needs Review"),
            "priority": CONTENT_PRIORITY.get(content, "high"),
            "destination_path": "",
            "latex_label": FIGURE_LABELS.get(content) or TABLE_LABELS.get(content, ""),
            "caption_or_table_title": TITLE_MAP.get(content, ""),
            "status": "missing",
            "notes": f"required output not found for selected run {selected_run.name if selected_run else ''}",
        })
    return rows


def load_json(path: Path | str) -> dict:
    try:
        with Path(path).open() as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(path: Path | str) -> list[dict[str, str]]:
    try:
        with Path(path).open(newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def find_source(rows: Sequence[dict[str, str]], content: str, main_only: bool = False) -> Path | None:
    statuses = {"main_report"} if main_only else {"main_report", "appendix"}
    candidates = [r for r in rows if r.get("detected_content") == content and r.get("status") in statuses and r.get("source_path")]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (0 if r.get("status") == "main_report" else 1, r.get("source_path", "")))
    return Path(candidates[0]["source_path"])


def find_dest(rows: Sequence[dict[str, str]], content: str) -> Path | None:
    candidates = [r for r in rows if r.get("detected_content") == content and r.get("destination_path")]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (0 if r.get("status") == "main_report" else 1, r.get("destination_path", "")))
    return Path(candidates[0]["destination_path"])


def latex_escape(value: object) -> str:
    if value is None:
        return "--"
    text = str(value)
    if text == "" or text.lower() == "nan":
        return "--"
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(value: object, digits: int = 2, scale: float = 1.0, signed: bool = False) -> str:
    try:
        x = float(value) * scale
        if math.isnan(x) or math.isinf(x):
            return "--"
        sign = "+" if signed and x > 0 else ""
        return f"{sign}{x:.{digits}f}"
    except Exception:
        return "--"


def intfmt(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "--"


def tabular(cols: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + r" \\" , "\\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def replace_or_append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text()
    begin = f"% BEGIN {marker}"
    end = f"% END {marker}"
    wrapped = f"{begin}\n{block.rstrip()}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        text = pattern.sub(lambda _m: wrapped, text)
    else:
        # Put generated definitions after generated_results_summary if possible.
        anchor = "\\IfFileExists{sections/generated_results_summary.tex}"
        idx = text.find("\\setlength{\\parskip}")
        if idx != -1:
            text = text[:idx] + wrapped + "\n\n" + text[idx:]
        else:
            text += "\n" + wrapped + "\n"
    path.write_text(text)
