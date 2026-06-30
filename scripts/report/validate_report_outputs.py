#!/usr/bin/env python3
"""Validate generated LaTeX report outputs for safety and consistency."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

BLOCKED_EXTS = {".pt", ".pth", ".ckpt", ".safetensors", ".parquet", ".feather", ".pkl", ".pickle", ".h5", ".hdf5", ".db", ".sqlite"}
BLOCKED_NAME_PATTERNS = ["predictions", "checkpoint", "best_model", "final_model", "raw_participant", "subject_ids", "patient_ids", "participant_id"]
ALLOWED_GENERATED_WITH_PARTICIPANT_LABEL = {"participant_level_metrics.csv", "participant_level_summary.json"}
MAX_REPORT_FILE_MB = 20.0


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def tex_files(report_dir: Path) -> list[Path]:
    return sorted(report_dir.rglob("*.tex"))


def referenced_graphics(tex_paths: list[Path]) -> list[tuple[Path, str]]:
    refs = []
    pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for tex in tex_paths:
        text = tex.read_text(errors="ignore")
        for m in pat.finditer(text):
            refs.append((tex, m.group(1)))
    return refs


def referenced_inputs(tex_paths: list[Path]) -> list[tuple[Path, str]]:
    refs = []
    pat = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    for tex in tex_paths:
        text = tex.read_text(errors="ignore")
        for m in pat.finditer(text):
            refs.append((tex, m.group(1)))
    return refs


def resolve_graphic(report_dir: Path, ref: str) -> Path | None:
    candidates = [report_dir / ref]
    if not Path(ref).suffix:
        for ext in [".pdf", ".png", ".jpg", ".jpeg", ".svg"]:
            candidates.append(report_dir / (ref + ext))
            candidates.append(report_dir / "figures" / (ref + ext))
            candidates.append(report_dir / "figures" / "generated" / (ref + ext))
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_input(report_dir: Path, ref: str) -> Path | None:
    candidates = [report_dir / ref]
    if not Path(ref).suffix:
        candidates.append(report_dir / (ref + ".tex"))
    for p in candidates:
        if p.exists():
            return p
    return None


def check_report_files(report_dir: Path) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    for p in report_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(report_dir)
        size_mb = p.stat().st_size / (1024 ** 2)
        if size_mb > MAX_REPORT_FILE_MB:
            errors.append(f"large report artifact {rel} is {size_mb:.1f} MB")
        if p.suffix.lower() in BLOCKED_EXTS:
            errors.append(f"blocked file type copied into report: {rel}")
        name_lower = p.name.lower()
        for pattern in BLOCKED_NAME_PATTERNS:
            if pattern in name_lower:
                if p.name in ALLOWED_GENERATED_WITH_PARTICIPANT_LABEL:
                    continue
                errors.append(f"suspicious report filename contains {pattern!r}: {rel}")
                break
        if p.suffix.lower() in {".tex", ".csv", ".json"}:
            text = p.read_text(errors="ignore")[:500000]
            if "participant_id" in text and p.name not in ALLOWED_GENERATED_WITH_PARTICIPANT_LABEL and p.name != "results_manifest.csv":
                errors.append(f"raw participant_id token appears in report file: {rel}")
            if re.search(r"\b(10[0-9]{2}|11[0-9]{2}|12[0-9]{2})\b", text) and "results_manifest" not in str(rel):
                # Raw AI-READI IDs in this project are numeric; anonymized labels are P00001-like.
                warnings.append(f"numeric participant-like tokens found in {rel}; verify they are not raw IDs")
    return errors, warnings


def check_latex_refs(report_dir: Path) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    tex = tex_files(report_dir)
    for source, ref in referenced_graphics(tex):
        if resolve_graphic(report_dir, ref) is None:
            errors.append(f"missing figure referenced by {source.relative_to(report_dir)}: {ref}")
    for source, ref in referenced_inputs(tex):
        if ref.startswith("sections/generated_results_summary"):
            continue
        if resolve_input(report_dir, ref) is None:
            errors.append(f"missing input referenced by {source.relative_to(report_dir)}: {ref}")
    labels = {}
    label_pat = re.compile(r"\\label\{([^}]+)\}")
    for source in tex:
        for label in label_pat.findall(source.read_text(errors="ignore")):
            labels.setdefault(label, []).append(source.relative_to(report_dir))
    for label, files in labels.items():
        if len(files) > 1:
            errors.append(f"duplicate LaTeX label {label}: {', '.join(map(str, files))}")
    return errors, warnings


def check_manifest(report_dir: Path, outputs_root: Path | None) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    manifest = report_dir / "results_manifest.csv"
    rows = read_manifest(manifest)
    if not rows:
        errors.append("manifest missing or empty: report/results_manifest.csv")
        return errors, warnings
    for i, row in enumerate(rows, start=2):
        status = row.get("status", "")
        content = row.get("detected_content", "")
        if status == "ignored" and not row.get("notes"):
            errors.append(f"manifest row {i} ignored without reason: {row.get('source_path')}")
        if content == "unknown_output" and status != "needs_review":
            errors.append(f"manifest row {i} unknown output not marked needs_review: {row.get('source_path')}")
        if status == "main_report" and row.get("file_type") in {"figure", "table_metric", "config"}:
            dest = row.get("destination_path")
            if not dest and content != "participant_level_metrics":
                warnings.append(f"main report row has no copied destination: {row.get('source_path')}")
    if outputs_root and outputs_root.exists():
        manifest_sources = {row.get("source_path") for row in rows if row.get("source_path")}
        discovered = {str(p).replace("\\", "/") for p in outputs_root.rglob("*") if p.is_file()}
        missing = sorted(discovered - manifest_sources)
        if missing:
            errors.append(f"manifest is missing {len(missing)} discovered output files; first missing: {missing[0]}")
    needs_review = sum(1 for r in rows if r.get("status") == "needs_review")
    missing_required = sum(1 for r in rows if r.get("status") == "missing")
    if needs_review:
        warnings.append(f"{needs_review} manifest rows need manual review")
    if missing_required:
        warnings.append(f"{missing_required} required report outputs are missing")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate report outputs after stream result sync")
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--outputs-root", default="outputs")
    args = ap.parse_args()
    report_dir = Path(args.report_dir)
    outputs_root = Path(args.outputs_root) if args.outputs_root else None
    if not report_dir.exists():
        print(f"ERROR: report directory not found: {report_dir}", file=sys.stderr)
        return 1

    errors = []
    warnings = []
    for e, w in [check_report_files(report_dir), check_latex_refs(report_dir), check_manifest(report_dir, outputs_root)]:
        errors.extend(e)
        warnings.extend(w)

    print(f"[validate] checked {report_dir}")
    for w in warnings:
        print(f"[validate] WARNING: {w}")
    if errors:
        print(f"[validate] FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[validate] passed safety/reference checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
