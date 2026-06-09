#!/usr/bin/env python3
"""
validate_report_inputs.py
Check that the report/ folder does not contain raw protected files,
participant-level identifiable data, or large data artifacts.

Usage:
    python scripts/report/validate_report_inputs.py
    python scripts/report/validate_report_inputs.py --report-dir report

Exit code 0 = pass, 1 = violations found.
"""
import argparse
import sys
from pathlib import Path


BLOCKED_EXTENSIONS = {
    ".parquet", ".feather", ".pkl", ".pickle", ".h5", ".hdf5",
    ".csv",  # raw CSV data — generated table CSVs are allowed; see allowlist below
    ".json",  # raw JSON data — generated JSON is not committed to report/
    ".pth", ".pt", ".ckpt", ".safetensors",
    ".xlsx", ".xls",
    ".db", ".sqlite",
}

BLOCKED_NAME_PATTERNS = [
    "participant_level",   # e.g., per_participant_level_data
    "per_participant",
    "per_subject",
    "per_patient",
    "subject_ids",
    "patient_ids",
    "identifiable",
    "password",
    "credential",
    "token",
    "secret",
    "api_key",
    "apikey",
    "confidential",
    "hipaa",
    "_phi_",               # protected health information
]

# Generated LaTeX table files (.tex) in tables/generated/ are allowed
# Generated figure files (.pdf/.png) in figures/generated/ are allowed
ALLOWED_CSV_DIRS = set()  # no raw CSVs allowed in report/
ALLOWED_JSON_DIRS = set()  # no raw JSONs allowed in report/

MAX_FILE_SIZE_MB = 20  # alert on suspiciously large files


def check_report_dir(report_dir: Path) -> list[str]:
    violations = []
    for path in report_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(report_dir)
        ext = path.suffix.lower()
        name_lower = path.stem.lower()

        # size check
        size_mb = path.stat().st_size / (1024 ** 2)
        if size_mb > MAX_FILE_SIZE_MB:
            violations.append(
                f"LARGE FILE ({size_mb:.1f} MB): {rel} — "
                "may be a data file; verify it is a generated figure/table only."
            )

        # extension check
        if ext in BLOCKED_EXTENSIONS:
            if ext == ".csv" and "generated" in str(rel):
                pass  # skip generated table CSVs if any
            elif ext == ".json" and "generated" in str(rel):
                pass
            else:
                violations.append(
                    f"BLOCKED EXTENSION ({ext}): {rel}"
                )

        # name pattern check
        for pattern in BLOCKED_NAME_PATTERNS:
            if pattern in name_lower:
                violations.append(
                    f"SUSPICIOUS NAME (contains '{pattern}'): {rel}"
                )
                break

    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"ERROR: report directory {report_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {report_dir}/ ...")
    violations = check_report_dir(report_dir)

    if violations:
        print(f"\nFOUND {len(violations)} VIOLATION(S):")
        for v in violations:
            print(f"  [FAIL] {v}")
        print(
            "\nFix: remove or move blocked files before committing report/ to git.\n"
            "Only .tex, .bib, .pdf (figures), .png (figures), and Makefile "
            "belong in report/."
        )
        sys.exit(1)
    else:
        print("  All checks passed.  report/ contains no blocked files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
