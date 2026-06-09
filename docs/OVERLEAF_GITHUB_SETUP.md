# Overleaf ↔ GitHub Sync Setup

This document explains how to connect the `report/` folder to an Overleaf project so you can
edit LaTeX in Overleaf and keep changes in sync with GitHub.

## Prerequisites

- A GitHub account with push access to this repo.
- An Overleaf account with at least the **free** tier
  (GitHub sync requires Overleaf **premium**; see the workaround below for free accounts).

---

## Option A — Overleaf Premium: Direct GitHub Sync

1. **Create a new Overleaf project** (blank or from a ZIP).

2. In Overleaf, click **Menu → GitHub → Link to a GitHub repository**.

3. Select this repository (`SSM-CGM`) and the branch you want to sync (e.g., `main`).

4. Set the **main document** to `report/main.tex`.

5. **Pull from GitHub** to initialize: Menu → GitHub → Pull changes from GitHub.

6. Compile in Overleaf: click the green **Recompile** button.

7. **Push edits back to GitHub**: Menu → GitHub → Push changes to GitHub.
   Write a commit message and confirm.

> **Rule**: never commit raw AI-READI data files, participant-level outputs, or credentials
> through Overleaf.  Only `.tex`, `.bib`, `.pdf`/`.png` figures, and `Makefile` belong in
> `report/`.  Run `python scripts/report/validate_report_inputs.py` before every push.

---

## Option B — Free Overleaf: Initial ZIP Upload + Incremental File Updates

### First time only — create the Overleaf project from a ZIP

1. **Create a ZIP** of the `report/` folder:
   ```bash
   cd /home/myriamcharfeddine/CGM/SSM-CGM
   zip -r report_snapshot.zip report/ --exclude '*.pdf' --exclude '*.aux' \
     --exclude '*.log' --exclude '*.out' --exclude '*.bbl' --exclude '*.blg'
   ```

2. In Overleaf: **New Project → Upload Project → select `report_snapshot.zip`**.

3. Set the main file to `report/main.tex` in Overleaf, then click **Recompile**.

> **Do this only once.** Uploading a ZIP always creates a *new* project.
> For all subsequent updates, use the incremental file upload described below.

---

### Subsequent updates — upload only changed files (no new project)

After running the automation pipeline, only a handful of files change.
Upload them **into the existing project** — no new project is created:

1. **List what to re-upload** (auto-generated files always change after a new run):
   ```bash
   cd /home/myriamcharfeddine/CGM/SSM-CGM
   find report/sections/generated_results_summary.tex \
        report/tables/generated/ \
        report/figures/generated/ \
        -type f 2>/dev/null | sort
   ```

2. **In Overleaf**, open your existing project.

3. In the **file tree** on the left, click the **upload icon (↑ arrow)**.

4. **Drag in only the files that changed** — typically:
   - `sections/generated_results_summary.tex`
   - `tables/generated/*.tex`
   - `figures/generated/*.png`
   - Any section `.tex` file you edited manually

5. When Overleaf asks **"Overwrite existing file?"** — click **Yes**.
   The file is replaced in place; the project is not recreated.

6. Click **Recompile**.

### Pulling Overleaf edits back to the repo

If you edit `.tex` files directly in Overleaf:

1. In Overleaf: **Menu → Download → Source (.zip)**.
2. Unzip and copy the changed `.tex` / `.bib` files back into `report/`.
3. Commit normally.

---

## Directory layout for Overleaf

When Overleaf opens the project, it sees:
```
main.tex            ← set this as the main file
macros.tex
references.bib
sections/
  00_abstract.tex
  01_introduction.tex
  ...
  generated_results_summary.tex   ← auto-generated; may be absent initially
figures/
  generated/        ← place figure PDFs/PNGs here before compiling
tables/
  generated/        ← place generated LaTeX table fragments here
```

Missing `sections/generated_results_summary.tex` is handled gracefully — the report compiles
without it and uses `\PENDING` placeholders.

---

## Updating tables and figures after new experiments

> **All commands must be run from the repo root** (`SSM-CGM/`), not from inside `report/`.

1. Run the automation pipeline:
   ```bash
   # ── make sure you are in the repo root ──────────────────────────────────
   cd /home/myriamcharfeddine/CGM/SSM-CGM

   # 1. Collect latest metrics from outputs/
   python scripts/report/collect_latest_results.py --outputs-root outputs

   # 2. Generate LaTeX tables into report/tables/generated/
   python scripts/report/make_report_tables.py

   # 3. Generate figures into report/figures/generated/  (needs matplotlib)
   python scripts/report/make_report_figures.py

   # 4. Update the results snapshot (writes real values into report/sections/)
   python scripts/report/update_report_snapshot.py --outputs-root outputs

   # 5. Validate — no raw data or credentials in report/
   python scripts/report/validate_report_inputs.py

   # 6. Compile the PDF  (needs pdflatex — install with: sudo apt install texlive-full)
   cd report && make
   ```

2. Commit the updated `.tex` files and figures to GitHub.
   - **Option A (Premium):** Menu → GitHub → Pull changes from GitHub.
   - **Option B (Free):** upload only the changed files into your existing project
     (see "Subsequent updates" above — do NOT upload a new ZIP).

---

## What NOT to commit

| File type | Action |
|-----------|--------|
| `.parquet`, `.feather`, `.pkl` | Never commit — keep in `outputs/` only |
| Participant-level tables (e.g., per-PID CSVs) | Never commit |
| `.pth`, `.ckpt`, model weights | Never commit |
| `.env`, credentials, API keys | Never commit |
| `report/*.pdf` (compiled output) | Excluded by `.gitignore` |
| `report/*.aux`, `*.log`, `*.bbl` | Excluded by `.gitignore` |

The `validate_report_inputs.py` script checks for these before allowing a commit.
