#!/usr/bin/env bash
# run_full_pipeline_smoke.sh
# ==========================
# End-to-end smoke test of the full Experiment C pipeline.
# Runs everything in reduced form to validate the pipeline before full runs.
#
# What it runs (in order):
#
#   STEP 1 — Exp C training smoke
#     mamba288_static_participant_split_local.py --smoke --epochs 3
#     ~200 train participants, 3 epochs
#     Output: Data/results/exp_C_smoke/
#
#   STEP 2 — Personalization smoke (single run, 48h adapt, output_layer)
#     personalize_exp_C_head.py --smoke --adapt-epochs 1
#     5 test participants, 1 epoch
#     Output: Data/results/exp_C_personalization_smoke/test/
#
#   STEP 3 — Adaptation sweep smoke (0h/6h/12h/24h/48h)
#     sweep_adaptation.py --smoke
#     5 participants × 5 adapt lengths × 1 epoch
#     Output: Data/results/exp_C_personalization/sweep_adapt{N}h_smoke/
#             Data/results/exp_C_personalization/adaptation_sweep_summary_smoke.csv
#
#   STEP 4 — Subgroup analysis
#     subgroup_analysis.py (reads from step 2 output)
#     Output: Data/results/exp_C_personalization_smoke/test/subgroup_analysis_test*.csv
#
# Usage:
#   bash scripts/run_full_pipeline_smoke.sh
#
# Skip a step:
#   SKIP_TRAIN=1 bash scripts/run_full_pipeline_smoke.sh
#   SKIP_PERS=1 SKIP_SWEEP=1 bash scripts/run_full_pipeline_smoke.sh

set -e

CGM_ROOT="/home/myriamcharfeddine/CGM"
PYTHON="/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python"

TRAIN_SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/mamba288_static_participant_split_local.py"
PERS_SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/personalize_exp_C_head.py"
SWEEP_SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/sweep_adaptation.py"
SUBGROUP_SCRIPT="${CGM_ROOT}/SSM-CGM/Benchmarking/Day1/subgroup_analysis.py"

PERS_SMOKE_OUT="${CGM_ROOT}/Data/results/exp_C_personalization_smoke"

echo "========================================================"
echo " Experiment C — Full Pipeline Smoke Test"
echo " $(date)"
echo "========================================================"

# ── STEP 1: Exp C training ────────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo ""
    echo "[SKIP] Step 1: Training (SKIP_TRAIN=1)"
else
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 1/4 — Exp C Training Smoke"
    echo "   ~200 participants · 3 epochs · 4 GPUs"
    echo "════════════════════════════════════════════════════════"
    conda run -n ssmcgm python "${TRAIN_SCRIPT}" \
        --smoke \
        --epochs 3
    echo ""
    echo "  ✓ Training done → Data/results/exp_C_smoke/"
fi

# ── STEP 2: Personalization smoke (48h, output_layer) ────────────────────────
if [ "${SKIP_PERS:-0}" = "1" ]; then
    echo ""
    echo "[SKIP] Step 2: Personalization smoke (SKIP_PERS=1)"
else
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 2/4 — Personalization Smoke (48h adapt, output_layer)"
    echo "   5 participants · 1 epoch"
    echo "════════════════════════════════════════════════════════"
    conda run -n ssmcgm python "${PERS_SCRIPT}" \
        --split test \
        --smoke \
        --adapt-epochs 1 \
        --unfreeze-mode output_layer \
        --overwrite
    echo ""
    echo "  ✓ Personalization done → ${PERS_SMOKE_OUT}/test/"
fi

# ── STEP 3: Adaptation sweep smoke (0h/6h/12h/24h/48h) ──────────────────────
if [ "${SKIP_SWEEP:-0}" = "1" ]; then
    echo ""
    echo "[SKIP] Step 3: Adaptation sweep (SKIP_SWEEP=1)"
else
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 3/4 — Adaptation Sweep Smoke"
    echo "   0h / 6h / 12h / 24h / 48h · 5 participants each · 1 epoch"
    echo "════════════════════════════════════════════════════════"
    conda run -n ssmcgm python "${SWEEP_SCRIPT}" \
        --split test \
        --smoke \
        --overwrite
    echo ""
    echo "  ✓ Sweep done → Data/results/exp_C_personalization/adaptation_sweep_summary_smoke.csv"
fi

# ── STEP 4: Subgroup analysis ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " STEP 4/4 — Subgroup Analysis"
echo "   Merges personalization metrics with static features + cohort"
echo "════════════════════════════════════════════════════════"
conda run -n ssmcgm python "${SUBGROUP_SCRIPT}" \
    --split test \
    --metrics-dir "${PERS_SMOKE_OUT}/test" \
    --out-dir     "${PERS_SMOKE_OUT}/test"
echo ""
echo "  ✓ Subgroup analysis done → ${PERS_SMOKE_OUT}/test/subgroup_analysis_test*.csv"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Pipeline smoke complete!"
echo " $(date)"
echo ""
echo " Outputs:"
echo "   Training        : Data/results/exp_C_smoke/"
echo "   Personalization : Data/results/exp_C_personalization_smoke/test/"
echo "   Sweep summary   : Data/results/exp_C_personalization/adaptation_sweep_summary_smoke.csv"
echo "   Subgroup        : Data/results/exp_C_personalization_smoke/test/subgroup_analysis_test*.csv"
echo ""
echo " Next — full runs (Cloud Batch for training, then local):"
echo "   bash scripts/submit_exp_C_full_batch.sh    # 30-epoch training on 4×A100"
echo "   bash scripts/run_adaptation_sweep.sh       # full sweep (all participants)"
echo "   python SSM-CGM/Benchmarking/Day1/subgroup_analysis.py  # full subgroup analysis"
echo "========================================================"
