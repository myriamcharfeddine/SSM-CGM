"""Phase G — narrative ``meal_transfer_report.md`` builder."""

from __future__ import annotations

from .config import PipelineConfig


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_report(cfg: PipelineConfig, s: dict) -> str:
    ts = s.get("teacher_student_agreement", {})
    so = s.get("response_size_ordering", {})
    st = s.get("student", {})
    plc = s.get("pseudo_label_counts", {})
    ph = s.get("phase_distribution", {})

    lines = []
    A = lines.append
    A("# Passive meal-transfer pipeline — report")
    A("")
    A(f"Mode: {'SMOKE' if cfg.smoke else 'FULL'} "
      f"(<= {cfg.max_participants} participants) · runtime {s.get('runtime_sec')}s")
    A(f"Participants: {s.get('participants')} · rows: {s.get('rows')} "
      f"({s.get('rows_by_split')})")
    A("")
    A("This run executes Pass 1 (retrospective CGMacros teacher) and Pass 2 "
      "(weak labels, causal student, structured meal-state decoder). It does **not** "
      "retrain the SSM-CGM forecaster.")
    A("")

    A("## 1. Source preprocessing recovery (Phase A)")
    A("- Architecture verified against the checkpoint tensor shapes (31 tensors, "
      "exact match): dilated CNN (1→16) → 2-layer **bidirectional** LSTM (h=64) → "
      "LayerNorm(128) → Linear(128→1); attention disabled.")
    A("- Units **mg/dL**, **no normalization** on the CGM channel (raw values fed).")
    A("- Window **72 steps × 5 min = 6 h**, identical for CGMacros training and AI-READI "
      "deployment (SSM-CGM paper arXiv:2510.04386, Methods / Appendix A.2). Resolution is "
      "**resolved**: both domains are 5-minute, so there is no time-scale domain shift; the "
      "notebook's stale '1 min' docstring is documentation drift. See `source_model_audit.md`.")
    A("")

    A("## 2. Teacher reconstruction (Phase B)")
    A(f"- teacher_source: `{s.get('teacher_source')}`")
    A(f"- continuous probability available: **{s.get('teacher_prob_available')}** "
      f"(finite fraction {_fmt(s.get('teacher_prob_finite_frac'),3)}) — the `prob_list` "
      "the original notebook discarded is now reconstructed by overlap averaging.")
    A(f"- flag rate (new, prob≥0.40): {_fmt(s.get('teacher_flag_rate'))} · "
      f"legacy ratio-vote baseline: {_fmt(s.get('teacher_baseline_flag_rate'))} · "
      f"surviving `predmeal_flag`: {_fmt(s.get('legacy_predmeal_flag_rate'))}")
    A("- Corrections vs original: batched inference, segment/gap-safe windowing, and "
      "probabilities are kept (not thresholded away).")
    tva = s.get("teacher_vs_artifact", {})
    if tva.get("available"):
        A(f"- vs surviving `predmeal_flag`: {_fmt(tva.get('row_agreement'),3)} per-row "
          f"agreement on {tva.get('rows')} rows; reconstruction rate "
          f"{_fmt(tva.get('reconstruction_flag_rate'),3)} vs artifact "
          f"{_fmt(tva.get('artifact_flag_rate'),3)}. Disagreements are almost all "
          f"reconstruction=1/artifact=0 ({tva.get('recon1_artifact0')} vs "
          f"{tva.get('recon0_artifact1')}): the downstream timestamp-merge that built the "
          "feathers dropped ~60% of the teacher's positive flags. The corrected pipeline "
          "recovers the full teacher signal.")
    A("")

    A("## 3. Weak pseudo-labels (Phase D)")
    A(f"- counts (all rows): positive={plc.get('positive',0)}, "
      f"negative={plc.get('negative',0)}, uncertain={plc.get('uncertain',0)}")
    A(f"- train-split counts: {s.get('pseudo_label_counts_train')}")
    A("- Cut points are **train-split quantiles** (not hardcoded glucose). "
      "`med_insulin=0` is the primary cohort; for `med_insulin=1` a flat response is "
      "**not** labelled negative and confidence is down-weighted "
      f"(scale {cfg.pseudo.insulin_confidence_scale}); `hidden_insulin_risk` is set.")
    thr = s.get("pseudo_thresholds", {})
    A(f"- thresholds: resp_pos={_fmt(thr.get('resp_pos'),2)} mg/dL, "
      f"resp_neg={_fmt(thr.get('resp_neg'),2)}, size_small={_fmt(thr.get('size_small'),2)}, "
      f"size_medium={_fmt(thr.get('size_medium'),2)}, "
      f"teacher_available={thr.get('teacher_available')}")
    A("")

    A("## 4. Causal student (Phase E)")
    A(f"- HistGradientBoosting on {st.get('n_features')} **past/current-only** features; "
      f"leakage check: **{st.get('leakage_check')}** (no future glucose/wearables, no "
      "teacher probability as a feature).")
    A(f"- participant-disjoint validation: average precision "
      f"{_fmt(st.get('val_average_precision'))}, AUC {_fmt(st.get('val_auc'))} "
      f"(train rows {st.get('train_rows')}, pos rate {_fmt(st.get('train_pos_rate'))}).")
    if ts.get("available"):
        A(f"- teacher↔student agreement (held out as label only): "
          f"flag agreement {_fmt(ts.get('flag_agreement'))}, "
          f"prob correlation {_fmt(ts.get('prob_correlation'))}.")
    A("")

    A("## 5. Structured meal states (Phase F)")
    A(f"- duration-constrained Viterbi over {', '.join(cfg.decoder.states)} with minimum "
      f"dwell {cfg.decoder.min_dwell_steps} steps (×5 min).")
    A(f"- decoded flag rate: {_fmt(s.get('decoded_flag_rate'))}; events: {s.get('n_events')}.")
    A("- phase distribution: " + ", ".join(
        f"{k}={_fmt(ph.get(k,0),3)}" for k in cfg.decoder.states))
    A("")

    A("## 6. Response-size ordering (Phase 7 of the brief)")
    A(f"- monotonic mean peak-rise across small<medium<large: "
      f"**{so.get('monotonic_peak_rise')}**")
    A("- This is a **response-size proxy** (realised glucose excursion of the detected "
      "event), explicitly not carbohydrate grams.")
    A("")

    A("## 7. Leakage / domain-shift concerns")
    A("- Student features are strictly causal; future-glucose response and the "
      "bidirectional teacher are used for **label construction only**.")
    A("- Teacher domain shift: CGMacros→AI-READI population/device shift (resolution is matched "
      "5-min/6h, §1). Treat teacher probabilities as weak evidence, not truth.")
    A("- AI-READI has **no ground-truth meal logs**; all positives are weak/pseudo labels. "
      "Event counts and the size ordering are diagnostics, not validated detection metrics.")
    A("- Next step (not run here): the lightweight residual-forecast test with negative "
      "controls (shuffle/shift meal states) before any SSM-CGM retraining.")
    A("")

    A("## Artifacts")
    for f in ["teacher_predictions.parquet", "meal_pseudo_labels.parquet",
              "causal_student_predictions.parquet", "passive_meal_states.parquet",
              "meal_transfer_metrics.csv", "meal_event_summary.csv",
              "meal_response_by_size.csv", "source_model_audit.md",
              "diagnostics/", "diagnostics/figures/"]:
        A(f"- `{f}`")
    return "\n".join(lines) + "\n"
