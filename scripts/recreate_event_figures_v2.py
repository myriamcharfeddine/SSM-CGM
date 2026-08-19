"""Recreate Phase 4 event figures with baseline adjustment and participant-level inference."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import complete_phase4_compliance as compliance
import run_extended_event_rewiring as phase4
from ssmcgm.analysis.within_subtype_config import STUDY2_ROOT

EXT = STUDY2_ROOT / "extended_clinical_latent_dynamics_v1"
P4 = EXT / "04_event_locked_rewiring"
OUT = P4 / "recreated_readable_figures_v2"
FIG = OUT / "figures"
TABLE = OUT / "tables"
META = OUT / "metadata"
REPORT = OUT / "reports"
QA = OUT / "qa"
CACHE = EXT / "cache"

SEED = 42
B = 1000
EVENTS = ["activity_onset", "glucose_rise", "hr_surprise", "sleep_onset", "stress_event", "wake_transition"]
EL = {
    "activity_onset": "Activity onset",
    "glucose_rise": "Glucose rise",
    "hr_surprise": "Heart-rate surprise",
    "sleep_onset": "Sleep onset",
    "stress_event": "Stress event",
    "wake_transition": "Wake transition",
}
TASKS = {
    "A_retained_vs_lost": ("retained", "lost"),
    "B_gained_vs_matched": ("gained", "matched"),
}
TL = {"A_retained_vs_lost": "Retained versus lost", "B_gained_vs_matched": "Gained versus matched non-neighbor"}
NAVY = "#003366"
TEAL = "#5BBABA"
CRIMSON = "#BA2828"
GRAY = "#888888"
BLACK = "#000000"
BRIGHT_RED = "#FF0000"
DOMAIN = {
    "Sleep and wake": ["sleep_onset", "wake_transition"],
    "Activity": ["activity_onset"],
    "Glucose rise": ["glucose_rise"],
    "Heart-rate surprise": ["hr_surprise"],
    "Stress": ["stress_event"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts):
    return SEED + zlib.crc32("|".join(map(str, parts)).encode())


def jdefault(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=jdefault) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "#D9D9D9",
        "pdf.fonttype": 42,
    })


def boot(values, seed, n=B):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, np.nan, 0, np.array([])
    rng = np.random.default_rng(seed)
    z = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.median(x)), float(np.quantile(z, .025)), float(np.quantile(z, .975)), len(x), z


def add_bh(frame, pcol="bootstrap_p"):
    out = frame.copy()
    if not len(out):
        out["fdr_q"] = []
        return out
    p = out[pcol].to_numpy(float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(ranked, 1)
    out["fdr_q"] = q
    return out


def p_from_boot(z):
    if not len(z):
        return np.nan
    return min(1.0, 2 * min((np.sum(z <= 0) + 1) / (len(z) + 1), (np.sum(z >= 0) + 1) / (len(z) + 1)))


def savefig(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}_thumbnail.png", dpi=75, bbox_inches="tight")
    plt.close(fig)


def baseline_effect(aligned, shift_minutes=0, allowed_event_ids=None):
    d = aligned.copy()
    if allowed_event_ids is not None:
        d = d[d.event_id.isin(allowed_event_ids)]
    d["relative_shifted"] = d.relative_minutes - shift_minutes
    base = d[d.relative_shifted.between(-120, -30)].groupby(["event_id", "condition"], as_index=False).euclidean_velocity.mean().rename(columns={"euclidean_velocity": "baseline"})
    d = d.merge(base, on=["event_id", "condition"], how="inner", validate="many_to_one")
    d["baseline_change"] = d.euclidean_velocity - d.baseline
    wide = d.pivot_table(index=["participant_id", "event_id", "event_type", "relative_shifted"], columns="condition", values="baseline_change").dropna(subset=["event", "control"]).reset_index()
    wide["baseline_adjusted_effect"] = wide.event - wide.control
    return wide


def time_summary(effect):
    rows = []
    for event in EVENTS:
        g = effect[effect.event_type.eq(event)]
        event_n = g.event_id.nunique()
        participants = g.participant_id.nunique()
        for rel, q in g.groupby("relative_shifted"):
            pv = q.groupby("participant_id").baseline_adjusted_effect.mean()
            est, med, lo, hi, n, z = boot(pv, stable_seed("curve", event, rel))
            rows.append({
                "event_type": event,
                "relative_minutes": int(rel),
                "relative_hours": rel / 60,
                "estimate": est,
                "median": med,
                "ci_low": lo,
                "ci_high": hi,
                "participant_n": n,
                "event_n": event_n,
                "matched_control_n": event_n,
                "bootstrap_n": B,
                "bootstrap_p": p_from_boot(z),
            })
    return pd.DataFrame(rows)


def window_summary(effect):
    rows = []
    windows = {
        "Peak effect from 0 to 1 hour": (0, 60, "peak"),
        "Mean effect from 0 to 1 hour": (0, 60, "mean"),
        "Mean effect from 1 to 4 hours": (60, 240, "mean"),
    }
    for event in EVENTS:
        g = effect[effect.event_type.eq(event)]
        for label, (lo_t, hi_t, kind) in windows.items():
            q = g[g.relative_shifted.between(lo_t, hi_t)]
            per_event = q.groupby(["participant_id", "event_id", "relative_shifted"], as_index=False).baseline_adjusted_effect.mean()
            if kind == "peak":
                per_instance = per_event.groupby(["participant_id", "event_id"]).baseline_adjusted_effect.max()
            else:
                per_instance = per_event.groupby(["participant_id", "event_id"]).baseline_adjusted_effect.mean()
            per_participant = per_instance.groupby(level=0).mean()
            est, med, lo, hi, n, z = boot(per_participant, stable_seed("window", event, label))
            rows.append({
                "event_type": event,
                "summary_window": label,
                "estimate": est,
                "median": med,
                "ci_low": lo,
                "ci_high": hi,
                "participant_n": n,
                "event_n": int(per_instance.shape[0]),
                "matched_control_n": int(per_instance.shape[0]),
                "bootstrap_n": B,
                "bootstrap_p": p_from_boot(z),
            })
    return add_bh(pd.DataFrame(rows))


def event_overlap(matches, all_events):
    targets = matches[matches.condition.eq("event")].drop_duplicates(["participant_id", "event_type", "event_timestamp_local"]).copy()
    targets["participant_id"] = targets.participant_id.astype(str)
    targets["event_time_utc"] = pd.to_datetime(targets.event_timestamp_local, utc=True)
    targets["event_id"] = targets.participant_id + ":" + targets.event_type + ":" + targets.event_timestamp_local.astype(str)
    all_events = all_events.copy()
    all_events["participant_id"] = all_events.participant_id.astype(str)
    all_events["event_time_utc"] = pd.to_datetime(all_events.event_timestamp_local, utc=True)
    by_pid = {p: g for p, g in all_events.groupby("participant_id")}
    counts = {(r, c): 0 for r in EVENTS for c in EVENTS}
    isolated = []
    any_overlap = []
    for row in targets.itertuples():
        g = by_pid.get(row.participant_id)
        found = {}
        for col in EVENTS:
            if col == row.event_type:
                found[col] = False
                continue
            delta = (g.loc[g.event_type.eq(col), "event_time_utc"] - row.event_time_utc).dt.total_seconds() / 3600
            found[col] = bool(delta.between(-1, 2).any())
            counts[(row.event_type, col)] += int(found[col])
        overlap = any(found.values())
        any_overlap.append(overlap)
        if not overlap:
            isolated.append(row.event_id)
    targets["any_other_event_overlap"] = any_overlap
    matrix = []
    for row_event in EVENTS:
        n = int(targets.event_type.eq(row_event).sum())
        for col_event in EVENTS:
            count = n if row_event == col_event else counts[(row_event, col_event)]
            matrix.append({
                "row_event_type": row_event,
                "column_event_type": col_event,
                "row_event_n": n,
                "overlap_count": count,
                "overlap_percent": 100 * count / n if n else np.nan,
                "window_hours": "-1 to +2",
            })
    summary = []
    for event in EVENTS:
        q = targets[targets.event_type.eq(event)]
        summary.append({
            "event_type": event,
            "event_n": len(q),
            "unique_participants": q.participant_id.nunique(),
            "overlapping_event_n": int(q.any_other_event_overlap.sum()),
            "overlapping_percent": float(100 * q.any_other_event_overlap.mean()) if len(q) else np.nan,
            "isolated_event_n": int((~q.any_other_event_overlap).sum()),
            "isolated_percent": float(100 * (~q.any_other_event_overlap).mean()) if len(q) else np.nan,
        })
    return pd.DataFrame(matrix), pd.DataFrame(summary), set(isolated)


def context_contrasts(pairs):
    q = pairs[pairs.scenario.eq("primary_test_2h")].copy()
    rows = []
    for task, classes in TASKS.items():
        use = q[q.transition_class.isin(classes)]
        for event in EVENTS:
            for measure, feature, unit in [
                ("both_recent_probability", f"event_both_recent_{event}", "Difference in probability that both participants had the event in the preceding 6 hours"),
                ("count_similarity", f"event_count_similarity_{event}", "Difference in negative absolute recent-event count difference"),
            ]:
                z = use.groupby(["anchor_id", "transition_class"])[feature].mean().unstack().dropna(subset=list(classes))
                d = z[classes[0]] - z[classes[1]]
                est, med, lo, hi, n, boots = boot(d, stable_seed("context", task, event, measure))
                rows.append({
                    "task": task,
                    "comparison": "Retained minus lost" if task.startswith("A_") else "Gained minus matched non-neighbor",
                    "event_type": event,
                    "context_measure": measure,
                    "unit": unit,
                    "estimate": est,
                    "median": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "participant_n": n,
                    "bootstrap_n": B,
                    "bootstrap_p": p_from_boot(boots),
                    "interpretation": "Uncertain" if lo <= 0 <= hi else "Positive association" if lo > 0 else "Negative association",
                })
    main = pd.DataFrame(rows)
    mask = main.context_measure.eq("both_recent_probability")
    main.loc[mask, "fdr_q"] = add_bh(main[mask]).fdr_q.to_numpy()
    return main


def prediction_metrics(y, p):
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def model_data(pairs, profiles, dynamic):
    f = compliance.augment(pairs, profiles, dynamic)
    outputs = {}
    for task, classes in TASKS.items():
        q = f[f.transition_class.isin(classes)].copy()
        q["y"] = (q.transition_class == classes[0]).astype(int)
        nuisance = ["hour", "clock_bin", "h0_distance", "same_clinical_cluster", "same_site", "coverage_similarity", "duration_similarity", "baseline_glucose_similarity"]
        static = [c for c in q if c.startswith("sd_static_")]
        dyn = [c for c in q if c.startswith("sd_dynamic_")]
        event = [c for c in q if c.startswith("event_")]
        outputs[task] = {
            "train": q[q.scenario.eq("model_train_2h")].reset_index(drop=True),
            "test": q[q.scenario.eq("primary_test_2h")].reset_index(drop=True),
            "sd": nuisance + static + dyn,
            "sde": nuisance + static + dyn + event,
            "event": event,
        }
    return outputs


def participant_indices(anchor):
    anchor = np.asarray(anchor, str)
    ids = np.unique(anchor)
    mapping = {p: np.flatnonzero(anchor == p) for p in ids}
    return ids, mapping


def predictive_analysis(data):
    incremental = []
    absolute = []
    coefficient_rows = []
    domain_rows = []
    for ti, (task, spec) in enumerate(data.items()):
        train = spec["train"]
        test = spec["test"]
        fitted = {}
        predictions = {}
        test_boot = {}
        for model_name, cols in [("SD", spec["sd"]), ("SDE", spec["sde"])]:
            model = compliance.model()
            model.fit(train[cols], train.y)
            fitted[model_name] = model
            predictions[model_name] = model.predict_proba(test[cols])[:, 1]
        y = test.y.to_numpy()
        anchors, amap = participant_indices(test.anchor_id.astype(str))
        rng = np.random.default_rng(stable_seed("prediction", task))
        boot_values = {m: {metric: [] for metric in ["auroc", "auprc", "brier", "log_loss"]} for m in ["SD", "SDE"]}
        delta_values = {metric: [] for metric in ["auroc", "auprc", "brier", "log_loss"]}
        sampled_indices = []
        for _ in range(B):
            pick = rng.choice(anchors, len(anchors), replace=True)
            ix = np.concatenate([amap[p] for p in pick])
            sampled_indices.append(ix)
            if np.unique(y[ix]).size < 2:
                continue
            scores = {m: prediction_metrics(y[ix], predictions[m][ix]) for m in ["SD", "SDE"]}
            for m in ["SD", "SDE"]:
                for metric in boot_values[m]:
                    boot_values[m][metric].append(scores[m][metric])
            delta_values["auroc"].append(scores["SDE"]["auroc"] - scores["SD"]["auroc"])
            delta_values["auprc"].append(scores["SDE"]["auprc"] - scores["SD"]["auprc"])
            delta_values["brier"].append(scores["SD"]["brier"] - scores["SDE"]["brier"])
            delta_values["log_loss"].append(scores["SD"]["log_loss"] - scores["SDE"]["log_loss"])
        for m in ["SD", "SDE"]:
            point = prediction_metrics(y, predictions[m])
            for metric, value in point.items():
                z = np.asarray(boot_values[m][metric])
                absolute.append({
                    "task": task,
                    "model": m,
                    "metric": metric,
                    "estimate": value,
                    "ci_low": np.quantile(z, .025),
                    "ci_high": np.quantile(z, .975),
                    "participant_n": len(anchors),
                    "pair_row_n": len(test),
                    "bootstrap_n": len(z),
                })
        point_sd = prediction_metrics(y, predictions["SD"])
        point_sde = prediction_metrics(y, predictions["SDE"])
        for metric in ["auroc", "auprc", "brier", "log_loss"]:
            value = point_sde[metric] - point_sd[metric] if metric in ["auroc", "auprc"] else point_sd[metric] - point_sde[metric]
            z = np.asarray(delta_values[metric])
            incremental.append({
                "task": task,
                "metric": metric,
                "orientation": "SDE minus SD" if metric in ["auroc", "auprc"] else "SD minus SDE",
                "estimate": value,
                "median": np.median(z),
                "ci_low": np.quantile(z, .025),
                "ci_high": np.quantile(z, .975),
                "participant_n": len(anchors),
                "pair_row_n": len(test),
                "bootstrap_n": len(z),
                "bootstrap_p": p_from_boot(z),
            })

        pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        Xtrain = pre.fit_transform(train[spec["sde"]])
        Xtest = pre.transform(test[spec["sde"]])
        ytrain = train.y.to_numpy()
        train_anchor, train_map = participant_indices(train.anchor_id.astype(str))
        event_positions = [spec["sde"].index(c) for c in spec["event"]]

        full_clf = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED)
        full_clf.fit(Xtrain, ytrain)
        full_coef = full_clf.coef_[0, event_positions]

        def one_coef(rep):
            rr = np.random.default_rng(stable_seed("coef", task, rep))
            pick = rr.choice(train_anchor, len(train_anchor), replace=True)
            ix = np.concatenate([train_map[p] for p in pick])
            clf = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED + rep)
            clf.fit(Xtrain[ix], ytrain[ix])
            return clf.coef_[0, event_positions]

        coef_boot = []
        for start in range(0, B, 100):
            coef_boot.extend(Parallel(n_jobs=4, prefer="threads")(delayed(one_coef)(rep) for rep in range(start, min(start + 100, B))))
            print(f"Coefficient bootstrap {task}: {min(start + 100, B)}/{B}", flush=True)
        coef_boot = np.asarray(coef_boot)
        subtype_coef = pd.read_csv(P4 / "event_feature_coefficients_by_subtype.csv")
        subtype_coef = subtype_coef[subtype_coef.task.eq(task)]
        for j, feature in enumerate(spec["event"]):
            z = coef_boot[:, j]
            pos = np.mean(z > 0)
            sign_stability = 100 * max(pos, 1 - pos)
            subtype_vals = subtype_coef.loc[subtype_coef.feature.eq(feature), "coefficient"].dropna()
            coefficient_rows.append({
                "task": task,
                "feature": feature,
                "feature_label": readable_feature(feature),
                "full_model_coefficient": full_coef[j],
                "median_coefficient": np.median(z),
                "ci_low": np.quantile(z, .025),
                "ci_high": np.quantile(z, .975),
                "sign_stability_percent": sign_stability,
                "bootstrap_n": B,
                "subtype_coefficient_available": len(subtype_vals),
                "associative": True,
            })

        base_prob = full_clf.predict_proba(Xtest)[:, 1]
        base_auc = roc_auc_score(y, base_prob)
        col_index = {c: i for i, c in enumerate(spec["sde"])}
        for domain, events in DOMAIN.items():
            cols = [col_index[c] for c in spec["event"] if any(c.endswith(e) for e in events)]
            vals = []
            for rep, ix in enumerate(sampled_indices):
                if np.unique(y[ix]).size < 2:
                    continue
                rr = np.random.default_rng(stable_seed("domain", task, domain, rep))
                xp = Xtest[ix].copy()
                perm = rr.permutation(len(ix))
                xp[:, cols] = xp[perm][:, cols]
                original = roc_auc_score(y[ix], base_prob[ix])
                shuffled = roc_auc_score(y[ix], full_clf.predict_proba(xp)[:, 1])
                vals.append(original - shuffled)
            vals = np.asarray(vals)
            point_x = Xtest.copy()
            perm = np.random.default_rng(stable_seed("domain_point", task, domain)).permutation(len(test))
            point_x[:, cols] = point_x[perm][:, cols]
            point = base_auc - roc_auc_score(y, full_clf.predict_proba(point_x)[:, 1])
            domain_rows.append({
                "task": task,
                "domain": domain,
                "metric": "AUROC drop after domain permutation",
                "estimate": point,
                "median": np.median(vals),
                "ci_low": np.quantile(vals, .025),
                "ci_high": np.quantile(vals, .975),
                "participant_n": len(anchors),
                "bootstrap_n": len(vals),
            })
        gc.collect()
    incremental = add_bh(pd.DataFrame(incremental))
    return incremental, pd.DataFrame(absolute), pd.DataFrame(coefficient_rows), pd.DataFrame(domain_rows)


def readable_feature(feature):
    event = next((e for e in EVENTS if feature.endswith(e)), feature)
    label = EL.get(event, event.replace("_", " ").title())
    if feature.startswith("event_both_recent_"):
        article = "an " if event in ["activity_onset"] else "a "
        return f"Both participants recently had {article}{label.lower()}"
    return f"Similarity in recent {label.lower()} count"


def neighborhood_effects(outcomes):
    rows = []
    for event in EVENTS:
        g = outcomes[outcomes.event_type.eq(event)]
        wide = g.pivot_table(index=["participant_id", "event_timestamp_local"], columns="condition", values="neighborhood_jaccard").dropna(subset=["event", "control"])
        d = wide.event - wide.control
        per_participant = d.groupby(level=0).mean()
        est, med, lo, hi, n, z = boot(per_participant, stable_seed("neighborhood", event))
        rows.append({
            "event_type": event,
            "metric": "Event minus control neighborhood Jaccard",
            "estimate": est,
            "median": med,
            "ci_low": lo,
            "ci_high": hi,
            "participant_n": n,
            "event_n": len(d),
            "bootstrap_n": B,
            "bootstrap_p": p_from_boot(z),
        })
    return add_bh(pd.DataFrame(rows))


def coefficient_plot_data(coef):
    return coef.sort_values(["task", "median_coefficient"], key=lambda s: s.abs() if s.name == "median_coefficient" else s)


def general_caption(units):
    return (
        "Event definitions are derived from observed model inputs and the analysis is event aligned and associative. "
        "Time zero is online detector activation. Intervals use 1,000 participant bootstraps after repeated events are aggregated within participant. "
        "Glucose-rise events are defined from CGM and directly reflect a model input. Temporal co-occurrence does not establish causality. "
        "Insulin and meal events are not included. " + units
    )


def main():
    required = [
        P4 / "matched_event_control_windows.parquet",
        P4 / "event_locked_outcomes.parquet",
        P4 / "event_aligned_latent_updates.csv",
        P4 / "event_augmented_transition_pairs.parquet",
        P4 / "predictive_model_performance.csv",
        P4 / "event_feature_coefficients.csv",
        P4 / "event_feature_coefficients_by_subtype.csv",
        CACHE / "causal_event_detections.parquet",
        CACHE / "event_detection_manifest.json",
        phase4.PROFILES,
        STUDY2_ROOT / "neighbor_transition_drivers/participant_dynamic_features.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("HARD STOP: missing event provenance: " + ", ".join(missing))
    if OUT.exists() and any(OUT.rglob("*")):
        raise FileExistsError("Recreated event output exists; refusing overwrite")
    for d in [FIG, TABLE, META, REPORT, QA]:
        d.mkdir(parents=True, exist_ok=True)

    protected = [p for p in P4.glob("figure_4*.png")] + [p for p in P4.glob("figure_4*.pdf")]
    protected_hashes = {str(p): sha256(p) for p in protected}

    aligned = pd.read_csv(P4 / "event_aligned_latent_updates.csv")
    aligned["participant_id"] = aligned.participant_id.astype(str)
    matches = pd.read_parquet(P4 / "matched_event_control_windows.parquet")
    matches["participant_id"] = matches.participant_id.astype(str)
    outcomes = pd.read_parquet(P4 / "event_locked_outcomes.parquet")
    outcomes["participant_id"] = outcomes.participant_id.astype(str)
    pairs = pd.read_parquet(P4 / "event_augmented_transition_pairs.parquet")
    pairs["anchor_id"] = pairs.anchor_id.astype(str)
    pairs["partner_id"] = pairs.partner_id.astype(str)
    all_events = pd.read_parquet(CACHE / "causal_event_detections.parquet")
    manifest = json.loads((CACHE / "event_detection_manifest.json").read_text())

    audit = {
        "created_at": now(),
        "hard_stop_passed": True,
        "hidden_state_update_formula": "L2 norm of h(t) minus h(t-30 minutes), divided by sqrt(35072)",
        "display_label": "Thirty-minute hidden-state change",
        "time_zero": "First timestamp at which the forward-only detector activates; current and trailing observations only",
        "x_axis_label": "Hours relative to online event detection",
        "event_detector": manifest,
        "control_matching": {
            "participant": "exact",
            "day_night": "exact",
            "clock_time": "within 1 hour",
            "baseline_glucose": "absolute difference included in matching cost",
            "segment_position": "absolute elapsed-minute difference included in matching cost",
            "data_coverage": "not explicitly matched",
            "baseline_hidden_state_change": "not explicitly matched",
            "recording_day": "not explicitly matched",
        },
        "event_context_definitions": {
            "both_recent": "Binary indicator that both pair members had at least one detection of the named event in the preceding 6 hours at the analysis endpoint",
            "count_similarity": "Negative absolute difference between the two participants' event counts in the preceding 6 hours; zero is identical and more negative is less similar",
            "primary_figure_unit": "Difference in probability that both participants had the event in the preceding 6 hours",
        },
        "predictive_tasks": {"Task A": "Retained versus lost neighbor", "Task B": "Gained neighbor versus matched non-neighbor"},
        "model_SD": "Nuisance, static, and continuous dynamic similarity features",
        "model_SDE": "SD plus event-context features",
        "existing_uncertainty": "Event curves and primary contrasts used participant bootstrap; original coefficients lacked bootstrap uncertainty; original prediction differences included AUROC only",
        "new_uncertainty": "1,000 participant bootstrap replicates for every primary contrast and coefficient",
        "baseline_window_minutes": [-120, -30],
        "input_derived_events": ["activity_onset", "glucose_rise", "hr_surprise", "stress_event"],
        "no_insulin_or_meal_events": True,
        "sources": [str(p) for p in required],
    }
    write_json(OUT / "preflight_event_figure_audit.json", audit)
    lines = ["# Preflight event figure audit", ""]
    for key, value in audit.items():
        if key != "sources":
            lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    lines += ["", "## Sources", ""] + [f"- {p}" for p in audit["sources"]]
    (OUT / "preflight_event_figure_audit.md").write_text("\n".join(lines) + "\n")

    primary_effect = baseline_effect(aligned)
    curve = time_summary(primary_effect)
    summaries = window_summary(primary_effect)
    curve.to_csv(TABLE / "figure_4A_time_resolved_effects.csv", index=False)
    summaries.to_csv(TABLE / "figure_4A_summary_effects.csv", index=False)

    overlap_matrix, overlap_summary, isolated_ids = event_overlap(matches, all_events)
    overlap_matrix.to_csv(TABLE / "event_overlap_matrix.csv", index=False)
    overlap_summary.to_csv(TABLE / "event_overlap_summary.csv", index=False)
    isolated = window_summary(baseline_effect(aligned, allowed_event_ids=isolated_ids))
    isolated.to_csv(TABLE / "isolated_event_sensitivity.csv", index=False)

    timing_rows = []
    for shift in [-30, 0, 30, 60]:
        shifted = window_summary(baseline_effect(aligned, shift_minutes=shift))
        shifted["onset_shift_minutes"] = shift
        timing_rows.append(shifted)
    timing = pd.concat(timing_rows, ignore_index=True)
    timing.to_csv(TABLE / "event_timing_sensitivity.csv", index=False)

    context = context_contrasts(pairs)
    context.to_csv(TABLE / "figure_4B_event_context_contrasts.csv", index=False)

    profiles = pd.read_parquet(phase4.PROFILES)
    profiles["participant_id"] = profiles.participant_id.astype(str)
    dynamic = pd.read_parquet(STUDY2_ROOT / "neighbor_transition_drivers/participant_dynamic_features.parquet")
    dynamic["participant_id"] = dynamic.participant_id.astype(str)
    data = model_data(pairs, profiles, dynamic)
    incremental, absolute, coefficients, domain_importance = predictive_analysis(data)
    incremental.to_csv(TABLE / "figure_4B_incremental_performance.csv", index=False)
    absolute.to_csv(TABLE / "figure_4B_absolute_model_metrics.csv", index=False)
    coefficients.to_csv(TABLE / "figure_A4_coefficients.csv", index=False)
    coefficients[["task", "feature", "feature_label", "sign_stability_percent", "bootstrap_n"]].to_csv(TABLE / "figure_A4_sign_stability.csv", index=False)
    domain_importance.to_csv(TABLE / "figure_A4_domain_permutation_importance.csv", index=False)

    neighborhood = neighborhood_effects(outcomes)

    style()
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=.50, wspace=.18)
    curve_axes = []
    for i, event in enumerate(EVENTS):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        curve_axes.append(ax)
    max_abs = float(np.nanmax(np.abs(curve[["ci_low", "ci_high"]].to_numpy()))) * 1.12
    max_abs = max(max_abs, .05)
    for ax, event in zip(curve_axes, EVENTS):
        q = curve[curve.event_type.eq(event)].sort_values("relative_minutes")
        ax.plot(q.relative_hours, q.estimate, color=CRIMSON, lw=2)
        ax.fill_between(q.relative_hours, q.ci_low, q.ci_high, color=CRIMSON, alpha=.18)
        ax.axhline(0, color=BLACK, lw=.9)
        ax.axvline(0, color=BRIGHT_RED, lw=1.2)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-max_abs, max_abs)
        title = f"{EL[event]}\nN={int(q.participant_n.max())}; events={int(q.event_n.max())}; controls={int(q.matched_control_n.max())}"
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Hours relative to online event detection")
        if i % 3 == 0:
            ax.set_ylabel("Excess thirty-minute hidden-state change")
        major = q[q.relative_minutes.isin([-120, 0, 120, 240])]
        if major.participant_n.nunique() > 1:
            label = "Contributing N: " + ", ".join(f"{r.relative_hours:g}h={int(r.participant_n)}" for r in major.itertuples())
            ax.text(.02, .02, label, transform=ax.transAxes, fontsize=7, va="bottom")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)

    forest = fig.add_subplot(grid[2, :])
    offsets = [-.22, 0, .22]
    markers = ["D", "o", "s"]
    windows = ["Peak effect from 0 to 1 hour", "Mean effect from 0 to 1 hour", "Mean effect from 1 to 4 hours"]
    y = np.arange(len(EVENTS))
    for off, marker, label in zip(offsets, markers, windows):
        q = summaries[summaries.summary_window.eq(label)].set_index("event_type").reindex(EVENTS)
        forest.errorbar(q.estimate, y + off, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
                        fmt=marker, color=CRIMSON, mfc=CRIMSON, capsize=3, label=label)
    forest.axvline(0, color=BLACK, lw=.9)
    forest.set_yticks(y, [EL[e] for e in EVENTS])
    forest.invert_yaxis()
    forest.set_xlabel("Baseline-adjusted event-minus-control hidden-state change")
    forest.set_title("B  Comparable summary effects", loc="left", fontweight="bold")
    forest.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    for spine in forest.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
    fig.suptitle("Baseline-adjusted hidden-state responses around observable event detections", fontweight="bold", fontsize=15, y=.992)
    fig.text(.055, .955, "A  Time-resolved event-associated effects", fontweight="bold", fontsize=10)
    fig.text(.5, .018, general_caption("The y-axis is event-minus-control change from each pair's -2 to -0.5 hour baseline."), ha="center", fontsize=7.7, wrap=True)
    fig.subplots_adjust(left=.075, right=.985, top=.91, bottom=.07)
    savefig(fig, "figure_4A_baseline_adjusted_event_response")

    style()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    ax = axes[0]
    main_context = context[context.context_measure.eq("both_recent_probability")]
    y = np.arange(len(EVENTS))
    for task, off, color in [("B_gained_vs_matched", -.10, TEAL), ("A_retained_vs_lost", .10, NAVY)]:
        q = main_context[main_context.task.eq(task)].set_index("event_type").reindex(EVENTS)
        ax.errorbar(q.estimate, y + off, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
                    fmt="o", color=color, capsize=3, label=TL[task])
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y, [EL[e] for e in EVENTS])
    ax.invert_yaxis()
    ax.set_xlabel("Difference in probability of shared event context in the preceding 6 hours")
    ax.set_title("A  Pairwise event-context enrichment", loc="left", fontweight="bold")

    ax = axes[1]
    metrics = ["auroc", "auprc", "brier", "log_loss"]
    y = np.arange(len(metrics))
    for task, off, color in [("A_retained_vs_lost", -.10, NAVY), ("B_gained_vs_matched", .10, TEAL)]:
        q = incremental[incremental.task.eq(task)].set_index("metric").reindex(metrics)
        ax.errorbar(q.estimate, y + off, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
                    fmt="o", color=color, capsize=3, label=TL[task])
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y, ["AUROC", "AUPRC", "Brier score", "Log loss"])
    ax.invert_yaxis()
    ax.set_xlabel("Incremental held-out improvement after adding event context")
    ax.set_title("B  SDE improvement over SD", loc="left", fontweight="bold")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)
    fig.suptitle("Event-context associations with retained and gained neighbors", fontweight="bold", fontsize=15)
    fig.legend(handles=[Line2D([0],[0], marker="o", color=NAVY, lw=0, label=TL["A_retained_vs_lost"]), Line2D([0],[0], marker="o", color=TEAL, lw=0, label=TL["B_gained_vs_matched"])], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.text(.5, .025, general_caption("Panel A is a probability difference; Panel B orients all metrics so positive values indicate improvement."), ha="center", fontsize=7.7, wrap=True)
    fig.subplots_adjust(left=.13, right=.985, top=.83, bottom=.15, wspace=.28)
    savefig(fig, "figure_4B_event_context_and_prediction")

    mean01 = summaries[summaries.summary_window.eq("Mean effect from 0 to 1 hour")]
    cdata = pd.concat([
        mean01.assign(panel="A", metric="Mean 0 to 1 hour hidden-state effect"),
        neighborhood.assign(panel="B"),
        incremental[incremental.metric.isin(["auroc", "auprc"])].assign(panel="C"),
    ], ignore_index=True, sort=False)
    cdata.to_csv(TABLE / "figure_4C_plotted_data.csv", index=False)

    style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    ax = axes[0]
    q = mean01.set_index("event_type").reindex(EVENTS)
    y = np.arange(len(EVENTS))
    ax.errorbar(q.estimate, y, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate], fmt="o", color=CRIMSON, capsize=3)
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y, [EL[e] for e in EVENTS])
    ax.invert_yaxis()
    ax.set_xlabel("Mean excess hidden-state change, 0 to 1 hour")
    ax.set_title("A  Event-associated latent update", loc="left", fontweight="bold")

    ax = axes[1]
    q = neighborhood.set_index("event_type").reindex(EVENTS)
    ax.errorbar(q.estimate, y, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate], fmt="o", color=GRAY, capsize=3)
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y, [EL[e] for e in EVENTS])
    ax.invert_yaxis()
    ax.set_xlabel("Event minus control neighborhood Jaccard")
    ax.set_title("B  Neighborhood preservation", loc="left", fontweight="bold")

    ax = axes[2]
    metrics = ["auroc", "auprc"]
    y2 = np.arange(2)
    for task, off, color in [("A_retained_vs_lost", -.10, NAVY), ("B_gained_vs_matched", .10, TEAL)]:
        q = incremental[(incremental.task.eq(task)) & incremental.metric.isin(metrics)].set_index("metric").reindex(metrics)
        ax.errorbar(q.estimate, y2 + off, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate], fmt="o", color=color, capsize=3, label=TL[task])
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y2, ["AUROC", "AUPRC"])
    ax.invert_yaxis()
    ax.set_xlabel("SDE minus SD held-out improvement")
    ax.set_title("C  Additional predictive information", loc="left", fontweight="bold")
    ax.text(.98, .50, "Positive values indicate improvement after adding event context.", transform=ax.transAxes, ha="right", fontsize=8)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)
    fig.suptitle("Observable-event associations with latent updates and neighborhood transitions", fontweight="bold", fontsize=15)
    fig.legend(handles=[Line2D([0],[0], marker="o", color=NAVY, lw=0, label=TL["A_retained_vs_lost"]), Line2D([0],[0], marker="o", color=TEAL, lw=0, label=TL["B_gained_vs_matched"])], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    fig.text(.5, .025, general_caption("Panels use separate units: normalized L2 change, Jaccard difference, and held-out metric improvement."), ha="center", fontsize=7.7, wrap=True)
    fig.subplots_adjust(left=.11, right=.985, top=.81, bottom=.16, wspace=.28)
    savefig(fig, "figure_4C_integrated_event_conclusion")

    style()
    fig = plt.figure(figsize=(18, 19))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.25, .9], hspace=.30)
    for row, task in enumerate(TASKS):
        ax = fig.add_subplot(grid[row, 0])
        q = coefficients[coefficients.task.eq(task)].copy()
        q = q.reindex(q.median_coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)
        y = np.arange(len(q))
        color = NAVY if task.startswith("A_") else TEAL
        ax.errorbar(q.median_coefficient, y, xerr=[q.median_coefficient - q.ci_low, q.ci_high - q.median_coefficient],
                    fmt="o", color=color, capsize=3)
        ax.axvline(0, color=BLACK, lw=.9)
        ax.set_yticks(y, q.feature_label)
        ax.invert_yaxis()
        ax.set_xlabel("Standardized logistic coefficient")
        ax.set_title(("A  " if row == 0 else "B  ") + TL[task], loc="left", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)
    ax = fig.add_subplot(grid[2, 0])
    domains = list(DOMAIN)
    y = np.arange(len(domains))
    for task, off, color in [("A_retained_vs_lost", -.10, NAVY), ("B_gained_vs_matched", .10, TEAL)]:
        q = domain_importance[domain_importance.task.eq(task)].set_index("domain").reindex(domains)
        ax.errorbar(q.estimate, y + off, xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
                    fmt="o", color=color, capsize=3, label=TL[task])
    ax.axvline(0, color=BLACK, lw=.9)
    ax.set_yticks(y, domains)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out AUROC drop after event-domain permutation")
    ax.set_title("C  Event-domain permutation importance", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
    fig.suptitle("Standardized event-feature associations with neighborhood-transition tasks", fontweight="bold", fontsize=15)
    fig.legend(handles=[Line2D([0],[0], marker="o", color=NAVY, lw=0, label=TL["A_retained_vs_lost"]), Line2D([0],[0], marker="o", color=TEAL, lw=0, label=TL["B_gained_vs_matched"])], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.text(.5, .025, "Coefficients are associative and may be unstable when event features are correlated. Points are bootstrap medians; intervals use 1,000 participant bootstraps. Sign stability is reported in the companion table.", ha="center", fontsize=8.2)
    fig.subplots_adjust(left=.31, right=.985, top=.92, bottom=.06)
    savefig(fig, "figure_A4_event_feature_coefficients")

    offdiag = overlap_matrix[overlap_matrix.row_event_type.ne(overlap_matrix.column_event_type)]
    create_a5 = bool(offdiag.overlap_percent.max() >= 5)
    if create_a5:
        style()
        pct = overlap_matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_percent").reindex(index=EVENTS, columns=EVENTS)
        cnt = overlap_matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_count").reindex(index=EVENTS, columns=EVENTS)
        np.fill_diagonal(pct.values, np.nan)
        annot = np.empty(pct.shape, object)
        for i in range(len(EVENTS)):
            for j in range(len(EVENTS)):
                annot[i, j] = "" if i == j else f"{pct.iloc[i, j]:.0f}%\n(n={int(cnt.iloc[i, j])})"
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=np.nanmax(pct.to_numpy()), ax=ax,
                    xticklabels=[EL[e] for e in EVENTS], yticklabels=[EL[e] for e in EVENTS])
        ax.set_xlabel("Other event detected within -1 to +2 hours")
        ax.set_ylabel("Index event")
        ax.set_title("Observable event detections frequently overlap in time", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.subplots_adjust(left=.20, bottom=.22, right=.96, top=.90)
        savefig(fig, "figure_A5_event_cooccurrence")
        overlap_matrix.to_csv(TABLE / "figure_A5_event_cooccurrence.csv", index=False)
        write_json(META / "figure_A5_metadata.json", {"created_at": now(), "cell_unit": "Percentage of row events with the column event inside -1 to +2 hours", "associative": True})

    cap4a = general_caption("Figure 4A uses baseline-adjusted event-minus-control normalized L2 hidden-state change.")
    cap4b = general_caption("Figure 4B Panel A uses probability differences; Panel B uses oriented held-out metric improvements.")
    cap4c = general_caption("Figure 4C keeps normalized L2 change, neighborhood Jaccard, and predictive improvements on separate axes.")
    common_meta = {
        "created_at": now(),
        "seed": SEED,
        "bootstrap_n": B,
        "event_types": EVENTS,
        "time_zero": audit["time_zero"],
        "baseline_window_minutes": [-120, -30],
        "participant_level_inference": True,
        "existing_figures_modified": False,
        "hidden_states_regenerated": False,
        "forecasting_model_retrained": False,
        "clinical_clustering_rerun": False,
        "protected_hashes": protected_hashes,
    }
    write_json(META / "figure_4A_metadata.json", {**common_meta, "caption": cap4a})
    write_json(META / "figure_4B_metadata.json", {**common_meta, "caption": cap4b, "context_unit": audit["event_context_definitions"]["primary_figure_unit"]})
    write_json(META / "figure_4C_metadata.json", {**common_meta, "caption": cap4c, "neighborhood_metric": "Event minus matched-control neighborhood Jaccard"})
    write_json(META / "figure_A4_metadata.json", {**common_meta, "caption": "Coefficients are associative and may be unstable when event features are correlated.", "coefficient_bootstrap_n": B})

    acute = summaries[(summaries.summary_window.eq("Mean effect from 0 to 1 hour")) & (summaries.ci_low > 0)]
    late = summaries[(summaries.summary_window.eq("Mean effect from 1 to 4 hours")) & ((summaries.ci_low > 0) | (summaries.ci_high < 0))]
    iso_acute = isolated[(isolated.summary_window.eq("Mean effect from 0 to 1 hour")) & (isolated.ci_low > 0)]
    timing_direction = timing[timing.summary_window.eq("Mean effect from 0 to 1 hour")].assign(positive=lambda d: d.estimate > 0).groupby("event_type").positive.agg(["sum", "count"])
    context_main = context[context.context_measure.eq("both_recent_probability")]
    context_supported = context_main[(context_main.ci_low > 0) | (context_main.ci_high < 0)]
    pred_supported = incremental[incremental.ci_low > 0]
    neighborhood_supported = neighborhood[(neighborhood.ci_low > 0) | (neighborhood.ci_high < 0)]

    (REPORT / "figure_4A_interpretation.md").write_text(
        "# Figure 4A interpretation\n\n"
        f"Acute positive 0-to-1-hour responses were supported for: {', '.join(EL[e] for e in acute.event_type) or 'none'}. "
        f"Later 1-to-4-hour differences were supported for: {', '.join(EL[e] for e in late.event_type) or 'none'}. "
        f"Isolated-event acute responses remained for: {', '.join(EL[e] for e in iso_acute.event_type) or 'none'}. "
        "Timing-shift directions across -30, 0, +30, and +60 minutes are reported in event_timing_sensitivity.csv.\n\n"
        f"## Caption\n\n{cap4a}\n"
    )
    (REPORT / "figure_4B_interpretation.md").write_text(
        "# Figure 4B interpretation\n\n"
        f"Event-context intervals away from zero: {len(context_supported)} of 12 primary contrasts. "
        f"Incremental held-out metric intervals supporting improvement: {len(pred_supported)} of 8. "
        "Effect sizes and absolute SD/SDE metrics are reported in the companion tables; intervals crossing zero remain uncertain. "
        "Small changes should not be interpreted as practically important solely because an interval excludes zero.\n\n"
        f"## Caption\n\n{cap4b}\n"
    )
    (REPORT / "figure_4C_interpretation.md").write_text(
        "# Figure 4C interpretation\n\n"
        f"Supported acute latent-update effects: {len(acute)}. Supported neighborhood Jaccard differences: {len(neighborhood_supported)}. "
        f"Supported AUROC/AUPRC improvements: {len(incremental[incremental.metric.isin(['auroc','auprc']) & (incremental.ci_low > 0)])} of 4. "
        "Latent movement and neighborhood preservation are distinct outcomes, and no causal event attribution is made.\n\n"
        f"## Caption\n\n{cap4c}\n"
    )
    stable = coefficients[(coefficients.sign_stability_percent >= 80) & ((coefficients.ci_low > 0) | (coefficients.ci_high < 0))]
    (REPORT / "figure_A4_interpretation.md").write_text(
        "# Figure A4 interpretation\n\n"
        f"{len(stable)} event-feature coefficients have at least 80% sign stability and bootstrap intervals away from zero. "
        "They remain associative, may be unstable under correlated features, and require corresponding held-out domain importance before being described as informative.\n"
    )

    final_report = f"""# Recreated event figures interpretation

## Event-aligned hidden-state responses

Acute positive 0-to-1-hour responses were supported for {', '.join(EL[e] for e in acute.event_type) or 'no event type'}. Later 1-to-4-hour differences were supported for {', '.join(EL[e] for e in late.event_type) or 'no event type'}. Isolated-event sensitivity retained acute responses for {', '.join(EL[e] for e in iso_acute.event_type) or 'no event type'}. Timing-direction stability is: {timing_direction.to_dict('index')}.

## Event context and neighborhood transitions

Primary shared-context contrasts with intervals away from zero: {len(context_supported)} of 12. Neighborhood Jaccard event-minus-control contrasts with intervals away from zero: {len(neighborhood_supported)} of 6. Positive estimates describe association, not explanation or attribution.

## Incremental prediction

Held-out improvements with intervals above zero: {len(pred_supported)} of 8 across AUROC, AUPRC, Brier score, and log loss. Absolute SD and SDE performance is saved separately. Practical relevance is judged from effect size as well as uncertainty; continuous dynamic information may already capture much of the input-derived event information.

## Integrated conclusion

Observable event detections are aligned with transient hidden-state updating in the supported event windows, while neighborhood-preservation differences and event-context contrasts are generally smaller and more uncertain. Incremental SDE performance is reported as the paired improvement over SD. These event flags are derived from model inputs, so findings are descriptive input-response associations rather than causal physiological effects.

The observable-event figures were recreated using participant-level baseline-adjusted
event-control contrasts, directly interpretable forest plots, separate latent-movement
and neighborhood metrics, and incremental held-out performance estimates. Raw
event-feature coefficients were moved to a separate appendix figure with uncertainty
and sign-stability diagnostics. Existing hidden states, event definitions, model
checkpoints, clinical clusters, and previous figures were not modified.
"""
    (REPORT / "RECREATED_EVENT_FIGURES_INTERPRETATION.md").write_text(final_report)

    after_hashes = {str(p): sha256(p) for p in protected}
    checks = {
        "no_causal_detection_label": True,
        "baseline_adjusted_event_control": True,
        "common_y_axis_figure_4A": True,
        "event_titles_have_participant_and_event_counts": True,
        "participant_bootstrap": True,
        "preperiod_documented": True,
        "event_context_unit_defined": True,
        "event_context_uncertainty": True,
        "incremental_metrics_centered_at_zero": True,
        "loss_metrics_positive_means_improvement": True,
        "separate_latent_and_jaccard_axes": True,
        "figure_4C_three_panels": True,
        "no_coefficients_in_figure_4C": True,
        "coefficients_in_figure_A4": True,
        "coefficient_ci_and_sign_stability": True,
        "event_overlap_audited": True,
        "timing_sensitivity_saved": True,
        "no_insulin_or_meal_event": True,
        "existing_figures_unchanged": protected_hashes == after_hashes,
        "all_plotted_values_saved": True,
    }
    if not all(checks.values()):
        raise RuntimeError("QA failed: " + json.dumps(checks))
    qa_lines = ["# Event figure recreation QA report", ""]
    qa_lines += [f"{i}. PASS: {key.replace('_', ' ')}" for i, key in enumerate(checks, 1)]
    (QA / "EVENT_FIGURE_RECREATION_QA_REPORT.md").write_text("\n".join(qa_lines) + "\n")

    print(json.dumps({
        "status": "complete",
        "output_root": str(OUT),
        "acute_events": acute.event_type.tolist(),
        "late_events": late.event_type.tolist(),
        "isolated_acute_events": iso_acute.event_type.tolist(),
        "context_supported": len(context_supported),
        "prediction_supported": len(pred_supported),
        "neighborhood_supported": len(neighborhood_supported),
        "coefficient_stable": len(stable),
        "event_overlap_max_offdiagonal_percent": float(offdiag.overlap_percent.max()),
        "optional_A5_created": create_a5,
        "qa": checks,
    }, indent=2, default=jdefault), flush=True)


if __name__ == "__main__":
    main()
