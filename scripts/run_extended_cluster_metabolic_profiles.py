"""Phase 1 of the extended clinical hidden-state interpretation.

Reuses the immutable within-subtype preprocessing pipelines and centroids.  No
clustering model is fitted.  CGM summaries are recomputed only within the clean
segments used by the frozen streaming study.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import neighbor_transition_drivers as drivers  # noqa: E402
import neighbor_transition_dynamic_extract as dynamic_extract  # noqa: E402
import posthoc_phase1_literature_alignment as posthoc  # noqa: E402
from ssmcgm.analysis.within_subtype_config import DATASET, STUDY2_ROOT  # noqa: E402
from ssmcgm.data.aireadi import AireadiSchema, prepare_aireadi_panel  # noqa: E402

EXTENDED = STUDY2_ROOT / "extended_clinical_latent_dynamics_v1"
OUT = EXTENDED / "01_cluster_metabolic_profiles"
REPORTS = EXTENDED / "reports"
LOGS = EXTENDED / "logs"
SEED = 42
SUBTYPES = list(posthoc.CANONICAL_STRATA)
PALETTES = {
    "healthy": ["#003366", "#5B7FA3"],
    "pre_diabetes": ["#2F7F7F", "#7BC4C4"],
    "t2d_oral_non_insulin": ["#7A1F1F", "#BA2828", "#E57373"],
    "insulin_dependent": ["#4A5568", "#7A8798", "#B0BAC6"],
}
SUBTYPE_LABEL = {
    "healthy": "Healthy", "pre_diabetes": "Pre-diabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent*",
}
FACTORS = posthoc.FACTORS
FACTOR_LABELS = {
    "participants_age": "Study-visit age\n(years)",
    "bmi_baseline": "BMI\n(kg/m²)",
    "hba1c_percent_baseline": "HbA1c\n(%)",
    "c_peptide_ngml_baseline": "C-peptide\n(ng/mL)",
    "tg_hdl_ratio": "TG/HDL\n(ratio)",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
}
METABOLIC = [
    ("cgm_mean", "Mean CGM"), ("cgm_cv", "CGM CV"),
    ("cgm_time_in_range", "Time in range"),
    ("cgm_time_above_180", "Time above 180"),
    ("cgm_time_below_70", "Time below 70"),
    ("cgm_masd", "Mean absolute\nsuccessive difference"),
    ("hba1c_percent_baseline", "HbA1c"),
    ("c_peptide_ngml_baseline", "C-peptide"), ("tg_hdl_ratio", "TG/HDL"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x)) + "\n")


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.spines.top": True,
        "axes.spines.right": True, "grid.color": "#D9D9D9",
        "grid.linewidth": 0.7, "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42,
    })


def clean_cgm_summaries() -> tuple[pd.DataFrame, dict]:
    available = set(pq.read_schema(DATASET).names)
    columns = [c for c in dynamic_extract.DYNAMIC_COLUMNS if c in available]
    data = pd.read_parquet(DATASET, columns=columns)
    panel = prepare_aireadi_panel(data, AireadiSchema())
    panel["participant_id"] = panel.participant_id.astype(str)
    rows = []
    for pid, group in panel.groupby("participant_id", sort=True):
        glucose = pd.to_numeric(group.cgm_glucose_mean, errors="coerce").to_numpy(float)
        glucose = glucose[np.isfinite(glucose)]
        diffs = []
        for _, segment in group.groupby("segment_id", sort=True):
            x = pd.to_numeric(segment.cgm_glucose_mean, errors="coerce").to_numpy(float)
            valid = np.isfinite(x[1:]) & np.isfinite(x[:-1])
            diffs.extend(np.diff(x)[valid].tolist())
        diffs = np.asarray(diffs, float)
        mean = float(np.mean(glucose))
        sd = float(np.std(glucose, ddof=1)) if len(glucose) > 1 else np.nan
        rows.append({
            "participant_id": pid,
            "clean_segment_count": int(group.segment_id.nunique()),
            "clean_cgm_observation_count": int(len(glucose)),
            "clean_recording_hours": float(len(group) * 5 / 60),
            "cgm_mean": mean, "cgm_median": float(np.median(glucose)),
            "cgm_sd": sd, "cgm_cv": float(sd / mean) if mean else np.nan,
            "cgm_time_in_range": float(np.mean((glucose >= 70) & (glucose <= 180))),
            "cgm_time_above_180": float(np.mean(glucose > 180)),
            "cgm_time_above_250": float(np.mean(glucose > 250)),
            "cgm_time_below_70": float(np.mean(glucose < 70)),
            "cgm_masd": float(np.mean(np.abs(diffs))) if len(diffs) else np.nan,
            "cgm_range": float(np.max(glucose) - np.min(glucose)),
            "hyperglycemic_excursion_count": np.nan,
        })
    report = {
        "participant_count": len(rows), "clean_segment_count": int(panel[["participant_id", "segment_id"]].drop_duplicates().shape[0]),
        "clean_segment_definition": "Frozen streaming preprocessing: contiguous 5-minute rows, split at long core-modality gaps, minimum valid segment length retained.",
        "aggregation": "All valid observations across every retained clean segment per participant; successive differences never cross segment boundaries.",
        "hyperglycemic_excursion_count": "Not estimated because no previously validated hyperglycemic-excursion onset definition was found. The existing slope-excursion count was not relabeled as hyperglycemia.",
    }
    return pd.DataFrame(rows), report


def recover_frozen_memberships() -> tuple[pd.DataFrame, dict]:
    frame, provenance = posthoc.load_frame()
    labeled, manifest = drivers.recover_labels(frame)
    labeled["participant_id"] = labeled.participant_id.astype(str)
    expected = {s: int(manifest["clusters"][s]["selected_k"]) for s in SUBTYPES}
    got = labeled.groupby("canonical_stratum").display_cluster.nunique().astype(int).to_dict()
    if got != expected:
        raise RuntimeError(f"Frozen k mismatch: observed {got}, expected {expected}")
    return labeled, {"source": provenance, "selected_k": expected, "frozen_manifest": str(STUDY2_ROOT / "phase1_clinical_clustering/frozen_clustering_manifest.json")}


def clinical_figure(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    long = frame.melt(
        id_vars=["participant_id", "split", "canonical_stratum", "display_cluster"],
        value_vars=FACTORS, var_name="factor", value_name="value",
    ).dropna(subset=["value"])
    fig, axes = plt.subplots(4, 6, figsize=(19, 13), constrained_layout=False)
    rng = np.random.default_rng(SEED)
    for i, subtype in enumerate(SUBTYPES):
        sf = frame[frame.canonical_stratum == subtype]
        clusters = sorted(sf.display_cluster.astype(int).unique())
        counts = sf.groupby("display_cluster").participant_id.nunique().to_dict()
        for j, factor in enumerate(FACTORS):
            ax = axes[i, j]
            values = [pd.to_numeric(sf.loc[sf.display_cluster == c, factor], errors="coerce").dropna().to_numpy(float) for c in clusters]
            bp = ax.boxplot(values, positions=np.arange(len(clusters)), widths=.55, patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "linewidth": 1.2}, whiskerprops={"color": "black"}, capprops={"color": "black"})
            for patch, color in zip(bp["boxes"], PALETTES[subtype]):
                patch.set_facecolor(color); patch.set_alpha(.78); patch.set_edgecolor("black")
            for x, vals, color in zip(range(len(clusters)), values, PALETTES[subtype]):
                jitter = rng.uniform(-.16, .16, len(vals))
                ax.scatter(x + jitter, vals, s=10, alpha=.25, color=color, edgecolors="none", rasterized=True)
            ax.set_xticks(range(len(clusters)), [f"C{c}\nN={counts[c]}" for c in clusters])
            ax.set_xlabel("")
            ax.set_ylabel(FACTOR_LABELS[factor] if j == 0 else "")
            if i == 0: ax.set_title(FACTOR_LABELS[factor].replace("\n", " "), fontsize=10, fontweight="normal")
            # Subtype labels are positioned at figure level below to keep them separate from y-axis labels.
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values(): spine.set_visible(True); spine.set_color("black")
    for row_y, subtype in zip([.825, .615, .405, .195], SUBTYPES):
        fig.text(.018, row_y, SUBTYPE_LABEL[subtype], rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Clinical factor distributions define distinct profiles within diagnostic subtypes", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .012, "Age is study-visit age. Fasting status is unconfirmed for C-peptide and triglycerides. Clusters were frozen before post hoc interpretation.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.115, right=.99, top=.93, bottom=.075, hspace=.58, wspace=.34)
    png = OUT / "figure_1A_clinical_factor_distributions_recreated.png"
    pdf = OUT / "figure_1A_clinical_factor_distributions_recreated.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "figure_1A_clinical_factor_distributions_recreated_thumbnail.png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    long.to_csv(OUT / "figure_1A_plotted_data.csv", index=False)
    meta = {"created_at": now(), "participant_count": int(frame.participant_id.nunique()), "rows": 4, "columns": 6,
            "cluster_membership": "Recovered with immutable train-fitted pipelines and centroids; no clustering refit.", "palette": PALETTES,
            "raw_units": True, "insulin_dependent_exploratory": True}
    write_json(OUT / "figure_1A_metadata.json", meta)
    return long, meta


def metabolic_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    raw_rows = []
    for subtype in SUBTYPES:
        sf = frame[frame.canonical_stratum == subtype]
        subtype_mean = sf[[x[0] for x in METABOLIC]].mean()
        subtype_sd = sf[[x[0] for x in METABOLIC]].std(ddof=1).replace(0, np.nan)
        for cluster, group in sf.groupby("display_cluster", sort=True):
            row = {"canonical_stratum": subtype, "display_cluster": int(cluster), "cluster_n": int(group.participant_id.nunique()), "cgm_n": int(group.cgm_mean.notna().sum())}
            for variable, label in METABOLIC:
                vals = pd.to_numeric(group[variable], errors="coerce").dropna().to_numpy(float)
                row[variable] = float((np.mean(vals) - subtype_mean[variable]) / subtype_sd[variable]) if len(vals) and np.isfinite(subtype_sd[variable]) else np.nan
                if len(vals): q1, med, q3 = np.percentile(vals, [25, 50, 75])
                else: q1 = med = q3 = np.nan
                raw_rows.append({"canonical_stratum": subtype, "display_cluster": int(cluster), "cluster_n": int(group.participant_id.nunique()),
                                 "variable": variable, "label": label.replace("\n", " "), "n_nonmissing": len(vals), "median": med, "q1": q1, "q3": q3, "iqr": q3-q1})
            rows.append(row)
    z = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    z.to_csv(OUT / "cluster_metabolic_expression.csv", index=False)
    raw.to_csv(OUT / "cluster_metabolic_expression_raw_summary.csv", index=False)

    matrix = z.set_index(["canonical_stratum", "display_cluster"])[[x[0] for x in METABOLIC]]
    labels = [f"{SUBTYPE_LABEL[s]} C{c} (N={int(z[(z.canonical_stratum==s)&(z.display_cluster==c)].cluster_n.iloc[0])})" for s, c in matrix.index]
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    sns.heatmap(matrix.to_numpy(), cmap="vlag", center=0, vmin=-1.5, vmax=1.5, annot=True, fmt=".2f", linewidths=.7, linecolor="white",
                xticklabels=[x[1] for x in METABOLIC], yticklabels=labels, cbar_kws={"label": "Within-subtype standardized cluster mean"}, ax=ax)
    ax.set_title("Frozen clinical clusters show distinct glucose and metabolic-expression profiles", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=28, labelsize=9); ax.tick_params(axis="y", rotation=0, labelsize=9)
    for tick, (subtype, _) in zip(ax.get_yticklabels(), matrix.index): tick.set_color(PALETTES[subtype][0])
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_color("black")
    fig.tight_layout()
    fig.savefig(OUT / "figure_1B_cluster_metabolic_expression.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figure_1B_cluster_metabolic_expression.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure_1B_cluster_metabolic_expression_thumbnail.png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    write_json(OUT / "figure_1B_metadata.json", {"created_at": now(), "standardization": "Within-subtype across eligible participants; cluster cell is standardized arithmetic mean.",
        "raw_summary": "Median and IQR saved separately.", "baseline_insulin_included": False,
        "baseline_insulin_audit": "No participant-level baseline insulin laboratory measurement exists. med_insulin was not substituted.",
        "hyperglycemic_excursion_count_included": False, "palette": PALETTES})
    return z, raw


def interpretation(frame: pd.DataFrame, z: pd.DataFrame, provenance: dict, cgm_report: dict) -> None:
    lines = ["# Phase 1: frozen cluster clinical and metabolic profiles", "",
        "This descriptive post hoc phase recovered memberships from the immutable train-fitted preprocessing pipelines and centroids. It did not refit clustering, alter selected k, or revise C1/C2/C3 labels. CGM summaries use only retained clean segments and successive differences never cross segment boundaries.", ""]
    for subtype in SUBTYPES:
        lines += [f"## {SUBTYPE_LABEL[subtype]}", ""]
        parts=[]
        for _, row in z[z.canonical_stratum == subtype].sort_values("display_cluster").iterrows():
            c=int(row.display_cluster); group=frame[(frame.canonical_stratum==subtype)&(frame.display_cluster==c)]
            factor_means=[]
            for f in FACTORS:
                sm=(group[f].mean()-frame.loc[frame.canonical_stratum==subtype,f].mean())/frame.loc[frame.canonical_stratum==subtype,f].std(ddof=1)
                factor_means.append((f,float(sm)))
            defining=", ".join(f"{posthoc.LABEL[f]} {v:+.2f} SD" for f,v in sorted(factor_means,key=lambda x:abs(x[1]),reverse=True)[:3])
            external=f"mean CGM {row.cgm_mean:+.2f} SD, CV {row.cgm_cv:+.2f} SD, time in range {row.cgm_time_in_range:+.2f} SD, and time above 180 {row.cgm_time_above_180:+.2f} SD"
            prior=posthoc.DECISIONS[(subtype,c)][0]
            if prior in ("Insulin-resistance-aligned profile","Obesity-dominant profile","Insulin-deficiency-aligned profile"):
                burden=np.nanmean([row.cgm_mean,row.cgm_time_above_180,-row.cgm_time_in_range])
                change="strengthens" if burden>.25 else "weakens" if burden<-.25 else "does not materially change"
            else: change="does not materially change"
            contradiction="CGM burden is directionally lower than expected" if change=="weakens" else "no single external CGM characteristic overturns the mixed within-subtype profile"
            parts.append(f"**C{c} (N={int(row.cluster_n)}, CGM N={int(row.cgm_n)}):** defining-variable contrasts were {defining}. External CGM characteristics were {external}. HbA1c, C-peptide, and TG/HDL provide supporting metabolic context but were also clustering inputs, so they are not independent validation. {contradiction}. The frozen interpretation remains compatible with a **{prior.lower()}**, and the new CGM information {change} that interpretation; this is not a formal published-subtype assignment.")
        lines += [" ".join(parts), ""]
    lines += ["## Audits", "", "No sufficiently complete participant-level baseline insulin laboratory measurement was available. `med_insulin` is a treatment indicator and was not substituted. Timed insulin events were not used. No validated hyperglycemic-excursion onset definition was found, so no new count was invented; threshold occupancy and the previously specified CGM summaries are reported instead.", "",
              f"Clean-stream coverage: {cgm_report['participant_count']} participants and {cgm_report['clean_segment_count']} segments.", ""]
    (REPORTS / "phase1_cluster_metabolic_profiles.md").write_text("\n".join(lines))


def main() -> None:
    started=time.time()
    for p in [OUT, REPORTS, LOGS]: p.mkdir(parents=True, exist_ok=True)
    marker=EXTENDED/"PHASE1_COMPLETE.json"
    if marker.exists():
        print(f"Phase 1 already complete: {marker}")
        return
    style()
    labeled, provenance = recover_frozen_memberships()
    cgm, cgm_report = clean_cgm_summaries()
    cgm.to_parquet(OUT / "participant_clean_segment_cgm_summaries.parquet", index=False)
    frame=labeled.merge(cgm,on="participant_id",how="left",validate="one_to_one")
    frame.to_parquet(OUT / "participant_frozen_cluster_profiles.parquet", index=False)
    clinical_figure(frame)
    z, raw = metabolic_outputs(frame)
    interpretation(frame,z,provenance,cgm_report)
    qa={"phase":"phase1","status":"complete","created_at":now(),"elapsed_seconds":time.time()-started,
        "participant_count":int(frame.participant_id.nunique()),"cgm_participant_count":int(frame.cgm_mean.notna().sum()),
        "selected_k":provenance["selected_k"],"cluster_refit":False,"test_preprocessing_fit":False,
        "baseline_insulin_included":False,"timed_insulin_used":False,"meal_event_used":False,
        "outputs":[p.name for p in OUT.iterdir() if p.is_file()]}
    write_json(marker,qa)
    write_json(OUT/"phase1_qc.json",qa)
    print(json.dumps(qa,indent=2))


if __name__ == "__main__":
    main()
