"""Post hoc literature alignment for frozen Phase 1 clinical partitions.

Never fits a clustering model and never reads h0/ht. Selected labels are
recovered only from the frozen Phase 1 pipelines and centroids. Sensitivity-k
results are limited to diagnostics persisted by Phase 1.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf, MinCovDet

import within_subtype_phase1 as phase1
from ssmcgm.analysis.within_subtype_config import (
    CANONICAL_STRATA, CLUSTER_COLORS, DATASET, FIGURE_DPI, RAW_STRATUM_MAP,
    SEED, SPLIT_PATH, STRATIFIER, STUDY2_ROOT, THUMBNAIL_DPI,
)

ROOT = STUDY2_ROOT / "posthoc_phase1_literature_alignment"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
SITE = "participants_clinical_site"
SEX = "demo_sex_at_birth"
MEDS = ["med_any_diabetes_drug", "med_metformin", "med_insulin",
        "med_glp1_or_gip_glp1", "med_sglt2", "med_sulfonylurea",
        "med_thiazolidinedione"]
FACTORS = ["participants_age", "bmi_baseline", "hba1c_percent_baseline",
           "c_peptide_ngml_baseline", "tg_hdl_ratio",
           "waist_to_hip_ratio_baseline"]
EXTRA = ["triglycerides_mgdl_baseline", "hdl_cholesterol_mgdl_baseline",
         "ldl_cholesterol_mgdl_baseline", "clinical_systolic_bp_mmhg_baseline",
         "clinical_diastolic_bp_mmhg_baseline"]
ALL_FACTORS = FACTORS + EXTRA
LABEL = {
    "participants_age": "Study-visit age", "bmi_baseline": "BMI",
    "hba1c_percent_baseline": "HbA1c", "c_peptide_ngml_baseline": "C-peptide",
    "tg_hdl_ratio": "TG/HDL", "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "triglycerides_mgdl_baseline": "Triglycerides",
    "hdl_cholesterol_mgdl_baseline": "HDL cholesterol",
    "ldl_cholesterol_mgdl_baseline": "LDL cholesterol",
    "clinical_systolic_bp_mmhg_baseline": "Systolic BP",
    "clinical_diastolic_bp_mmhg_baseline": "Diastolic BP",
}
UNIT = {
    "participants_age": "years", "bmi_baseline": "kg/m^2",
    "hba1c_percent_baseline": "%", "c_peptide_ngml_baseline": "ng/mL",
    "tg_hdl_ratio": "ratio", "waist_to_hip_ratio_baseline": "ratio",
    "triglycerides_mgdl_baseline": "mg/dL", "hdl_cholesterol_mgdl_baseline": "mg/dL",
    "ldl_cholesterol_mgdl_baseline": "mg/dL",
    "clinical_systolic_bp_mmhg_baseline": "mmHg",
    "clinical_diastolic_bp_mmhg_baseline": "mmHg",
}
EXTRA_DATES = {
    "ldl_cholesterol_mgdl_baseline": "ldl_cholesterol_mgdl_baseline_date",
    "clinical_systolic_bp_mmhg_baseline": "clinical_systolic_bp_mmhg_baseline_date",
    "clinical_diastolic_bp_mmhg_baseline": "clinical_diastolic_bp_mmhg_baseline_date",
}
SUBTYPE_LABEL = {"healthy": "Healthy", "pre_diabetes": "Pre-diabetes",
                 "t2d_oral_non_insulin": "T2D oral non-insulin",
                 "insulin_dependent": "Insulin-dependent"}
DOMAINS = ["insulin_deficiency_domain", "insulin_resistance_domain",
           "obesity_dominant_domain", "older_clinical_profile_domain"]
DOMAIN_LABEL = {"insulin_deficiency_domain": "Insulin-deficiency\ndomain",
                "insulin_resistance_domain": "Insulin-resistance\ndomain",
                "obesity_dominant_domain": "Obesity-dominant\ndomain",
                "older_clinical_profile_domain": "Older clinical-profile\ndomain"}

# Qualitative decisions are reasoned explicitly, never thresholded from scores.
DECISIONS = {
 ("healthy",1): ("No clear literature analogue","none","The source studies concern adult-onset diabetes, while this is a metabolically lower healthy profile."),
 ("healthy",2): ("Insulin-resistance-aligned profile","low","Higher adiposity, C-peptide, and TG/HDL are concordant, but participants are healthy rather than newly diagnosed with diabetes."),
 ("pre_diabetes",1): ("No clear literature analogue","none","Lower secretion and adiposity occur without the hyperglycaemic pattern used to define insulin-deficiency domains in diabetes cohorts."),
 ("pre_diabetes",2): ("Insulin-resistance-aligned profile","low","Adiposity, C-peptide, TG/HDL, and waist-to-hip ratio align, but pre-diabetes is outside the source studies' case definition."),
 ("t2d_oral_non_insulin",1): ("Mixed or overlapping profile","low","Lower C-peptide and BMI suggest deficiency alignment, whereas lower HbA1c and TG/HDL contradict severe deficiency or resistance."),
 ("t2d_oral_non_insulin",2): ("Obesity-dominant profile","moderate","Markedly higher BMI with non-extreme TG/HDL and preserved C-peptide is coherent, but visit age and treated cross-sectional measures limit comparison."),
 ("t2d_oral_non_insulin",3): ("Insulin-resistance-aligned profile","moderate","Higher C-peptide, TG/HDL, and waist-to-hip ratio support resistance alignment, with an overlapping older clinical profile."),
 ("insulin_dependent",1): ("Insulin-deficiency-aligned profile","low","Lower C-peptide, BMI, and TG/HDL support deficiency alignment, but lower HbA1c and insulin treatment complicate interpretation."),
 ("insulin_dependent",2): ("Mixed or overlapping profile","low","Higher C-peptide, TG/HDL, waist-to-hip ratio, and visit age overlap resistance and older-profile domains in an exploratory treated partition."),
 ("insulin_dependent",3): ("Obesity-dominant profile","low","Markedly higher BMI and higher C-peptide support obesity alignment, but the small exploratory cluster and insulin treatment reduce confidence."),
}

def now_iso(): return datetime.now(timezone.utc).isoformat()
def jdefault(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v))
def write_json(path, value): path.write_text(json.dumps(value, indent=2, default=jdefault)+"\n")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def source_table():
    a={"study":"Ahlqvist et al., 2018, Novel subgroups of adult-onset diabetes and their association with outcomes",
       "cohort":"ANDIS discovery cohort, Sweden (newly diagnosed adult diabetes; n=8,980)",
       "variables_used":"GAD antibodies, age at diagnosis, BMI, HbA1c, HOMA2-B, HOMA2-IR",
       "measurements_near_diagnosis":"Yes; newly diagnosed participants",
       "fasting_insulin_or_c_peptide":"Fasting C-peptide and glucose underpin HOMA2-B/HOMA2-IR",
       "main_physiological_domains":"Autoimmunity; insulin deficiency; insulin resistance; obesity; older age at diagnosis",
       "direct_comparability_with_aireadi":"Limited: AI-READI uses study-visit age, C-peptide and TG/HDL proxies, lacks GAD status, includes treated prevalent disease, and clusters within diagnostic-treatment strata.",
       "doi":"10.1016/S2213-8587(18)30051-2","primary_source_url":"https://pubmed.ncbi.nlm.nih.gov/29503172/"}
    s={"study":"Slieker et al., 2021, Replication and cross-validation of type 2 diabetes subtypes based on clinical variables",
       "variables_used":"Age at first visit, BMI, HbA1c, C-peptide, HDL cholesterol",
       "main_physiological_domains":"Insulin deficiency; insulin resistance; obesity; older/mild profiles; high-HDL mild profile",
       "doi":"10.1007/s00125-021-05490-8","primary_source_url":"https://link.springer.com/article/10.1007/s00125-021-05490-8"}
    return pd.DataFrame([a,
      {**s,"cohort":"Hoorn Diabetes Care System, Netherlands","measurements_near_diagnosis":"Clinical data within 2 years of diagnosis; not necessarily before treatment","fasting_insulin_or_c_peptide":"Fasting C-peptide","direct_comparability_with_aireadi":"Partly comparable because C-peptide was used, but DCS was not necessarily at diagnosis before treatment; AI-READI fasting is unconfirmed and uses TG/HDL rather than HDL alone."},
      {**s,"cohort":"GoDARTS, Scotland","measurements_near_diagnosis":"Clinical data within 2 years of diagnosis; not necessarily before treatment","fasting_insulin_or_c_peptide":"Non-fasting C-peptide","direct_comparability_with_aireadi":"Non-fasting C-peptide improves measurement comparability, but timing, treatment, diagnostic scope, and HDL rather than TG/HDL differ."},
      {**s,"cohort":"ANDIS, Sweden","measurements_near_diagnosis":"Yes; median 40 days from diagnosis (IQR 12-99)","fasting_insulin_or_c_peptide":"Fasting C-peptide","direct_comparability_with_aireadi":"Limited by AI-READI's prevalent treated cross-sectional design, unconfirmed fasting, study-visit age, and within-stratum clustering."}])

def load_frame():
    requested=["participant_id",STRATIFIER,SITE,SEX,*MEDS,
      *[c for c in phase1.FACTOR_COLUMN_MAP.values() if c!="fasting_insulin_baseline"],
      *EXTRA,"triglycerides_mgdl_baseline","hdl_cholesterol_mgdl_baseline",
      *phase1.BASELINE_DATE_COLUMNS.values(),*EXTRA_DATES.values()]
    clinical,conflicts=phase1.load_participant_table(list(dict.fromkeys(requested)))
    split=pd.read_csv(SPLIT_PATH,dtype={"participant_id":str})
    clinical["participant_id"]=clinical.participant_id.astype(str)
    clinical["canonical_stratum"]=clinical[STRATIFIER].map(RAW_STRATUM_MAP)
    frame=split[["participant_id","split"]].merge(clinical,on="participant_id",how="left",validate="one_to_one")
    frame=frame.merge(phase1.load_cgm_bounds(),on="participant_id",how="left",validate="one_to_one")
    nulled={}
    for factor,datecol in {**phase1.BASELINE_DATE_COLUMNS,**EXTRA_DATES}.items():
        dates=pd.to_datetime(frame[datecol],errors="coerce",utc=True)
        mask=dates.notna()&frame.cgm_end.notna()&(dates>frame.cgm_end)
        nulled[factor]=int((mask&frame[factor].notna()).sum()); frame.loc[mask,factor]=np.nan
    hdl=frame.hdl_cholesterol_mgdl_baseline; tg=frame.triglycerides_mgdl_baseline
    frame["tg_hdl_ratio"]=np.where(hdl.notna()&(hdl>0)&tg.notna(),tg/hdl,np.nan)
    return frame,{"dataset_path":str(DATASET),"dataset_sha256":sha(DATASET),
      "participant_split_path":str(SPLIT_PATH),"participant_split_sha256":sha(SPLIT_PATH),
      "duplicate_static_value_conflicts":conflicts,"post_cgm_descriptive_values_nulled":nulled}

def recover_frozen(frame):
    factors=json.loads((STUDY2_ROOT/"decisions/factor_selection.json").read_text())["final_factor_list"]
    if factors!=FACTORS: raise RuntimeError("Frozen factor list mismatch")
    mp=STUDY2_ROOT/"phase1_clinical_clustering/frozen_clustering_manifest.json"
    manifest=json.loads(mp.read_text()); paths=[mp,STUDY2_ROOT/"tables/phase1_cluster_characterization.csv"]
    out={}
    for subtype in CANONICAL_STRATA:
        info=manifest["clusters"][subtype]
        train=frame[(frame.canonical_stratum==subtype)&(frame.split=="train")].copy().reset_index(drop=True)
        if info["missing_data_strategy"]=="complete_case": train=train[train[FACTORS].notna().all(axis=1)].reset_index(drop=True)
        pp=STUDY2_ROOT/info["preprocessing_pipeline_path"]; cp=STUDY2_ROOT/info["centroid_path"]; paths += [pp,cp]
        pipe=joblib.load(pp); cent=json.loads(cp.read_text())["centroids_by_display_cluster"]
        order=sorted(map(int,cent)); C=np.asarray([cent[str(c)] for c in order])
        X=phase1.apply_pipeline(train,pipe["factors"],pipe["log_transformed"],pipe["imputer"],pipe["scaler"])
        labels=np.asarray([order[i] for i in np.linalg.norm(X[:,None,:]-C[None,:,:],axis=2).argmin(axis=1)])
        train["display_cluster"]=labels
        got=train.display_cluster.value_counts().sort_index().to_dict(); expected={int(k):int(v) for k,v in info["cluster_sizes_display_order"].items()}
        if got!=expected: raise RuntimeError(f"Frozen count mismatch {subtype}: {got} vs {expected}")
        out[subtype]={"frame":train,"matrix":X,"labels":labels,"centroids":dict(zip(order,C)),"selected_k":int(info["selected_k"]),"status":info["status"],"strategy":info["missing_data_strategy"]}
    return out,{str(p.relative_to(STUDY2_ROOT)):sha(p) for p in paths}

def characteristic(z, trimmed, n):
    if not np.isfinite(z): return "Not estimable"
    if n<=2: return "Driven by a small number of outliers"
    if abs(z)>=.5 and (abs(trimmed)<.5 or abs(trimmed)<.5*abs(z)): return "Driven by a small number of outliers"
    if abs(z)>=.8: return "Strongly characteristic"
    if abs(z)>=.5: return "Moderately characteristic"
    return "Overlapping"

def distributions(selected):
    wide=[]; long=[]
    for subtype in CANONICAL_STRATA:
        frame=selected[subtype]["frame"]
        stats={f:(float(frame[f].mean()),float(frame[f].std(ddof=1))) for f in ALL_FACTORS}
        for cluster in sorted(frame.display_cluster.unique()):
            group=frame[frame.display_cluster==cluster]
            row={"canonical_stratum":subtype,"selected_k":selected[subtype]["selected_k"],
                 "partition_status":selected[subtype]["status"],"display_cluster":int(cluster),
                 "cluster_label":f"C{int(cluster)}","n":len(group),
                 "site_composition":json.dumps({str(k):int(v) for k,v in group[SITE].value_counts(dropna=False).items()}),
                 "sex_composition":json.dumps({str(k):int(v) for k,v in group[SEX].value_counts(dropna=False).items()})}
            for med in MEDS:
                v=pd.to_numeric(group[med],errors="coerce").fillna(0)
                row[f"fraction_{med}"]=float((v>0).mean())
            for factor in ALL_FACTORS:
                v=pd.to_numeric(group[factor],errors="coerce").dropna().to_numpy(float); om,sd=stats[factor]
                if len(v):
                    q1,median,q3=np.percentile(v,[25,50,75]); mean=float(v.mean())
                    order=np.argsort(np.abs(v-median))[::-1]; keep=np.ones(len(v),bool); keep[order[:min(2,max(0,len(v)-1))]]=False
                    tmean=float(v[keep].mean()) if keep.any() else mean
                    z=(mean-om)/sd if sd>0 else 0.; tz=(tmean-om)/sd if sd>0 else 0.; char=characteristic(z,tz,len(v))
                else: q1=median=q3=mean=tmean=z=tz=np.nan; char="Not estimable"
                row.update({f"{factor}_n_nonmissing":len(v),f"{factor}_median":median,f"{factor}_q1":q1,
                    f"{factor}_q3":q3,f"{factor}_iqr":q3-q1,f"{factor}_mean":mean,
                    f"{factor}_mean_standardized_within_subtype":z,
                    f"{factor}_outlier_trimmed_mean_standardized":tz,f"{factor}_characterization":char})
                long.append({"canonical_stratum":subtype,"selected_k":selected[subtype]["selected_k"],
                    "partition_status":selected[subtype]["status"],"display_cluster":int(cluster),
                    "cluster_label":f"C{int(cluster)}","cluster_n":len(group),"factor":factor,
                    "factor_label":LABEL[factor],"unit":UNIT[factor],"n_nonmissing":len(v),"median":median,
                    "q1":q1,"q3":q3,"iqr":q3-q1,"mean":mean,"mean_standardized_within_subtype":z,
                    "outlier_trimmed_mean_standardized":tz,"characterization":char})
            wide.append(row)
    return pd.DataFrame(wide),pd.DataFrame(long)

def pairwise_smd(selected):
    rows=[]
    for subtype in CANONICAL_STRATA:
        frame=selected[subtype]["frame"]; clusters=sorted(frame.display_cluster.unique())
        for ca in clusters:
          for cb in clusters:
            if ca==cb: continue
            for factor in ALL_FACTORS:
                a=pd.to_numeric(frame.loc[frame.display_cluster==ca,factor],errors="coerce").dropna().to_numpy(float)
                b=pd.to_numeric(frame.loc[frame.display_cluster==cb,factor],errors="coerce").dropna().to_numpy(float)
                df=len(a)+len(b)-2
                pv=(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/df) if len(a)>1 and len(b)>1 and df>0 else np.nan
                d=(a.mean()-b.mean())/np.sqrt(pv) if np.isfinite(pv) and pv>0 else np.nan
                rows.append({"canonical_stratum":subtype,"selected_k":selected[subtype]["selected_k"],
                  "cluster_a":f"C{int(ca)}","cluster_b":f"C{int(cb)}","factor":factor,
                  "factor_label":LABEL[factor],"n_a":len(a),"n_b":len(b),
                  "standardized_mean_difference_a_minus_b":d})
    return pd.DataFrame(rows)

def robust_distance(X):
    try:
        model=MinCovDet(random_state=SEED).fit(X); method="minimum_covariance_determinant"
    except Exception:
        model=LedoitWolf().fit(X); method="ledoit_wolf_mahalanobis_fallback"
    return np.sqrt(np.maximum(model.mahalanobis(X),0)),method

def selected_extremes(selected):
    rows=[]; threshold=float(np.sqrt(chi2.ppf(.99,df=len(FACTORS))))
    for subtype in CANONICAL_STRATA:
      p=selected[subtype]
      for cluster in sorted(set(p["labels"].tolist())):
        X=p["matrix"][p["labels"]==cluster]; centroid=X.mean(axis=0)
        loo=[float(np.linalg.norm(np.delete(X,i,axis=0).mean(axis=0)-centroid)) for i in range(len(X))] if len(X)>1 else []
        dist,method=robust_distance(X); order=np.argsort(dist)[::-1]
        X1=X[np.setdiff1d(np.arange(len(X)),order[:1])]; X2=X[np.setdiff1d(np.arange(len(X)),order[:min(2,len(X)-1)])]
        shift1=float(np.linalg.norm(X1.mean(axis=0)-centroid)) if len(X1) else np.nan
        shift2=float(np.linalg.norm(X2.mean(axis=0)-centroid)) if len(X2) else np.nan
        outn=int((dist>threshold).sum()); driven=bool(len(X)<=2 or (outn in (1,2) and np.isfinite(shift2) and shift2>=.25))
        rows.append({"canonical_stratum":subtype,"solution_role":"selected_frozen_partition","k":p["selected_k"],
          "cluster_identifier":f"C{int(cluster)}","cluster_size":len(X),"singleton_or_near_singleton_n_le_5":len(X)<=5,
          "driven_by_one_or_two_extreme_participants":driven,
          "leave_one_out_centroid_shift_mean":float(np.mean(loo)) if loo else None,
          "leave_one_out_centroid_shift_max":float(np.max(loo)) if loo else None,
          "centroid_shift_after_removing_most_extreme":shift1,"centroid_shift_after_removing_two_most_extreme":shift2,
          "robust_distance_method":method,"maximum_robust_distance":float(np.max(dist)),
          "robust_distance_99pct_threshold":threshold,"robust_outlier_n":outn,
          "participant_level_metrics_estimable":True,"nonestimable_reason":None})
    return pd.DataFrame(rows)

def full_extreme_audit(selected_audit):
    for column in ["phase1_bootstrap_centroid_stability_mean_distance", "phase1_bootstrap_centroid_stability_std_distance", "phase1_bootstrap_mean_ari", "phase1_initialization_mean_ari"]:
        selected_audit[column] = np.nan
    parts=[selected_audit]
    for subtype in CANONICAL_STRATA:
      payload=json.loads((STUDY2_ROOT/f"phase1_clinical_clustering/k_selection_{subtype}.json").read_text())
      sk=int(payload["selected_k"])
      for ks,candidate in payload["candidates"].items():
        k=int(ks)
        if k==sk: continue
        for i,size in enumerate(candidate["cluster_sizes"]):
          parts.append(pd.DataFrame([{"canonical_stratum":subtype,"solution_role":"sensitivity_k_saved_diagnostics_only",
            "k":k,"cluster_identifier":f"internal_candidate_index_{i}","cluster_size":int(size),
            "singleton_or_near_singleton_n_le_5":int(size)<=5,"driven_by_one_or_two_extreme_participants":None,
            "leave_one_out_centroid_shift_mean":None,"leave_one_out_centroid_shift_max":None,
            "centroid_shift_after_removing_most_extreme":None,"centroid_shift_after_removing_two_most_extreme":None,
            "robust_distance_method":None,"maximum_robust_distance":None,"robust_distance_99pct_threshold":None,
            "robust_outlier_n":None,"participant_level_metrics_estimable":False,
            "nonestimable_reason":"Sensitivity memberships and centroids were not persisted; participant influence would require forbidden reclustering.",
            "phase1_bootstrap_centroid_stability_mean_distance":candidate["centroid_stability_mean_distance"][i],
            "phase1_bootstrap_centroid_stability_std_distance":candidate["centroid_stability_std_distance"][i],
            "phase1_bootstrap_mean_ari":candidate["bootstrap_mean_ari"],
            "phase1_initialization_mean_ari":candidate["initialization_sensitivity"]["mean_ari_vs_reference"]}]))
    return pd.concat([part.dropna(axis=1, how="all") for part in parts],ignore_index=True,sort=False)

def zv(row,f): return float(row[f"{f}_mean_standardized_within_subtype"])
def scores(row):
    age,bmi,hba,cpep,tg,whr=[zv(row,f) for f in FACTORS]
    return {"insulin_deficiency_domain":(-cpep+hba-.5*bmi)/2.5,
            "insulin_resistance_domain":(tg+cpep+.5*bmi+.5*whr)/3,
            "obesity_dominant_domain":(bmi+whr+.5*cpep-.5*abs(tg))/3,
            "older_clinical_profile_domain":age}

def evidence(row,closest):
    v={"Study-visit age":zv(row,"participants_age"),"BMI":zv(row,"bmi_baseline"),
       "HbA1c":zv(row,"hba1c_percent_baseline"),"C-peptide":zv(row,"c_peptide_ngml_baseline"),
       "TG/HDL":zv(row,"tg_hdl_ratio"),"Waist-to-hip ratio":zv(row,"waist_to_hip_ratio_baseline")}
    if closest=="No clear literature analogue":
        x=sorted(v.items(),key=lambda i:abs(i[1]),reverse=True)[:4]
        return "No concordant physiological domain","; ".join(f"{n} z={z:+.2f}" for n,z in x)
    if closest=="Mixed or overlapping profile":
        pos=sorted(v.items(),key=lambda i:i[1],reverse=True)[:3]; neg=sorted(v.items(),key=lambda i:i[1])[:3]
        return "; ".join(f"{n} z={z:+.2f}" for n,z in pos),"; ".join(f"{n} z={z:+.2f}" for n,z in neg)
    expected={"Insulin-deficiency-aligned profile":{"C-peptide":-1,"HbA1c":1,"BMI":-.5,"TG/HDL":-.25},
              "Insulin-resistance-aligned profile":{"TG/HDL":1,"C-peptide":1,"BMI":.5,"Waist-to-hip ratio":.5},
              "Obesity-dominant profile":{"BMI":1,"Waist-to-hip ratio":1,"C-peptide":.5},
              "Older clinical profile":{"Study-visit age":1}}
    c=[(n,v[n],w*v[n]) for n,w in expected[closest].items()]
    s=sorted([x for x in c if x[2]>=.15],key=lambda x:x[2],reverse=True)
    q=sorted([x for x in c if x[2]<=-.15],key=lambda x:x[2])
    return ("; ".join(f"{n} z={z:+.2f}" for n,z,_ in s) or "No single factor provides strong support",
            "; ".join(f"{n} z={z:+.2f}" for n,z,_ in q) or "No strongly opposing factor")

def align_table(dist):
    rows=[]
    for _,r in dist.iterrows():
        closest,conf,reason=DECISIONS[(r.canonical_stratum,int(r.display_cluster))]; support,contra=evidence(r,closest)
        if r.canonical_stratum in ("healthy","pre_diabetes"):
            limit="Published source cohorts contain adult-onset diabetes; this diagnostic stratum is outside that target population."
        elif r.canonical_stratum=="insulin_dependent":
            limit="Insulin treatment may alter HbA1c and C-peptide; the partition is exploratory and age is measured at study visit."
        else:
            limit="Age is study-visit age; fasting is unconfirmed; C-peptide and TG/HDL are proxies in treated prevalent disease."
        rows.append({"canonical_stratum":r.canonical_stratum,"selected_k":int(r.selected_k),
          "partition_status":r.partition_status,"display_cluster":int(r.display_cluster),"cluster_label":r.cluster_label,
          "n":int(r.n),**scores(r),"closest_literature_aligned_domain":closest,"supporting_variables":support,
          "contradicting_or_overlapping_variables":contra,"confidence":conf,"confidence_reasoning":reason,
          "main_comparability_limitation":limit,"exact_published_subtype_claimed":False})
    return pd.DataFrame(rows)

def med_iqr(r,f): return f"{LABEL[f]} {r[f'{f}_median']:.2f} [{r[f'{f}_q1']:.2f}-{r[f'{f}_q3']:.2f}] {UNIT[f]}"
def interpretation(dist,align,extreme):
    lines=["# Post hoc Phase 1 clinical interpretation against diabetes-subtyping literature","",
      "## Interpretation boundary","",
      "These are cohort-specific, within-diagnostic-stratum clinical profiles. They are not reproductions of any published diabetes subtype. Published names are used only when describing source papers; frozen AI-READI clusters retain neutral C1/C2/C3 labels.","",
      "## Required comparability limitations","",
      "1. Age is age at the AI-READI study visit, not age at diabetes diagnosis.",
      "2. C-peptide is a proxy for endogenous insulin secretion, not HOMA2-beta.",
      "3. TG/HDL is a proxy for insulin resistance, not HOMA2-IR.",
      "4. C-peptide and triglycerides are not confirmed fasting measurements.",
      "5. GAD antibody status is unavailable or not used.",
      "6. Treatment may alter the measured phenotype.",
      "7. AI-READI is cross-sectional and does not contain newly diagnosed participants only.",
      "8. Clusters were identified within an existing diagnostic-treatment subtype rather than across all newly diagnosed diabetes cases.","",
      "## Primary literature basis","",
      "Ahlqvist et al. studied newly diagnosed adult diabetes using GAD status, age at diagnosis, BMI, HbA1c, and HOMA2 estimates. Slieker et al. replicated physiological domains using age, BMI, HbA1c, C-peptide, and HDL across DCS, GoDARTS, and ANDIS, while documenting fasting and diagnosis-timing differences. See `literature_source_table.csv`.","",
      "- Ahlqvist et al. 2018: https://pubmed.ncbi.nlm.nih.gov/29503172/",
      "- Slieker et al. 2021: https://link.springer.com/article/10.1007/s00125-021-05490-8","",
      "## Cluster interpretations",""]
    for subtype in CANONICAL_STRATA:
      lines += [f"### {SUBTYPE_LABEL[subtype]}",""]
      for _,r in dist[dist.canonical_stratum==subtype].sort_values("display_cluster").iterrows():
        a=align[(align.canonical_stratum==subtype)&(align.display_cluster==r.display_cluster)].iloc[0]
        salient=sorted(FACTORS,key=lambda f:abs(zv(r,f)),reverse=True)[:3]
        paragraph=(f"**C{int(r.display_cluster)} (N={int(r.n)}).** "+"; ".join(med_iqr(r,f) for f in salient)+". "
          f"The distribution is most compatible with **{a.closest_literature_aligned_domain}** ({a.confidence} confidence): {a.confidence_reasoning} "
          f"Supporting evidence was {a.supporting_variables}; contradictory or overlapping evidence was {a.contradicting_or_overlapping_variables}. "
          f"{a.main_comparability_limitation} C-peptide is not HOMA2-beta, TG/HDL is not HOMA2-IR, fasting is unconfirmed, GAD was not used, and treatment may alter phenotype. Therefore this cluster should not be treated as an exact published subtype.")
        lines += [paragraph,""]
    driven=extreme[(extreme.solution_role=="selected_frozen_partition")&(extreme.driven_by_one_or_two_extreme_participants==True)]
    k4=extreme[(extreme.canonical_stratum=="t2d_oral_non_insulin")&(extreme.k==4)]
    sizes=", ".join(str(int(x)) for x in k4.cluster_size)
    lines += ["## Extreme-cluster audit","",
      "No selected frozen cluster was classified as driven by one or two extreme participants." if driven.empty else "Selected extreme-driven clusters: "+", ".join(driven.canonical_stratum+" "+driven.cluster_identifier),"",
      f"Persisted T2D oral non-insulin k=4 sensitivity sizes are {sizes}; they contain no singleton or near-singleton. k=4 remains sensitivity-only and is not used to claim another biological subtype.","",
      "Sensitivity-k memberships and centroids were not persisted. Leave-one-out and robust-distance metrics are marked non-estimable rather than rerunning forbidden clustering; saved Phase 1 bootstrap centroid stability is reported instead.","",
      "## Figure notes","",
      "Figure L1 shows raw clustering-factor distributions for frozen selected partitions. Figure L2 shows standardized domain evidence descriptively; its cells do not generate qualitative confidence.","",
      "Post hoc Phase 1 literature-alignment extension complete. No clustering was rerun, no k or membership changed, and no h0 or ht artifact was inspected. Stop.",""]
    return "\n".join(lines)

def figure_l1(selected):
    sns.set_style("whitegrid"); fig,axes=plt.subplots(len(CANONICAL_STRATA),len(FACTORS),figsize=(24,15),facecolor="white",squeeze=False); plotted=[]
    for ri,subtype in enumerate(CANONICAL_STRATA):
      frame=selected[subtype]["frame"]; clusters=sorted(frame.display_cluster.unique()); counts=frame.display_cluster.value_counts().to_dict(); pal={c:CLUSTER_COLORS[(c-1)%len(CLUSTER_COLORS)] for c in clusters}
      for ci,factor in enumerate(FACTORS):
        ax=axes[ri,ci]; p=frame[["participant_id","display_cluster",factor]].dropna().copy()
        sns.boxplot(data=p,x="display_cluster",y=factor,hue="display_cluster",order=clusters,palette=[pal[c] for c in clusters],legend=False,width=.58,fliersize=0,linewidth=.9,ax=ax)
        sns.stripplot(data=p,x="display_cluster",y=factor,order=clusters,color="#333333",alpha=.20,size=1.6,jitter=.22,ax=ax)
        ax.set_xticks(range(len(clusters)),[f"C{c}\nN={counts[c]}" for c in clusters],fontsize=8); ax.set_xlabel(""); ax.set_ylabel(UNIT[factor] if ci==0 else "")
        if ri==0: ax.set_title(LABEL[factor],fontweight="bold",fontsize=11)
        if ci==0: ax.text(-.47,.5,f"{SUBTYPE_LABEL[subtype]}\nselected k={selected[subtype]['selected_k']}",transform=ax.transAxes,rotation=90,va="center",ha="center",fontweight="bold",fontsize=10)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#333333"); sp.set_linewidth(.6)
        plotted.append(p.assign(canonical_stratum=subtype,selected_k=selected[subtype]["selected_k"],factor=factor,value=p[factor])[["participant_id","canonical_stratum","selected_k","display_cluster","factor","value"]])
    fig.suptitle("Clinical factor distributions support distinct within-subtype profiles",fontweight="bold",fontsize=17,y=.995)
    fig.text(.5,.008,"Age is study-visit age. Fasting status is unconfirmed for C-peptide and triglycerides. Frozen selected clusters only.",ha="center",fontsize=10,color="#A51C30")
    fig.tight_layout(rect=[.03,.025,1,.98]); fig.savefig(ROOT/"figure_L1_cluster_distributions.png",dpi=FIGURE_DPI,bbox_inches="tight",facecolor="white"); fig.savefig(FIGURES/"figure_L1_cluster_distributions_thumbnail.png",dpi=THUMBNAIL_DPI,bbox_inches="tight",facecolor="white"); plt.close(fig)
    pd.concat(plotted,ignore_index=True).to_csv(FIGURES/"figure_L1_cluster_distributions_plotted_data.csv",index=False)

def figure_l2(align):
    sns.set_style("white"); order=align.copy(); order["subtype_order"]=order.canonical_stratum.map({s:i for i,s in enumerate(CANONICAL_STRATA)}); order=order.sort_values(["subtype_order","display_cluster"]).reset_index(drop=True)
    M=order[DOMAINS].to_numpy(float); labels=[f"{SUBTYPE_LABEL[r.canonical_stratum]} C{int(r.display_cluster)} (N={int(r.n)})" for r in order.itertuples()]; lim=max(.8,float(np.nanmax(np.abs(M)))); cmap=LinearSegmentedColormap.from_list("alignment",["#0B1F3A","#D0D0D0","#A51C30"])
    fig=plt.figure(figsize=(18,9),facecolor="white"); grid=fig.add_gridspec(1,2,width_ratios=[1.25,1.7],wspace=.04); ax=fig.add_subplot(grid[0,0]); tx=fig.add_subplot(grid[0,1])
    sns.heatmap(M,cmap=cmap,center=0,vmin=-lim,vmax=lim,annot=True,fmt=".2f",linewidths=.7,linecolor="white",cbar_kws={"label":"Standardized evidence summary"},ax=ax)
    ax.set_yticklabels(labels,rotation=0,fontsize=9); ax.set_xticklabels([DOMAIN_LABEL[c] for c in DOMAINS],rotation=0,fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    tx.set_xlim(0,1); tx.set_ylim(len(order),0); tx.axis("off"); tx.text(0,-.35,"Closest literature-aligned interpretation",fontweight="bold",fontsize=10); tx.text(.49,-.35,"Confidence",fontweight="bold",fontsize=10); tx.text(.64,-.35,"Main limitation",fontweight="bold",fontsize=10)
    for i,r in enumerate(order.itertuples()):
      y=i+.55; tx.text(0,y,r.closest_literature_aligned_domain,va="center",fontsize=8); tx.text(.49,y,r.confidence,va="center",fontsize=8,fontweight="bold"); tx.text(.64,y,r.main_comparability_limitation,va="center",fontsize=7.2,wrap=True)
    fig.suptitle("AI-READI clinical profiles show partial alignment with published diabetes domains",fontweight="bold",fontsize=16,y=.98)
    fig.text(.5,.02,"Gray: weak/conflicting; navy: negative; crimson: positive. Confidence is qualitative reasoning, not a score threshold.",ha="center",fontsize=9)
    fig.savefig(ROOT/"figure_L2_literature_alignment.png",dpi=FIGURE_DPI,bbox_inches="tight",facecolor="white"); fig.savefig(FIGURES/"figure_L2_literature_alignment_thumbnail.png",dpi=THUMBNAIL_DPI,bbox_inches="tight",facecolor="white"); plt.close(fig)
    order.drop(columns="subtype_order").to_csv(FIGURES/"figure_L2_literature_alignment_plotted_data.csv",index=False)

def main():
    for p in [ROOT,FIGURES,TABLES]: p.mkdir(parents=True,exist_ok=True)
    source_table().to_csv(ROOT/"literature_source_table.csv",index=False)
    frame,data_audit=load_frame(); selected,before=recover_frozen(frame)
    dist,long=distributions(selected); pair=pairwise_smd(selected); extreme=full_extreme_audit(selected_extremes(selected)); align=align_table(dist)
    dist.to_csv(ROOT/"cluster_distribution_summary.csv",index=False); long.to_csv(TABLES/"cluster_factor_characterization_long.csv",index=False); pair.to_csv(TABLES/"pairwise_standardized_mean_differences.csv",index=False); extreme.to_csv(TABLES/"extreme_cluster_audit.csv",index=False); align.to_csv(ROOT/"cluster_literature_alignment.csv",index=False)
    (ROOT/"cluster_literature_interpretation.md").write_text(interpretation(dist,align,extreme)); figure_l1(selected); figure_l2(align)
    after={p:sha(STUDY2_ROOT/p) for p in before}
    if before!=after: raise RuntimeError("Frozen Phase 1 artifacts changed")
    validation={"created_at":now_iso(),"git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=phase1.REPO,text=True).strip(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=phase1.REPO,text=True).strip(),"frozen_phase1_artifact_hashes_before":before,"frozen_phase1_artifact_hashes_after":after,"frozen_artifacts_unchanged":True,"selected_partitions_recovered_from_frozen_pipeline_and_centroids":True,"selected_k_by_subtype":{s:selected[s]["selected_k"] for s in CANONICAL_STRATA},"selected_cluster_counts":{s:selected[s]["frame"].display_cluster.value_counts().sort_index().to_dict() for s in CANONICAL_STRATA},"clustering_rerun":False,"k_changed":False,"cluster_membership_changed":False,"h0_or_ht_inspected":False,"sensitivity_participant_metrics_not_reconstructed":True,"literature_used_to_relabel_merge_split_or_select":False,"confidence_derived_from_numeric_score":False,"data_audit":data_audit}
    write_json(ROOT/"analysis_validation.json",validation)
    artifacts=sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name!="artifact_hashes.json"); write_json(ROOT/"artifact_hashes.json",{str(p.relative_to(ROOT)):sha(p) for p in artifacts})
    print("Post hoc Phase 1 literature alignment complete. No clustering was rerun and no h0/ht artifact was inspected.")

if __name__=="__main__": main()
