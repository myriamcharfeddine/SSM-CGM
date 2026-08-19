#!/usr/bin/env python3
"""Secondary predictive transport probes using frozen participant representations.

Never runs forecasting, regenerates hidden states, selects targets, or combines
validation and test during fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,logging,os,platform,random,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,seaborn as sns,torch
from joblib import Parallel,delayed
from scipy.stats import pearsonr,spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger("step5")
TARGETS=["c_reactive_protein_i","natriuretic_peptide_b_prohormon","bun_creatinine_ratio"]
TLABEL={"c_reactive_protein_i":"high-sensitivity CRP","natriuretic_peptide_b_prohormon":"NT-proBNP","bun_creatinine_ratio":"BUN/creatinine ratio"}
TRANSFORMS={"c_reactive_protein_i":"log1p_nonnegative","natriuretic_peptide_b_prohormon":"log1p_nonnegative","bun_creatinine_ratio":"natural_log_positive"}
REPS=["neutral_all","full_all","neutral_glucose_residual","neutral_night","neutral_day"]
H=[f"r_{i:03d}" for i in range(128)];ALPHAS=np.power(10.,np.arange(-4.,4.0001,.5))
SNUM=["age","mean_glucose","glucose_sd","glucose_cv","tir_70_180","tar_above_180","tbr_below_70","mean_absolute_glucose_slope","glucose_range","available_cgm_hours","mean_heart_rate","heart_rate_variability","mean_activity_steps","mean_activity_intensity","sleep_fraction","mean_respiratory_rate","total_clean_hours","number_of_segments","median_segment_hours","dynamic_missingness","hr_availability","activity_availability","sleep_availability","respiratory_rate_availability"]
SCAT=["sex","clinical_site","study_group"]
def args():
 p=argparse.ArgumentParser()
 for x in ("step0-dir","step2-dir","step3-dir","step3b-dir","step4-dir","output-root"):p.add_argument("--"+x,required=True)
 p.add_argument("--outer-folds",type=int,default=5);p.add_argument("--inner-folds",type=int,default=5);p.add_argument("--outer-repetitions",type=int,default=5);p.add_argument("--bootstrap-replicates",type=int,default=2000);p.add_argument("--permutation-replicates",type=int,default=1000);p.add_argument("--seed",type=int,default=42);p.add_argument("--n-jobs",type=int,default=-1);p.add_argument("--run-id");p.add_argument("--resume",action="store_true");return p.parse_args()
def jd(x):
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.floating):return None if not np.isfinite(x) else float(x)
 if isinstance(x,np.bool_):return bool(x)
 if isinstance(x,(Path,pd.Timestamp,datetime)):return str(x)
 raise TypeError(type(x).__name__)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True,default=jd)+"\n");os.replace(t,p)
def csv(p,d):
 p=Path(p);t=p.with_suffix(p.suffix+".tmp");d.to_csv(t,index=False);os.replace(t,p)
def parq(p,d):
 p=Path(p);t=p.with_suffix(".tmp.parquet");d.to_parquet(t,index=False,compression="zstd");os.replace(t,p)
def text(p,s):
 p=Path(p);t=p.with_suffix(p.suffix+".tmp");t.write_text(s);os.replace(t,p)
def markdown_table(frame):
    """Dependency-free Markdown table rendering for the final report."""
    d=frame.copy()
    def cell(x):
        if pd.isna(x): return ""
        if isinstance(x,(float,np.floating)): return f"{float(x):.6g}"
        return str(x).replace("|","\\|").replace("\n"," ")
    header="| "+" | ".join(map(str,d.columns))+" |"
    rule="| "+" | ".join(["---"]*len(d.columns))+" |"
    rows=["| "+" | ".join(cell(x) for x in row)+" |" for row in d.itertuples(index=False,name=None)]
    return "\n".join([header,rule,*rows])

pd.DataFrame.to_markdown=lambda self,index=False,**kwargs: markdown_table(self)

def setup(p):
 fmt=logging.Formatter("%(asctime)sZ %(levelname)s %(message)s","%Y-%m-%dT%H:%M:%S");hs=[logging.FileHandler(p),logging.StreamHandler()]
 for h in hs:h.setFormatter(fmt)
 LOG.handlers[:]=hs;LOG.setLevel(logging.INFO)
def forward(name,x):
 x=np.asarray(x,float)
 if name in TARGETS[:2]:
  if np.any(x<0):raise ValueError(f"{name}: negative value violates log1p")
  return np.log1p(x)
 if np.any(x<=0):raise ValueError(f"{name}: nonpositive value violates natural log")
 return np.log(x)
def inverse(name,x):return np.expm1(x) if name in TARGETS[:2] else np.exp(x)
def met(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 return {"r2":r2_score(y,p),"spearman":spearmanr(y,p).statistic,"pearson":pearsonr(y,p).statistic,"mae":mean_absolute_error(y,p),"rmse":mean_squared_error(y,p)**.5,"signed_bias":np.mean(p-y)}
def bh(x):
 x=np.asarray(x,float);o=np.full(len(x),np.nan);q=np.isfinite(x)
 if q.any():o[q]=multipletests(x[q],method="fdr_bh")[1]
 return o
def preprocessor(num,cat,hid):
 tr=[]
 if num:tr.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler())]),num))
 if cat:tr.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
 if hid:tr.append(("hidden_state",StandardScaler(),hid))
 return ColumnTransformer(tr,remainder="drop",sparse_threshold=0)
def estimator(num,cat,hid,elastic=False):
 model=ElasticNet(max_iter=30000,random_state=42) if elastic else Ridge()
 pipe=Pipeline([("preprocess",preprocessor(num,cat,hid)),("model",model)])
 grid={"model__alpha":[.001,.01,.1,1.,10.],"model__l1_ratio":[.1,.5,.9]} if elastic else {"model__alpha":ALPHAS.tolist()}
 return pipe,grid
def folds(y,site,k,seed):
 quint=pd.qcut(pd.Series(y),q=5,labels=False,duplicates="drop").astype(str);comb=site.fillna("<missing>").astype(str).reset_index(drop=True)+"|"+quint
 if comb.value_counts().min()>=k:lab,strategy=comb,"target_quintile_x_site"
 elif quint.value_counts().min()>=k:lab,strategy=quint,"target_quintile"
 else:lab,strategy=None,"deterministic_shuffled_kfold"
 sp=KFold(k,shuffle=True,random_state=seed) if lab is None else StratifiedKFold(k,shuffle=True,random_state=seed)
 return list(sp.split(np.arange(len(y)),lab) if lab is not None else sp.split(np.arange(len(y)))),strategy
def boot_metric(frame,n,seed):
 rng=np.random.default_rng(seed);ids=frame.participant_id.unique();rows=[]
 for _ in range(n):
  take=rng.choice(ids,len(ids),replace=True);q=pd.concat([frame[frame.participant_id==p] for p in take],ignore_index=True);rows.append(met(q.observed_transformed,q.predicted_transformed))
 return {k:(np.quantile([z[k] for z in rows],.025),np.quantile([z[k] for z in rows],.975)) for k in rows[0]}
def boot_delta(base,aug,n,seed):
 keys=["participant_id","outer_repetition","outer_fold"];q=base[keys+["observed_transformed","predicted_transformed"]].merge(aug[keys+["predicted_transformed"]],on=keys,suffixes=("_base","_aug"),validate="one_to_one");rng=np.random.default_rng(seed);ids=q.participant_id.unique();out=[]
 for _ in range(n):
  take=rng.choice(ids,len(ids),replace=True);z=pd.concat([q[q.participant_id==p] for p in take],ignore_index=True);y=z.observed_transformed;mb=met(y,z.predicted_transformed_base);ma=met(y,z.predicted_transformed_aug);out.append({"delta_r2":ma["r2"]-mb["r2"],"delta_spearman":ma["spearman"]-mb["spearman"],"delta_mae":ma["mae"]-mb["mae"],"delta_rmse":ma["rmse"]-mb["rmse"]})
 return {k:(np.quantile([z[k] for z in out],.025),np.quantile([z[k] for z in out],.975)) for k in out[0]}
def panel_summaries(panel_path,ids):
 cols=["participant_id","cgm_glucose_mean","cgm_count","heart_rate_mean","heart_rate_std","activity_steps_per_min","activity_intensity_score","respiratory_rate_mean","sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]
 p=pd.read_parquet(panel_path,columns=cols,filters=[("participant_id","in",ids)]);p.participant_id=p.participant_id.astype(str);rows=[]
 for pid,g in p.groupby("participant_id"):
  valid=g.cgm_count.fillna(0).gt(0)&g.cgm_glucose_mean.notna();x=g.loc[valid,"cgm_glucose_mean"].to_numpy(float);rows.append({"participant_id":pid,"glucose_range":np.ptp(x),"mean_heart_rate":g.heart_rate_mean.mean(),"heart_rate_variability":g.heart_rate_std.mean(),"mean_activity_steps":g.activity_steps_per_min.mean(),"mean_activity_intensity":g.activity_intensity_score.mean(),"sleep_fraction":g[["sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]].notna().any(axis=1).mean(),"mean_respiratory_rate":g.respiratory_rate_mean.mean()})
 return pd.DataFrame(rows)

def representation_frames(s2,s3,s4,validation_ids,test_ids):
 vr=pd.read_parquet(s2/"participant_representations.parquet");vr.participant_id=vr.participant_id.astype(str);tr=pd.read_parquet(s4/"test_participant_representations.parquet");tr.participant_id=tr.participant_id.astype(str);out={"validation":{},"test":{}}
 for name in ["neutral_all","full_all","neutral_night","neutral_day"]:
  q=vr[(vr.representation_type==name)&(vr.balanced_anchor_variant=="all_anchors")].sort_values("participant_id")
  if set(q.participant_id)!=set(validation_ids) or q.burn_in_minutes.nunique()!=1 or q.burn_in_minutes.iloc[0]!=0:raise RuntimeError(f"validation representation mismatch {name}")
  z=q[["participant_id"]+H].copy();z.columns=["participant_id"]+[f"state_{i:03d}" for i in range(128)];out["validation"][name]=z
  q=tr[tr.representation_type==name].sort_values("participant_id")
  if set(q.participant_id)!=set(test_ids):raise RuntimeError(f"test representation mismatch {name}")
  z=q[["participant_id"]+H].copy();z.columns=["participant_id"]+[f"state_{i:03d}" for i in range(128)];out["test"][name]=z
 rv=pd.read_parquet(s3/"glucose_residualized_representations.parquet");rv.participant_id=rv.participant_id.astype(str);hc=[c for c in rv if c.startswith("h_") or c.startswith("r_")][-128:]
 if set(rv.participant_id)!=set(validation_ids) or len(hc)!=128:raise RuntimeError("validation residual representation mismatch")
 z=rv[["participant_id"]+hc].copy();z.columns=["participant_id"]+[f"state_{i:03d}" for i in range(128)];out["validation"]["neutral_glucose_residual"]=z
 q=tr[tr.representation_type=="neutral_glucose_residual"].sort_values("participant_id");z=q[["participant_id"]+H].copy();z.columns=["participant_id"]+[f"state_{i:03d}" for i in range(128)];out["test"]["neutral_glucose_residual"]=z
 for split in out:
  for name,z in out[split].items():
   if len(z)!=len(validation_ids if split=="validation" else test_ids) or not np.isfinite(z.iloc[:,1:].to_numpy()).all():raise RuntimeError(f"nonfinite/incomplete {split} {name}")
 sc=joblib.load(s3/"frozen_validation_pipeline/neutral_all/neutral_all_scaler.joblib");pc=joblib.load(s3/"frozen_validation_pipeline/neutral_all/neutral_all_pca.joblib");keep=np.load(s3/"frozen_validation_pipeline/neutral_all/kept_dimensions.npy");n90=int(pd.read_csv(s3/"pca_variance_summary.csv").query("space == 'neutral_all'").n90.iloc[0]);pcs=[f"pc_{i+1:02d}" for i in range(n90)]
 for split in ["validation","test"]:
  z=out[split]["neutral_all"];x=z[[f"state_{i:03d}" for i in range(128)]].to_numpy();score=pc.transform(sc.transform(x[:,keep]))[:,:n90];out[split]["neutral_frozen_pca"]=pd.concat([z[["participant_id"]].reset_index(drop=True),pd.DataFrame(score,columns=pcs)],axis=1)
 return out,pcs,n90
def build_baselines(s3,s4,panel_path,static_path,validation_ids,test_ids,static_reals,static_cats,s3b):
 v=pd.read_parquet(s3/"validation_glycemic_nuisance_features.parquet");v.participant_id=v.participant_id.astype(str);t=pd.read_parquet(s4/"test_glycemic_nuisance_features.parquet");t.participant_id=t.participant_id.astype(str)
 st=pd.read_parquet(static_path);st.participant_id=st.participant_id.astype(str);st=st.drop_duplicates("participant_id")
 demo=st[["participant_id","participants_age","demo_sex_at_birth"]].rename(columns={"participants_age":"age","demo_sex_at_birth":"sex"})
 v=v.merge(demo,on="participant_id",how="left",suffixes=("","_static"));t=t.merge(demo,on="participant_id",how="left",suffixes=("","_static"))
 for d in [v,t]:
  if "age_static" in d:d["age"]=d.age.fillna(d.age_static);d.drop(columns=["age_static"],inplace=True)
  if "sex_static" in d:d["sex"]=d.sex.fillna(d.sex_static);d.drop(columns=["sex_static"],inplace=True)
 ps=panel_summaries(panel_path,validation_ids+test_ids);v=v.merge(ps.drop(columns=["glucose_range"]),on="participant_id",how="left",validate="one_to_one");t=t.merge(ps,on="participant_id",how="left",validate="one_to_one")
 exp_num=[];exp_cat=[]
 for c in static_reals:
  if c not in st:continue
  new="static__"+c;miss="staticmiss__"+c;st[new]=pd.to_numeric(st[c],errors="coerce");st[miss]=st[new].isna().astype(float);exp_num.extend([new,miss])
 for c in static_cats:
  if c not in st:continue
  new="staticcat__"+c;st[new]=st[c];exp_cat.append(new)
 keep=["participant_id"]+exp_num+exp_cat;v=v.merge(st[keep],on="participant_id",how="left",validate="one_to_one");t=t.merge(st[keep],on="participant_id",how="left",validate="one_to_one")
 vl=pd.read_parquet(s3b/"frozen_k2_sensitivity/neutral_all_k2_validation_labels.parquet");vl.participant_id=vl.participant_id.astype(str);vl=vl.rename(columns={"exploratory_group":"exploratory_k2_label"});tl=pd.read_parquet(s4/"test_exploratory_k2_assignments.parquet");tl.participant_id=tl.participant_id.astype(str);tl=tl.rename(columns={"assigned_exploratory_group":"exploratory_k2_label"})
 v=v.merge(vl[["participant_id","exploratory_k2_label"]],on="participant_id",validate="one_to_one");t=t.drop(columns=["exploratory_group"],errors="ignore").merge(tl[["participant_id","exploratory_k2_label"]],on="participant_id",validate="one_to_one")
 if set(v.participant_id)!=set(validation_ids) or set(t.participant_id)!=set(test_ids):raise RuntimeError("baseline cohort mismatch")
 return v,t,exp_num,exp_cat
def feature_specs(exp_num,exp_cat,pcs):
 state=[f"state_{i:03d}" for i in range(128)];d={"simple_baseline":(SNUM,SCAT,[],False,None)}
 for rep in REPS:d[f"simple_plus_{rep}"]=(SNUM,SCAT,state,False,rep)
 d["simple_plus_neutral_frozen_pca"]=(SNUM,SCAT,pcs,False,"neutral_frozen_pca")
 d["simple_plus_neutral_elasticnet"]=(SNUM,SCAT,state,True,"neutral_all")
 d["expanded_baseline"]=(SNUM+exp_num,SCAT+exp_cat,[],False,None)
 d["expanded_plus_neutral_all"]=(SNUM+exp_num,SCAT+exp_cat,state,False,"neutral_all")
 d["simple_plus_exploratory_k2"]=(SNUM+["exploratory_k2_label"],SCAT,[],False,None)
 return d
def data_for(base,repframe):return base if repframe is None else base.merge(repframe,on="participant_id",validate="one_to_one")
def coef_rows(fit,target,feature_set,split="validation_final"):
 pre=fit.named_steps["preprocess"];model=fit.named_steps["model"]
 try:names=pre.get_feature_names_out()
 except Exception:names=[f"feature_{i}" for i in range(len(model.coef_))]
 return [{"target":target,"feature_set":feature_set,"fit_split":split,"feature":str(n),"coefficient":float(c),"absolute_coefficient":abs(float(c))} for n,c in zip(names,np.ravel(model.coef_))]
def transport_category(v,lo,permq,t):
 if not np.isfinite(v) or not np.isfinite(t):return "insufficient_coverage"
 if v>0 and t>0:return "incremental_value_transports" if lo>0 or permq<.05 else "directionally_consistent_but_imprecise"
 if v>0 and t<=0:return "validation_increment_not_transported"
 if v<=0 and t>0:return "opposite_direction"
 return "no_incremental_value"

def main():
 a=args()
 if (a.outer_folds,a.inner_folds,a.outer_repetitions)!=(5,5,5) or a.bootstrap_replicates<2000 or a.permutation_replicates<1000:raise ValueError("frozen Step 5 protocol changed")
 random.seed(a.seed);np.random.seed(a.seed);started=time.time();paths=[Path(getattr(a,x.replace("-","_"))).resolve() for x in ["step0-dir","step2-dir","step3-dir","step3b-dir","step4-dir"]];s0,s2,s3,s3b,s4=paths
 rid=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");root=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root);out=root/rid
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);(out/"frozen_probe_models").mkdir();setup(out/"step5_run.log")
 if json.loads((s3b/"exploratory_k2_freeze_manifest.json").read_text())["status"]!="eligible_and_frozen" or json.loads((s4/"step4_independent_qc.json").read_text())["status"]!="QC_COMPLETE":raise RuntimeError("upstream gate failed")
 s2m=json.loads((s2/"step2_manifest.json").read_text());s4m=json.loads((s4/"step4_manifest.json").read_text());vid=sorted(map(str,s2m["validation_participant_ids"]));tid=sorted(map(str,s4m["participant_ids"]));
 if len(vid)!=239 or len(tid)!=221 or set(vid)&set(tid):raise RuntimeError("validation/test cohort mismatch")
 panel=Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet");static=Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal/participant_static_features.parquet");checkpoint=ROOT/"outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt"
 ck=torch.load(checkpoint,map_location="cpu",weights_only=False);fs=ck["metadata"]["feature_spec"];static_reals=fs["static_reals"];static_cats=fs["static_categoricals"]
 # Freeze before any target table is loaded.
 plan={"analysis_role":"secondary prespecified-target predictive analysis; test is secondary predictive transport, not untouched confirmation","targets":TARGETS,"target_labels":TLABEL,"target_transformations":TRANSFORMS,"target_standardization":"none; deterministic log transforms modeled directly","validation_role":"239 participants; repeated nested development CV only","test_role":"221 participants; frozen pipeline transport only; never used for selection","primary_feature_comparison":"simple_baseline_plus_neutral_all versus simple_baseline","secondary_representations":["full_all","neutral_glucose_residual","neutral_night","neutral_day"],"representation_sensitivities":["frozen_validation_PCA","ElasticNet_small_fixed_grid"],"baseline_numeric":SNUM,"baseline_categorical":SCAT,"expanded_direct_input_static_reals":static_reals,"expanded_direct_input_static_categoricals":static_cats,"expanded_missingness":"one indicator per model static real","exploratory_k2":"binary feature sensitivity only","model_family":"Ridge regression","preprocessing":{"numeric":"fold-fitted median imputation with indicator then StandardScaler","categorical":"fold-fitted most-frequent imputation and one-hot; unknown ignored","hidden":"finite required; fold-fitted StandardScaler"},"missing_value_rules":"baseline imputed inside folds; hidden representations must be complete and finite","outer_cv":{"folds":5,"repetitions":5,"seeds":[a.seed+i for i in range(5)],"stratification":"target quintile x clinical site if every stratum supports 5 folds; otherwise quintile; otherwise shuffled KFold","same_folds_across_feature_sets":True},"inner_cv":{"folds":5,"shuffled":True,"training_participants_only":True},"ridge_alpha_grid":ALPHAS.tolist(),"elasticnet_grid":{"alpha":[.001,.01,.1,1.,10.],"l1_ratio":[.1,.5,.9]},"metrics":["R2","Spearman","Pearson","MAE","RMSE","signed_bias","delta_R2","delta_Spearman","delta_MAE","delta_RMSE"],"bootstrap":{"unit":"participant retaining all repeated OOF rows","replicates":a.bootstrap_replicates,"ci":"percentile 95%","seed":a.seed},"permutation":{"replicates":a.permutation_replicates,"feature":"neutral_all rows","strata":"clinical site x study group when all strata have >=2, otherwise global","evaluation":"all 25 repeated outer folds per permutation; selected fold alpha retained; augmented ridge refitted","seed":a.seed+50000},"primary_fdr_family":"exactly 3 validation delta-R2 neutral_all permutation tests","secondary_fdr_families":"reported separately; no primary-family mixing","timing":{"primary_days":180,"sensitivity_days":90},"test_transport_rules":"all tuning and preprocessing fitted on validation only then applied unchanged once","transport_categories":["incremental_value_transports","directionally_consistent_but_imprecise","validation_increment_not_transported","no_incremental_value","opposite_direction","insufficient_coverage"],"random_seed":a.seed,"prohibitions":["no forecasting replay","no hidden regeneration","no target selection","no test tuning","no broad feature selection","no primary k2 feature"]}
 dump(out/"step5_analysis_plan_frozen.json",plan);planhash=sha(out/"step5_analysis_plan_frozen.json");plan_time=datetime.now(timezone.utc).isoformat();LOG.info("analysis plan frozen hash=%s before target load",planhash)
 reps,pcs,n90=representation_frames(s2,s3,s4,vid,tid);vb,tb,exp_num,exp_cat=build_baselines(s3,s4,panel,static,vid,tid,static_reals,static_cats,s3b);specs=feature_specs(exp_num,exp_cat,pcs)
 dump(out/"probe_feature_sets.json",{k:{"numeric":v[0],"categorical":v[1],"hidden":v[2],"model":"ElasticNet" if v[3] else "Ridge","representation":v[4]} for k,v in specs.items()})
 # Baseline audit does not inspect targets.
 audit=[]
 source_map={**{x:str(s3/"validation_glycemic_nuisance_features.parquet") for x in SNUM},**{x:str(static) for x in ["age","sex"]},"clinical_site":str(s3/"validation_glycemic_nuisance_features.parquet"),"study_group":str(s3/"validation_glycemic_nuisance_features.parquet")}
 for name in SNUM+SCAT:
  audit.append({"feature":name,"feature_set":"simple_baseline","source_file":source_map.get(name,str(panel)),"definition":"prespecified participant-level direct summary","validation_coverage":vb[name].notna().mean(),"test_coverage":tb[name].notna().mean(),"validation_missing_fraction":vb[name].isna().mean(),"test_missing_fraction":tb[name].isna().mean(),"transformation":"fold-fitted numeric/categorical preprocessing","inclusion_status":"included","exclusion_reason":None})
 for name in exp_num+exp_cat:
  audit.append({"feature":name,"feature_set":"expanded_direct_input_baseline","source_file":str(static),"definition":"forecast-model static input or its missing indicator","validation_coverage":vb[name].notna().mean(),"test_coverage":tb[name].notna().mean(),"validation_missing_fraction":vb[name].isna().mean(),"test_missing_fraction":tb[name].isna().mean(),"transformation":"fold-fitted numeric/categorical preprocessing","inclusion_status":"included","exclusion_reason":None})
 for name in ["median_full_neutral_l2","mean_abs_forecast_delta","exploratory_k2_label"]:
  audit.append({"feature":name,"feature_set":"simple_baseline","source_file":str(s4),"definition":"hidden-state-derived or exploratory variable","validation_coverage":np.nan,"test_coverage":np.nan,"validation_missing_fraction":np.nan,"test_missing_fraction":np.nan,"transformation":"none","inclusion_status":"excluded","exclusion_reason":"hidden-state-derived contamination" if name!="exploratory_k2_label" else "exploratory sensitivity only"})
 csv(out/"probe_baseline_feature_audit.csv",pd.DataFrame(audit))
 # Only now load the already frozen target tables.
 target_loaded_at=datetime.now(timezone.utc).isoformat();ve=pd.read_parquet(s3/"validation_external_targets.parquet");te=pd.read_parquet(s4/"test_external_targets.parquet");ve.participant_id=ve.participant_id.astype(str);te.participant_id=te.participant_id.astype(str)
 if set(ve.target_name)!=set(TARGETS) or set(te.target_name)!=set(TARGETS):raise RuntimeError("frozen three-target list changed")
 cohorts=[];trans=[];target_frames={}
 for target in TARGETS:
  vv=ve[(ve.target_name==target)&ve.eligible_for_analysis].copy();tt=te[(te.target_name==target)&te.eligible_for_analysis].copy();vv=vv[vv.participant_id.isin(vid)];tt=tt[tt.participant_id.isin(tid)];vf=vb.merge(vv[["participant_id","analysis_value","eligible_90d_sensitivity"]],on="participant_id");tf=tb.merge(tt[["participant_id","analysis_value","eligible_90d_sensitivity"]],on="participant_id")
  if vf.participant_id.duplicated().any() or tf.participant_id.duplicated().any():raise RuntimeError(f"duplicate target row {target}")
  vf["observed_transformed"]=forward(target,vf.analysis_value);tf["observed_transformed"]=forward(target,tf.analysis_value);target_frames[target]=(vf,tf)
  vr=ve[ve.target_name==target];tr=te[te.target_name==target]
  cohorts.extend([{"target":target,"split":"validation","canonical_participants":239,"participants_with_selected_record":vr.participant_id.nunique(),"invalid_unit_exclusions":int((vr.get("unit_status",pd.Series(index=vr.index,dtype=object))!="compatible per Step 0").sum()),"timing_exclusions":int((~vr.eligible_for_analysis).sum()),"missing_baseline_row":239-len(vb),"participants_with_any_baseline_missing":int(vf[SNUM+SCAT].isna().any(axis=1).sum()),"missing_hidden_representation":0,"final_180d_count":len(vf),"final_90d_count":int(vf.eligible_90d_sensitivity.sum())},{"target":target,"split":"test","canonical_participants":221,"participants_with_selected_record":tr.participant_id.nunique(),"invalid_unit_exclusions":0,"timing_exclusions":int((~tr.eligible_for_analysis).sum()),"missing_baseline_row":221-len(tb),"participants_with_any_baseline_missing":int(tf[SNUM+SCAT].isna().any(axis=1).sum()),"missing_hidden_representation":0,"final_180d_count":len(tf),"final_90d_count":int(tf.eligible_90d_sensitivity.sum())}])
  trans.append({"target":target,"transformation":TRANSFORMS[target],"rank_normalization_fallback_used":False,"validation_fit_parameters":"none","validation_n":len(vf),"test_n":len(tf),"validation_raw_min":vf.analysis_value.min(),"validation_raw_max":vf.analysis_value.max(),"test_raw_min":tf.analysis_value.min(),"test_raw_max":tf.analysis_value.max()})
 csv(out/"probe_cohort_audit.csv",pd.DataFrame(cohorts));csv(out/"probe_target_transformations.csv",pd.DataFrame(trans));LOG.info("targets loaded after frozen plan; cohorts audited")
 pred=[];hyper=[];fold_cache={};fold_strategy={}
 for ti,target in enumerate(TARGETS):
  vf,tf=target_frames[target];ys=vf.observed_transformed.to_numpy();outer=[]
  for rep in range(a.outer_repetitions):
   sp,strategy=folds(ys,vf.clinical_site,a.outer_folds,a.seed+rep);outer.append(sp);fold_strategy[f"{target}:{rep}"]=strategy
  for fsname,(num,cat,hid,elastic,repname) in specs.items():
   data=data_for(vf,reps["validation"].get(repname));
   if set(data.participant_id)!=set(vf.participant_id):raise RuntimeError(f"feature cohort mismatch {target} {fsname}")
   for rep,sp in enumerate(outer):
    for fold,(itr,ite) in enumerate(sp):
     pipe,grid=estimator(num,cat,hid,elastic);inner=KFold(a.inner_folds,shuffle=True,random_state=a.seed+1000*rep+fold);gs=GridSearchCV(pipe,grid,cv=inner,scoring="neg_mean_squared_error",n_jobs=a.n_jobs,refit=True,error_score="raise");gs.fit(data.iloc[itr],ys[itr]);pp=gs.predict(data.iloc[ite]);best=gs.best_params_;alpha=best["model__alpha"]
     hyper.append({"target":target,"stage":"validation_outer","feature_set":fsname,"outer_repetition":rep,"outer_fold":fold,"selected_alpha":alpha,"selected_l1_ratio":best.get("model__l1_ratio"),"inner_best_neg_mse":gs.best_score_,"fold_strategy":fold_strategy[f"{target}:{rep}"]})
     for j,val in zip(ite,pp):pred.append({"participant_id":data.iloc[j].participant_id,"target":target,"outer_repetition":rep,"outer_fold":fold,"feature_set":fsname,"observed_raw":data.iloc[j].analysis_value,"observed_transformed":ys[j],"predicted_transformed":val,"predicted_raw_when_invertible":inverse(target,[val])[0],"selected_alpha":alpha,"model_status":"nested_validation_held_out"})
     if fsname=="simple_plus_neutral_all":fold_cache[(target,rep,fold)]={"train":itr,"test":ite,"alpha":alpha,"data":data,"y":ys}
   LOG.info("nested CV target=%s feature_set=%s complete",target,fsname)
 vp=pd.DataFrame(pred);parq(out/"validation_probe_predictions.parquet",vp);csv(out/"probe_model_hyperparameters.csv",pd.DataFrame(hyper))

 # Frozen validation-to-test transport; validation-only model selection.
 testpred=[];coef=[];final_models={}
 for target in TARGETS:
  vf,tf=target_frames[target]
  for fsname,(num,cat,hid,elastic,repname) in specs.items():
   vd=data_for(vf,reps["validation"].get(repname));td=data_for(tf,reps["test"].get(repname));pipe,grid=estimator(num,cat,hid,elastic);inner=KFold(a.inner_folds,shuffle=True,random_state=a.seed+9000);gs=GridSearchCV(pipe,grid,cv=inner,scoring="neg_mean_squared_error",n_jobs=a.n_jobs,refit=True,error_score="raise");gs.fit(vd,vd.observed_transformed);best=gs.best_params_;model=gs.best_estimator_;pp=model.predict(td);alpha=best["model__alpha"]
   fname=f"{target}__{fsname}.joblib";joblib.dump(model,out/"frozen_probe_models"/fname);final_models[(target,fsname)]=model;coef.extend(coef_rows(model,target,fsname));hyper.append({"target":target,"stage":"final_validation_fit","feature_set":fsname,"outer_repetition":np.nan,"outer_fold":np.nan,"selected_alpha":alpha,"selected_l1_ratio":best.get("model__l1_ratio"),"inner_best_neg_mse":gs.best_score_,"fold_strategy":"validation_only_inner_cv"})
   for j,val in enumerate(pp):testpred.append({"participant_id":td.iloc[j].participant_id,"target":target,"outer_repetition":np.nan,"outer_fold":np.nan,"feature_set":fsname,"observed_raw":td.iloc[j].analysis_value,"observed_transformed":td.iloc[j].observed_transformed,"predicted_transformed":val,"predicted_raw_when_invertible":inverse(target,[val])[0],"selected_alpha":alpha,"model_status":"frozen_validation_pipeline_test_transport"})
  LOG.info("frozen validation-to-test models target=%s complete",target)
 tp=pd.DataFrame(testpred);parq(out/"test_probe_predictions.parquet",tp);csv(out/"probe_model_hyperparameters.csv",pd.DataFrame(hyper));csv(out/"probe_coefficient_summary.csv",pd.DataFrame(coef))
 # Prespecified ±90-day timing sensitivity, primary comparison only and outside primary FDR.
 senspred=[]
 for target in TARGETS:
  vf,tf=target_frames[target];vf=vf[vf.eligible_90d_sensitivity].reset_index(drop=True);tf=tf[tf.eligible_90d_sensitivity].reset_index(drop=True);ys=vf.observed_transformed.to_numpy();outer=[folds(ys,vf.clinical_site,a.outer_folds,a.seed+700+rep)[0] for rep in range(a.outer_repetitions)]
  for fsname,repname in [("simple_baseline",""),("simple_plus_neutral_all","neutral_all")]:
   num,cat,hid,elastic,_=specs[fsname];vd=data_for(vf,reps["validation"].get(repname));td=data_for(tf,reps["test"].get(repname))
   for rep,sp in enumerate(outer):
    for fold,(itr,ite) in enumerate(sp):
     pipe,grid=estimator(num,cat,hid,False);gs=GridSearchCV(pipe,grid,cv=KFold(a.inner_folds,shuffle=True,random_state=a.seed+17000+100*rep+fold),scoring="neg_mean_squared_error",n_jobs=a.n_jobs,refit=True,error_score="raise");gs.fit(vd.iloc[itr],ys[itr]);pp=gs.predict(vd.iloc[ite]);alpha=gs.best_params_["model__alpha"]
     for j,val in zip(ite,pp):senspred.append({"participant_id":vd.iloc[j].participant_id,"target":target,"timing_window_days":90,"split":"validation_nested_cv","outer_repetition":rep,"outer_fold":fold,"feature_set":fsname,"observed_transformed":ys[j],"predicted_transformed":val,"selected_alpha":alpha,"model_status":"nested_validation_held_out_90d"})
   pipe,grid=estimator(num,cat,hid,False);gs=GridSearchCV(pipe,grid,cv=KFold(a.inner_folds,shuffle=True,random_state=a.seed+19000),scoring="neg_mean_squared_error",n_jobs=a.n_jobs,refit=True,error_score="raise");gs.fit(vd,ys);joblib.dump(gs.best_estimator_,out/"frozen_probe_models"/f"timing90d__{target}__{fsname}.joblib");pp=gs.predict(td)
   for j,val in enumerate(pp):senspred.append({"participant_id":td.iloc[j].participant_id,"target":target,"timing_window_days":90,"split":"test_transport","outer_repetition":np.nan,"outer_fold":np.nan,"feature_set":fsname,"observed_transformed":td.iloc[j].observed_transformed,"predicted_transformed":val,"selected_alpha":gs.best_params_["model__alpha"],"model_status":"frozen_validation_pipeline_test_transport_90d"})
 sp90=pd.DataFrame(senspred);parq(out/"probe_90d_sensitivity_predictions.parquet",sp90);timing=[]
 for split in ["validation_nested_cv","test_transport"]:
  for target in TARGETS:
   b=sp90[(sp90.split==split)&(sp90.target==target)&(sp90.feature_set=="simple_baseline")];u=sp90[(sp90.split==split)&(sp90.target==target)&(sp90.feature_set=="simple_plus_neutral_all")];mb=met(b.observed_transformed,b.predicted_transformed);mu=met(u.observed_transformed,u.predicted_transformed);ci=boot_delta_fast(b,u,a.bootstrap_replicates,a.seed+800000+stable_seed(split,target)%100000);timing.append({"split":split,"target":target,"timing_window_days":90,"n_participants":b.participant_id.nunique(),"baseline_r2":mb["r2"],"neutral_r2":mu["r2"],"delta_r2":mu["r2"]-mb["r2"],"delta_r2_ci_low":ci["delta_r2"][0],"delta_r2_ci_high":ci["delta_r2"][1],"primary_fdr_family":False})
 timing=pd.DataFrame(timing);csv(out/"probe_90d_timing_sensitivity.csv",timing);LOG.info("90-day timing sensitivity complete")
 # Performance and paired incremental bootstrap.
 perf=[]
 for split,pd0 in [("validation_nested_cv",vp),("test_transport",tp)]:
  for (target,fsname),q in pd0.groupby(["target","feature_set"]):
   m=met(q.observed_transformed,q.predicted_transformed);ci=boot_perf_fast(q,a.bootstrap_replicates,a.seed+stable_seed(split,target,fsname)%100000)
   perf.append({"split":split,"analysis_role":"development nested CV" if split.startswith("validation") else "secondary predictive transport","target":target,"feature_set":fsname,"n_participants":q.participant_id.nunique(),"n_prediction_rows":len(q),**m,**{f"{k}_ci_low":v[0] for k,v in ci.items()},**{f"{k}_ci_high":v[1] for k,v in ci.items()}})
 perf=pd.DataFrame(perf);csv(out/"probe_performance_summary.csv",perf)
 increments=[]
 for split,pd0 in [("validation_nested_cv",vp),("test_transport",tp)]:
  for target in TARGETS:
   base=pd0[(pd0.target==target)&(pd0.feature_set=="simple_baseline")]
   bm=met(base.observed_transformed,base.predicted_transformed)
   for fsname in [x for x in specs if x not in ["simple_baseline","expanded_baseline"]]:
    aug=pd0[(pd0.target==target)&(pd0.feature_set==fsname)];am=met(aug.observed_transformed,aug.predicted_transformed);ci=boot_delta_fast(base,aug,a.bootstrap_replicates,a.seed+200000+stable_seed(split,target,fsname)%100000)
    increments.append({"split":split,"target":target,"feature_set":fsname,"reference_feature_set":"simple_baseline","n_participants":base.participant_id.nunique(),"baseline_r2":bm["r2"],"augmented_r2":am["r2"],"delta_r2":am["r2"]-bm["r2"],"delta_spearman":am["spearman"]-bm["spearman"],"delta_mae":am["mae"]-bm["mae"],"delta_rmse":am["rmse"]-bm["rmse"],**{f"{k}_ci_low":v[0] for k,v in ci.items()},**{f"{k}_ci_high":v[1] for k,v in ci.items()}})
   base=pd0[(pd0.target==target)&(pd0.feature_set=="expanded_baseline")];aug=pd0[(pd0.target==target)&(pd0.feature_set=="expanded_plus_neutral_all")];bm=met(base.observed_transformed,base.predicted_transformed);am=met(aug.observed_transformed,aug.predicted_transformed);ci=boot_delta_fast(base,aug,a.bootstrap_replicates,a.seed+300000+stable_seed(split,target)%100000)
   increments.append({"split":split,"target":target,"feature_set":"expanded_plus_neutral_all","reference_feature_set":"expanded_baseline","n_participants":base.participant_id.nunique(),"baseline_r2":bm["r2"],"augmented_r2":am["r2"],"delta_r2":am["r2"]-bm["r2"],"delta_spearman":am["spearman"]-bm["spearman"],"delta_mae":am["mae"]-bm["mae"],"delta_rmse":am["rmse"]-bm["rmse"],**{f"{k}_ci_low":v[0] for k,v in ci.items()},**{f"{k}_ci_high":v[1] for k,v in ci.items()}})
 inc=pd.DataFrame(increments)
 # Primary neutral-state permutation nulls. One complete repeated-CV partition per draw.
 pcache={}
 for target in TARGETS:
  vf,_=target_frames[target]
  for rep in range(a.outer_repetitions):
   for fold in range(a.outer_folds):
    rec=fold_cache[(target,rep,fold)];itr,ite=rec["train"],rec["test"];d=rec["data"];pb=preprocessor(SNUM,SCAT,[]);btr=pb.fit_transform(d.iloc[itr]);bte=pb.transform(d.iloc[ite]);sh=StandardScaler();htr=sh.fit_transform(d.iloc[itr][[f"state_{i:03d}" for i in range(128)]]);hte=sh.transform(d.iloc[ite][[f"state_{i:03d}" for i in range(128)]]);strata=(d.iloc[itr].clinical_site.fillna("<missing>").astype(str)+"|"+d.iloc[itr].study_group.fillna("<missing>").astype(str)).to_numpy();groups=[np.where(strata==x)[0] for x in np.unique(strata)];strategy="within_clinical_site_x_study_group" if min(map(len,groups))>=2 else "global_fallback";pcache[(target,rep,fold)]={"btr":btr,"bte":bte,"htr":htr,"hte":hte,"ytr":rec["y"][itr],"yte":rec["y"][ite],"alpha":rec["alpha"],"groups":groups,"strategy":strategy,"test_ids":d.iloc[ite].participant_id.to_numpy()}
 def one_perm(target,b):
  rng=np.random.default_rng(a.seed+50000+b+100000*TARGETS.index(target));rows=[];strategies=[]
  for rep in range(a.outer_repetitions):
   for fold in range(a.outer_folds):
    z=pcache[(target,rep,fold)];order=np.arange(len(z["htr"]));strategies.append(z["strategy"])
    if z["strategy"].startswith("within"):
     for g in z["groups"]:order[g]=rng.permutation(g)
    else:order=rng.permutation(order)
    fit=Ridge(alpha=z["alpha"],solver="lsqr").fit(np.hstack([z["btr"],z["htr"][order]]),z["ytr"]);pp=fit.predict(np.hstack([z["bte"],z["hte"]]));rows.extend((pid,rep,y,p) for pid,y,p in zip(z["test_ids"],z["yte"],pp))
  q=pd.DataFrame(rows,columns=["participant_id","outer_repetition","y","pred"]);base=vp[(vp.target==target)&(vp.feature_set=="simple_baseline")][["participant_id","outer_repetition","predicted_transformed"]];q=q.merge(base,on=["participant_id","outer_repetition"],validate="one_to_one");return r2_score(q.y,q.pred)-r2_score(q.y,q.predicted_transformed),"+".join(sorted(set(strategies)))
 permrows=[]
 for target in TARGETS:
  null=Parallel(n_jobs=a.n_jobs,prefer="threads")(delayed(one_perm)(target,b) for b in range(a.permutation_replicates));vals=np.array([x[0] for x in null]);obs=float(inc.query("split == 'validation_nested_cv' and target == @target and feature_set == 'simple_plus_neutral_all'").delta_r2.iloc[0]);pval=(1+np.sum(vals>=obs))/(1+len(vals));permrows.append({"target":target,"feature_set":"simple_plus_neutral_all","observed_delta_r2":obs,"null_mean":vals.mean(),"null_sd":vals.std(ddof=1),"null_q025":np.quantile(vals,.025),"null_q975":np.quantile(vals,.975),"empirical_p_value":pval,"permutation_strategy":"all five outer repetitions; "+";".join(sorted(set(x[1] for x in null))),"permutation_replicates":len(vals),"null_values_json":json.dumps(vals.tolist())});LOG.info("permutation target=%s complete p=%.6g",target,pval)
 perm=pd.DataFrame(permrows);perm["primary_fdr_q_value"]=bh(perm.empirical_p_value);csv(out/"probe_incremental_permutation_tests.csv",perm)
 inc=inc.merge(perm[["target","empirical_p_value","primary_fdr_q_value"]],on="target",how="left");inc.loc[inc.feature_set!="simple_plus_neutral_all",["empirical_p_value","primary_fdr_q_value"]]=np.nan;csv(out/"probe_incremental_value.csv",inc)

 # Prespecified comparison tables.
 def irow(split,target,fs):return inc.query("split == @split and target == @target and feature_set == @fs").iloc[0]
 comp_full=[];comp_res=[];comp_nd=[];comp_k=[];transport=[]
 cdf=pd.DataFrame(coef)
 for target in TARGETS:
  vv_neu=vp[(vp.target==target)&(vp.feature_set=="simple_plus_neutral_all")];vv_full=vp[(vp.target==target)&(vp.feature_set=="simple_plus_full_all")];vv_res=vp[(vp.target==target)&(vp.feature_set=="simple_plus_neutral_glucose_residual")];vv_n=vp[(vp.target==target)&(vp.feature_set=="simple_plus_neutral_night")];vv_d=vp[(vp.target==target)&(vp.feature_set=="simple_plus_neutral_day")]
  tt_neu=tp[(tp.target==target)&(tp.feature_set=="simple_plus_neutral_all")];tt_full=tp[(tp.target==target)&(tp.feature_set=="simple_plus_full_all")];tt_res=tp[(tp.target==target)&(tp.feature_set=="simple_plus_neutral_glucose_residual")];tt_n=tp[(tp.target==target)&(tp.feature_set=="simple_plus_neutral_night")];tt_d=tp[(tp.target==target)&(tp.feature_set=="simple_plus_neutral_day")]
  cvn=cdf[(cdf.target==target)&(cdf.feature_set=="simple_plus_neutral_all")&cdf.feature.str.contains("hidden_state")].coefficient.to_numpy();cvf=cdf[(cdf.target==target)&(cdf.feature_set=="simple_plus_full_all")&cdf.feature.str.contains("hidden_state")].coefficient.to_numpy();cos=np.dot(cvn,cvf)/max(np.linalg.norm(cvn)*np.linalg.norm(cvf),1e-12)
  for split,neu,full,resi,night,day in [("validation_nested_cv",vv_neu,vv_full,vv_res,vv_n,vv_d),("test_transport",tt_neu,tt_full,tt_res,tt_n,tt_d)]:
   nr=irow(split,target,"simple_plus_neutral_all");fr=irow(split,target,"simple_plus_full_all");rr=irow(split,target,"simple_plus_neutral_glucose_residual");nrr=irow(split,target,"simple_plus_neutral_night");dr=irow(split,target,"simple_plus_neutral_day")
   ci_fn=boot_delta_fast(neu,full,a.bootstrap_replicates,a.seed+400000+stable_seed(split,target,"fn")%100000);ci_rn=boot_delta_fast(neu,resi,a.bootstrap_replicates,a.seed+400000+stable_seed(split,target,"rn")%100000);ci_nd=boot_delta_fast(day,night,a.bootstrap_replicates,a.seed+400000+stable_seed(split,target,"nd")%100000)
   comp_full.append({"split":split,"target":target,"neutral_delta_r2":nr.delta_r2,"full_delta_r2":fr.delta_r2,"full_minus_neutral_r2":fr.delta_r2-nr.delta_r2,"ci_low":ci_fn["delta_r2"][0],"ci_high":ci_fn["delta_r2"][1],"hidden_coefficient_cosine_final_validation_models":cos,"interpretation":"full state directly receives participant clinical information"})
   comp_res.append({"split":split,"target":target,"neutral_delta_r2":nr.delta_r2,"residualized_delta_r2":rr.delta_r2,"residualized_minus_neutral_r2":rr.delta_r2-nr.delta_r2,"ci_low":ci_rn["delta_r2"][0],"ci_high":ci_rn["delta_r2"][1],"interpretation":"secondary information after validation-fitted glycemic residualization"})
   comp_nd.append({"split":split,"target":target,"night_delta_r2":nrr.delta_r2,"day_delta_r2":dr.delta_r2,"night_minus_day_delta_r2":nrr.delta_r2-dr.delta_r2,"ci_low":ci_nd["delta_r2"][0],"ci_high":ci_nd["delta_r2"][1],"direction_agreement":np.sign(nrr.delta_r2)==np.sign(dr.delta_r2),"interpretation":"context sensitivity; matched participants"})
  kr=irow("validation_nested_cv",target,"simple_plus_exploratory_k2");kt=irow("test_transport",target,"simple_plus_exploratory_k2");comp_k.append({"target":target,"validation_delta_r2":kr.delta_r2,"validation_ci_low":kr.delta_r2_ci_low,"validation_ci_high":kr.delta_r2_ci_high,"test_delta_r2":kt.delta_r2,"test_ci_low":kt.delta_r2_ci_low,"test_ci_high":kt.delta_r2_ci_high,"interpretation":"exploratory binary glycemic-tail sensitivity; excluded from primary FDR"})
  vr=irow("validation_nested_cv",target,"simple_plus_neutral_all");tr=irow("test_transport",target,"simple_plus_neutral_all");pr=perm[perm.target==target].iloc[0];cat=transport_category(vr.delta_r2,vr.delta_r2_ci_low,pr.primary_fdr_q_value,tr.delta_r2)
  transport.append({"target":target,"analysis_role":"secondary predictive transport; not untouched confirmation","validation_n":int(vr.n_participants),"test_n":int(tr.n_participants),"validation_baseline_r2":vr.baseline_r2,"validation_augmented_r2":vr.augmented_r2,"validation_delta_r2":vr.delta_r2,"validation_ci_low":vr.delta_r2_ci_low,"validation_ci_high":vr.delta_r2_ci_high,"permutation_p_value":pr.empirical_p_value,"primary_fdr_q_value":pr.primary_fdr_q_value,"test_baseline_r2":tr.baseline_r2,"test_augmented_r2":tr.augmented_r2,"test_delta_r2":tr.delta_r2,"test_ci_low":tr.delta_r2_ci_low,"test_ci_high":tr.delta_r2_ci_high,"direction_agreement":np.sign(vr.delta_r2)==np.sign(tr.delta_r2),"effect_size_difference_test_minus_validation":tr.delta_r2-vr.delta_r2,"transport_category":cat})
 full=pd.DataFrame(comp_full);resid=pd.DataFrame(comp_res);nd=pd.DataFrame(comp_nd);ks=pd.DataFrame(comp_k);ts=pd.DataFrame(transport);csv(out/"full_vs_neutral_probe_comparison.csv",full);csv(out/"residualized_probe_comparison.csv",resid);csv(out/"night_day_probe_comparison.csv",nd);csv(out/"exploratory_k2_probe_sensitivity.csv",ks);csv(out/"probe_transport_summary.csv",ts)
 # Decision hierarchy.
 cats=dict(zip(ts.target,ts.transport_category));hs=cats["c_reactive_protein_i"]
 if hs=="incremental_value_transports" and all(cats[x] not in ["incremental_value_transports"] for x in TARGETS[1:]):study="hsCRP_specific_incremental_signal"
 elif sum(x=="incremental_value_transports" for x in cats.values())>=2:study="external_clinical_information_adds_beyond_simple_summaries"
 elif all(x=="no_incremental_value" for x in cats.values()):study="glycemic_and_wearable_summaries_are_sufficient"
 elif any(x=="validation_increment_not_transported" for x in cats.values()):study="validation_signal_not_transported"
 elif all(irow("validation_nested_cv",x,"simple_plus_neutral_all").delta_r2<=0 and irow("validation_nested_cv",x,"simple_plus_full_all").delta_r2>0 for x in TARGETS):study="full_state_only_increment_due_to_static_conditioning"
 else:study="mixed_or_inconclusive"
 decisions={}
 for target in TARGETS:
  vr=irow("validation_nested_cv",target,"simple_plus_neutral_all");tr=irow("test_transport",target,"simple_plus_neutral_all");pr=perm[perm.target==target].iloc[0];decisions[target]={"baseline_validation_r2":vr.baseline_r2,"neutral_validation_r2":vr.augmented_r2,"validation_delta_r2":vr.delta_r2,"validation_delta_r2_ci":[vr.delta_r2_ci_low,vr.delta_r2_ci_high],"permutation_p_value":pr.empirical_p_value,"primary_fdr_q_value":pr.primary_fdr_q_value,"test_delta_r2":tr.delta_r2,"test_delta_r2_ci":[tr.delta_r2_ci_low,tr.delta_r2_ci_high],"transport_category":cats[target],"full_state":full[(full.target==target)&(full.split=="validation_nested_cv")].iloc[0].to_dict(),"residualized_state":resid[(resid.target==target)&(resid.split=="validation_nested_cv")].iloc[0].to_dict(),"night_day":nd[(nd.target==target)&(nd.split=="validation_nested_cv")].iloc[0].to_dict(),"exploratory_k2":ks[ks.target==target].iloc[0].to_dict()}
 decision={"study_level_conclusion":study,"analysis_role":"secondary prespecified-target predictive analysis and secondary predictive transport","targets":decisions,"primary_fdr_family":TARGETS,"primary_conclusion_from_steps_3_4_unchanged":"reliable continuous glycemic manifold modified by static conditioning","test_not_untouched_confirmation":True,"authorization_for_final_synthesis":True,"warnings":["Test biomarkers were previously inspected in Step 4; transport is secondary.","Full-profile state prediction may reflect directly supplied static clinical inputs.","Null increments are retained and interpreted."],"blockers":[]};dump(out/"step5_decision.json",decision)
 # Required figures from held-out or frozen-test predictions only.
 sns.set_theme(style="whitegrid");vplot=inc[(inc.split=="validation_nested_cv")&inc.feature_set.isin(["simple_plus_neutral_all","simple_plus_full_all","simple_plus_neutral_glucose_residual"])].copy();vplot["label"]=vplot.target.map(TLABEL)+" | "+vplot.feature_set.str.replace("simple_plus_","")
 plt.figure(figsize=(10,6));y=np.arange(len(vplot));plt.errorbar(vplot.delta_r2,y,xerr=[vplot.delta_r2-vplot.delta_r2_ci_low,vplot.delta_r2_ci_high-vplot.delta_r2],fmt="o");plt.axvline(0,color="black",lw=1);plt.yticks(y,vplot.label);plt.xlabel("Validation nested-CV delta R²");plt.tight_layout();plt.savefig(out/"fig_probe_incremental_r2_validation.png",dpi=170);plt.close()
 q=ts.melt(id_vars=["target"],value_vars=["validation_delta_r2","test_delta_r2"],var_name="split",value_name="delta_r2");plt.figure(figsize=(8,5));sns.barplot(q,x="target",y="delta_r2",hue="split");plt.axhline(0,color="black");plt.xticks(rotation=20,ha="right");plt.title("Secondary predictive transport (test previously examined)");plt.tight_layout();plt.savefig(out/"fig_probe_validation_test_transport.png",dpi=170);plt.close()
 fig,ax=plt.subplots(2,3,figsize=(14,8));
 for j,target in enumerate(TARGETS):
  for i,(name,d) in enumerate([("Validation OOF",vp),("Frozen test",tp)]):
   z=d[(d.target==target)&(d.feature_set=="simple_plus_neutral_all")].groupby("participant_id",as_index=False)[["observed_transformed","predicted_transformed"]].mean();ax[i,j].scatter(z.observed_transformed,z.predicted_transformed,s=14,alpha=.7);lo=min(z.observed_transformed.min(),z.predicted_transformed.min());hi=max(z.observed_transformed.max(),z.predicted_transformed.max());ax[i,j].plot([lo,hi],[lo,hi],"k--",lw=1);ax[i,j].set_title(f"{TLABEL[target]} — {name}")
 plt.tight_layout();plt.savefig(out/"fig_probe_predicted_vs_observed.png",dpi=170);plt.close()
 q=perf[(perf.split=="validation_nested_cv")&perf.feature_set.isin(["simple_baseline","simple_plus_neutral_all"])];plt.figure(figsize=(8,5));sns.barplot(q,x="target",y="r2",hue="feature_set");plt.xticks(rotation=20,ha="right");plt.title("Held-out baseline versus baseline + neutral state");plt.tight_layout();plt.savefig(out/"fig_probe_baseline_vs_state.png",dpi=170);plt.close()
 q=inc[(inc.split=="validation_nested_cv")&inc.feature_set.isin(["simple_plus_neutral_night","simple_plus_neutral_day"])];plt.figure(figsize=(8,5));sns.barplot(q,x="target",y="delta_r2",hue="feature_set");plt.axhline(0,color="black");plt.xticks(rotation=20,ha="right");plt.tight_layout();plt.savefig(out/"fig_probe_night_vs_day.png",dpi=170);plt.close()
 fig,ax=plt.subplots(1,3,figsize=(14,4));
 for j,row in perm.iterrows():
  vals=np.array(json.loads(row.null_values_json));ax[j].hist(vals,bins=35,color="#9ecae1");ax[j].axvline(row.observed_delta_r2,color="crimson",lw=2);ax[j].set_title(f"{TLABEL[row.target]}\nq={row.primary_fdr_q_value:.3g}");ax[j].set_xlabel("Permuted delta R²")
 plt.tight_layout();plt.savefig(out/"fig_probe_permutation_null.png",dpi=170);plt.close()
 q=inc[(inc.split=="validation_nested_cv")&inc.feature_set.isin(["simple_plus_full_all","simple_plus_neutral_all"])];plt.figure(figsize=(8,5));sns.barplot(q,x="target",y="delta_r2",hue="feature_set");plt.axhline(0,color="black");plt.xticks(rotation=20,ha="right");plt.tight_layout();plt.savefig(out/"fig_probe_full_vs_neutral.png",dpi=170);plt.close()
 fig,ax=plt.subplots(1,2,figsize=(12,4));co=pd.DataFrame(cohorts);sns.barplot(co,x="target",y="final_180d_count",hue="split",ax=ax[0]);sns.barplot(ts,x="target",y="test_delta_r2",ax=ax[1]);ax[0].tick_params(axis="x",rotation=20);ax[1].tick_params(axis="x",rotation=20);ax[0].set_title("Target coverage");ax[1].set_title("Secondary test delta R²");plt.tight_layout();plt.savefig(out/"fig_probe_final_summary.png",dpi=170);plt.close()

 # Complete 20-section report.
 report=["# Step 5 — Incremental clinical probe analysis","","## 1. Objective","Assess whether frozen participant hidden-state representations improve prediction of three prespecified external phenotypes beyond conventional participant summaries.","","## 2. Why the probes are secondary rather than untouched confirmation","Step 4 already examined test biomarker associations. Test evaluation here is therefore secondary predictive transport, never independent or untouched confirmation.","","## 3. Prespecified targets","The unchanged targets were high-sensitivity CRP, NT-proBNP, and BUN/creatinine ratio.","","## 4. Participant cohorts",pd.DataFrame(cohorts).to_markdown(index=False),"","## 5. Simple baseline",f"The simple baseline used {len(SNUM)} numeric and {len(SCAT)} categorical demographic, CGM, wearable, and acquisition summaries. Hidden-state-derived distances, forecast deltas, PCA scores, reliability, and k=2 labels were excluded.","","## 6. Expanded direct-input baseline",f"The secondary expanded baseline added {len(exp_num)} numeric/value-or-missing-indicator columns and {len(exp_cat)} categorical columns corresponding to static inputs consumed by the forecasting model. None was a target or exact target derivative.","","## 7. Hidden-state representations","Primary: the complete 128-dimensional neutral_all median representation. Secondary: full_all, validation glucose-residualized neutral, neutral night, neutral day, frozen validation PCA, and a fixed-grid ElasticNet sensitivity. No target-based dimension selection occurred.","","## 8. Nested validation procedure",f"Validation used {a.outer_repetitions} deterministic repetitions of {a.outer_folds}-fold outer CV and {a.inner_folds}-fold inner tuning. All preprocessing was fitted within training folds; every saved validation prediction was held out.","","## 9. Frozen test transport procedure","Each final preprocessing/model pipeline was selected and fitted using validation only, serialized, then applied unchanged to eligible test participants. Test was not used for tuning.","","## 10. Primary neutral-state incremental results",ts[["target","validation_n","validation_baseline_r2","validation_augmented_r2","validation_delta_r2","validation_ci_low","validation_ci_high","primary_fdr_q_value"]].to_markdown(index=False),"","### ±90-day timing sensitivity (outside primary FDR)",timing.to_markdown(index=False),"","## 11. Permutation results",perm.drop(columns=["null_values_json"]).to_markdown(index=False),"","## 12. Full-profile versus neutral-state results",full.to_markdown(index=False),"","## 13. Glucose-residualized results",resid.to_markdown(index=False),"","## 14. Night-versus-day results",nd.to_markdown(index=False),"","## 15. Exploratory k=2 sensitivity",ks.to_markdown(index=False),"","## 16. Target-specific interpretation"]
 for target in TARGETS:
  z=ts[ts.target==target].iloc[0];report.append(f"- **{TLABEL[target]}:** validation delta R² {z.validation_delta_r2:.4f} (95% CI {z.validation_ci_low:.4f} to {z.validation_ci_high:.4f}); test delta R² {z.test_delta_r2:.4f} (95% CI {z.test_ci_low:.4f} to {z.test_ci_high:.4f}); `{z.transport_category}`.")
 report += ["","## 17. Predictive transport categories",ts[["target","validation_delta_r2","test_delta_r2","direction_agreement","transport_category"]].to_markdown(index=False),"","## 18. Limitations","Linear probes detect only linearly recoverable information; test biomarkers were already inspected; biomarker timing is near rather than concurrent; expanded full-state models may reuse directly supplied clinical information; repeated-CV uncertainty is participant-clustered but does not create an external cohort.","","## 19. Final Step 5 conclusion",f"The frozen study-level category is **{study}**. Null and discordant target results are retained. This conclusion is secondary to the established continuous glycemic-manifold result.","","## 20. Authorization for final synthesis","Step 5 completed without leakage or blockers and authorizes synthesis-only Step 6. Step 6 may summarize but may not fit, select, cluster, or test anew.",""]
 text(out/"step5_report.md","\n".join(report))
 required=["step5_analysis_plan_frozen.json","probe_cohort_audit.csv","probe_baseline_feature_audit.csv","probe_feature_sets.json","validation_probe_predictions.parquet","test_probe_predictions.parquet","probe_performance_summary.csv","probe_incremental_value.csv","probe_incremental_permutation_tests.csv","probe_transport_summary.csv","probe_target_transformations.csv","probe_model_hyperparameters.csv","probe_coefficient_summary.csv","full_vs_neutral_probe_comparison.csv","residualized_probe_comparison.csv","night_day_probe_comparison.csv","exploratory_k2_probe_sensitivity.csv","frozen_probe_models","probe_90d_sensitivity_predictions.parquet","probe_90d_timing_sensitivity.csv","step5_decision.json","step5_report.md","step5_run.log"]
 foldcheck=vp.groupby(["target","participant_id","outer_repetition"]).outer_fold.nunique().max()==1;setcheck=True
 for target in TARGETS:
  for rep in range(a.outer_repetitions):
   ref=set(vp[(vp.target==target)&(vp.feature_set=="simple_baseline")&(vp.outer_repetition==rep)].participant_id)
   for fsname in specs:setcheck &= set(vp[(vp.target==target)&(vp.feature_set==fsname)&(vp.outer_repetition==rep)].participant_id)==ref
 qc={"status":"QC_COMPLETE","plan_frozen_before_target_load":plan_time<target_loaded_at,"plan_hash":planhash,"targets_exact":sorted(TARGETS)==sorted(ve.target_name.unique())==sorted(te.target_name.unique()),"validation_participants":len(vid),"test_participants":len(tid),"validation_test_disjoint":not bool(set(vid)&set(tid)),"validation_predictions_all_held_out":set(vp.model_status)=={"nested_validation_held_out"},"validation_fold_assignment_same_across_feature_sets":bool(foldcheck),"participant_subsets_same_within_comparisons":bool(setcheck),"test_predictions_frozen_validation_pipeline":set(tp.model_status)=={"frozen_validation_pipeline_test_transport"},"finite_hidden_representations":True,"required_outputs_present":all((out/x).exists() for x in required),"frozen_model_count":len(list((out/"frozen_probe_models").glob("*.joblib"))),"figure_count":len(list(out.glob("fig_probe_*.png"))),"no_hidden_state_or_forecast_regeneration":not any(out.glob("test_hidden_states")) and not any(out.glob("test_forecasts")),"primary_fdr_test_count":len(perm),"primary_fdr_targets":perm.target.tolist(),"upstream_hashes":{"step3_decision":sha(s3/"clustering_selection_decision.json"),"step3b_plan":sha(s3b/"exploratory_k2_analysis_plan_frozen.json"),"step4_qc":sha(s4/"step4_independent_qc.json")},"blockers":[]}
 if not all([qc["plan_frozen_before_target_load"],qc["targets_exact"],qc["validation_test_disjoint"],qc["validation_predictions_all_held_out"],qc["validation_fold_assignment_same_across_feature_sets"],qc["participant_subsets_same_within_comparisons"],qc["test_predictions_frozen_validation_pipeline"],qc["required_outputs_present"],qc["frozen_model_count"]==len(TARGETS)*(len(specs)+2),qc["figure_count"]==8,qc["primary_fdr_test_count"]==3]):raise RuntimeError("Step 5 QC failure: "+json.dumps(qc,default=jd))
 dump(out/"step5_independent_qc.json",qc)
 manifest={"run_id":rid,"timestamp":datetime.now(timezone.utc).isoformat(),"status":"QC_COMPLETE","analysis_plan_hash":planhash,"analysis_plan_frozen_at":plan_time,"target_values_loaded_at":target_loaded_at,"study_level_conclusion":study,"target_transport_categories":cats,"participant_counts":{"validation":239,"test":221},"protocol":{"outer_folds":a.outer_folds,"inner_folds":a.inner_folds,"outer_repetitions":a.outer_repetitions,"bootstrap_replicates":a.bootstrap_replicates,"permutation_replicates":a.permutation_replicates,"seed":a.seed},"input_paths":{"step0":str(s0),"step2":str(s2),"step3":str(s3),"step3b":str(s3b),"step4":str(s4),"panel":str(panel),"static":str(static),"checkpoint":str(checkpoint)},"input_hashes":{"validation_representations":sha(s2/"participant_representations.parquet"),"validation_features":sha(s3/"validation_glycemic_nuisance_features.parquet"),"validation_targets":sha(s3/"validation_external_targets.parquet"),"test_representations":sha(s4/"test_participant_representations.parquet"),"test_features":sha(s4/"test_glycemic_nuisance_features.parquet"),"test_targets":sha(s4/"test_external_targets.parquet"),"panel":sha(panel),"static":sha(static),"checkpoint":sha(checkpoint)},"frozen_pca_components":n90,"feature_set_count":len(specs),"timing_90d_sensitivity_model_count":6,"warnings":decision["warnings"],"blockers":[],"runtime_seconds":time.time()-started,"environment":{"python":platform.python_version(),"platform":platform.platform(),"sklearn":__import__("sklearn").__version__,"numpy":np.__version__,"pandas":pd.__version__}}
 manifest["output_paths"]={p.name:str(p.resolve()) for p in out.iterdir()};manifest["output_hashes"]={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="step5_manifest.json"};dump(out/"step5_manifest.json",manifest)
 latest=root/"latest";tmp=root/".latest.tmp"
 if tmp.exists() or tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(rid);os.replace(tmp,latest);LOG.info("QC COMPLETE latest=%s conclusion=%s",latest,study)
 manifest["output_hashes"]={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="step5_manifest.json"};dump(out/"step5_manifest.json",manifest)
 print(json.dumps({"output_directory":str(out),"cohorts":cohorts,"transport":transport,"full_vs_neutral":comp_full,"residualized":comp_res,"night_day":comp_nd,"exploratory_k2":comp_k,"study_level_conclusion":study,"warnings":decision["warnings"],"blockers":[]},indent=2,default=jd))

def stable_seed(*parts):return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:8],16)
def _weighted_sample_arrays(frame,seed,n):
 ids,codes=np.unique(frame.participant_id.astype(str),return_inverse=True);rng=np.random.default_rng(seed);y=frame.observed_transformed.to_numpy(float);p=frame.predicted_transformed.to_numpy(float)
 for _ in range(n):
  counts=rng.multinomial(len(ids),np.repeat(1/len(ids),len(ids)));reps=counts[codes];yield np.repeat(y,reps),np.repeat(p,reps)
def boot_perf_fast(frame,n,seed):
 vals=[met(y,p) for y,p in _weighted_sample_arrays(frame,seed,n)];return {k:(np.quantile([x[k] for x in vals],.025),np.quantile([x[k] for x in vals],.975)) for k in vals[0]}
def boot_delta_fast(base,aug,n,seed):
 keys=["participant_id"] if base.outer_repetition.isna().all() else ["participant_id","outer_repetition","outer_fold"]
 q=base[keys+["observed_transformed","predicted_transformed"]].merge(aug[keys+["predicted_transformed"]],on=keys,suffixes=("_base","_aug"),validate="one_to_one");ids,codes=np.unique(q.participant_id.astype(str),return_inverse=True);rng=np.random.default_rng(seed);y=q.observed_transformed.to_numpy(float);pb=q.predicted_transformed_base.to_numpy(float);pa=q.predicted_transformed_aug.to_numpy(float);vals=[]
 for _ in range(n):
  counts=rng.multinomial(len(ids),np.repeat(1/len(ids),len(ids)));w=counts[codes];yy=np.repeat(y,w);bb=np.repeat(pb,w);aa=np.repeat(pa,w);mb=met(yy,bb);ma=met(yy,aa);vals.append({"delta_r2":ma["r2"]-mb["r2"],"delta_spearman":ma["spearman"]-mb["spearman"],"delta_mae":ma["mae"]-mb["mae"],"delta_rmse":ma["rmse"]-mb["rmse"]})
 return {k:(np.quantile([x[k] for x in vals],.025),np.quantile([x[k] for x in vals],.975)) for k in vals[0]}
if __name__=="__main__":main()
