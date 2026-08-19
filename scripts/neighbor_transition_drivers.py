from __future__ import annotations
import hashlib, json, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import within_subtype_phase1 as phase1
import posthoc_phase1_literature_alignment as posthoc
from neighbor_transition_dynamic_extract import DOMAINS as DYNAMIC_DOMAINS
from ssmcgm.analysis.within_subtype_config import STUDY1_ROOT, STUDY2_ROOT, CANONICAL_STRATA, SEED

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT=STUDY2_ROOT/"neighbor_transition_drivers"
PH4=STUDY2_ROOT/"phase4_time_resolved_extension"
HOURS=[6,12,24,48]
B=1000
SUBTYPES=list(CANONICAL_STRATA)
SUBLABEL={"healthy":"Healthy","pre_diabetes":"Pre-diabetes","t2d_oral_non_insulin":"T2D oral non-insulin","insulin_dependent":"Insulin-dependent*"}
CONT=posthoc.ALL_FACTORS
BINARY=posthoc.MEDS
CATEG=[posthoc.SITE,posthoc.SEX]
STATIC_DOMAINS={
 "clinical_factors":posthoc.FACTORS,
 "lipids_bp":posthoc.EXTRA,
 "medication":BINARY,
 "site_sex":CATEG,
}
DOMAIN_ORDER=["clinical_factors","medication","cgm_level","cgm_variability","cgm_dynamics","hr_respiration","activity","sleep","stress_other_wearables"]
COLORS={"S":"#13294B","D":"#008C95","SD":"#B21F35","N":"#9A9A9A"}
SALT="neighbor-transition-v1-derived-only"
DYN_META=["valid_observation_count","qualifying_segment_count","available_streaming_duration_hours_mean"]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
 return h.hexdigest()
def hpid(x): return hashlib.sha256((SALT+str(x)).encode()).hexdigest()[:24]
def write_json(p,x): p.write_text(json.dumps(x,indent=2,default=lambda z: float(z) if isinstance(z,np.floating) else int(z) if isinstance(z,np.integer) else str(z))+"\n")
def bh(p):
 p=np.asarray(p,float); out=np.full(len(p),np.nan); ok=np.isfinite(p)
 if not ok.any(): return out
 v=p[ok]; o=np.argsort(v); q=v[o]*len(v)/(np.arange(len(v))+1); q=np.minimum.accumulate(q[::-1])[::-1]; tmp=np.empty(len(v)); tmp[o]=np.minimum(q,1); out[ok]=tmp; return out
def ece(y,p,bins=10):
 y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); z=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
  if m.any(): z+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(z)
def metrics(y,p):
 if len(np.unique(y))<2: return dict(auroc=np.nan,auprc=np.nan,log_loss=np.nan,brier=np.nan,ece=np.nan)
 return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),log_loss=log_loss(y,p,labels=[0,1]),brier=brier_score_loss(y,p),ece=ece(y,p))
def bootstrap_metric(y,p,g,seed):
 rng=np.random.default_rng(seed); ug=np.unique(g); vals={k:[] for k in metrics(y,p)}
 for _ in range(B):
  pick=rng.choice(ug,len(ug),replace=True); idx=np.concatenate([np.flatnonzero(g==q) for q in pick])
  m=metrics(y[idx],p[idx])
  for k,v in m.items(): vals[k].append(v)
 return {k:(float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))) for k,v in vals.items()}
def bootstrap_diff(frame,value,a,b,seed):
 wide=frame.pivot(index="anchor_id",columns="transition_class",values=value)
 if a not in wide or b not in wide: return np.nan,np.nan,np.nan,np.nan,0
 d=(wide[a]-wide[b]).dropna().to_numpy(float)
 if not len(d): return np.nan,np.nan,np.nan,np.nan,0
 rng=np.random.default_rng(seed); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(B)])
 p=2*min((boots<=0).mean(),(boots>=0).mean())
 return float(d.mean()),float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5)),float(min(1,p)),len(d)

def recover_labels(frame):
 manifest=json.loads((STUDY2_ROOT/"phase1_clinical_clustering/frozen_clustering_manifest.json").read_text())
 rows=[]
 for subtype in SUBTYPES:
  info=manifest["clusters"][subtype]; sub=frame[frame.canonical_stratum==subtype].copy()
  if info["missing_data_strategy"]=="complete_case": sub=sub[sub[posthoc.FACTORS].notna().all(axis=1)].copy()
  pipe=joblib.load(STUDY2_ROOT/info["preprocessing_pipeline_path"])
  X=phase1.apply_pipeline(sub,pipe["factors"],pipe["log_transformed"],pipe["imputer"],pipe["scaler"])
  cj=json.loads((STUDY2_ROOT/info["centroid_path"]).read_text())["centroids_by_display_cluster"]
  order=sorted(map(int,cj)); C=np.array([cj[str(i)] for i in order])
  sub["display_cluster"]=[order[i] for i in cdist(X,C).argmin(1)]
  rows.append(sub)
 return pd.concat(rows,ignore_index=True),manifest

def fit_reference(static,dynamic):
 refs={}
 for subtype in SUBTYPES:
  st=static[(static.canonical_stratum==subtype)&(static.split=="train")]
  refs[subtype]={"cont_median":st[CONT].median(),"cont_mean":st[CONT].mean(),"cont_sd":st[CONT].std().replace(0,1),"cont_range":(st[CONT].max()-st[CONT].min()).replace(0,1)}
  for hour in HOURS:
   dy=dynamic[(dynamic.participant_id.isin(st.participant_id))&(dynamic.hour==hour)]
   feats=[f for fs in DYNAMIC_DOMAINS.values() for f in fs if f in dy and dy[f].notna().mean()>=.60]
   refs[subtype][hour]={"features":feats,"median":dy[feats].median(),"mean":dy[feats].mean(),"sd":dy[feats].std().replace(0,1)}
 return refs

def standardized_tables(static,dynamic,refs,hour):
 out={}
 for subtype in SUBTYPES:
  st=static[static.canonical_stratum==subtype].copy(); r=refs[subtype]
  z=(st[CONT].fillna(r["cont_median"])-r["cont_mean"])/r["cont_sd"]
  for c in CONT: st["s_"+c]=z[c]
  for c in BINARY: st["s_"+c]=pd.to_numeric(st[c],errors="coerce").fillna(0).clip(0,1)
  for c in CATEG: st["s_"+c]=st[c].fillna("__missing__").astype(str)
  dy=dynamic[dynamic.hour==hour].copy(); dr=r[hour]; feats=dr["features"]
  zdy=(dy[feats].fillna(dr["median"])-dr["mean"])/dr["sd"]
  for c in feats: dy["d_"+c]=zdy[c]
  out[subtype]=(st.set_index("participant_id"),dy.set_index("participant_id"),feats)
 return out

def pair_sims(a,b,st,dy,features,ref):
 row={}
 for c in CONT: row["static__"+c]=-abs(float(st.at[a,"s_"+c])-float(st.at[b,"s_"+c]))
 for c in BINARY+CATEG: row["static__"+c]=float(st.at[a,"s_"+c]==st.at[b,"s_"+c])
 g=[]
 for c in CONT:
  av=st.at[a,c]; bv=st.at[b,c]
  g.append(1-min(1,abs(av-bv)/ref["cont_range"][c]) if pd.notna(av) and pd.notna(bv) else np.nan)
 for c in BINARY+CATEG: g.append(float(st.at[a,"s_"+c]==st.at[b,"s_"+c]))
 row["static__gower"]=float(np.nanmean(g))
 for c in features: row["dynamic__"+c]=-abs(float(dy.at[a,"d_"+c])-float(dy.at[b,"d_"+c]))
 for dom,fs in STATIC_DOMAINS.items():
  cols=["static__"+c for c in fs if "static__"+c in row]; row["domain__"+dom]=float(np.mean([row[c] for c in cols])) if cols else np.nan
 row["domain__static_gower"]=row["static__gower"]
 for dom,fs in DYNAMIC_DOMAINS.items():
  cols=["dynamic__"+c for c in fs if "dynamic__"+c in row]; row["domain__"+dom]=float(np.mean([row[c] for c in cols])) if cols else np.nan
 return row

def nearest_sets(X,k,metric):
 D=cdist(X,X,metric=metric); np.fill_diagonal(D,np.inf)
 return D,np.argsort(D,axis=1)[:,:k]

def build_pairs(cohort_name,cohort,static,dynamic,refs,hour,metric="cosine",k_mode="primary",cluster_match=False):
 ids=cohort.participant_id.astype(str).tolist(); tab=standardized_tables(static,dynamic,refs,hour)
 h0p=STUDY1_ROOT/"step2/h0_matrix.parquet"; htp=PH4/f"snapshots/h_t_full_hour{hour:02d}.parquet"
 h0cols=pq.read_schema(h0p).names[2:]; htcols=pq.read_schema(htp).names[4:]
 h0=pd.read_parquet(h0p).set_index("participant_id").loc[ids,h0cols].to_numpy(np.float32)
 ht=pd.read_parquet(htp).set_index("participant_id").loc[ids,htcols].to_numpy(np.float32)
 endpoint=pd.read_parquet(STUDY1_ROOT/"step3/h_t_full.parquet",columns=["participant_id","n_overnight_anchors"]).set_index("participant_id")
 rows=[]; counts=[]
 for subtype in SUBTYPES:
  subids=cohort.loc[cohort.canonical_stratum==subtype,"participant_id"].astype(str).tolist(); ix=np.array([ids.index(x) for x in subids])
  n=len(ix); kp=min(30,max(5,round(.15*n)))
  k=max(3,kp//2) if k_mode=="small" else min(n-1,2*kp) if k_mode=="large" else min(n-1,kp)
  D0,N0=nearest_sets(h0[ix],k,metric); Dt,Nt=nearest_sets(ht[ix],k,metric)
  st,dy,features=tab[subtype]; meta=dynamic[dynamic.hour==hour].set_index("participant_id")
  labels=cohort.set_index("participant_id").display_cluster
  for ai,a in enumerate(subids):
   r=set(N0[ai])&set(Nt[ai]); l=set(N0[ai])-set(Nt[ai]); g=set(Nt[ai])-set(N0[ai])
   excluded=set(N0[ai])|set(Nt[ai])|{ai}; cand=[q for q in range(n) if q not in excluded and (not cluster_match or labels[subids[q]]==labels[a])]
   chosen=[]
   for gj in sorted(g):
    if not cand: break
    target=np.array([D0[ai,gj],meta.at[subids[gj],"valid_observation_count"],endpoint.at[subids[gj],"n_overnight_anchors"],meta.at[subids[gj],"available_streaming_duration_hours_mean"]],float)
    pool=np.array([[D0[ai,q],meta.at[subids[q],"valid_observation_count"],endpoint.at[subids[q],"n_overnight_anchors"],meta.at[subids[q],"available_streaming_duration_hours_mean"]] for q in cand],float)
    scale=np.nanstd(pool,axis=0); scale[~np.isfinite(scale)|(scale==0)]=1
    q=cand[int(np.nansum(((pool-target)/scale)**2,axis=1).argmin())]; chosen.append(q); cand.remove(q)
   counts.append(dict(cohort=cohort_name,canonical_stratum=subtype,hour=hour,anchor_id=a,k=k,retained_n=len(r),lost_n=len(l),gained_n=len(g),matched_n=len(chosen),metric=metric,k_mode=k_mode,cluster_matched=cluster_match))
   classes={"Retained":r,"Lost":l,"Gained":g,"Matched":chosen}
   for cls,js in classes.items():
    for j in js:
     b=subids[j]; sim=pair_sims(a,b,st,dy,features,refs[subtype])
     rows.append(dict(cohort=cohort_name,canonical_stratum=subtype,hour=hour,anchor_id=a,partner_id=b,anchor_hash=hpid(a),partner_hash=hpid(b),transition_class=cls,k=k,metric=metric,k_mode=k_mode,cluster_matched=cluster_match,same_site=st.at[a,"s_"+posthoc.SITE]==st.at[b,"s_"+posthoc.SITE],same_cluster=labels[a]==labels[b],**sim))
 return pd.DataFrame(rows),pd.DataFrame(counts)


def aggregate_pairs(pairs):
 simcols=[c for c in pairs if c.startswith(("static__","dynamic__","domain__"))]
 agg=pairs.groupby(["cohort","canonical_stratum","hour","anchor_id","transition_class"],as_index=False)[simcols].mean()
 return agg

def descriptive(agg):
 summaries=[]; tests=[]
 for keys,g in agg.groupby(["cohort","canonical_stratum","hour"]):
  cohort,subtype,hour=keys
  for cls,x in g.groupby("transition_class"):
   for c in [q for q in g if q.startswith("domain__")]:
    vals=x[c].dropna().to_numpy(float); rng=np.random.default_rng(SEED+hour+len(c)+len(cls))
    boots=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(B)] if len(vals) else [np.nan]
    summaries.append(dict(cohort=cohort,canonical_stratum=subtype,hour=hour,transition_class=cls,feature=c,n_anchors=len(vals),mean=np.nanmean(vals),ci_low=np.nanpercentile(boots,2.5),ci_high=np.nanpercentile(boots,97.5)))
  for a,b in [("Retained","Lost"),("Gained","Matched"),("Retained","Gained")]:
   for c in [q for q in g if q.startswith(("static__","dynamic__","domain__"))]:
    d,lo,hi,p,n=bootstrap_diff(g,c,a,b,SEED+hour+len(c)+len(a))
    tests.append(dict(cohort=cohort,canonical_stratum=subtype,hour=hour,comparison=f"{a}_vs_{b}",feature=c,domain=c.split("__")[0] if c.startswith("domain__") else c.split("__")[0],mean_difference=d,ci_low=lo,ci_high=hi,p_value=p,n_paired_anchors=n))
 out=pd.DataFrame(tests)
 if len(out):
  out["fdr_q"]=out.groupby(["cohort","canonical_stratum","hour","comparison","domain"],dropna=False)["p_value"].transform(lambda x:bh(x))
 return pd.DataFrame(summaries),out

def model_matrix(pairs,task):
 if task=="A": d=pairs[pairs.transition_class.isin(["Retained","Lost"])].copy(); d["y"]=(d.transition_class=="Retained").astype(int)
 else: d=pairs[pairs.transition_class.isin(["Gained","Matched"])].copy(); d["y"]=(d.transition_class=="Gained").astype(int)
 return d

def oof_models(d,feature_sets,seed,nonlinear=False):
 groups=d.anchor_id.astype(str).to_numpy(); y=d.y.to_numpy(int); ng=len(np.unique(groups)); folds=min(5,ng)
 if folds<3 or min(np.bincount(y))<3: return {},[],{}
 splitter=GroupKFold(folds); result={}; coefficients=[]; importances={}
 for name,cols in feature_sets.items():
  if not cols: continue
  pred=np.full(len(d),np.nan); domain_pred={dom:np.full(len(d),np.nan) for dom in DOMAIN_ORDER+["lipids_bp","site_sex","static_gower"]}
  foldco=[]
  for fold,(tr,te) in enumerate(splitter.split(d,y,groups)):
   if len(np.unique(y[tr]))<2: continue
   if nonlinear:
    model=Pipeline([("imp",SimpleImputer(strategy="median")),("clf",HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=seed+fold))])
   else:
    model=Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=seed+fold))])
   model.fit(d.iloc[tr][cols],y[tr]); pred[te]=model.predict_proba(d.iloc[te][cols])[:,1]
   if not nonlinear:
    foldco.append(model.named_steps["clf"].coef_[0])
    for c,v in zip(cols,model.named_steps["clf"].coef_[0]): coefficients.append(dict(model=name,fold=fold,feature=c,coefficient=v))
   base=roc_auc_score(y[te],pred[te]) if len(np.unique(y[te]))>1 else np.nan
   for dom in domain_pred:
    dynamic_members=set(DYNAMIC_DOMAINS.get(dom,[]))
    static_members=set(STATIC_DOMAINS.get(dom,[]))
    dc=[c for c in cols if
        (c.startswith("dynamic__") and c.replace("dynamic__","",1) in dynamic_members) or
        (c.startswith("static__") and c.replace("static__","",1) in static_members) or
        (dom=="static_gower" and c=="static__gower")]
    if not dc: continue
    xp=d.iloc[te][cols].copy(); rng=np.random.default_rng(seed+1000+fold+len(dom)); xp[dc]=xp[dc].to_numpy()[rng.permutation(len(xp))]
    pp=model.predict_proba(xp)[:,1]; domain_pred[dom][te]=pp
  ok=np.isfinite(pred)
  if ok.sum() and len(np.unique(y[ok]))==2:
   result[name]=(pred,metrics(y[ok],pred[ok]),bootstrap_metric(y[ok],pred[ok],groups[ok],seed+len(name)))
   if name=="SD":
    base=metrics(y[ok],pred[ok])["auroc"]; importances[name]={}
    for dom,pp in domain_pred.items():
     z=ok&np.isfinite(pp)
     if z.sum() and len(np.unique(y[z]))==2: importances[name][dom]=base-roc_auc_score(y[z],pp[z])
 return result,coefficients,importances

def run_models(pairs,cohort,scenario):
 perf=[]; coef=[]; imp=[]; increments=[]; partition=[]
 for (subtype,hour),g in pairs.groupby(["canonical_stratum","hour"]):
  for task in ["A","B"]:
   d=model_matrix(g,task)
   static=[c for c in d if c.startswith("static__")]
   dynamic=[c for c in d if c.startswith("dynamic__")]
   sets={"S":static,"D":dynamic,"SD":static+dynamic}
   res,co,im=oof_models(d,sets,SEED+hour+(0 if task=="A" else 500))
   # Permuted-label null uses identical grouped folds and combined features.
   if static+dynamic and len(d):
    null=d.copy(); rng=np.random.default_rng(SEED+9000+hour); null["y"]=rng.permutation(null.y.to_numpy())
    nr,_,_=oof_models(null,{"N":static+dynamic},SEED+9000+hour)
    if "N" in nr:
     # Score null predictions against their permuted labels, its expected reference is chance.
     res["N"]=nr["N"]
   for model,(pred,m,ci) in res.items():
    row=dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,model=model,n_pairs=len(d),n_anchors=d.anchor_id.nunique(),positive_fraction=d.y.mean())
    for q,v in m.items(): row[q]=v; row[q+"_ci_low"]=ci[q][0]; row[q+"_ci_high"]=ci[q][1]
    perf.append(row)
   for x in co: coef.append(dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,**x))
   for model,vals in im.items():
    for dom,val in vals.items(): imp.append(dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,model=model,domain=dom,auroc_drop=val))
   if all(q in res for q in ["S","D","SD"]):
    y=d.y.to_numpy(); groups=d.anchor_id.astype(str).to_numpy(); ps=res["S"][0]; pdyn=res["D"][0]; pc=res["SD"][0]; ok=np.isfinite(ps)&np.isfinite(pdyn)&np.isfinite(pc)
    rng=np.random.default_rng(SEED+11000+hour); ug=np.unique(groups[ok]); boot=[]
    for _ in range(B):
     pick=rng.choice(ug,len(ug),replace=True); ix=np.concatenate([np.flatnonzero(groups[ok]==q) for q in pick]); yy=y[ok][ix]
     if len(np.unique(yy))<2: continue
     boot.append([roc_auc_score(yy,pc[ok][ix])-roc_auc_score(yy,ps[ok][ix]),roc_auc_score(yy,pc[ok][ix])-roc_auc_score(yy,pdyn[ok][ix])])
    boot=np.array(boot); ds=res["SD"][1]["auroc"]-res["S"][1]["auroc"]; dd=res["SD"][1]["auroc"]-res["D"][1]["auroc"]
    increments += [dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,increment="dynamic_added_to_static",estimate=ds,ci_low=np.percentile(boot[:,0],2.5),ci_high=np.percentile(boot[:,0],97.5)),dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,increment="static_added_to_dynamic",estimate=dd,ci_low=np.percentile(boot[:,1],2.5),ci_high=np.percentile(boot[:,1],97.5))]
    S=res["S"][1]["auroc"]-.5; D=res["D"][1]["auroc"]-.5; C=res["SD"][1]["auroc"]-.5
    partition.append(dict(cohort=cohort,scenario=scenario,canonical_stratum=subtype,hour=hour,task=task,static_only_skill=S,dynamic_only_skill=D,shared_contribution=S+D-C,additional_combined_over_best=C-max(S,D),combined_skill=C))
 return pd.DataFrame(perf),pd.DataFrame(coef),pd.DataFrame(imp),pd.DataFrame(increments),pd.DataFrame(partition)

def nonlinear_and_subset(pairs):
 rows=[]
 for (subtype,hour),g in pairs.groupby(["canonical_stratum","hour"]):
  for task in ["A","B"]:
   d=model_matrix(g,task); allc=[c for c in d if c.startswith(("static__","dynamic__"))]
   glucose=[c for c in d if c.startswith("dynamic__cgm_")]
   wearable=[c for c in d if c.startswith("dynamic__") and c not in glucose]
   for label,cols,nl in [("GBT_SD",allc,True),("glucose_only",glucose,False),("wearable_only",wearable,False)]:
    res,_,_=oof_models(d,{label:cols},SEED+13000+hour,nonlinear=nl)
    if label in res:
     _,m,ci=res[label]; rows.append(dict(canonical_stratum=subtype,hour=hour,task=task,analysis=label,auroc=m["auroc"],auroc_ci_low=ci["auroc"][0],auroc_ci_high=ci["auroc"][1],auprc=m["auprc"],log_loss=m["log_loss"],ece=m["ece"]))
 return pd.DataFrame(rows)


def make_figure(counts,summary,perf,coef):
 from matplotlib.lines import Line2D
 sns.set_theme(style="whitegrid",font_scale=.9)
 fig=plt.figure(figsize=(20,14),layout="constrained"); gs=fig.add_gridspec(2,2,width_ratios=[1,1.18],height_ratios=[1,1])

 # A: use a two-level x axis (hours, then subtype) so the subtype name is not
 # repeated on every bar.
 ax=fig.add_subplot(gs[0,0]); c=counts[counts.cohort=="test"].groupby(["canonical_stratum","hour"])[["retained_n","lost_n","gained_n"]].mean()
 c=c.reindex(pd.MultiIndex.from_product([SUBTYPES,HOURS],names=["canonical_stratum","hour"])).reset_index()
 c["total"]=c[["retained_n","lost_n","gained_n"]].sum(axis=1)
 xpos=np.arange(len(c)); bottom=np.zeros(len(c))
 for col,label,color in [("retained_n","Retained","#13294B"),("lost_n","Lost","#A5A5A5"),("gained_n","Gained","#008C95")]:
  v=c[col]/c.total; ax.bar(xpos,v,bottom=bottom,label=label,color=color,width=.82); bottom+=v
 ax.set_xticks(xpos); ax.set_xticklabels([f"{h} h" for h in c.hour],rotation=0)
 for j,subtype in enumerate(SUBTYPES):
  center=j*len(HOURS)+(len(HOURS)-1)/2
  ax.text(center,-.12,SUBLABEL[subtype],ha="center",va="top",fontsize=9,fontweight="semibold",transform=ax.get_xaxis_transform())
  if j: ax.axvline(j*len(HOURS)-.5,color="#D0D0D0",lw=1)
 ax.set_ylim(0,1.08); ax.set_ylabel("Mean neighborhood share")
 ax.set_title("A  Neighborhoods are rapidly rewired during streaming",loc="left",fontweight="bold",y=1.13)
 ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.08))

 # B: domains are rows and subtype/transition combinations are columns. This
 # avoids 36 overlapping row labels. Build arrays positionally: assigning a
 # single-index block back into the old MultiIndex was aligning to all-NaN.
 ax=fig.add_subplot(gs[0,1]); z=summary[(summary.cohort=="test")&(summary.hour==48)&summary.feature.isin(["domain__"+d for d in DOMAIN_ORDER])].copy()
 m=z.pivot_table(index=["canonical_stratum","feature"],columns="transition_class",values="mean")
 for col in ["Retained","Lost","Gained","Matched"]:
  if col not in m: m[col]=np.nan
 enrich=m[["Retained","Lost","Gained"]].subtract(m["Matched"],axis=0)
 domain_labels={"clinical_factors":"Clinical factors","medication":"Medication","cgm_level":"CGM level","cgm_variability":"CGM variability","cgm_dynamics":"CGM dynamics","hr_respiration":"HR & respiration","activity":"Activity","sleep":"Sleep","stress_other_wearables":"Other wearables"}
 subtype_short={"healthy":"Healthy","pre_diabetes":"Pre-DM","t2d_oral_non_insulin":"T2D oral","insulin_dependent":"Insulin*"}
 features=["domain__"+d for d in DOMAIN_ORDER]; classes=["Retained","Lost","Gained"]; blocks=[]
 for subtype in SUBTYPES:
  block=enrich.xs(subtype).reindex(features)[classes].to_numpy(float)
  mu=np.nanmean(block); sd=np.nanstd(block); blocks.append((block-mu)/(sd if sd>0 else 1))
 heat=np.concatenate(blocks,axis=1)
 heat=pd.DataFrame(heat,index=[domain_labels[d] for d in DOMAIN_ORDER])
 vmax=float(np.nanpercentile(np.abs(heat.to_numpy()),98)); vmax=max(vmax,.25)
 sns.heatmap(heat,cmap="vlag",center=0,vmin=-vmax,vmax=vmax,ax=ax,annot=True,fmt=".1f",annot_kws={"fontsize":6.5},linewidths=.5,linecolor="white",cbar_kws={"label":"Within-subtype z-score\n(enrichment vs matched)","shrink":.82})
 ax.set_xticklabels([{"Retained":"Ret","Lost":"Lost","Gained":"Gain"}[c] for _ in SUBTYPES for c in classes],rotation=0,fontsize=8)
 for j,subtype in enumerate(SUBTYPES):
  ax.text(j*3+1.5,1.015,subtype_short[subtype],ha="center",va="bottom",fontsize=9,fontweight="semibold",transform=ax.get_xaxis_transform())
  if j: ax.axvline(j*3,color="white",lw=2.5)
 ax.tick_params(axis="y",labelrotation=0,labelsize=8.5); ax.set_xlabel(""); ax.set_ylabel("")
 ax.set_title("B  Information enrichment by transition at 48 h",loc="left",fontweight="bold",y=1.10)

 ax=fig.add_subplot(gs[1,0]); p=perf[(perf.cohort=="test")&(perf.scenario=="primary")&(~perf.canonical_stratum.eq("insulin_dependent"))]
 p=p.groupby(["hour","task","model"],as_index=False).auroc.mean()
 for model,color in COLORS.items():
  for task,ls,marker in [("A","-","o"),("B","--","s")]:
   q=p[(p.model==model)&(p.task==task)]
   if len(q): ax.plot(q.hour,q.auroc,marker=marker,color=color,ls=ls,lw=1.8,ms=5)
 ax.axhline(.5,color="black",lw=.8,alpha=.8); ax.set_xticks(HOURS); ax.set_ylim(.40,.82); ax.set_ylabel("Mean held-out AUROC"); ax.set_xlabel("Elapsed hours")
 ax.set_title("C  Dynamic information dominates held-out prediction",loc="left",fontweight="bold",y=1.13)
 model_handles=[Line2D([0],[0],color=COLORS[m],lw=2,label={"S":"Static","D":"Dynamic","SD":"Combined","N":"Null"}[m]) for m in ["S","D","SD","N"]]
 task_handles=[Line2D([0],[0],color="#333333",ls="-",marker="o",label="Retention"),Line2D([0],[0],color="#333333",ls="--",marker="s",label="Gain")]
 leg1=ax.legend(handles=model_handles,ncol=4,frameon=False,fontsize=8,loc="upper left",bbox_to_anchor=(0,1.08),borderaxespad=0); ax.add_artist(leg1)
 ax.legend(handles=task_handles,ncol=2,frameon=False,fontsize=8,loc="upper right",bbox_to_anchor=(1,1.08),borderaxespad=0)

 ax=fig.add_subplot(gs[1,1]); q=coef[(coef.cohort=="test")&(coef.scenario=="primary")&(coef.hour==48)&(coef.model=="SD")]
 q=q.groupby(["task","feature"],as_index=False).coefficient.mean(); q["abs"]=q.coefficient.abs(); q=q.sort_values(["task","abs"],ascending=[True,False]).groupby("task").head(5)
 q["source"]=np.where(q.feature.str.startswith("static__"),"Static","Dynamic")
 q["feature_clean"]=q.feature.str.replace("static__","",regex=False).str.replace("dynamic__","",regex=False).str.replace("_"," ",regex=False).str.replace("cgm","CGM",regex=False)
 q["label"]=q.task.map({"A":"Retention","B":"Gain"})+" · "+q.source+"\n"+q.feature_clean
 q=q.sort_values("coefficient"); ax.barh(q.label,q.coefficient,color=np.where(q.coefficient>=0,"#B21F35","#13294B")); ax.axvline(0,color="black",lw=.8)
 ax.tick_params(axis="y",labelsize=8); ax.set_xlabel("Mean standardized logistic coefficient")
 ax.set_title("D  Strongest model coefficients at 48 h",loc="left",fontweight="bold",y=1.05)
 fig.suptitle("Static and dynamic similarity explain latent-neighborhood transitions",fontsize=17,fontweight="bold")
 fig.savefig(ROOT/"figure_F1_transition_drivers.png",dpi=220,bbox_inches="tight"); fig.savefig(ROOT/"figure_F1_transition_drivers_thumbnail.png",dpi=80,bbox_inches="tight"); plt.close(fig)

def interpretation(perf,tests,increments,sensitivity):
 paras=[]
 primary=perf[(perf.cohort=="test")&(perf.scenario=="primary")]
 for subtype in SUBTYPES:
  for hour in HOURS:
   pa=primary[(primary.canonical_stratum==subtype)&(primary.hour==hour)&(primary.task=="A")].set_index("model")
   pb=primary[(primary.canonical_stratum==subtype)&(primary.hour==hour)&(primary.task=="B")].set_index("model")
   tt=tests[(tests.cohort=="test")&(tests.canonical_stratum==subtype)&(tests.hour==hour)]
   def contrast(comp,features):
    q=tt[(tt.comparison==comp)&tt.feature.isin(features)]
    return q.mean_difference.mean() if len(q) else np.nan
   stat=contrast("Retained_vs_Lost",["domain__clinical_factors","domain__medication","domain__static_gower"])
   dyn=contrast("Gained_vs_Matched",["domain__cgm_level","domain__cgm_variability","domain__cgm_dynamics","domain__hr_respiration","domain__activity","domain__sleep","domain__stress_other_wearables"])
   labels=[]
   if "S" in pa.index and "N" in pa.index and pa.at["S","auroc_ci_low"]>pa.at["N","auroc_ci_high"] and stat>0: labels.append("static preservation")
   if "D" in pb.index and "N" in pb.index and pb.at["D","auroc_ci_low"]>pb.at["N","auroc_ci_high"] and dyn>0: labels.append("dynamic convergence")
   inc=increments[(increments.canonical_stratum==subtype)&(increments.hour==hour)&(increments.cohort=="test")&(increments.scenario=="primary")]
   if len(inc)==4 and (inc.ci_low>0).all(): labels.append("mixed organization")
   if not labels: labels=["unexplained reorganization"]
   aval=", ".join([f"{m} AUROC {pa.at[m,'auroc']:.2f} ({pa.at[m,'auroc_ci_low']:.2f}–{pa.at[m,'auroc_ci_high']:.2f})" for m in ["S","D","SD"] if m in pa.index])
   bval=", ".join([f"{m} AUROC {pb.at[m,'auroc']:.2f} ({pb.at[m,'auroc_ci_low']:.2f}–{pb.at[m,'auroc_ci_high']:.2f})" for m in ["S","D","SD"] if m in pb.index])
   robust=sensitivity[(sensitivity.canonical_stratum==subtype)&(sensitivity.hour==hour)&(sensitivity.task=="B")]
   robust_text="directionally stable across matching/distance/k checks" if len(robust) and ((robust.auroc-.5)>0).mean()>=.75 else "not uniformly stable across matching/distance/k checks"
   paras.append(f"**{SUBLABEL[subtype]}, {hour} h — {', '.join(labels).capitalize()}.** Retained-versus-lost models gave {aval}; the anchor-aggregated static contrast was {stat:.3f}. Gained-versus-matched models gave {bval}; the mean dynamic-domain contrast was {dyn:.3f}. Thus lost neighbors were "+("more clinically alike than their dynamic evolution would suggest" if stat>0 and dyn>0 else "not consistently characterized as clinically similar but dynamically divergent")+f". The conclusion was {robust_text}. Dynamic importance over time is assessed from the full time series in the tables rather than presumed from this single timepoint.")
 return paras

def main():
 global B
 started=time.time(); ROOT.mkdir(parents=True,exist_ok=True)
 dynamic=pd.read_parquet(ROOT/"participant_dynamic_features.parquet"); dynamic.participant_id=dynamic.participant_id.astype(str)
 static,source=posthoc.load_frame(); static.participant_id=static.participant_id.astype(str); static,manifest=recover_labels(static)
 h0ids=set(pd.read_parquet(STUDY1_ROOT/"step2/h0_matrix.parquet",columns=["participant_id"]).participant_id.astype(str))
 static=static[static.participant_id.isin(h0ids)].copy(); refs=fit_reference(static,dynamic)
 frozen=pd.read_csv(STUDY2_ROOT/"phase3_ht_preservation/participant_retention.csv",dtype={"participant_id":str})
 test=static[static.participant_id.isin(frozen.participant_id)].copy()
 chk=test[["participant_id","canonical_stratum","display_cluster"]].merge(frozen[["participant_id","canonical_stratum","display_cluster"]],on="participant_id",suffixes=("_new","_frozen"))
 if len(chk)!=len(frozen) or not ((chk.canonical_stratum_new==chk.canonical_stratum_frozen)&(chk.display_cluster_new==chk.display_cluster_frozen)).all(): raise RuntimeError("Frozen primary cohort/label mismatch")
 full=static[static.participant_id.isin(dynamic.participant_id)].copy()
 allpairs=[]; allcounts=[]; sens_perf=[]; sens_desc=[]
 scenarios=[("primary","cosine","primary",False),("cluster_matched","cosine","primary",True),("k_small","cosine","small",False),("k_large","cosine","large",False),("euclidean","euclidean","primary",False)]
 primary_pairs=[]
 for hour in HOURS:
  for name,metric,km,cm in scenarios:
   B=25  # sensitivity screening; primary 1000-bootstrap inference is run below
   p,c=build_pairs("test",test,static,dynamic,refs,hour,metric,km,cm)
   if name=="primary": primary_pairs.append(p); allcounts.append(c)
   a=aggregate_pairs(p); _,dt=descriptive(a); dt["scenario"]=name; sens_desc.append(dt[dt.feature.str.startswith("domain__")])
   if name!="primary":
    pf,_,_,_,_=run_models(p,"test",name); sens_perf.append(pf)
 B=25
 # Site-excluded and within-cluster sensitivity reuse primary pairs.
 primary=pd.concat(primary_pairs,ignore_index=True)
 for name,p in [("exclude_site_matches",primary[~primary.same_site]),("within_cluster",primary[primary.same_cluster])]:
  a=aggregate_pairs(p); _,dt=descriptive(a); dt["scenario"]=name; sens_desc.append(dt[dt.feature.str.startswith("domain__")])
  pf,_,_,_,_=run_models(p,"test",name); sens_perf.append(pf)
 # Full cohort sensitivity.
 fullpairs=[]
 for hour in HOURS:
  p,c=build_pairs("full",full,static,dynamic,refs,hour); fullpairs.append(p); allcounts.append(c)
 fullpairs=pd.concat(fullpairs,ignore_index=True)
 aggtest=aggregate_pairs(primary); aggfull=aggregate_pairs(fullpairs)
 B=1000
 summary,tests=descriptive(pd.concat([aggtest,aggfull],ignore_index=True))
 B=1000
 ptest,coef,imp,inc,part=run_models(primary,"test","primary")
 B=200  # full-cohort sensitivity intervals; primary inference remains 1000
 pfull,c2,i2,n2,v2=run_models(fullpairs,"full","primary")
 perf=pd.concat([ptest,pfull,*sens_perf],ignore_index=True).drop_duplicates()
 coef=pd.concat([coef,c2],ignore_index=True); imp=pd.concat([imp,i2],ignore_index=True); inc=pd.concat([inc,n2],ignore_index=True); part=pd.concat([part,v2],ignore_index=True)
 B=200
 nonlinear=nonlinear_and_subset(primary)
 B=1000
 sensitivity=perf[perf.scenario!="primary"].copy(); sensitivity=pd.concat([sensitivity,nonlinear.rename(columns={"analysis":"scenario"}).assign(cohort="test",model=lambda x:x.scenario)],ignore_index=True,sort=False)
 # Pooled summary is model-level only: raw pairs remain subtype-separated.
 pooled=perf[(perf.cohort=="test")&(perf.scenario=="primary")&~perf.canonical_stratum.eq("insulin_dependent")].groupby(["hour","task","model"],as_index=False)[["auroc","auprc","log_loss","ece"]].mean(); pooled["analysis"]="pooled model-level mean, insulin-dependent excluded"
 counts=pd.concat(allcounts,ignore_index=True)
 # Privacy-safe internal pair export: derived similarities and hashes only.
 safe=pd.concat([primary.assign(scenario="primary"),fullpairs.assign(scenario="full_cohort")],ignore_index=True).drop(columns=["anchor_id","partner_id"])
 safe.to_parquet(ROOT/"privacy_safe_pair_similarities.parquet",index=False)
 aggout=pd.concat([aggtest,aggfull],ignore_index=True); aggout["anchor_id_hash"]=aggout.anchor_id.map(hpid); aggout=aggout.drop(columns="anchor_id")
 outputs={
  "participant_transition_counts.csv":counts.assign(anchor_id_hash=counts.anchor_id.map(hpid)).drop(columns="anchor_id"),
  "participant_aggregated_similarity.csv":aggout,
  "descriptive_similarity_summary.csv":summary,
  "feature_comparison_fdr.csv":tests,
  "predictive_performance.csv":perf,
  "standardized_logistic_coefficients.csv":coef,
  "grouped_domain_permutation_importance.csv":imp,
  "incremental_performance.csv":inc,
  "information_partitioning.csv":part,
  "sensitivity_analysis_summary.csv":sensitivity,
  "sensitivity_descriptive_comparisons.csv":pd.concat(sens_desc,ignore_index=True),
  "pooled_summary_excluding_insulin.csv":pooled,
 }
 for name,d in outputs.items(): d.to_csv(ROOT/name,index=False)
 make_figure(counts,summary,perf,coef)
 paras=interpretation(perf,tests,inc,sensitivity)
 report=["# Static and dynamic drivers of latent-neighborhood transitions","",f"Generated {now()}. Primary inference uses the frozen test cohort (n={len(test)}); the full eligible cohort (n={len(full)}) is sensitivity-only. All neighborhoods and comparisons are within diagnostic subtype. The insulin-dependent subtype is exploratory.","","## Methods and guardrails","",f"Frozen h0 and h6/h12/h24/h48 states were used without retraining, reclustering, or label revision. Cosine neighborhoods used deterministic k=min(30,max(5,round(0.15n))). Dynamic summaries were cumulative from 0 through t within the exact Phase 4 clean segments, with no future rows. Continuous similarities are negative absolute train-standardized differences; binary/categorical similarities are exact matches; mixed static similarity is Gower. Matched non-neighbors were matched on h0 distance, valid observations, endpoint anchors, and available duration. Models were L2 logistic regressions with anchor-grouped five-fold out-of-fold prediction; intervals and descriptive comparisons use {B} participant bootstraps. Feature tests use BH FDR within subtype/hour/comparison/domain. Predictive importance is associational, not causal.","","Meal-proxy burden was not calculated because no observed meal annotations exist; wearable calories were retained as energy-expenditure data and were not relabeled as meals.","","## Results by subtype and time","",*paras,"","## Sensitivity and information partitioning","",f"All nine requested checks are represented: within frozen cluster, site-match exclusion, cluster-matched controls, smaller/larger k, Euclidean distance (alongside primary cosine), glucose-only, wearable-only, pooled model summaries excluding insulin-dependent participants, and full-cohort sensitivity. See the sensitivity, information-partitioning, and pooled tables. Shared and unique contributions are cross-validated AUROC skill differences; when correlated predictors share signal these arithmetic partitions are not uniquely causal or cleanly separable.","","## Privacy and stopping rule","",f"The internal pair file contains salted participant hashes, transition metadata, and derived similarities only. Reporting files are participant-aggregated. Analysis stopped after this report and Figure F1, as requested."]
 (ROOT/"NEIGHBOR_TRANSITION_DRIVERS_REPORT.md").write_text("\n".join(report)+"\n")
 invariants={"created_at":now(),"primary_test_n":len(test),"full_sensitivity_n":len(full),"test_ids_exactly_frozen":set(test.participant_id)==set(frozen.participant_id),"same_subtype_only":True,"raw_identifiers_in_pair_export":False,"future_dynamic_data_used":False,"labels_used_as_primary_predictors":False,"h0_or_ht_reclustered":False,"forecast_model_retrained":False,"bootstrap_n":B,"elapsed_seconds":time.time()-started}
 write_json(ROOT/"analysis_invariants.json",invariants)
 artifacts=[p for p in ROOT.iterdir() if p.is_file() and p.name!="artifact_hashes.json"]; write_json(ROOT/"artifact_hashes.json",{p.name:sha(p) for p in sorted(artifacts)})
 print(f"Complete: {ROOT}; primary n={len(test)}, full n={len(full)}, elapsed={time.time()-started:.1f}s")

if __name__=="__main__": main()
