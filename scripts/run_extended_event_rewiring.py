"""Phase 4: causal-onset observable events and associative latent rewiring."""
from __future__ import annotations
import json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import neighbor_transition_dynamic_extract as de  # noqa:E402
from ssmcgm.analysis.within_subtype_config import DATASET,STUDY2_ROOT  # noqa:E402
from ssmcgm.data.aireadi import AireadiSchema,prepare_aireadi_panel  # noqa:E402

EXT=STUDY2_ROOT/"extended_clinical_latent_dynamics_v1"; ARCH=EXT/"timestamped_state_archive_35072d"; P2=EXT/"02_circadian_matched_reorganization"; OUT=EXT/"04_event_locked_rewiring"; REPORTS=EXT/"reports"; CACHE=EXT/"cache"
PROFILES=EXT/"01_cluster_metabolic_profiles/participant_frozen_cluster_profiles.parquet"; DIM=35072; SEED=42; B=1000
EVENTS=["sleep_onset","wake_transition","activity_onset","glucose_rise","hr_surprise","stress_event"]
SUBTYPES=["healthy","pre_diabetes","t2d_oral_non_insulin","insulin_dependent"]
NAVY="#003366"; TEAL="#5BBABA"; CRIMSON="#BA2828"; GRAY="#888888"

def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p,x): p.write_text(json.dumps(x,indent=2,default=lambda z:float(z) if isinstance(z,np.floating) else int(z) if isinstance(z,np.integer) else str(z))+"\n")
def state_matrix(pid):
 arr=pq.ParquetFile(ARCH/"participants"/f"participant_id={pid}"/"states.parquet").read(columns=["state"])["state"].combine_chunks(); return np.asarray(arr.values).reshape(len(arr),DIM).astype(np.float32,copy=False)
def cosine(a,b):
 den=np.linalg.norm(a)*np.linalg.norm(b); return float(1-np.dot(a,b)/den) if den else np.nan
def cyc_hour(a,b): d=abs(int(a)-int(b)); return min(d,24-d)
def ece(y,p,bins=10):
 out=0.; edges=np.linspace(0,1,bins+1)
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&(p<(hi if hi<1 else 1.000001));
  if m.any(): out+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(out)
def metrics(y,p): return {"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p),"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),"ece":ece(y,p)}

def raw_panel(cohort):
 cache=CACHE/"event_state_context.parquet"; event_cache=CACHE/"causal_event_detections.parquet"
 if cache.exists() and event_cache.exists(): return pd.read_parquet(cache),pd.read_parquet(event_cache)
 available=set(pq.read_schema(DATASET).names); cols=[c for c in de.DYNAMIC_COLUMNS if c in available]
 raw=pd.read_parquet(DATASET,columns=cols); raw.participant_id=raw.participant_id.astype(str); raw=raw[raw.participant_id.isin(cohort)].copy(); panel=prepare_aireadi_panel(raw,AireadiSchema()); panel.participant_id=panel.participant_id.astype(str)
 splits=pd.read_csv('/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt6h_seed42/split_participants.csv',dtype={"participant_id":str}).set_index("participant_id").split
 # Train-fitted strictly trailing event thresholds.
 panel["g15"]=panel.groupby(["participant_id","segment_id"]).cgm_glucose_mean.diff(3)
 train=panel.participant_id.map(splits).eq("train"); positive=panel.loc[train&(panel.g15>0),"g15"].dropna(); glucose_threshold=max(15.,float(positive.quantile(.90)))
 events=[]
 for (pid,seg),g in panel.groupby(["participant_id","segment_id"],sort=True):
  g=g.sort_values("time_idx").copy(); sleep=g[["sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]].fillna(0).max(axis=1)>0; awake=g.sleep_stage_awake.fillna(0)>0; active=(g.activity_steps_per_min.fillna(0)>0)|(g.activity_intensity_score.fillna(0)>0)
  glucose=pd.to_numeric(g.cgm_glucose_mean,errors="coerce"); dif=glucose.diff(); causal_rise=(dif.gt(0).rolling(3,min_periods=3).sum()==3)&(glucose-glucose.shift(3)>=glucose_threshold)
  hr=pd.to_numeric(g.heart_rate_mean,errors="coerce"); past_mean=hr.shift(1).rolling(72,min_periods=12).mean(); past_sd=hr.shift(1).rolling(72,min_periods=12).std().replace(0,np.nan); hz=(hr-past_mean)/past_sd; hrflag=hz>2.5
  stress=pd.to_numeric(g.stress_level_mean,errors="coerce"); sq=stress.shift(1).rolling(72,min_periods=12).quantile(.90); stressflag=stress>sq
  flags={"sleep_onset":(sleep.rolling(3,min_periods=3).sum()==3)&(~sleep.shift(3).fillna(False)),"wake_transition":(awake.rolling(3,min_periods=3).sum()==3)&(~awake.shift(3).fillna(False)),"activity_onset":(active.rolling(3,min_periods=3).sum()==3)&(~active.shift(3).fillna(False)),"glucose_rise":causal_rise,"hr_surprise":(hrflag.rolling(3,min_periods=3).sum()==3)&(~hrflag.shift(3).fillna(False)),"stress_event":(stressflag.rolling(3,min_periods=3).sum()==3)&(~stressflag.shift(3).fillna(False))}
  for typ,flag in flags.items():
   last=-10**9
   for pos in np.flatnonzero(flag.fillna(False).to_numpy()):
    ti=int(g.iloc[pos].time_idx)
    if ti-last<72: continue
    last=ti; r=g.iloc[pos]; events.append({"participant_id":pid,"split":splits.get(pid),"segment_id":int(seg),"event_type":typ,"event_timestamp_local":r["_stream_timestamp"],"event_time_idx":ti,"event_elapsed_minutes":ti*5,"event_local_hour":pd.Timestamp(r["_stream_timestamp"]).hour})
 ev=pd.DataFrame(events)
 idx=pd.read_parquet(ARCH/"state_index.parquet"); idx.participant_id=idx.participant_id.astype(str); idx["state_row"]=idx.groupby("participant_id").cumcount()
 keep=["participant_id","segment_id","_stream_timestamp","cgm_glucose_mean","heart_rate_mean","activity_steps_per_min","activity_intensity_score","stress_level_mean","sleep_stage_awake","sleep_stage_light","sleep_stage_deep","sleep_stage_rem","heart_rate_count","stress_level_count"]
 context=idx.merge(panel[keep].rename(columns={"_stream_timestamp":"timestamp_local"}),on=["participant_id","segment_id","timestamp_local"],how="left",validate="one_to_one")
 context.to_parquet(cache,index=False); ev.to_parquet(event_cache,index=False)
 write_json(CACHE/"event_detection_manifest.json",{"created_at":now(),"future_values_used":False,"glucose_15min_train_90pct_threshold_mgdl":glucose_threshold,"sustained_detection":"Three current/past observations; detection time is used as onset surrogate.","washout_hours":6,"exercise_detector":"No independently validated exercise detector found; generic activity was not called exercise.","meal_events":"Unavailable and excluded.","insulin_events":"Unavailable and excluded."})
 return context,ev

def map_events_and_controls(context,events,exclusion_events=None):
 if exclusion_events is None: exclusion_events=events
 rows=[]; used=set()
 for _,e in events.iterrows():
  g=context[(context.participant_id==e.participant_id)&(context.segment_id==e.segment_id)].sort_values("elapsed_minutes")
  target=int(np.ceil(e.event_elapsed_minutes/30)*30); hit=g[g.elapsed_minutes==target]
  if hit.empty or target<150 or target>2640: continue
  er=hit.iloc[0]; allp=context[context.participant_id==e.participant_id].copy(); blocked=exclusion_events[(exclusion_events.participant_id==e.participant_id)&(exclusion_events.event_type==e.event_type)]
  cand=allp[(allp.elapsed_minutes>=150)&(allp.elapsed_minutes<=2640)&(allp.day_night==er.day_night)&(allp.apply(lambda r:cyc_hour(r.local_hour,er.local_hour)<=1,axis=1))].copy()
  if len(blocked):
   bad=np.zeros(len(cand),bool)
   for _,b in blocked.iterrows(): bad|=(cand.segment_id.eq(b.segment_id)&((cand.elapsed_minutes-b.event_elapsed_minutes).abs()<=240)).to_numpy()
   cand=cand.loc[~bad]
  cand=cand[~cand.apply(lambda r:(str(e.event_type),str(e.participant_id),int(r.segment_id),int(r.elapsed_minutes)) in used,axis=1)]
  if cand.empty: continue
  base=float(er.cgm_glucose_mean) if pd.notna(er.cgm_glucose_mean) else np.nan
  cand["cost"]=(cand.local_hour.apply(lambda x:cyc_hour(x,er.local_hour))+.02*(cand.elapsed_minutes-er.elapsed_minutes).abs()+.02*(cand.cgm_glucose_mean-base).abs().fillna(50))
  cr=cand.sort_values(["cost","segment_id","elapsed_minutes"]).iloc[0]; used.add((str(e.event_type),str(e.participant_id),int(cr.segment_id),int(cr.elapsed_minutes)))
  for condition,r in [("event",er),("control",cr)]: rows.append({**e.to_dict(),"condition":condition,"aligned_segment_id":int(r.segment_id),"aligned_elapsed_minutes":int(r.elapsed_minutes),"aligned_state_row":int(r.state_row),"aligned_local_hour":int(r.local_hour),"baseline_glucose":float(r.cgm_glucose_mean) if pd.notna(r.cgm_glucose_mean) else np.nan,"match_cost":0. if condition=="event" else float(cr.cost)})
 return pd.DataFrame(rows)

def event_outcomes(context,matches,profiles):
 outcome=[]; aligned=[]; pmap=profiles.set_index("participant_id"); kmap={s:min(30,max(5,round(.15*profiles[(profiles.split=="test")&(profiles.canonical_stratum==s)].participant_id.nunique()))) for s in SUBTYPES}
 for subtype in SUBTYPES:
  ms=matches[matches.participant_id.map(pmap.canonical_stratum).eq(subtype) & matches.split.eq("test")]
  candidates=sorted(profiles[(profiles.split=="test")&(profiles.canonical_stratum==subtype)&profiles.participant_id.isin(context.participant_id)].participant_id.astype(str).unique())
  pdata={pid:(context[context.participant_id==pid].sort_values("state_row").reset_index(drop=True),state_matrix(pid)) for pid in candidates}
  for _,m in ms.iterrows():
   idx,X=pdata[m.participant_id]; anchor=int(m.aligned_state_row); seg=int(m.aligned_segment_id); el=int(m.aligned_elapsed_minutes)
   win=idx[(idx.segment_id==seg)&(idx.elapsed_minutes.between(el-120,el+240))].copy(); win["relative_minutes"]=win.elapsed_minutes-el; win["condition"]=m.condition; win["event_type"]=m.event_type; win["event_id"]=f"{m.participant_id}:{m.event_type}:{m.event_timestamp_local}"; aligned.append(win[["participant_id","event_id","event_type","condition","relative_minutes","euclidean_velocity","cosine_velocity"]])
   pre=idx[(idx.segment_id==seg)&(idx.elapsed_minutes==el-30)]; post=idx[(idx.segment_id==seg)&(idx.elapsed_minutes==el+120)]
   if pre.empty or post.empty: continue
   ip=int(pre.iloc[0].state_row); ia=int(post.iloc[0].state_row); a0=X[ip]; a1=X[ia]
   pool0=[]; pool1=[]; poolids=[]
   for qpid in candidates:
    if qpid==m.participant_id: continue
    qidx,qX=pdata[qpid]; q0=qidx[(qidx.elapsed_minutes==el-30)&(qidx.clock_bin_2h==int(pre.iloc[0].clock_bin_2h))]; q1=qidx[(qidx.elapsed_minutes==el+120)&(qidx.clock_bin_2h==int(post.iloc[0].clock_bin_2h))]
    if q0.empty or q1.empty: continue
    pool0.append(qX[q0.state_row.astype(int)].mean(axis=0)); pool1.append(qX[q1.state_row.astype(int)].mean(axis=0)); poolids.append(qpid)
   if len(poolids)>=3:
    P0=np.stack(pool0); P1=np.stack(pool1); d0=cdist(a0[None],P0,metric="cosine")[0]; d1=cdist(a1[None],P1,metric="cosine")[0]; kk=min(kmap[subtype],len(poolids)); n0=set(np.argsort(d0)[:kk]); n1=set(np.argsort(d1)[:kk]); jac=len(n0&n1)/len(n0|n1); retained=len(n0&n1)/kk; gained=len(n1-n0)/kk; anchor_label=int(pmap.at[m.participant_id,"display_cluster"]); pool_labels=np.array([int(pmap.at[q,"display_cluster"]) for q in poolids]); purity=float(np.mean(pool_labels[list(n1)]==anchor_label)); same=pool_labels==anchor_label; centroid_distance=float(np.linalg.norm(a1-P1[same].mean(axis=0))/np.sqrt(DIM)) if same.any() else np.nan
   else: jac=retained=gained=purity=centroid_distance=np.nan
   outcome.append({"participant_id":m.participant_id,"event_type":m.event_type,"event_timestamp_local":m.event_timestamp_local,"condition":m.condition,"canonical_stratum":subtype,"pre_to_post_euclidean":float(np.linalg.norm(a1-a0)/np.sqrt(DIM)),"pre_to_post_cosine":cosine(a1,a0),"neighborhood_jaccard":jac,"retained_neighbor_fraction":retained,"lost_neighbor_fraction":1-retained if np.isfinite(retained) else np.nan,"gained_neighbor_fraction":gained,"fixed_label_purity":purity,"distance_to_clinical_cluster_centroid":centroid_distance,"distance_to_pre_event_state":float(np.linalg.norm(a1-a0)/np.sqrt(DIM))})
 return pd.DataFrame(outcome),pd.concat(aligned,ignore_index=True) if aligned else pd.DataFrame()

def summarize(outcomes,aligned):
 curves=[]
 for (typ,cond,rel),g in aligned.groupby(["event_type","condition","relative_minutes"]):
  x=g.groupby("participant_id").euclidean_velocity.mean().dropna().to_numpy(); rng=np.random.default_rng(SEED+len(typ)+int(rel)+10000); boots=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(B)]) if len(x) else np.array([np.nan]); curves.append({"event_type":typ,"condition":cond,"relative_minutes":rel,"estimate":np.nanmean(x),"ci_low":np.nanpercentile(boots,2.5),"ci_high":np.nanpercentile(boots,97.5),"participant_n":len(x)})
 curve=pd.DataFrame(curves)
 effects=[]
 for typ,g in outcomes.groupby("event_type"):
  wide=g.pivot_table(index=["participant_id","event_timestamp_local"],columns="condition",values=["pre_to_post_euclidean","neighborhood_jaccard"])
  for metric in ["pre_to_post_euclidean","neighborhood_jaccard"]:
   if (metric,"event") not in wide or (metric,"control") not in wide: continue
   d=(wide[(metric,"event")]-wide[(metric,"control")]).groupby(level=0).mean().dropna().to_numpy(); rng=np.random.default_rng(SEED+len(typ)+len(metric)); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(B)]) if len(d) else np.array([np.nan]); effects.append({"event_type":typ,"metric":metric,"event_minus_control":np.nanmean(d),"ci_low":np.nanpercentile(boots,2.5),"ci_high":np.nanpercentile(boots,97.5),"participant_n":len(d)})
 return curve,pd.DataFrame(effects)

def pair_models(context,events,profiles):
 edges=pd.read_parquet(P2/"circadian_transition_edges.parquet"); edges=edges[edges.scenario.isin(["model_train_2h","primary_test_2h"])].copy(); p=profiles.set_index("participant_id"); dyn=pd.read_parquet(STUDY2_ROOT/"neighbor_transition_drivers/participant_dynamic_features.parquet"); dyn.participant_id=dyn.participant_id.astype(str); dmap=dyn.set_index(["participant_id","hour"])
 static=["participants_age","bmi_baseline","hba1c_percent_baseline","c_peptide_ngml_baseline","tg_hdl_ratio","waist_to_hip_ratio_baseline"]
 rows=[]
 burden={}
 for pid,g in events.groupby("participant_id"):
  for typ in EVENTS:
   times=g.loc[g.event_type.eq(typ),"event_elapsed_minutes"].to_numpy(float)
   for hour in [6,12,24,48]: burden[(str(pid),typ,hour)]=int(np.sum((times>=hour*60-360)&(times<=hour*60)))
 for _,r in edges.iterrows():
  a,b=str(r.anchor_id),str(r.partner_id); row=r.to_dict();
  for f in static: row["sd_static_"+f]=-abs(float(p.at[a,f])-float(p.at[b,f])) if pd.notna(p.at[a,f]) and pd.notna(p.at[b,f]) else np.nan
  for f in ["cgm_mean","cgm_cv","cgm_masd","heart_rate_mean_summary","activity_intensity_mean","sleep_awake_proportion","stress_mean_summary"]:
   row["sd_dynamic_"+f]=-abs(float(dmap.at[(a,int(r.hour)),f])-float(dmap.at[(b,int(r.hour)),f])) if (a,int(r.hour)) in dmap.index and (b,int(r.hour)) in dmap.index else np.nan
  for typ in EVENTS:
   av=burden.get((a,typ,int(r.hour)),0); bv=burden.get((b,typ,int(r.hour)),0)
   row["event_both_recent_"+typ]=int(av>0 and bv>0); row["event_count_similarity_"+typ]=-abs(av-bv)
  rows.append(row)
 f=pd.DataFrame(rows); f.to_parquet(OUT/"event_augmented_transition_pairs.parquet",index=False)
 results=[]; coefs=[]; predictions=[]
 for task,classes in {"A_retained_vs_lost":("retained","lost"),"B_gained_vs_matched":("gained","matched")}.items():
  q=f[f.transition_class.isin(classes)].copy(); q["clock_bin"]=pd.to_numeric(q.clock_bin,errors="coerce"); q["y"]=(q.transition_class==classes[0]).astype(int); train=q[q.scenario=="model_train_2h"]; test=q[q.scenario=="primary_test_2h"]
  nuisance=["hour","clock_bin","h0_distance"]; sd=[c for c in q if c.startswith("sd_")]; ev=[c for c in q if c.startswith("event_")]
  model_preds={}
  for name,cols in {"N":nuisance,"SD":nuisance+sd,"SDE":nuisance+sd+ev}.items():
   model=Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=SEED))]); model.fit(train[cols],train.y); pred=model.predict_proba(test[cols])[:,1]; model_preds[name]=pred; met=metrics(test.y.to_numpy(),pred); results.append({"task":task,"model":name,"test_n":len(test),**met})
   if name=="SDE":
    for c,v in zip(cols,model.named_steps["clf"].coef_[0]): coefs.append({"task":task,"feature":c,"coefficient":v})
   predictions.extend({"task":task,"model":name,"anchor_id":a,"y":int(y),"probability":float(pr)} for a,y,pr in zip(test.anchor_id,test.y,pred))
  anchors=test.anchor_id.astype(str).unique(); rng=np.random.default_rng(SEED+len(task)); diffs=[]
  for _ in range(B):
   pick=rng.choice(anchors,len(anchors),replace=True); ix=np.concatenate([np.flatnonzero(test.anchor_id.astype(str).to_numpy()==a) for a in pick]);
   if len(np.unique(test.y.to_numpy()[ix]))>1: diffs.append(roc_auc_score(test.y.to_numpy()[ix],model_preds["SDE"][ix])-roc_auc_score(test.y.to_numpy()[ix],model_preds["SD"][ix]))
  results.append({"task":task,"model":"SDE_minus_SD","test_n":len(test),"auroc":float(np.mean(diffs)),"auroc_ci_low":float(np.percentile(diffs,2.5)),"auroc_ci_high":float(np.percentile(diffs,97.5))})
 return pd.DataFrame(results),pd.DataFrame(coefs),pd.DataFrame(predictions),f

def figures(curve,effects,perf,coef,pairs):
 sns.set_theme(style="whitegrid"); valid=curve.event_type.unique(); n=max(1,len(valid)); fig,axes=plt.subplots(int(np.ceil(n/2)),2,figsize=(14,3.8*np.ceil(n/2)),squeeze=False)
 for ax,typ in zip(axes.flat,valid):
  g=curve[curve.event_type==typ]
  for cond,color in [("event",CRIMSON),("control",NAVY)]:
   q=g[g.condition==cond]; ax.plot(q.relative_minutes/60,q.estimate,color=color,label=cond.title()); ax.fill_between(q.relative_minutes/60,q.ci_low,q.ci_high,color=color,alpha=.18)
  ax.axvline(0,color="#FF0000",lw=1); ax.set_title(typ.replace("_"," ").title(),fontweight="bold"); ax.set_xlabel("Hours relative to causal detection"); ax.set_ylabel("Stepwise update / sqrt(d)"); ax.legend(frameon=False)
 for ax in axes.flat[len(valid):]: ax.axis("off")
 fig.suptitle("Observable event detections and matched-control latent updates",fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4A_event_aligned_latent_updates.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4A_event_aligned_latent_updates.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure_4A_event_aligned_latent_updates_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 effects.to_csv(OUT/"event_context_transition_metrics.csv",index=False); pivot=effects.pivot(index="event_type",columns="metric",values="event_minus_control") if len(effects) else pd.DataFrame(); fig,axes=plt.subplots(1,2,figsize=(13,5)); sns.heatmap(pivot,annot=True,fmt=".3g",center=0,cmap="vlag",ax=axes[0]); axes[0].set_title("Event minus matched control",fontweight="bold"); inc=perf[perf.model=="SDE_minus_SD"]; axes[1].bar(np.arange(len(inc)),inc.auroc,color=TEAL,edgecolor="black"); axes[1].errorbar(np.arange(len(inc)),inc.auroc,yerr=[inc.auroc-inc.auroc_ci_low,inc.auroc_ci_high-inc.auroc],fmt="none",ecolor="black",capsize=3); axes[1].set_xticks(np.arange(len(inc)),inc.task,rotation=20,ha="right"); axes[1].set_ylabel("Test AUROC: SDE minus SD"); axes[1].set_title("Incremental event context",fontweight="bold"); fig.suptitle("Event context provides an additional test of latent-neighborhood rewiring",fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(14,9));
 if len(curve):
  q=curve.groupby(["relative_minutes","condition"]).estimate.mean().unstack(); q.plot(ax=axes[0,0],color=[NAVY,CRIMSON]); axes[0,0].axvline(0,color="#FF0000"); axes[0,0].set_title("A  Event-aligned update",loc="left",fontweight="bold")
 if len(effects): sns.barplot(data=effects,x="event_type",y="event_minus_control",hue="metric",ax=axes[0,1]); axes[0,1].tick_params(axis="x",rotation=25); axes[0,1].set_title("B  Event versus control",loc="left",fontweight="bold")
 base=perf[perf.model.isin(["SD","SDE"])]; sns.barplot(data=base,x="task",y="auroc",hue="model",ax=axes[1,0]); axes[1,0].tick_params(axis="x",rotation=20); axes[1,0].set_title("C  Held-out prediction",loc="left",fontweight="bold")
 ec=coef[coef.feature.str.startswith("event_")].copy(); ec=ec.reindex(ec.coefficient.abs().sort_values(ascending=False).index).head(12); sns.barplot(data=ec,y="feature",x="coefficient",hue="task",ax=axes[1,1]); axes[1,1].set_title("D  Event-feature coefficients",loc="left",fontweight="bold"); fig.suptitle("Measured event context and latent-state rewiring",fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4C_integrated_event_attribution.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4C_integrated_event_attribution.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure_4C_integrated_event_attribution_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)

def main():
 for p in [OUT,REPORTS,CACHE]: p.mkdir(parents=True,exist_ok=True)
 if not json.loads((EXT/"PHASE3_COMPLETE.json").read_text()).get("status")=="complete": raise SystemExit("STOP: Phase 3 incomplete")
 profiles=pd.read_parquet(PROFILES); profiles.participant_id=profiles.participant_id.astype(str); cohort=set(pd.read_parquet(ARCH/"state_index.parquet",columns=["participant_id"]).participant_id.astype(str))
 context,events=raw_panel(cohort)
 matched_path=OUT/"matched_event_control_windows.parquet"; matches=pd.read_parquet(matched_path) if matched_path.exists() else map_events_and_controls(context,events)
 if not matched_path.exists(): matches.to_parquet(matched_path,index=False)
 outcome_path=OUT/"event_locked_outcomes.parquet"; aligned_path=OUT/"event_aligned_latent_updates.csv"
 if outcome_path.exists() and aligned_path.exists(): outcomes=pd.read_parquet(outcome_path); aligned=pd.read_csv(aligned_path)
 else:
  outcomes,aligned=event_outcomes(context,matches,profiles); outcomes.to_parquet(outcome_path,index=False); aligned.to_csv(aligned_path,index=False)
 curve,effects=summarize(outcomes,aligned); curve.to_csv(OUT/"event_aligned_summary.csv",index=False); perf,coef,preds,pairs=pair_models(context,events,profiles); perf.to_csv(OUT/"predictive_model_performance.csv",index=False); coef.to_csv(OUT/"event_feature_coefficients.csv",index=False); preds.to_parquet(OUT/"heldout_predictions.parquet",index=False)
 figures(curve,effects,perf,coef,pairs)
 inc=perf[perf.model=="SDE_minus_SD"]; robust=bool(len(inc) and (inc.auroc_ci_low>0).any()); category="Incremental event information" if robust else "Continuous dynamics sufficient"
 report=["# Phase 4: observable-event drivers of latent rewiring","",f"Interpretation category: **{category}**.","","All onsets were detected causally from current and trailing observations. The detection timestamp, rather than an inferred biological onset, anchors each window. Events were matched within participant on clock time (±1 hour), day/night context, glucose, and recording position. Results are associative.","","No independently validated exercise detector was found, so activity episodes were labeled activity onsets, not exercise. No meal or insulin event was used.","",f"Valid matched event/control rows: {len(matches):,}; outcome rows: {len(outcomes):,}. Event-augmented test AUROC improvements and participant-bootstrap intervals are in `predictive_model_performance.csv`. Insulin-dependent findings are exploratory."]
 (REPORTS/"phase4_event_locked_rewiring.md").write_text("\n".join(report)+"\n")
 qa={"phase":"phase4","status":"complete","created_at":now(),"category":category,"future_values_used_for_onset":False,"timed_insulin_used":False,"meal_event_used":False,"exercise_claimed":False,"within_participant_controls":True,"clock_time_tolerance_hours":1,"bootstrap_n":B,"matched_rows":len(matches),"outcome_rows":len(outcomes)}; write_json(OUT/"figure_4A_metadata.json",qa); write_json(OUT/"figure_4B_metadata.json",qa); write_json(OUT/"figure_4C_metadata.json",qa); write_json(EXT/"PHASE4_COMPLETE.json",qa)
 print(json.dumps(qa,indent=2))
if __name__=="__main__": main()
