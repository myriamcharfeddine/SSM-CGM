"""Phases 2 and 3 after validated 35,072-D timestamped-state export."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import within_subtype_phase1 as phase1  # noqa:E402
from ssmcgm.analysis.within_subtype_config import STUDY1_ROOT,STUDY2_ROOT,DATASET  # noqa:E402

EXT=STUDY2_ROOT/"extended_clinical_latent_dynamics_v1"
ARCH=EXT/"timestamped_state_archive_35072d"
P2=EXT/"02_circadian_matched_reorganization"; P3=EXT/"03_latent_update_dynamics"
REPORTS=EXT/"reports"; CACHE=EXT/"cache"; LOGS=EXT/"logs"
PROFILES=EXT/"01_cluster_metabolic_profiles/participant_frozen_cluster_profiles.parquet"
H0=STUDY1_ROOT/"step2/h0_matrix.parquet"
DIM=35072; HOURS=[6,12,24,48]; SEED=42; B=1000; BS=250
SUBTYPES=["healthy","pre_diabetes","t2d_oral_non_insulin","insulin_dependent"]
SLABEL={"healthy":"Healthy","pre_diabetes":"Pre-diabetes","t2d_oral_non_insulin":"T2D oral non-insulin","insulin_dependent":"Insulin-dependent*"}
COLORS={"healthy":"#003366","pre_diabetes":"#2F7F7F","t2d_oral_non_insulin":"#BA2828","insulin_dependent":"#7A8798"}
METRICS=["clinical_to_h0_jaccard","clinical_to_ht_jaccard","h0_to_ht_jaccard",
         "clinical_purity","h0_purity","ht_purity","neighbor_retention","gained_neighbor_fraction"]

def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p,x): p.write_text(json.dumps(x,indent=2,default=lambda z:float(z) if isinstance(z,np.floating) else int(z) if isinstance(z,np.integer) else str(z))+"\n")
def style():
 sns.set_theme(style="whitegrid",context="notebook")
 plt.rcParams.update({"font.family":"sans-serif","axes.edgecolor":"black","axes.spines.top":True,"axes.spines.right":True,"grid.color":"#D9D9D9","figure.facecolor":"white","axes.facecolor":"white","pdf.fonttype":42})
def state_matrix(pid):
 p=ARCH/"participants"/f"participant_id={pid}"/"states.parquet"
 arr=pq.ParquetFile(p).read(columns=["state"])["state"].combine_chunks()
 return np.asarray(arr.values).reshape(len(arr),DIM).astype(np.float32,copy=False)
def jaccard(a,b):
 a=set(a); b=set(b); return len(a&b)/len(a|b) if a|b else np.nan
def nearest(X,k,metric):
 D=cdist(X,X,metric=metric); np.fill_diagonal(D,np.inf); return D,np.argsort(D,axis=1)[:,:k]
def bootstrap_mean(g,value,nboot,seed):
 x=g.groupby("participant_id")[value].mean().dropna().to_numpy(float)
 if not len(x): return np.nan,np.nan,np.nan,0
 rng=np.random.default_rng(seed); boots=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(nboot)])
 return float(x.mean()),float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5)),len(x)

def build_timepoint_cache():
 paths={h:CACHE/f"clock_states_hour{h:02d}.npz" for h in HOURS}
 if all(p.exists() for p in paths.values()): return paths
 buckets={h:{"state":[],"participant_id":[],"split":[],"segment_id":[],"clock2":[],"clock1":[],"day_night":[],"sleep_wake":[]} for h in HOURS}
 pids=sorted(p.name.split("=",1)[1] for p in (ARCH/"participants").glob("participant_id=*"))
 sleep_cols=["participant_id","timestamp_local","sleep_stage_awake","sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]
 sleep=pd.read_parquet(DATASET,columns=sleep_cols); sleep.participant_id=sleep.participant_id.astype(str); sleep.timestamp_local=pd.to_datetime(sleep.timestamp_local)
 asleep=sleep[["sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]].fillna(0).max(axis=1).gt(0); awake=sleep.sleep_stage_awake.fillna(0).gt(0)
 sleep["sleep_wake"]=np.where(asleep,"sleep",np.where(awake,"wake","unknown"))
 sleep_map=sleep.drop_duplicates(["participant_id","timestamp_local"]).set_index(["participant_id","timestamp_local"])["sleep_wake"]
 started=time.time()
 for i,pid in enumerate(pids,1):
  idx=pd.read_parquet(ARCH/"participants"/f"participant_id={pid}"/"index.parquet")
  wanted=idx.elapsed_minutes.isin([h*60 for h in HOURS]).to_numpy(); X=state_matrix(pid)[wanted]; m=idx.loc[wanted].reset_index(drop=True)
  sw=sleep_map.reindex(pd.MultiIndex.from_frame(m[["participant_id","timestamp_local"]])).fillna("unknown").to_numpy(str)
  for h in HOURS:
   q=m.elapsed_minutes.eq(h*60).to_numpy()
   for _,r in m.loc[q].reset_index(drop=True).iterrows():
    k=int(np.flatnonzero(q)[_])
    b=buckets[h]; b["state"].append(X[k]); b["participant_id"].append(str(r.participant_id)); b["split"].append(str(r.split)); b["segment_id"].append(int(r.segment_id)); b["clock2"].append(int(r.clock_bin_2h)); b["clock1"].append(int(r.clock_bin_1h)); b["day_night"].append(str(r.day_night)); b["sleep_wake"].append(sw[k])
  if i%100==0: print(f"Timepoint cache {i}/{len(pids)} in {time.time()-started:.1f}s",flush=True)
 for h,p in paths.items():
  b=buckets[h]; np.savez(p,state=np.stack(b.pop("state")).astype(np.float32),**{k:np.asarray(v) for k,v in b.items()})
 return paths

def clinical_and_h0(profiles):
 h=pd.read_parquet(H0); h.participant_id=h.participant_id.astype(str); cols=[c for c in h if str(c).isdigit()]
 hmap={pid:i for i,pid in enumerate(h.participant_id)}; H=h[cols].to_numpy(np.float32)
 clinical={}
 manifest=json.loads((STUDY2_ROOT/"phase1_clinical_clustering/frozen_clustering_manifest.json").read_text())
 for subtype in SUBTYPES:
  sub=profiles[profiles.canonical_stratum==subtype].copy(); pipe=joblib.load(STUDY2_ROOT/manifest["clusters"][subtype]["preprocessing_pipeline_path"])
  X=phase1.apply_pipeline(sub,pipe["factors"],pipe["log_transformed"],pipe["imputer"],pipe["scaler"])
  clinical[subtype]={pid:X[i] for i,pid in enumerate(sub.participant_id.astype(str))}
 return clinical,H,hmap

def original_k(profiles):
 counts=profiles[profiles.split.eq("test")].groupby("canonical_stratum").participant_id.nunique()
 return {s:min(int(counts[s])-1,min(30,max(5,round(.15*int(counts[s]))))) for s in SUBTYPES}

def graph_block(ids,labels,C,H,T,k,metadata,edge_context):
 n=len(ids); ke=min(k,n-1)
 Dc,Nc=nearest(C,ke,"euclidean"); D0,N0=nearest(H,ke,"cosine"); Dt,Nt=nearest(T,ke,"cosine")
 rows=[]; edges=[]
 for i,pid in enumerate(ids):
  nc=Nc[i]; n0=N0[i]; nt=Nt[i]; lab=labels[i]
  retained=set(n0)&set(nt); lost=set(n0)-set(nt); gained=set(nt)-set(n0)
  row=dict(metadata,participant_id=pid,candidate_pool_n=n-1,knn_k=k,effective_k=ke,underpowered=(n-1<k+5),
   clinical_to_h0_jaccard=jaccard(nc,n0),clinical_to_ht_jaccard=jaccard(nc,nt),h0_to_ht_jaccard=jaccard(n0,nt),
   clinical_purity=float(np.mean(labels[nc]==lab)),h0_purity=float(np.mean(labels[n0]==lab)),ht_purity=float(np.mean(labels[nt]==lab)),
   neighbor_retention=len(retained)/ke,gained_neighbor_fraction=len(gained)/ke)
  rows.append(row)
  excluded=set(n0)|set(nt)|{i}; pool=[q for q in range(n) if q not in excluded]
  matched=[]
  for g in sorted(gained):
   if not pool: break
   q=min(pool,key=lambda z:abs(D0[i,z]-D0[i,g])); matched.append(q); pool.remove(q)
  for cls,js in [("retained",retained),("lost",lost),("gained",gained),("matched",matched)]:
   for j in js: edges.append(dict(edge_context,anchor_id=pid,partner_id=ids[j],transition_class=cls,anchor_cluster=int(lab),partner_cluster=int(labels[j]),h0_distance=float(D0[i,j]),ht_distance=float(Dt[i,j])))
 return rows,edges

def run_phase2(paths,profiles,clinical,H,hmap):
 marker=EXT/"PHASE2_COMPLETE.json"
 if marker.exists(): return
 kmap=original_k(profiles); profile_idx=profiles.set_index("participant_id")
 metrics=[]; edges=[]
 specs=[("primary_test_2h","test","clock2",B),("model_train_2h","train","clock2",BS),("sensitivity_test_1h","test","clock1",BS),("sensitivity_full_2h","all","clock2",BS)]
 for h in HOURS:
  z=np.load(paths[h],allow_pickle=False); S=z["state"]; pid=z["participant_id"].astype(str); split=z["split"].astype(str)
  for scenario,cohort,binkey,_ in specs:
   bins=z[binkey]
   for subtype in SUBTYPES:
    allowed=np.array([(p in profile_idx.index and profile_idx.at[p,"canonical_stratum"]==subtype and (cohort=="all" or split[i]==cohort)) for i,p in enumerate(pid)])
    for clockbin in sorted(set(bins[allowed].tolist())):
     take=np.flatnonzero(allowed&(bins==clockbin)); groups={}
     for q in take: groups.setdefault(pid[q],[]).append(q)
     ids=sorted(groups)
     if len(ids)<2: continue
     T=np.stack([S[groups[p]].mean(axis=0,dtype=np.float32) for p in ids]); C=np.stack([clinical[subtype][p] for p in ids]); Hx=np.stack([H[hmap[p]] for p in ids]); labels=np.array([profile_idx.at[p,"display_cluster"] for p in ids],int)
     meta={"scenario":scenario,"canonical_stratum":subtype,"hour":h,"clock_bin":str(int(clockbin)),"bin_width_hours":1 if binkey=="clock1" else 2}
     r,e=graph_block(ids,labels,C,Hx,T,kmap[subtype],meta,meta); metrics+=r; edges+=e
  # Day/night primary uses participant averages within stratum.
  for subtype in SUBTYPES:
   base=np.array([(p in profile_idx.index and profile_idx.at[p,"canonical_stratum"]==subtype and split[i]=="test") for i,p in enumerate(pid)])
   for dn in ["day","night"]:
    take=np.flatnonzero(base&(z["day_night"].astype(str)==dn)); groups={}
    for q in take: groups.setdefault(pid[q],[]).append(q)
    ids=sorted(groups)
    if len(ids)<2: continue
    T=np.stack([S[groups[p]].mean(axis=0,dtype=np.float32) for p in ids]); C=np.stack([clinical[subtype][p] for p in ids]); Hx=np.stack([H[hmap[p]] for p in ids]); labels=np.array([profile_idx.at[p,"display_cluster"] for p in ids],int)
    meta={"scenario":"primary_test_day_night","canonical_stratum":subtype,"hour":h,"clock_bin":dn,"bin_width_hours":None}
    r,e=graph_block(ids,labels,C,Hx,T,kmap[subtype],meta,meta); metrics+=r; edges+=e
  # Observed sleep/wake sensitivity is run only when at least 60% of target
  # states have an explicit sleep-stage classification.
  if np.mean(z["sleep_wake"].astype(str)!="unknown")>=.60:
   for subtype in SUBTYPES:
    base=np.array([(p in profile_idx.index and profile_idx.at[p,"canonical_stratum"]==subtype and split[i]=="test") for i,p in enumerate(pid)])
    for sw in ["sleep","wake"]:
     take=np.flatnonzero(base&(z["sleep_wake"].astype(str)==sw)); groups={}
     for q in take: groups.setdefault(pid[q],[]).append(q)
     ids=sorted(groups)
     if len(ids)<2: continue
     T=np.stack([S[groups[p]].mean(axis=0,dtype=np.float32) for p in ids]); C=np.stack([clinical[subtype][p] for p in ids]); Hx=np.stack([H[hmap[p]] for p in ids]); labels=np.array([profile_idx.at[p,"display_cluster"] for p in ids],int); meta={"scenario":"sensitivity_test_sleep_wake","canonical_stratum":subtype,"hour":h,"clock_bin":sw,"bin_width_hours":None}
     r,e=graph_block(ids,labels,C,Hx,T,kmap[subtype],meta,meta); metrics+=r; edges+=e
  print(f"Phase 2 graphs complete through hour {h}",flush=True)
 m=pd.DataFrame(metrics); e=pd.DataFrame(edges); m.to_parquet(P2/"circadian_participant_metrics.parquet",index=False); e.to_parquet(P2/"circadian_transition_edges.parquet",index=False)
 summaries=[]
 for (scenario,subtype),g in m.groupby(["scenario","canonical_stratum"]):
  nboot=B if scenario.startswith("primary") else BS
  for metric in METRICS:
   est,lo,hi,n=bootstrap_mean(g,metric,nboot,SEED+len(metric)+len(scenario)); summaries.append({"scenario":scenario,"canonical_stratum":subtype,"metric":metric,"estimate":est,"ci_low":lo,"ci_high":hi,"participant_n":n,"bootstrap_n":nboot})
 s=pd.DataFrame(summaries); s.to_csv(P2/"figure_2A_plotted_data.csv",index=False)
 day=[]
 q=m[m.scenario.eq("primary_test_day_night")]
 for (subtype,dn),g in q.groupby(["canonical_stratum","clock_bin"]):
  for metric in METRICS[:3]:
   est,lo,hi,n=bootstrap_mean(g,metric,B,SEED+len(metric)+len(dn)); day.append({"canonical_stratum":subtype,"day_night":dn,"metric":metric,"estimate":est,"ci_low":lo,"ci_high":hi,"participant_n":n})
 day=pd.DataFrame(day); day.to_csv(P2/"day_night_reorganization_metrics.csv",index=False)
 plot_phase2(s,day)
 primary=s[s.scenario.eq("primary_test_2h")]; old=pd.read_csv(STUDY2_ROOT/"tables/phase3_three_space_preservation.csv"); old_mean=old[(old.view=="test")&(old.comparison=="h0_to_ht")].knn_jaccard_mean.mean(); matched_mean=primary[primary.metric.eq("h0_to_ht_jaccard")].estimate.mean(); dn_mean=day[day.metric.eq("h0_to_ht_jaccard")].groupby("day_night").estimate.mean()
 if matched_mean<=old_mean+.10 and len(dn_mean)==2 and dn_mean.max()<=old_mean+.15: decision="circadian robust"
 elif matched_mean<.5: decision="partially circadian-dependent"
 else: decision="primarily circadian-confounded"
 report=["# Phase 2: circadian-matched neighborhood reorganization","",f"Decision: **{decision}**.","",
  "h0 is constructed from the frozen participant static profile through `encode_static` and `init_stream`. It includes static clinical and site/category fields but no clock time, calendar variable, or segment-start time. One common h0 was therefore used, and clinical, h0, and ht graphs always shared the identical same-bin candidate pool.","",
  f"Primary two-hour-bin mean h0-to-ht Jaccard across subtype estimates was {matched_mean:.3f}, versus {old_mean:.3f} in the frozen unmatched-clock analysis. Analyses with fewer than k+5 candidates were flagged rather than silently treated as fully powered. One-hour bins, sleep/wake, and the full cohort are saved as sensitivities. Insulin-dependent results remain exploratory."]
 (REPORTS/"phase2_circadian_matched_reorganization.md").write_text("\n".join(report)+"\n")
 qa={"phase":"phase2","status":"complete","created_at":now(),"representation_validation_passed":True,"candidate_pools_identical_across_spaces":True,"k_by_subtype":kmap,"latent_metric":"cosine","clinical_metric":"euclidean","decision":decision,"participant_bootstrap_n":B,"sensitivity_bootstrap_n":BS}
 write_json(P2/"figure_2A_metadata.json",qa); write_json(P2/"figure_2B_metadata.json",qa); write_json(marker,qa)

def plot_phase2(s,day):
 style(); primary=s[s.scenario.eq("primary_test_2h")]
 fig,axes=plt.subplots(1,2,figsize=(14,5.5))
 specs=[(["clinical_to_h0_jaccard","clinical_to_ht_jaccard","h0_to_ht_jaccard"],["Clinical to h0","Clinical to ht","h0 to ht"],"Neighborhood overlap"),(["clinical_purity","h0_purity","ht_purity"],["Clinical","h0","ht"],"Fixed-label neighbor purity")]
 for ax,(metrics,labels,title) in zip(axes,specs):
  width=.18; x=np.arange(len(SUBTYPES))
  for j,(metric,label) in enumerate(zip(metrics,labels)):
   g=primary[primary.metric.eq(metric)].set_index("canonical_stratum").reindex(SUBTYPES); pos=x+(j-1)*width
   ax.bar(pos,g.estimate,width,label=label,color=[COLORS[s] for s in SUBTYPES],alpha=.45+j*.2,edgecolor="black")
   ax.errorbar(pos,g.estimate,yerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],fmt="none",ecolor="black",capsize=3,lw=1)
  ax.set_xticks(x,[SLABEL[s] for s in SUBTYPES],rotation=18,ha="right"); ax.set_ylim(0,1); ax.set_ylabel(title); ax.set_title(title,fontweight="bold"); ax.legend(frameon=False,fontsize=9)
  for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("black")
 fig.suptitle("Clinical and initial-state neighborhoods after circadian matching",fontweight="bold",fontsize=15); fig.tight_layout(rect=[0,0,1,.95])
 fig.savefig(P2/"figure_2A_circadian_matched_reorganization.png",dpi=200,bbox_inches="tight"); fig.savefig(P2/"figure_2A_circadian_matched_reorganization.pdf",bbox_inches="tight"); fig.savefig(P2/"figure_2A_circadian_matched_reorganization_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
 for ax,metric,label in zip(axes,["clinical_to_h0_jaccard","clinical_to_ht_jaccard","h0_to_ht_jaccard"],["Clinical to h0","Clinical to ht","h0 to ht"]):
  g=day[day.metric.eq(metric)]; x=np.arange(len(SUBTYPES)); width=.34
  for j,dn in enumerate(["day","night"]):
   q=g[g.day_night.eq(dn)].set_index("canonical_stratum").reindex(SUBTYPES); pos=x+(j-.5)*width
   ax.bar(pos,q.estimate,width,label=dn.title(),color="#5BBABA" if dn=="day" else "#003366",edgecolor="black")
   ax.errorbar(pos,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt="none",ecolor="black",capsize=3)
  ax.set_xticks(x,[SLABEL[s] for s in SUBTYPES],rotation=22,ha="right"); ax.set_title(label,fontweight="bold"); ax.set_ylim(0,1); ax.legend(frameon=False)
  for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("black")
 axes[0].set_ylabel("Participant-level Jaccard"); fig.suptitle("Latent-neighborhood reorganization during day and night",fontweight="bold",fontsize=15); fig.tight_layout(rect=[0,0,1,.94])
 fig.savefig(P2/"figure_2B_day_night_reorganization.png",dpi=200,bbox_inches="tight"); fig.savefig(P2/"figure_2B_day_night_reorganization.pdf",bbox_inches="tight"); fig.savefig(P2/"figure_2B_day_night_reorganization_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)

def representative_figure(dyn,profiles):
 test=dyn[dyn.split.eq("test")].copy(); coverage=test.groupby("participant_id").agg(rows=("elapsed_minutes","size"),segments=("segment_id","nunique"),has_day=("day_night",lambda x:(x=="day").any()),has_night=("day_night",lambda x:(x=="night").any()),final=("euclidean_cumulative",lambda x:float(x.iloc[-1])))
 valid=coverage[(coverage.has_day)&(coverage.has_night)&(coverage.rows>=96)]
 subtype_counts=profiles[profiles.split.eq("test")].groupby("canonical_stratum").participant_id.nunique(); target_subtype=subtype_counts.idxmax(); eligible=valid.join(profiles.set_index("participant_id")[["canonical_stratum"]]); eligible=eligible[eligible.canonical_stratum.eq(target_subtype)]
 median=eligible.final.median(); pid=sorted(eligible.index,key=lambda p:(abs(eligible.at[p,"final"]-median),p))[0]
 idx=pd.read_parquet(ARCH/"participants"/f"participant_id={pid}"/"index.parquet"); X=state_matrix(pid); seg=idx.groupby("segment_id").size().idxmax(); take=idx.segment_id.eq(seg).to_numpy(); m=idx.loc[take].reset_index(drop=True); Xt=X[take]
 archived={p.name.split("=",1)[1] for p in (ARCH/"participants").glob("participant_id=*")}
 train_ids=sorted(set(profiles.loc[profiles.split.eq("train"),"participant_id"].astype(str))&archived)[:80]; samples=[]
 for tp in train_ids:
  q=state_matrix(tp); choose=np.linspace(0,len(q)-1,min(6,len(q)),dtype=int); samples.append(q[choose])
 pca=PCA(n_components=2,svd_solver="randomized",random_state=SEED).fit(np.vstack(samples)); Z=pca.transform(Xt)
 hf=pd.read_parquet(H0,filters=[("participant_id","=",pid)]); hc=[c for c in hf if str(c).isdigit()]; h0z=pca.transform(hf[hc].to_numpy(np.float32))[0]
 raw=pd.read_parquet(DATASET,columns=["participant_id","timestamp_local","cgm_glucose_mean","heart_rate_mean","activity_steps_per_min","sleep_stage_awake","sleep_stage_light","sleep_stage_deep","sleep_stage_rem"],filters=[("participant_id","=",pid)]); raw.participant_id=raw.participant_id.astype(str); raw.timestamp_local=pd.to_datetime(raw.timestamp_local); m.timestamp_local=pd.to_datetime(m.timestamp_local); plot=m.merge(raw,on=["participant_id","timestamp_local"],how="left")
 plot["pc1"]=Z[:,0]; plot["pc2"]=Z[:,1]; plot.to_csv(P3/"representative_participant_data.csv",index=False)
 style(); fig,axes=plt.subplots(2,2,figsize=(14,8)); ax=axes[0,0]; ax.plot(plot.elapsed_minutes/60,plot.cgm_glucose_mean,color="#003366",label="CGM"); ax.set_ylabel("CGM (mg/dL)"); ax2=ax.twinx(); ax2.plot(plot.elapsed_minutes/60,plot.heart_rate_mean,color="#BA2828",alpha=.55,label="Heart rate"); ax2.set_ylabel("Heart rate (bpm)"); ax.set_title("A  Observed physiology",loc="left",fontweight="bold")
 axes[0,1].plot(plot.elapsed_minutes/60,plot.euclidean_velocity,color="#2F7F7F"); axes[0,1].set_title("B  Stepwise latent update",loc="left",fontweight="bold"); axes[0,1].set_ylabel("Euclidean update / sqrt(d)")
 axes[1,0].plot(plot.elapsed_minutes/60,plot.euclidean_cumulative,color="#7A1F1F"); axes[1,0].set_title("C  Cumulative displacement",loc="left",fontweight="bold"); axes[1,0].set_ylabel("Distance from h0 / sqrt(d)")
 sc=axes[1,1].scatter(Z[:,0],Z[:,1],c=plot.elapsed_minutes/60,cmap="viridis",s=22); axes[1,1].plot([h0z[0],Z[0,0]],[h0z[1],Z[0,1]],color="#888888",lw=.7); axes[1,1].plot(Z[:,0],Z[:,1],color="#888888",lw=.7); axes[1,1].scatter(h0z[0],h0z[1],marker="*",s=180,color="#FF0000",edgecolor="black",label="h0"); axes[1,1].legend(frameon=False); axes[1,1].set_title("D  Train-fitted PCA visualization",loc="left",fontweight="bold"); fig.colorbar(sc,ax=axes[1,1],label="Elapsed hours")
 # Clock-based night shading and an independent sleep overlay.
 x=plot.elapsed_minutes.to_numpy()/60; night=plot.day_night.eq("night").to_numpy(); sleep=plot[["sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]].fillna(0).max(axis=1).gt(0).to_numpy()
 for target_ax in [axes[0,0],axes[0,1],axes[1,0]]:
  for i in np.flatnonzero(night): target_ax.axvspan(x[i]-.25,x[i]+.25,color="#003366",alpha=.055,lw=0)
  for i in np.flatnonzero(sleep): target_ax.axvspan(x[i]-.25,x[i]+.25,color="#7A8798",alpha=.045,lw=0)
 for ax in axes.flat:
  ax.set_xlabel("Elapsed hours")
  for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("black")
 fig.suptitle("Example streaming trajectory links latent-state updates with observed physiology",fontweight="bold",fontsize=15); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(P3/"figure_3B_representative_participant_trajectory.png",dpi=200,bbox_inches="tight"); fig.savefig(P3/"figure_3B_representative_participant_trajectory.pdf",bbox_inches="tight"); fig.savefig(P3/"figure_3B_representative_participant_trajectory_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 write_json(P3/"figure_3B_metadata.json",{"participant_id":pid,"segment_id":int(seg),"selection_rule":"Largest primary-test subtype; complete 30-minute coverage, both day and night, final cumulative displacement closest to that subtype median; participant ID breaks ties.","pca_fit":"Six deterministic states from each of the first 80 frozen train participants; visualization only.","statistics_dimension":DIM,"pca_visualization_only":True,"explained_variance_ratio":pca.explained_variance_ratio_.tolist()})

def run_phase3(profiles):
 marker=EXT/"PHASE3_COMPLETE.json"
 if marker.exists(): return
 dyn=pd.read_parquet(ARCH/"state_index.parquet").merge(profiles[["participant_id","canonical_stratum","display_cluster"]],on="participant_id",how="inner",validate="many_to_one"); test=dyn[dyn.split.eq("test")].copy()
 sleep_cache=CACHE/"state_sleep_wake_context.parquet"
 if sleep_cache.exists(): sleep_context=pd.read_parquet(sleep_cache)
 else:
  cols=["participant_id","timestamp_local","sleep_stage_awake","sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]; raw_sleep=pd.read_parquet(DATASET,columns=cols); raw_sleep.participant_id=raw_sleep.participant_id.astype(str); raw_sleep.timestamp_local=pd.to_datetime(raw_sleep.timestamp_local); asleep=raw_sleep[["sleep_stage_light","sleep_stage_deep","sleep_stage_rem"]].fillna(0).max(axis=1).gt(0); awake=raw_sleep.sleep_stage_awake.fillna(0).gt(0); raw_sleep["sleep_wake"]=np.where(asleep,"sleep",np.where(awake,"wake","unknown")); sleep_context=raw_sleep[["participant_id","timestamp_local","sleep_wake"]].drop_duplicates(["participant_id","timestamp_local"]); sleep_context.to_parquet(sleep_cache,index=False)
 test.timestamp_local=pd.to_datetime(test.timestamp_local); test=test.merge(sleep_context,on=["participant_id","timestamp_local"],how="left",validate="many_to_one"); test.sleep_wake=test.sleep_wake.fillna("unknown")
 test.to_parquet(P3/"latent_update_dynamics.parquet",index=False); test.to_csv(P3/"latent_update_dynamics.csv",index=False)
 summary=[]
 for (s,t),g in test.groupby(["canonical_stratum","elapsed_minutes"]):
  for metric in ["euclidean_cumulative","cosine_cumulative","euclidean_velocity","cosine_velocity"]:
   v=g.groupby("participant_id")[metric].median(); summary.append({"canonical_stratum":s,"elapsed_minutes":t,"metric":metric,"median":v.median(),"q1":v.quantile(.25),"q3":v.quantile(.75),"participant_n":len(v)})
 sm=pd.DataFrame(summary); sm.to_csv(P3/"latent_update_summary.csv",index=False)
 test["period"]=pd.cut(test.elapsed_minutes,bins=[0,360,1440,2880],labels=["early","middle","late"],include_lowest=True)
 periods=test.groupby(["participant_id","canonical_stratum","period"],observed=True).euclidean_velocity.median().reset_index(); wide=periods.pivot(index=["participant_id","canonical_stratum"],columns="period",values="euclidean_velocity").dropna(); wide["late_to_early_ratio"]=wide.late/wide.early; wide.reset_index().to_csv(P3/"participant_stabilization_metrics.csv",index=False)
 rng=np.random.default_rng(SEED); diff=(wide.early-wide.late).to_numpy(); boots=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(B)]); euclid_ci=np.percentile(boots,[2.5,97.5]); cos=test.groupby(["participant_id","period"],observed=True).cosine_velocity.median().unstack().dropna(); cdiff=(cos.early-cos.late).to_numpy(); cboots=np.array([rng.choice(cdiff,len(cdiff),replace=True).mean() for _ in range(B)]); cosine_ci=np.percentile(cboots,[2.5,97.5])
 paired=test.groupby(["participant_id","day_night"]).euclidean_velocity.median().unstack().dropna(); paired.to_csv(P3/"day_night_paired_updates.csv")
 sleep_paired=test[test.sleep_wake.isin(["sleep","wake"])].groupby(["participant_id","sleep_wake"]).euclidean_velocity.median().unstack().dropna(); sleep_paired.to_csv(P3/"sleep_wake_paired_updates.csv")
 late_slope=[]
 for pid,g in test[test.elapsed_minutes>=1440].groupby("participant_id"):
  x=g.groupby("elapsed_minutes").euclidean_cumulative.median(); late_slope.append(np.polyfit(x.index/60,x.values,1)[0] if len(x)>2 else np.nan)
 stabilizing=euclid_ci[0]>0 and cosine_ci[0]>0 and np.nanmedian(late_slope)<np.nanmedian(test.euclidean_velocity)/24
 plot_phase3(sm,wide,paired)
 representative_figure(test,profiles)
 report=["# Phase 3: latent update dynamics","",f"The 30-minute archive supports geometric update-rate analysis in the full {DIM:,}-dimensional state. 'Velocity' and 'acceleration' are geometric shorthand, not physical quantities.","",f"The participant-bootstrap early-minus-late Euclidean update contrast was {diff.mean():.6g} (95% CI {euclid_ci[0]:.6g} to {euclid_ci[1]:.6g}); the cosine contrast was {cdiff.mean():.6g} (95% CI {cosine_ci[0]:.6g} to {cosine_ci[1]:.6g}). The median late-to-early ratio was {wide.late_to_early_ratio.median():.3f}. The predefined stabilization criteria were {'met' if stabilizing else 'not all met'}; lower local update magnitude is distinguished from continued cumulative drift.","",f"Day-minus-night paired median update was {(paired.day-paired.night).median():.6g}. Sleep-minus-wake paired median update was {(sleep_paired.sleep-sleep_paired.wake).median() if len(sleep_paired) else float('nan'):.6g}. Subtype curves and a one-hour sensitivity obtainable by taking alternate rows are saved with the plotted data. Insulin-dependent estimates are exploratory."]
 (REPORTS/"phase3_latent_update_dynamics.md").write_text("\n".join(report)+"\n")
 qa={"phase":"phase3","status":"complete","created_at":now(),"resolution_minutes":30,"sensitivity_resolution_minutes":60,"state_dimension":DIM,"participant_bootstrap_n":B,"stabilization_criteria_met":bool(stabilizing),"euclidean_early_minus_late_ci":euclid_ci.tolist(),"cosine_early_minus_late_ci":cosine_ci.tolist(),"pca_visualization_only":True}
 write_json(P3/"figure_3A_metadata.json",qa); write_json(marker,qa)

def plot_phase3(sm,wide,paired):
 style(); fig,axes=plt.subplots(2,2,figsize=(14,9))
 for s in SUBTYPES:
  g=sm[(sm.canonical_stratum==s)&(sm.metric=="euclidean_cumulative")]; axes[0,0].plot(g.elapsed_minutes/60,g["median"],color=COLORS[s],label=SLABEL[s]); axes[0,0].fill_between(g.elapsed_minutes/60,g.q1,g.q3,color=COLORS[s],alpha=.15)
  g=sm[(sm.canonical_stratum==s)&(sm.metric=="euclidean_velocity")]; axes[0,1].plot(g.elapsed_minutes/60,g["median"],color=COLORS[s],label=SLABEL[s]); axes[0,1].fill_between(g.elapsed_minutes/60,g.q1,g.q3,color=COLORS[s],alpha=.15)
 axes[0,0].set_title("A  Cumulative displacement from h0",loc="left",fontweight="bold"); axes[0,1].set_title("B  Stepwise 30-minute update",loc="left",fontweight="bold"); axes[0,0].legend(frameon=False,fontsize=8)
 for _,r in paired.reset_index().iterrows(): axes[1,0].plot([0,1],[r.day,r.night],color="#B0BAC6",alpha=.25,lw=.7)
 axes[1,0].scatter(np.zeros(len(paired)),paired.day,color="#5BBABA",s=14); axes[1,0].scatter(np.ones(len(paired)),paired.night,color="#003366",s=14); axes[1,0].set_xticks([0,1],["Day","Night"]); axes[1,0].set_title("C  Participant-paired day and night updates",loc="left",fontweight="bold")
 sns.histplot(wide.late_to_early_ratio,bins=25,color="#7A1F1F",ax=axes[1,1]); axes[1,1].axvline(1,color="#000000",ls="--",lw=1); axes[1,1].set_title("D  Late-to-early update ratio",loc="left",fontweight="bold"); axes[1,1].set_xlabel("Late-to-early update ratio"); axes[1,1].set_ylabel("Participant count")
 for ax in axes.flat:
  if ax not in [axes[1,0],axes[1,1]]: ax.set_xlabel("Elapsed hours")
  if ax is not axes[1,1]: ax.set_ylabel("Dimension-normalized Euclidean distance")
  for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("black")
 fig.suptitle("Streaming hidden-state update dynamics across elapsed time and circadian context",fontweight="bold",fontsize=15); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(P3/"figure_3A_latent_update_dynamics.png",dpi=200,bbox_inches="tight"); fig.savefig(P3/"figure_3A_latent_update_dynamics.pdf",bbox_inches="tight"); fig.savefig(P3/"figure_3A_latent_update_dynamics_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)

def main():
 for p in [P2,P3,REPORTS,CACHE,LOGS]: p.mkdir(parents=True,exist_ok=True)
 validation=json.loads((ARCH/"canonical_representation_validation.json").read_text())
 if not validation.get("passed"): raise SystemExit("STOP: canonical representation-equivalence validation did not pass")
 profiles=pd.read_parquet(PROFILES); profiles.participant_id=profiles.participant_id.astype(str)
 paths=build_timepoint_cache(); clinical,H,hmap=clinical_and_h0(profiles)
 run_phase2(paths,profiles,clinical,H,hmap); run_phase3(profiles)
 print("Phases 2 and 3 complete",flush=True)

if __name__=="__main__": main()
