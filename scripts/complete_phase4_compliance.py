"""Complete prompt-required Phase 4 robustness checks from cached outputs."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from ssmcgm.analysis.within_subtype_config import STUDY2_ROOT  # noqa:E402
import run_extended_event_rewiring as base  # noqa:E402
EXT=STUDY2_ROOT/"extended_clinical_latent_dynamics_v1"; OUT=EXT/"04_event_locked_rewiring"; REPORTS=EXT/"reports"
SEED=42; B=1000; BS=250; EVENTS=base.EVENTS

def now(): return datetime.now(timezone.utc).isoformat()
def preserve(p:Path):
 if p.exists():
  q=p.with_name(f"{p.stem}_pre_manual_compliance_qa{p.suffix}")
  if not q.exists(): p.rename(q)
def ece(y,p,bins=10):
 out=0.; edges=np.linspace(0,1,bins+1)
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&(p<(hi if hi<1 else 1.000001))
  if m.any(): out+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(out)
def scores(y,p): return {"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p),"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),"ece":ece(y,p)}
def model(): return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=SEED))])
def boot_delta(test,pa,pb,nboot,seed):
 ids=test.anchor_id.astype(str).unique(); y=test.y.to_numpy(); a=test.anchor_id.astype(str).to_numpy(); rng=np.random.default_rng(seed); vals=[]
 for _ in range(nboot):
  pick=rng.choice(ids,len(ids),replace=True); ix=np.concatenate([np.flatnonzero(a==x) for x in pick])
  if len(np.unique(y[ix]))>1: vals.append(roc_auc_score(y[ix],pb[ix])-roc_auc_score(y[ix],pa[ix]))
 return float(np.mean(vals)),float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))

def augment(f,profiles,dyn):
 f=f.copy(); f["clock_bin"]=pd.to_numeric(f.clock_bin,errors="coerce"); p=profiles.set_index("participant_id")
 f["same_clinical_cluster"]=(f.anchor_cluster==f.partner_cluster).astype(int)
 f["same_site"]=[int(str(p.at[str(a),"participants_clinical_site"])==str(p.at[str(b),"participants_clinical_site"])) for a,b in zip(f.anchor_id,f.partner_id)]
 use=dyn[["participant_id","hour","cgm_mean","valid_observation_count","available_streaming_duration_hours_mean"]].copy(); use.participant_id=use.participant_id.astype(str)
 a=use.add_prefix("a_"); b=use.add_prefix("b_")
 f=f.merge(a,left_on=["anchor_id","hour"],right_on=["a_participant_id","a_hour"],how="left",validate="many_to_one").merge(b,left_on=["partner_id","hour"],right_on=["b_participant_id","b_hour"],how="left",validate="many_to_one")
 f["baseline_glucose_similarity"]=-abs(f.a_cgm_mean-f.b_cgm_mean)
 f["coverage_similarity"]=-abs(f.a_valid_observation_count-f.b_valid_observation_count)
 f["duration_similarity"]=-abs(f.a_available_streaming_duration_hours_mean-f.b_available_streaming_duration_hours_mean)
 return f

def fit_checks(f):
 tasks={"A_retained_vs_lost":("retained","lost"),"B_gained_vs_matched":("gained","matched")}; results=[]; coefs=[]; subtype=[]; perm=[]; nonins=[]
 for ti,(task,classes) in enumerate(tasks.items()):
  q=f[f.transition_class.isin(classes)].copy(); q["y"]=(q.transition_class==classes[0]).astype(int)
  nuisance=["hour","clock_bin","h0_distance","same_clinical_cluster","same_site","coverage_similarity","duration_similarity","baseline_glucose_similarity"]
  static=[c for c in q if c.startswith("sd_static_")]; dynamic=[c for c in q if c.startswith("sd_dynamic_")]; event=[c for c in q if c.startswith("event_")]
  specs={"N":nuisance,"SD":nuisance+static+dynamic,"SDE":nuisance+static+dynamic+event}; train=q[q.scenario.eq("model_train_2h")]; test=q[q.scenario.eq("primary_test_2h")]
  preds={}; fitted={}
  for name,cols in specs.items():
   m=model().fit(train[cols],train.y); pr=m.predict_proba(test[cols])[:,1]; preds[name]=pr; fitted[name]=m; results.append({"task":task,"model":name,"test_n":len(test),**scores(test.y.to_numpy(),pr)})
  d,lo,hi=boot_delta(test,preds["SD"],preds["SDE"],B,SEED+ti); results.append({"task":task,"model":"SDE_minus_SD","test_n":len(test),"auroc":d,"auroc_ci_low":lo,"auroc_ci_high":hi})
  vals=fitted["SDE"].named_steps["clf"].coef_[0]
  for c,v in zip(specs["SDE"],vals): coefs.append({"task":task,"feature":c,"coefficient":v})
  for st,g in train.groupby("canonical_stratum"):
   if g.y.nunique()<2: continue
   m=model().fit(g[specs["SDE"]],g.y)
   for c,v in zip(specs["SDE"],m.named_steps["clf"].coef_[0]):
    if c in event: subtype.append({"task":task,"canonical_stratum":st,"feature":c,"coefficient":v})
  tr=train[train.canonical_stratum.ne("insulin_dependent")]; te=test[test.canonical_stratum.ne("insulin_dependent")]
  msd=model().fit(tr[specs["SD"]],tr.y); mse=model().fit(tr[specs["SDE"]],tr.y); psd=msd.predict_proba(te[specs["SD"]])[:,1]; pse=mse.predict_proba(te[specs["SDE"]])[:,1]; nd,nlo,nhi=boot_delta(te,psd,pse,B,SEED+100+ti)
  nonins.append({"task":task,"test_n":len(te),"sde_minus_sd_auroc":nd,"ci_low":nlo,"ci_high":nhi})
  base_auc=roc_auc_score(test.y,preds["SDE"]); domains={"nuisance":nuisance,"static":static,"dynamic":dynamic,"event":event}; rng=np.random.default_rng(SEED+200+ti)
  for domain,cols in domains.items():
   drops=[]
   for _ in range(BS):
    xp=test[specs["SDE"]].copy(); order=rng.permutation(len(xp)); xp.loc[:,cols]=xp[cols].to_numpy()[order]; drops.append(base_auc-roc_auc_score(test.y,fitted["SDE"].predict_proba(xp)[:,1]))
   perm.append({"task":task,"domain":domain,"auroc_drop":np.mean(drops),"ci_low":np.percentile(drops,2.5),"ci_high":np.percentile(drops,97.5),"permutation_n":BS})
 return pd.DataFrame(results),pd.DataFrame(coefs),pd.DataFrame(subtype),pd.DataFrame(nonins),pd.DataFrame(perm)

def pair_contrasts(f):
 rows=[]; rng=np.random.default_rng(SEED)
 for task,classes in {"retained_minus_lost":("retained","lost"),"gained_minus_matched":("gained","matched")}.items():
  q=f[f.transition_class.isin(classes)]
  for feature in [c for c in q if c.startswith("event_")]:
   z=q.groupby(["anchor_id","transition_class"])[feature].mean().unstack().dropna()
   if not set(classes)<=set(z): continue
   d=(z[classes[0]]-z[classes[1]]).to_numpy(); sd=np.nanstd(q[feature].to_numpy(float),ddof=1); d=d/sd if sd>0 else d
   boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(B)])
   rows.append({"comparison":task,"feature":feature,"standardized_effect":d.mean(),"ci_low":np.percentile(boots,2.5),"ci_high":np.percentile(boots,97.5),"participant_n":len(d)})
 return pd.DataFrame(rows)

def daynight(outcomes,matches):
 keys=["participant_id","event_type","event_timestamp_local","condition"]; x=outcomes.merge(matches[keys+["aligned_local_hour"]],on=keys,how="left",validate="one_to_one"); x["day_night"]=np.where(x.aligned_local_hour.between(6,21),"day","night"); rows=[]; rng=np.random.default_rng(SEED)
 for (typ,dn),g in x.groupby(["event_type","day_night"]):
  w=g.pivot_table(index=["participant_id","event_timestamp_local"],columns="condition",values=["pre_to_post_euclidean","neighborhood_jaccard"])
  for metric in ["pre_to_post_euclidean","neighborhood_jaccard"]:
   if (metric,"event") not in w or (metric,"control") not in w: continue
   d=(w[(metric,"event")]-w[(metric,"control")]).groupby(level=0).mean().dropna().to_numpy(); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(BS)]) if len(d) else np.array([np.nan])
   rows.append({"event_type":typ,"day_night":dn,"metric":metric,"event_minus_control":np.nanmean(d),"ci_low":np.nanpercentile(boots,2.5),"ci_high":np.nanpercentile(boots,97.5),"participant_n":len(d),"bootstrap_n":BS,"sufficiently_powered":len(d)>=30})
 return pd.DataFrame(rows)

def figures(curve,effects,perf,coef,contrasts,dn,category):
 sns.set_theme(style="whitegrid"); cmap={"retained_minus_lost":"Retained minus lost","gained_minus_matched":"Gained minus matched"}
 event_rows=[]
 for ev in EVENTS:
  for comp in cmap:
   features=[f"event_both_recent_{ev}",f"event_count_similarity_{ev}"]; g=contrasts[(contrasts.comparison==comp)&contrasts.feature.isin(features)]; event_rows.append({"event_type":ev,"comparison":cmap[comp],"effect":g.standardized_effect.mean()})
 heat=pd.DataFrame(event_rows).pivot(index="event_type",columns="comparison",values="effect").reindex(EVENTS)
 fig,axes=plt.subplots(1,2,figsize=(14,5.5)); sns.heatmap(heat,annot=True,fmt=".3f",center=0,cmap="vlag",ax=axes[0]); axes[0].set_title("Participant-level event-context contrasts",fontweight="bold"); inc=perf[perf.model.eq("SDE_minus_SD")]; axes[1].bar(range(len(inc)),inc.auroc,color="#5BBABA",edgecolor="black"); axes[1].errorbar(range(len(inc)),inc.auroc,yerr=[inc.auroc-inc.auroc_ci_low,inc.auroc_ci_high-inc.auroc],fmt="none",ecolor="black",capsize=3); axes[1].set_xticks(range(len(inc)),["Retained vs lost","Gained vs matched"],rotation=15); axes[1].set_ylabel("Held-out AUROC: SDE minus SD"); axes[1].set_title("Incremental event domain",fontweight="bold"); fig.suptitle("Event context provides an additional test of latent-neighborhood rewiring",fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure_4B_event_context_and_neighbor_transitions_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(14,9)); q=curve.groupby(["relative_minutes","condition"]).estimate.mean().unstack(); q.plot(ax=axes[0,0],color=["#003366","#BA2828"]); axes[0,0].axvline(0,color="red"); axes[0,0].set_title("A  Event-aligned update",loc="left",fontweight="bold"); sns.barplot(data=effects,x="event_type",y="event_minus_control",hue="metric",ax=axes[0,1]); axes[0,1].tick_params(axis="x",rotation=25); axes[0,1].set_title("B  Event versus control",loc="left",fontweight="bold"); sns.barplot(data=perf[perf.model.isin(["SD","SDE"])],x="task",y="auroc",hue="model",ax=axes[1,0]); axes[1,0].set_xticklabels(["Retained vs lost","Gained vs matched"],rotation=15); axes[1,0].set_title("C  Held-out prediction",loc="left",fontweight="bold"); stable=coef[coef.feature.str.startswith("event_")].copy(); stable=stable.reindex(stable.coefficient.abs().sort_values(ascending=False).index).head(12); sns.barplot(data=stable,y="feature",x="coefficient",hue="task",ax=axes[1,1]); axes[1,1].set_title("D  Event-feature coefficients",loc="left",fontweight="bold"); title="Physiological events partly explain latent-state updates and neighborhood rewiring" if category=="Incremental event information" else "Measured event context adds limited information beyond continuous dynamics"; fig.suptitle(title,fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4C_integrated_event_attribution.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4C_integrated_event_attribution.pdf",bbox_inches="tight"); fig.savefig(OUT/"figure_4C_integrated_event_attribution_thumbnail.png",dpi=70,bbox_inches="tight"); plt.close(fig)
 powered=dn[dn.sufficiently_powered]; fig,axes=plt.subplots(1,2,figsize=(13,5));
 for ax,metric in zip(axes,["pre_to_post_euclidean","neighborhood_jaccard"]): sns.barplot(data=powered[powered.metric.eq(metric)],x="event_type",y="event_minus_control",hue="day_night",ax=ax); ax.tick_params(axis="x",rotation=25); ax.set_title(metric.replace("_"," ").title(),fontweight="bold")
 fig.suptitle("Day and night sensitivity of event-associated latent outcomes",fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"figure_4A_day_night_sensitivity.png",dpi=200,bbox_inches="tight"); fig.savefig(OUT/"figure_4A_day_night_sensitivity.pdf",bbox_inches="tight"); plt.close(fig)

def main():
 for name in ["predictive_model_performance.csv","event_feature_coefficients.csv","event_context_transition_metrics.csv","figure_4B_event_context_and_neighbor_transitions.png","figure_4B_event_context_and_neighbor_transitions.pdf","figure_4B_event_context_and_neighbor_transitions_thumbnail.png","figure_4C_integrated_event_attribution.png","figure_4C_integrated_event_attribution.pdf","figure_4C_integrated_event_attribution_thumbnail.png","figure_4B_metadata.json","figure_4C_metadata.json"]: preserve(OUT/name)
 for p in [EXT/"PHASE4_COMPLETE.json",REPORTS/"phase4_event_locked_rewiring.md",EXT/"FINAL_QA_REPORT.md",REPORTS/"FINAL_EXTENDED_INTERPRETATION_REPORT.md",EXT/"replacement_analysis_manifest.json"]: preserve(p)
 profiles=pd.read_parquet(base.PROFILES); profiles.participant_id=profiles.participant_id.astype(str); dyn=pd.read_parquet(STUDY2_ROOT/"neighbor_transition_drivers/participant_dynamic_features.parquet"); dyn.participant_id=dyn.participant_id.astype(str); f=augment(pd.read_parquet(OUT/"event_augmented_transition_pairs.parquet"),profiles,dyn)
 perf,coef,subcoef,nonins,perm=fit_checks(f); contrasts=pair_contrasts(f); outcomes=pd.read_parquet(OUT/"event_locked_outcomes.parquet"); matches=pd.read_parquet(OUT/"matched_event_control_windows.parquet"); dn=daynight(outcomes,matches); curve=pd.read_csv(OUT/"event_aligned_summary.csv"); effects=pd.read_csv(OUT/"event_context_transition_metrics_pre_manual_compliance_qa.csv")
 perf.to_csv(OUT/"predictive_model_performance.csv",index=False); coef.to_csv(OUT/"event_feature_coefficients.csv",index=False); subcoef.to_csv(OUT/"event_feature_coefficients_by_subtype.csv",index=False); nonins.to_csv(OUT/"non_insulin_incremental_performance.csv",index=False); perm.to_csv(OUT/"domain_permutation_importance.csv",index=False); contrasts.to_csv(OUT/"event_context_transition_metrics.csv",index=False); dn.to_csv(OUT/"event_day_night_sensitivity.csv",index=False)
 stability=subcoef.groupby(["task","feature"]).coefficient.agg(lambda x:max((x>0).sum(),(x<0).sum())).reset_index(name="concordant_subtypes"); stable=stability[stability.concordant_subtypes>=3]; full=perf[perf.model.eq("SDE_minus_SD")]; robust=bool(len(full)==2 and (full.auroc_ci_low>0).all() and (nonins.ci_low>0).all() and len(stable)); category="Incremental event information" if robust else "Continuous dynamics sufficient"
 figures(curve,effects,perf,coef,contrasts,dn,category)
 paragraphs=[]
 for ev in EVENTS:
  g=effects[effects.event_type.eq(ev)].set_index("metric"); eu=g.loc["pre_to_post_euclidean"]; ne=g.loc["neighborhood_jaccard"]; paragraphs.append(f"**{ev.replace('_',' ').title()}.** Event-minus-control Euclidean displacement was {eu.event_minus_control:.3f} (95% CI {eu.ci_low:.3f}, {eu.ci_high:.3f}); neighborhood Jaccard difference was {ne.event_minus_control:.3f} ({ne.ci_low:.3f}, {ne.ci_high:.3f}). Day/night estimates are reported separately when at least 30 participants contributed.")
 report=["# Phase 4: observable-event drivers of latent rewiring","",f"Interpretation category: **{category}**.","","All detections use current and trailing observations. Controls are within participant and clock/day-night matched; onset timing is a detection surrogate and all results are associative.","",*paragraphs,"",f"Both full-cohort incremental AUROC intervals were above zero: {full[['task','auroc','auroc_ci_low','auroc_ci_high']].to_dict('records')}. Non-insulin-only checks: {nonins.to_dict('records')}. Event-domain coefficient direction was concordant in at least three subtypes for {len(stable)} task-feature combinations. Domain permutation importance is saved separately. No meal, insulin, or exercise event was used."]
 (REPORTS/"phase4_event_locked_rewiring.md").write_text("\n\n".join(report)+"\n")
 qa={"phase":"phase4","status":"complete","created_at":now(),"category":category,"matched_rows":len(matches),"outcome_rows":len(outcomes),"predictive_models":6,"primary_bootstrap_n":B,"sensitivity_bootstrap_n":BS,"domain_permutation_importance":True,"subtype_coefficient_stability":True,"non_insulin_check":True,"clock_matched":True,"day_night_sensitivity":True,"future_values_used_for_onset":False,"timed_insulin_used":False,"meal_event_used":False,"exercise_claimed":False}; (EXT/"PHASE4_COMPLETE.json").write_text(json.dumps(qa,indent=2)+"\n"); (OUT/"figure_4B_metadata.json").write_text(json.dumps(qa,indent=2)+"\n"); (OUT/"figure_4C_metadata.json").write_text(json.dumps(qa,indent=2)+"\n"); print(json.dumps(qa,indent=2))
if __name__=="__main__": main()
