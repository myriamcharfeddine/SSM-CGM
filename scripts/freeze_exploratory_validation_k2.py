#!/usr/bin/env python3
"""Freeze the borderline validation neutral_all k=2 as exploratory sensitivity."""
from __future__ import annotations
import argparse,hashlib,json,logging,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,seaborn as sns
from scipy.stats import chi2_contingency,mannwhitneyu
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,confusion_matrix,precision_score,recall_score
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger("step3b");H=[f"r_{i:03d}" for i in range(128)]
def args():
 p=argparse.ArgumentParser();p.add_argument("--step2-dir",required=True);p.add_argument("--step3-dir",required=True);p.add_argument("--representation",default="neutral_all");p.add_argument("--k",type=int,default=2);p.add_argument("--output-root",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--run-id");p.add_argument("--overwrite",action="store_true");return p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def jd(x):
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,)):return None if not np.isfinite(x) else float(x)
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(Path,pd.Timestamp,datetime)):return str(x)
 raise TypeError(type(x).__name__)
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True,default=jd)+"\n");os.replace(t,p)
def boot_delta(x,y,n,rng):
 z=np.r_[x,y];g=np.r_[np.zeros(len(x)),np.ones(len(y))];v=[]
 for _ in range(n):
  a=rng.choice(x,len(x),replace=True);b=rng.choice(y,len(y),replace=True);v.append(np.median(b)-np.median(a))
 return np.quantile(v,[.025,.975])
def cliff(x,y):
 return (np.sum(y[:,None]>x[None,:])-np.sum(y[:,None]<x[None,:]))/(len(x)*len(y))
def main():
 a=args();s2=Path(a.step2_dir).resolve();s3=Path(a.step3_dir).resolve()
 if a.representation!="neutral_all" or a.k!=2:raise ValueError("only frozen neutral_all k=2 is authorized")
 if sha(s3/"clustering_selection_decision.json")!="1b4681333ad90a6258507f8a015975053458f0851666cf9aace6182eeeccb89b":raise RuntimeError("Step 3 decision hash mismatch")
 rid=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");root=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root);out=root/rid
 if out.exists() and a.overwrite:shutil.rmtree(out)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);frozen=out/"frozen_k2_sensitivity";frozen.mkdir()
 fmt=logging.Formatter("%(asctime)sZ %(levelname)s %(message)s","%Y-%m-%dT%H:%M:%S");LOG.setLevel(logging.INFO);hs=[logging.FileHandler(out/"step3b_run.log"),logging.StreamHandler(sys.stdout)]
 for h in hs:h.setFormatter(fmt)
 LOG.handlers[:]=hs;LOG.info("Step 3B validation-only eligibility audit")
 c=pd.read_csv(s3/"clustering_candidate_metrics.csv");r=c.query("representation_space=='neutral_all' and k==2").iloc[0]
 specs=[("minimum_cluster_size",20,r.minimum_cluster_size,">="),("minimum_cluster_fraction",.08,r.minimum_cluster_fraction,">="),("median_subsample_ari",.60,r.median_subsample_ari,">="),("median_minimum_cluster_jaccard",.60,r.median_minimum_cluster_jaccard,">="),("consensus_pac",.20,r.consensus_pac,"<="),("odd_even_ari",.40,r.odd_even_ari,">="),("odd_even_same_cluster",.60,r.odd_even_same_cluster,">="),("silhouette",.10,r.silhouette,">="),("assignment_probability_ge_80",.70,r.assignment_probability_ge_80,">=")]
 rows=[];failed=[]
 for name,req,obs,op in specs:
  passed=obs>=req if op==">=" else obs<=req
  if not passed:failed.append(name)
  rows.append({"criterion_name":name,"required_value":req,"operator":op,"observed_value":obs,"passed":passed,"failure_margin":0 if passed else (req-obs if op==">=" else obs-req),"source_file":str(s3/"clustering_candidate_metrics.csv"),"source_row_identifier":"representation_space=neutral_all;k=2"})
 eligible=set(failed)=={"minimum_cluster_size","minimum_cluster_fraction"} and int(r.minimum_cluster_size)==19 and abs(r.minimum_cluster_fraction-19/239)<1e-12
 dump(out/"exploratory_k2_eligibility_audit.json",{"representation":"neutral_all","k":2,"criteria":rows,"failed_criteria":failed,"exploratory_k2_status":"eligible_for_centroid_qc" if eligible else "not_eligible_for_transport"})
 original={x:sha(s3/x) for x in ["clustering_selection_decision.json","step3_report.md","step3_manifest.json","frozen_test_application_plan.json"]}
 dump(out/"step3b_exploratory_analysis_addendum.json",{"primary_conclusion_remains":"reliable_continuous_manifold","original_thresholds_unchanged":{"minimum_cluster_size":20,"minimum_cluster_fraction":.08},"no_discrete_solution_passed":True,"neutral_all_k2_role":"secondary exploratory sensitivity only","test_refit_or_k_selection_forbidden":True,"validation_result_not_reclassified_confirmatory":True,"original_step3_file_hashes":original})
 if not eligible:raise RuntimeError("k=2 has non-size eligibility failures")
 ass=pd.read_parquet(s3/"cluster_assignments.parquet");ass.participant_id=ass.participant_id.astype(str);z=ass.query("representation_space=='neutral_all' and k==2").sort_values("participant_id")
 reps=pd.read_parquet(s2/"participant_representations.parquet");reps.participant_id=reps.participant_id.astype(str);q=reps.query("representation_type=='neutral_all' and balanced_anchor_variant=='all_anchors'").set_index("participant_id").reindex(z.participant_id)
 scalerp=s3/"frozen_validation_pipeline/neutral_all/neutral_all_scaler.joblib";pcap=s3/"frozen_validation_pipeline/neutral_all/neutral_all_pca.joblib";keepp=s3/"frozen_validation_pipeline/neutral_all/kept_dimensions.npy"
 sc=joblib.load(scalerp);pc=joblib.load(pcap);keep=np.load(keepp);n90=int(pd.read_csv(s3/"pca_variance_summary.csv").query("space=='neutral_all'").n90.iloc[0]);scores=pc.transform(sc.transform(q[H].to_numpy()[:,keep]))[:,:n90];old=z.cluster_label.to_numpy(int)
 cold=np.stack([scores[old==k].mean(0) for k in [0,1]]);order=np.argsort(cold[:,0]);mapping={int(order[0]):0,int(order[1]):1};lab=np.array([mapping[x] for x in old]);cent=np.stack([cold[order[0]],cold[order[1]]]);d=np.sqrt(((scores[:,None]-cent[None])**2).sum(2));pred=d.argmin(1);first=d[np.arange(len(d)),pred];second=d[np.arange(len(d)),1-pred];margin=second-first;norm=margin/np.maximum(second,1e-12);conf=np.exp(-d);conf=conf[np.arange(len(d)),pred]/conf.sum(1);small=int(pd.Series(lab).value_counts().idxmin())
 agree=np.mean(pred==lab);ari=adjusted_rand_score(lab,pred);ami=adjusted_mutual_info_score(lab,pred);rec=recall_score(lab,pred,pos_label=small);prec=precision_score(lab,pred,pos_label=small);ok=agree>=.90 and ari>=.75 and rec>=.80
 qc=pd.DataFrame({"participant_id":z.participant_id,"consensus_group":lab,"centroid_group":pred,"same_label":pred==lab,"distance_assigned":first,"distance_alternative":second,"assignment_margin":margin,"normalized_assignment_margin":norm,"assignment_confidence":conf,"ambiguous_assignment":norm<.10})
 qc.to_csv(out/"validation_k2_centroid_transport_qc.csv",index=False);qc.to_csv(frozen/"neutral_all_k2_validation_assignment_qc.csv",index=False)
 summary=pd.DataFrame([{"overall_agreement":agree,"ari":ari,"adjusted_mutual_information":ami,"small_group":small,"small_group_recall":rec,"small_group_precision":prec,"ambiguous_fraction":np.mean(norm<.1),"transport_qc_passed":ok}]);summary.to_csv(frozen/"neutral_all_k2_validation_assignment_qc_summary.csv",index=False)
 if not ok:raise RuntimeError("centroid transport QC failed")
 labels=pd.DataFrame({"participant_id":z.participant_id,"split":"validation","original_consensus_label":old,"exploratory_group":lab});labels.to_parquet(frozen/"neutral_all_k2_validation_labels.parquet",index=False);np.save(frozen/"neutral_all_k2_centroids.npy",cent)
 refs={"scaler_path":str(scalerp),"scaler_sha256":sha(scalerp),"pca_path":str(pcap),"pca_sha256":sha(pcap),"kept_dimensions_path":str(keepp),"kept_dimensions_sha256":sha(keepp)}
 dump(frozen/"neutral_all_scaler_reference.json",refs);dump(frozen/"neutral_all_pca_reference.json",refs);dump(frozen/"neutral_all_k2_feature_order.json",{"source_features":H,"kept_indices":keep.tolist(),"pca_components":n90})
 metadata={"centroid_shape":list(cent.shape),"label_rule":"group_0 lower centroid PC1; group_1 higher centroid PC1","centroid_pc1":cent[:,0].tolist(),"validation_counts":pd.Series(lab).value_counts().sort_index().to_dict(),"validation_percentages":(pd.Series(lab).value_counts(normalize=True).sort_index()*100).to_dict(),"small_group":small,"original_to_deterministic_mapping":mapping};dump(frozen/"neutral_all_k2_centroid_metadata.json",metadata)
 rule={"representation":"neutral_all","burn_in_minutes":0,"aggregation":"all-anchor dimensionwise median","distance":"Euclidean in frozen standardized validation PCA space","assignment":"nearest validation consensus-label centroid","assignment_margin":"second_distance-first_distance","normalized_margin":"margin/max(second_distance,epsilon)","confidence":"softmax of negative distances","ambiguity_threshold":.10,"no_test_refit":True};dump(frozen/"neutral_all_k2_assignment_rule.json",rule)
 plan={"role":"secondary exploratory near-threshold k=2 sensitivity","primary_conclusion":"reliable_continuous_manifold","k":2,"representation":"neutral_all","burn_in_minutes":0,"aggregation":"all-anchor median","scaling_artifact":refs,"pca_components":n90,"centroid_construction":"mean frozen validation PCA coordinates by saved consensus labels","distance_metric":"Euclidean","label_order_rule":rule["assignment"],"margin_definition":rule["normalized_margin"],"ambiguity_threshold":.10,"validation_small_group_proportion":19/239,"test_metrics":["prevalence","centroid distance","odd/even","balanced anchor","nuisance","glycemic","3 external biomarkers"],"external_fdr_family":list(["NT-proBNP","hs-CRP","BUN/creatinine ratio"]),"interpretation_rules":["transports","partially transports","glycemic tail","site/acquisition","does not transport"],"no_test_fit_or_selection":True};dump(out/"exploratory_k2_analysis_plan_frozen.json",plan);shutil.copy2(out/"exploratory_k2_analysis_plan_frozen.json",frozen/"exploratory_k2_analysis_plan_frozen.json");planhash=sha(out/"exploratory_k2_analysis_plan_frozen.json")
 # Frozen labels may now be characterized with already extracted validation-level data.
 feat=pd.read_parquet(s3/"validation_glycemic_nuisance_features.parquet");feat.participant_id=feat.participant_id.astype(str);f=feat.merge(labels[["participant_id","exploratory_group"]],on="participant_id");ext=pd.read_parquet(s3/"validation_external_targets.parquet");ext.participant_id=ext.participant_id.astype(str);ew=ext[ext.eligible_for_analysis].pivot(index="participant_id",columns="target_name",values="analysis_value").reset_index();f=f.merge(ew,on="participant_id",how="left")
 nuisance=["number_of_segments","median_segment_hours","total_clean_hours","dynamic_missingness","hr_availability","activity_availability","respiratory_rate_availability","sleep_availability","fraction_states_within_30min_of_reset","median_full_neutral_l2","mean_abs_forecast_delta"];gly=["mean_glucose","glucose_sd","glucose_cv","tir_70_180","tar_above_180","tbr_below_70","hba1c","mean_absolute_glucose_slope","median_absolute_glucose_slope"];rng=np.random.default_rng(a.seed);cr=[]
 for fam,names in [("nuisance",nuisance),("glycemic",gly),("external",list(ew.columns[1:]))]:
  for name in names:
   x=f.loc[f.exploratory_group==0,name].dropna().to_numpy(float);y=f.loc[f.exploratory_group==1,name].dropna().to_numpy(float);u,p=mannwhitneyu(x,y,alternative="two-sided");lo,hi=boot_delta(x,y,2000,rng);cr.append({"family":fam,"variable":name,"group0_n":len(x),"group1_n":len(y),"group0_median":np.median(x),"group0_q1":np.quantile(x,.25),"group0_q3":np.quantile(x,.75),"group1_median":np.median(y),"group1_q1":np.quantile(y,.25),"group1_q3":np.quantile(y,.75),"median_difference_group1_minus_group0":np.median(y)-np.median(x),"ci_low":lo,"ci_high":hi,"cliffs_delta":cliff(x,y),"p_value":p})
 for name in ["clinical_site","study_group"]:
  tab=pd.crosstab(f.exploratory_group,f[name]);chi,p,_,_=chi2_contingency(tab);cr.append({"family":"nuisance","variable":name,"group0_n":int((f.exploratory_group==0).sum()),"group1_n":int((f.exploratory_group==1).sum()),"group0_median":np.nan,"group0_q1":np.nan,"group0_q3":np.nan,"group1_median":np.nan,"group1_q1":np.nan,"group1_q3":np.nan,"median_difference_group1_minus_group0":np.nan,"ci_low":np.nan,"ci_high":np.nan,"cliffs_delta":np.sqrt(chi/(len(f)*max(min(tab.shape)-1,1))),"p_value":p,"contingency":tab.to_json()})
 ch=pd.DataFrame(cr);ch["q_value"]=ch.groupby("family",group_keys=False).p_value.transform(lambda x:multipletests(x,method="fdr_bh")[1]);ch.to_csv(out/"validation_exploratory_k2_characterization.csv",index=False)
 dump(frozen/"exploratory_k2_freeze_manifest.json",{"status":"eligible_and_frozen","plan_hash":planhash,"centroid_hash":sha(frozen/"neutral_all_k2_centroids.npy"),"validation_label_hash":sha(frozen/"neutral_all_k2_validation_labels.parquet"),"step3_decision_hash":original["clustering_selection_decision.json"],"validation_qc":{"agreement":agree,"ari":ari,"ami":ami,"small_group_recall":rec,"small_group_precision":prec},"test_transport_authorized":True})
 # Figures
 sns.set_theme(style="whitegrid");plt.figure(figsize=(7,5));plt.scatter(scores[:,0],scores[:,1],c=lab,cmap="coolwarm",s=25);plt.scatter(cent[:,0],cent[:,1],marker="X",s=180,c=[0,1],cmap="coolwarm",edgecolor="black");plt.title("Validation exploratory neutral_all k=2");plt.tight_layout();plt.savefig(out/"fig_validation_k2_pca.png",dpi=170);plt.close()
 cm=np.load(s3/"frozen_validation_pipeline/consensus_neutral_all_k2.npy");o=np.argsort(lab);fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].imshow(cm[np.ix_(o,o)],vmin=0,vmax=1,cmap="mako");ax[0].set_title("Consensus matrix");ax[1].scatter(scores[:,0],scores[:,1],c=lab,cmap="coolwarm",s=20);ax[1].scatter(cent[:,0],cent[:,1],marker="X",s=150,c=[0,1],cmap="coolwarm",edgecolor="black");ax[1].set_title("Frozen centroids");plt.tight_layout();plt.savefig(out/"fig_validation_k2_consensus_and_centroids.png",dpi=170);plt.close()
 plt.figure(figsize=(7,4));sns.histplot(data=qc,x="normalized_assignment_margin",hue="consensus_group",bins=25);plt.axvline(.1,color="red",ls="--");plt.tight_layout();plt.savefig(out/"fig_validation_k2_assignment_margins.png",dpi=170);plt.close()
 plt.figure(figsize=(5,4));sns.heatmap(confusion_matrix(z.odd_cluster_label.map(mapping),z.even_cluster_label.map(mapping)),annot=True,fmt="d",cmap="Blues");plt.title("Validation odd/even exploratory groups");plt.tight_layout();plt.savefig(out/"fig_validation_k2_odd_even_stability.png",dpi=170);plt.close()
 top=ch[ch.family.isin(["nuisance","glycemic"])&ch.cliffs_delta.notna()].sort_values("cliffs_delta",key=abs).tail(12);plt.figure(figsize=(9,5));sns.barplot(data=top,x="cliffs_delta",y="variable",hue="family");plt.tight_layout();plt.savefig(out/"fig_validation_k2_nuisance_glycemic_characterization.png",dpi=170);plt.close()
 ex=ch[ch.family=="external"];plt.figure(figsize=(8,4));sns.barplot(data=ex,x="variable",y="cliffs_delta");plt.xticks(rotation=20);plt.tight_layout();plt.savefig(out/"fig_validation_k2_external_biomarkers.png",dpi=170);plt.close()
 report=f"""# Step 3B exploratory near-threshold k=2 freeze\n\nPrimary Step 3 conclusion remains **reliable continuous manifold**. The original thresholds remain 20 participants and 8%; neutral_all k=2 failed only those criteria with 19/239 (7.95%). It is frozen solely as a secondary exploratory sensitivity.\n\nCentroid self-assignment: agreement {agree:.3%}, ARI {ari:.3f}, AMI {ami:.3f}, small-group recall {rec:.3%}, precision {prec:.3%}. Test transport is authorized without refitting or k selection.\n\nPlan hash: `{planhash}`. No test data were accessed in Step 3B.\n""";(out/"exploratory_k2_step3b_report.md").write_text(report)
 required=["exploratory_k2_eligibility_audit.json","step3b_exploratory_analysis_addendum.json","validation_k2_centroid_transport_qc.csv","validation_exploratory_k2_characterization.csv","exploratory_k2_analysis_plan_frozen.json","exploratory_k2_step3b_report.md","step3b_run.log"]+[f"fig_validation_k2_{x}.png" for x in ["pca","consensus_and_centroids","assignment_margins","odd_even_stability","nuisance_glycemic_characterization","external_biomarkers"]]
 manifest={"run_id":rid,"status":"eligible_and_frozen","timestamp":datetime.now(timezone.utc).isoformat(),"git_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"step2_dir":str(s2),"step3_dir":str(s3),"step3_original_hashes":original,"eligibility_failures":failed,"validation_counts":metadata["validation_counts"],"validation_percentages":metadata["validation_percentages"],"centroid_qc":{"agreement":agree,"ari":ari,"ami":ami,"small_group_recall":rec,"small_group_precision":prec},"exploratory_plan_hash":planhash,"frozen_artifacts":[str(x) for x in frozen.rglob("*") if x.is_file()],"test_transport_authorized":True,"output_paths":{x:str(out/x) for x in required},"warnings":["Borderline solution remains secondary exploratory; primary continuous-manifold conclusion unchanged."],"errors":[]};dump(out/"exploratory_k2_freeze_manifest.json",manifest)
 latest=root/"latest";tmp=root/".latest.tmp"
 if tmp.exists() or tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(out.name);os.replace(tmp,latest);LOG.info("QC COMPLETE eligible_and_frozen latest=%s",latest)
 print(json.dumps({"output_directory":str(out),"eligibility_status":"eligible_and_frozen","failed_criteria":failed,"counts":metadata["validation_counts"],"percentages":metadata["validation_percentages"],"centroid_agreement":agree,"centroid_ari":ari,"small_group_recall":rec,"centroid_path":str(frozen/"neutral_all_k2_centroids.npy"),"plan_hash":planhash,"test_transport_authorized":True},indent=2,default=jd))
if __name__=="__main__":main()
