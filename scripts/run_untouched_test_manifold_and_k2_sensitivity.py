#!/usr/bin/env python3
"""One-shot untouched-test transport of frozen continuous and exploratory k=2 pipelines."""
from __future__ import annotations
import argparse,hashlib,json,logging,os,platform,random,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,seaborn as sns,torch,yaml
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist,squareform
from scipy.stats import binomtest,chi2_contingency,mannwhitneyu,spearmanr
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,cohen_kappa_score
from statsmodels.stats.multitest import multipletests
from ssmcgm.data.aireadi import AireadiFeatureSpec,AireadiPreprocessor,build_stream_feature_spec,infer_or_validate_schema,make_aireadi_stream_splits,make_participant_streams,prepare_aireadi_panel
from ssmcgm.models.aireadi_stream import AireadiStreamModel,AireadiStreamModelConfig
from scripts.export_validation_hidden_states import replay_segment,context_frame,rep_from,reliability,HIDDEN
from scripts.run_static_neutralization_pilot import sha256
LOG=logging.getLogger("step4");H=[f"r_{i:03d}" for i in range(128)];PRIMARY=["full_all","neutral_all","neutral_glucose_residual"];GLU=["mean_glucose","glucose_cv","tir_70_180","tar_above_180","tbr_below_70"]
def args():
 p=argparse.ArgumentParser()
 for x in ("config","checkpoint","schema","multimodal-parquet","static-table","split-manifest","step0-dir","step1-dir","step2-dir","step3-dir","step3b-dir","aireadi-root","output-root"):p.add_argument("--"+x,required=True)
 p.add_argument("--split",default="test");p.add_argument("--state-save-frequency-minutes",type=int,default=5);p.add_argument("--representation-frequency-minutes",type=int,default=15);p.add_argument("--bootstrap-replicates",type=int,default=2000);p.add_argument("--permutation-replicates",type=int,default=1000);p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=42);p.add_argument("--run-id");p.add_argument("--resume",action="store_true");return p.parse_args()
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
def aparq(d,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp.parquet");d.to_parquet(t,index=False,compression="zstd");os.replace(t,p)
def nearest(x,c):
 d=np.sqrt(((x[:,None]-c[None])**2).sum(2));o=np.argsort(d,1);a=o[:,0];d1=d[np.arange(len(x)),a];d2=d[np.arange(len(x)),o[:,1]];return a,d,d1,d2,d2-d1
def cosrows(a,b):return np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
def cliff(x,y):return (np.sum(y[:,None]>x[None,:])-np.sum(y[:,None]<x[None,:]))/(len(x)*len(y))
def external(cache,ids,starts,lockhash):
 targets={3029187:"natriuretic_peptide_b_prohormon",3010156:"c_reactive_protein_i",4112223:"bun_creatinine_ratio"};parts=[];use=["person_id","measurement_concept_id","measurement_date","measurement_datetime","value_as_number","unit_source_value"]
 for ch in pd.read_csv(cache/"measurement.csv",usecols=use,chunksize=100000,low_memory=False):
  q=ch[ch.person_id.astype(str).isin(ids)&ch.measurement_concept_id.isin(targets)].copy()
  if len(q):parts.append(q)
 m=pd.concat(parts);m["participant_id"]=m.person_id.astype(str)
 if set(m.participant_id)-set(ids):raise RuntimeError("non-test target row")
 m["date"]=pd.to_datetime(m.measurement_datetime,errors="coerce").fillna(pd.to_datetime(m.measurement_date,errors="coerce"));m["start"]=m.participant_id.map(starts);m["days_to_cgm_start"]=(m.date-m.start).dt.total_seconds()/86400;m["target_name"]=m.measurement_concept_id.map(targets);rows=[]
 for (pid,t),g in m.groupby(["participant_id","target_name"]):
  g=g[pd.to_numeric(g.value_as_number,errors="coerce").notna()];b=g[g.days_to_cgm_start<=0].sort_values("date");q=b.iloc[0] if len(b) else g[g.days_to_cgm_start>0].sort_values("days_to_cgm_start").iloc[0];days=float(q.days_to_cgm_start);rows.append({"participant_id":pid,"target_name":t,"analysis_value":float(q.value_as_number),"unit":q.unit_source_value if pd.notna(q.unit_source_value) else "<missing>","measurement_date":q.date,"days_to_cgm_start":days,"record_selection_rule":"earliest at/before" if days<=0 else "nearest after","eligible_for_analysis":abs(days)<=180,"eligible_90d_sensitivity":abs(days)<=90,"step3b_plan_hash":lockhash})
 return pd.DataFrame(rows)
def main():
 a=args()
 if a.split!="test" or a.state_save_frequency_minutes!=5 or a.representation_frequency_minutes!=15:raise ValueError("frozen test protocol required")
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True);dev=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else a.device if a.device!="auto" else "cpu")
 s0,s1,s2,s3,s3b=map(lambda x:Path(x).resolve(),[a.step0_dir,a.step1_dir,a.step2_dir,a.step3_dir,a.step3b_dir]);bman=json.loads((s3b/"exploratory_k2_freeze_manifest.json").read_text())
 if not bman["test_transport_authorized"] or bman["status"]!="eligible_and_frozen":raise RuntimeError("Step 3B gate not passed")
 original={"step3_decision":sha(s3/"clustering_selection_decision.json"),"step3b_plan":sha(s3b/"exploratory_k2_analysis_plan_frozen.json"),"step3b_manifest":sha(s3b/"exploratory_k2_freeze_manifest.json")}
 if original["step3_decision"]!="1b4681333ad90a6258507f8a015975053458f0851666cf9aace6182eeeccb89b":raise RuntimeError("Step 3 decision changed")
 rid=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");root=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root);out=root/rid;out.mkdir(parents=True);setupfmt=logging.Formatter("%(asctime)sZ %(levelname)s %(message)s","%Y-%m-%dT%H:%M:%S");LOG.setLevel(logging.INFO);hs=[logging.FileHandler(out/"step4_run.log"),logging.StreamHandler(sys.stdout)]
 for h in hs:h.setFormatter(setupfmt)
 LOG.handlers[:]=hs
 for d in ["test_hidden_states","test_forecasts","_completion"]:(out/d).mkdir()
 split=pd.read_csv(a.split_manifest,dtype={"participant_id":str});split["split"]=split.split.replace({"val":"validation"});ids=sorted(split.loc[split.split=="test","participant_id"])
 if len(ids)!=221 or set(ids)&set(split.loc[split.split!="test","participant_id"]):raise RuntimeError("test split invalid")
 dump(out/"test_application_plan_verified.json",{"test_participants":221,"step3_original_hashes":original,"burn_in_minutes":0,"sampling_minutes":15,"aggregation":"all-anchor median","scalers_pca_residualizer_frozen":True,"clustering_or_k_refit":False,"exploratory_transport_authorized":True})
 cfg=yaml.safe_load(open(a.config));saved=json.load(open(a.schema));ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);md=ck["metadata"];spec=AireadiFeatureSpec(**md["feature_spec"]);pre=AireadiPreprocessor.from_jsonable(md["preprocessor"]);mcfg=AireadiStreamModelConfig(**md["model_config"]);model=AireadiStreamModel(spec,pre,mcfg).to(dev);model.load_state_dict(ck["model_state_dict"]);model.eval()
 ref=json.load(open(s1/"static_reference_profile.json"));ncont=np.asarray(ref["transformed_static_cont"],"float32");ncat=np.asarray(ref["transformed_static_cat"],"int64");panel=pd.read_parquet(a.multimodal_parquet,filters=[("participant_id","in",ids)]);panel.participant_id=panel.participant_id.astype(str);st=pd.read_parquet(a.static_table,filters=[("participant_id","in",ids)]);st.participant_id=st.participant_id.astype(str);st=st.drop_duplicates("participant_id");panel=panel.merge(st[[c for c in st if c=="participant_id" or c not in panel]],on="participant_id",how="left",validate="many_to_one")
 if set(panel.participant_id)!=set(ids):raise RuntimeError("test panel mismatch")
 schema=infer_or_validate_schema(panel,saved["schema"]);prepared=prepare_aireadi_panel(panel,schema,bin_minutes=5,clean_min_segment_hours=cfg["dataset"]["clean_min_segment_hours"]);spl=make_aireadi_stream_splits(prepared,existing_split_path=a.split_manifest,seed=a.seed);backend=f"{dev.type.upper()} Mamba scan_chunk ({mcfg.scan_mode})";status=[];repdict={x:{} for x in ["full_all","neutral_all","neutral_night","neutral_day"]};odd={x:{} for x in ["full_all","neutral_all"]};even={x:{} for x in odd};balanced={x:{} for x in ["full_all","neutral_all"]};staticrows=[]
 for ii,pid in enumerate(ids,1):
  marker=out/"_completion"/f"{pid}.json"
  if marker.exists() and a.resume:raise RuntimeError("resume cache loading not implemented safely")
  streams=make_participant_streams(prepared[prepared.participant_id==pid],spl,schema,feature_spec=spec,preprocessor=pre,splits=["test"],min_steps=14);
  for s in streams:s.metadata["_dynamic_names"]=spec.dynamic_reals
  dates=sorted(set(str(pd.Timestamp(t).date()) for s in streams for t in s.timestamps));daymap={d:i+1 for i,d in enumerate(dates)};hs0=[];fs0=[]
  for s in streams:
   F,Ff,_,_=replay_segment(model,s,"full_profile",ncont,ncat,dev,sha(Path(a.checkpoint)),3,daymap,backend);N,Nf,_,_=replay_segment(model,s,"static_neutral",ncont,ncat,dev,sha(Path(a.checkpoint)),3,daymap,backend);hs0.extend([F,N]);fs0.extend([Ff,Nf])
  hsdf=pd.concat(hs0,ignore_index=True);fsdf=pd.concat(fs0,ignore_index=True)
  for cond in ["full_profile","static_neutral"]:
   aparq(hsdf[hsdf.condition==cond],out/"test_hidden_states"/f"condition={cond}"/f"participant_id={pid}"/"data.parquet");aparq(fsdf[fsdf.condition==cond],out/"test_forecasts"/f"condition={cond}"/f"participant_id={pid}"/"data.parquet")
  for typ in repdict:
   g=context_frame(hsdf,typ);g=g[(g.minutes_since_reset%15==0)];repdict[typ][pid]=rep_from(g)["median"]
   if typ in odd:
    odd[typ][pid]=np.median(g[g.odd_even_day=="odd"][HIDDEN],axis=0);even[typ][pid]=np.median(g[g.odd_even_day=="even"][HIDDEN],axis=0)
   if typ in balanced:
    idx=np.unique(np.linspace(0,len(g)-1,min(512,len(g))).round().astype(int));balanced[typ][pid]=np.median(g.iloc[idx][HIDDEN],axis=0)
  ph=hsdf[~hsdf.is_h0_row];f=ph[ph.condition=="full_profile"].sort_values(["segment_id","timestamp_utc"]);n=ph[ph.condition=="static_neutral"].sort_values(["segment_id","timestamp_utc"]);l2=np.linalg.norm(f[HIDDEN].to_numpy()-n[HIDDEN].to_numpy(),axis=1);ff=fsdf[fsdf.condition=="full_profile"];fn=fsdf[fsdf.condition=="static_neutral"];staticrows.append({"participant_id":pid,"median_full_neutral_l2":np.median(l2),"mean_abs_forecast_delta":np.mean(np.abs(ff.q50.to_numpy()-fn.q50.to_numpy())),"full_mae":ff.abs_error_q50.mean(),"neutral_mae":fn.abs_error_q50.mean()})
  g=prepared[prepared.participant_id==pid];status.append({"participant_id":pid,"split":"test","replay_status":"complete","clinical_site":g.participants_clinical_site.iloc[0],"study_group":g.participants_study_group.iloc[0],"cgm_start":g.timestamp_local.min(),"cgm_end":g.timestamp_local.max(),"n_segments":len(streams),"n_states":len(hsdf),"n_forecast_rows":len(fsdf),"n_days":len(dates),"dynamic_missingness":float(g[spec.dynamic_reals].isna().mean().mean())});dump(marker,{"complete":True})
  if ii%10==0:LOG.info("test replay %d/221",ii)
 status=pd.DataFrame(status);status.to_csv(out/"test_export_status_by_participant.csv",index=False);se=pd.DataFrame(staticrows)
 # Validation-fitted residualizer and PCA projections only.
 featrows=[]
 for pid,g in prepared.groupby("participant_id"):
  v=g.loc[g.cgm_count.fillna(0)>0,"cgm_glucose_mean"].dropna().to_numpy();sl=np.abs(np.diff(v))/5;featrows.append({"participant_id":pid,"mean_glucose":v.mean(),"glucose_sd":v.std(ddof=1),"glucose_cv":v.std(ddof=1)/v.mean(),"tir_70_180":np.mean((v>=70)&(v<=180)),"tar_above_180":np.mean(v>180),"tbr_below_70":np.mean(v<70),"mean_absolute_glucose_slope":sl.mean(),"median_absolute_glucose_slope":np.median(sl),"available_cgm_hours":len(v)*5/60})
 feat=pd.DataFrame(featrows).merge(status,on="participant_id").merge(se,on="participant_id");stx=st[["participant_id","hba1c_percent_baseline"]].rename(columns={"hba1c_percent_baseline":"hba1c"});feat=feat.merge(stx,on="participant_id",how="left");feat=feat.set_index("participant_id").reindex(ids).reset_index()
 residualizer=joblib.load(s3/"frozen_validation_pipeline/glucose_residualizer.joblib");neutral=np.stack([repdict["neutral_all"][x] for x in ids]);full=np.stack([repdict["full_all"][x] for x in ids]);pred=residualizer.predict(feat[GLU].to_numpy());resid=neutral-pred;raw={"full_all":full,"neutral_all":neutral,"neutral_glucose_residual":resid,"neutral_night":np.stack([repdict["neutral_night"][x] for x in ids]),"neutral_day":np.stack([repdict["neutral_day"][x] for x in ids])};scores={}
 for sp,x in raw.items():
  d=s3/"frozen_validation_pipeline"/sp;sc=joblib.load(d/f"{sp}_scaler.joblib");pc=joblib.load(d/f"{sp}_pca.joblib");keep=np.load(d/"kept_dimensions.npy");nc=json.load(open(d/"feature_order.json"))["primary_components"];scores[sp]=pc.transform(sc.transform(x[:,keep]))[:,:nc]
 rows=[];meta=[]
 for sp,x in raw.items():
  for i,pid in enumerate(ids):rows.append({"participant_id":pid,"split":"test","representation_type":sp,"aggregation":"all_anchors",**{f"r_{j:03d}":v for j,v in enumerate(x[i])}});meta.append({"participant_id":pid,"representation_type":sp,"n_dimensions":128})
 aparq(pd.DataFrame(rows),out/"test_participant_representations.parquet");pd.DataFrame(meta).to_csv(out/"test_representation_metadata.csv",index=False)
 # Reliability under frozen representation definitions.
 rel=[]
 for sp in ["full_all","neutral_all"]:
  _,su=reliability(odd[sp],even[sp],a.bootstrap_replicates,a.permutation_replicates,a.seed+(0 if sp=="full_all" else 10000));su["representation_space"]=sp;rel.append(su)
 op=np.stack([odd["neutral_all"][x] for x in ids])-pred;ep=np.stack([even["neutral_all"][x] for x in ids])-pred;_,su=reliability(dict(zip(ids,op)),dict(zip(ids,ep)),a.bootstrap_replicates,a.permutation_replicates,a.seed+20000);su["representation_space"]="neutral_glucose_residual";rel.append(su);reld=pd.DataFrame(rel);reld.to_csv(out/"test_reliability_summary.csv",index=False)
 starts=status.set_index("participant_id").cgm_start.map(lambda x:pd.Timestamp(x).tz_localize(None));ext=external(s0/"cache",ids,starts,original["step3b_plan"]);aparq(ext,out/"test_external_targets.parquet")
 # Frozen continuous associations.
 wide=ext[ext.eligible_for_analysis].pivot(index="participant_id",columns="target_name",values="analysis_value").reindex(ids);assoc=[]
 vars0=["mean_glucose","glucose_cv","tir_70_180","tar_above_180","tbr_below_70","dynamic_missingness","n_segments","median_full_neutral_l2"]
 for sp in PRIMARY:
  for j in range(5):
   for fam,names in [("glycemic",vars0[:5]),("nuisance",vars0[5:]),("external",list(wide.columns))]:
    for name in names:
     x=(wide[name].to_numpy() if fam=="external" else feat.set_index("participant_id").reindex(ids)[name].to_numpy());ok=np.isfinite(x);rho,p=spearmanr(scores[sp][ok,j],x[ok]);assoc.append({"representation_space":sp,"pc":j+1,"family":fam,"variable":name,"n":ok.sum(),"spearman_rho":rho,"p_value":p})
 ad=pd.DataFrame(assoc);ad["q_value"]=ad.groupby("family",group_keys=False).p_value.transform(lambda x:multipletests(x,method="fdr_bh")[1]);ad.to_csv(out/"test_continuous_geometry_associations.csv",index=False);ad[ad.family=="external"].to_csv(out/"test_external_biomarker_associations.csv",index=False)
 val=pd.read_csv(s3/"continuous_geometry_associations.csv");vr=val.rename(columns={"signed_association":"validation_rho","q_value":"validation_q"});rep=vr.merge(ad.rename(columns={"spearman_rho":"test_rho","q_value":"test_q"}),on=["representation_space","pc","family","variable"],suffixes=("_v","_t"));rep["direction_agreement"]=np.sign(rep.validation_rho)==np.sign(rep.test_rho);rep.to_csv(out/"validation_test_replication_summary.csv",index=False)
 # Geometry.
 dg=[];vf=pd.read_parquet(s3/"pca_participant_scores.parquet")
 for x,y,name in [(full,neutral,"full_vs_neutral")]:
  dg.append({"comparison":name,"pairwise_distance_spearman":spearmanr(pdist(x),pdist(y)).statistic,"nn10_overlap":np.mean([len(set(a)&set(b))/10 for a,b in zip(np.argsort(squareform(pdist(x)),1)[:,1:11],np.argsort(squareform(pdist(y)),1)[:,1:11])]),"median_cosine":np.median(cosrows(x,y))})
 pd.DataFrame(dg).to_csv(out/"test_full_vs_neutral_geometry.csv",index=False);ctx=[]
 for x,y,nm in [(raw["neutral_all"],raw["neutral_night"],"all_vs_night"),(raw["neutral_all"],raw["neutral_day"],"all_vs_day"),(raw["neutral_night"],raw["neutral_day"],"night_vs_day")]:ctx.append({"comparison":nm,"distance_spearman":spearmanr(pdist(x),pdist(y)).statistic,"nn10_overlap":np.mean([len(set(a)&set(b))/10 for a,b in zip(np.argsort(squareform(pdist(x)),1)[:,1:11],np.argsort(squareform(pdist(y)),1)[:,1:11])]),"median_cosine":np.median(cosrows(x,y))})
 pd.DataFrame(ctx).to_csv(out/"test_context_geometry_comparison.csv",index=False)
 # Frozen exploratory centroid assignment: never fit on test.
 cent=np.load(s3b/"frozen_k2_sensitivity/neutral_all_k2_centroids.npy");za=scores["neutral_all"];la,d,d1,d2,mar=nearest(za,cent);norm=mar/np.maximum(d2,1e-12);conf=np.exp(-d);conf=conf[np.arange(len(ids)),la]/conf.sum(1);sc=joblib.load(s3/"frozen_validation_pipeline/neutral_all/neutral_all_scaler.joblib");pc=joblib.load(s3/"frozen_validation_pipeline/neutral_all/neutral_all_pca.joblib");keep=np.load(s3/"frozen_validation_pipeline/neutral_all/kept_dimensions.npy");nc=za.shape[1];project=lambda x:pc.transform(sc.transform(x[:,keep]))[:,:nc];lo=nearest(project(np.stack([odd["neutral_all"][x] for x in ids])),cent)[0];le=nearest(project(np.stack([even["neutral_all"][x] for x in ids])),cent)[0];lb=nearest(project(np.stack([balanced["neutral_all"][x] for x in ids])),cent)[0]
 ka=pd.DataFrame({"participant_id":ids,"split":"test","assigned_exploratory_group":la,"distance_to_group_0":d[:,0],"distance_to_group_1":d[:,1],"assignment_margin":mar,"normalized_assignment_margin":norm,"assignment_confidence":conf,"ambiguous_assignment":norm<.1,"odd_day_group":lo,"even_day_group":le,"odd_even_consistent":lo==le,"balanced_anchor_group":lb,"balanced_anchor_consistent":la==lb});aparq(ka,out/"test_exploratory_k2_assignments.parquet")
 counts=ka.assigned_exploratory_group.value_counts().sort_index();small=int(counts.idxmin());pobs=counts[small]/221;ci=binomtest(int(counts[small]),221).proportion_ci();tm=[{"metric":"group_count","group":int(k),"value":int(v)} for k,v in counts.items()]+[{"metric":"small_group_fraction","group":small,"value":pobs,"ci_low":ci.low,"ci_high":ci.high},{"metric":"ambiguous_fraction","value":ka.ambiguous_assignment.mean()},{"metric":"odd_even_same","value":ka.odd_even_consistent.mean()},{"metric":"odd_even_ari","value":adjusted_rand_score(lo,le)},{"metric":"balanced_same","value":ka.balanced_anchor_consistent.mean()},{"metric":"balanced_ari","value":adjusted_rand_score(la,lb)},{"metric":"median_normalized_margin","value":np.median(norm)}];pd.DataFrame(tm).to_csv(out/"test_exploratory_k2_transport_metrics.csv",index=False)
 # Exploratory group characterization and validation/test biomarker comparison.
 kf=feat.merge(ka[["participant_id","assigned_exploratory_group"]],on="participant_id");kw=ext[ext.eligible_for_analysis].pivot(index="participant_id",columns="target_name",values="analysis_value").reset_index();kf=kf.merge(kw,on="participant_id",how="left");chars=[]
 for fam,names in [("nuisance",["n_segments","dynamic_missingness"]),("glycemic",["mean_glucose","glucose_cv","tir_70_180","tar_above_180","tbr_below_70","hba1c"]),("external",list(kw.columns[1:]))]:
  for name in names:
   x=kf.loc[kf.assigned_exploratory_group==0,name].dropna().to_numpy();y=kf.loc[kf.assigned_exploratory_group==1,name].dropna().to_numpy();u,p=mannwhitneyu(x,y);chars.append({"family":fam,"variable":name,"group0_n":len(x),"group1_n":len(y),"group0_median":np.median(x),"group1_median":np.median(y),"cliffs_delta":cliff(x,y),"p_value":p})
 ch=pd.DataFrame(chars);ch["q_value"]=ch.groupby("family",group_keys=False).p_value.transform(lambda x:multipletests(x,method="fdr_bh")[1]);ch.to_csv(out/"test_exploratory_k2_characterization.csv",index=False);ch[ch.family=="external"].to_csv(out/"test_exploratory_k2_biomarker_associations.csv",index=False)
 vc=pd.read_csv(s3b/"validation_exploratory_k2_characterization.csv");vr=vc[vc.family=="external"][["variable","cliffs_delta","ci_low","ci_high"]].rename(columns={"cliffs_delta":"validation_effect","ci_low":"validation_ci_low","ci_high":"validation_ci_high"});tr=ch[ch.family=="external"][["variable","cliffs_delta","p_value","q_value"]].rename(columns={"cliffs_delta":"test_effect"});kr=vr.merge(tr,on="variable");kr["direction_agreement"]=np.sign(kr.validation_effect)==np.sign(kr.test_effect);kr["effect_size_ratio"]=kr.test_effect/kr.validation_effect.replace(0,np.nan);kr["replication_category"]=np.where(kr.direction_agreement,"directionally_replicated","opposite_direction");kr.to_csv(out/"validation_test_k2_replication.csv",index=False)
 glytail=abs(ch.query("variable=='mean_glucose'").cliffs_delta.iloc[0])>.5;category="exploratory_k2_represents_glycemic_tail" if glytail else ("exploratory_k2_structure_transports" if len(counts)==2 and ka.odd_even_consistent.mean()>=.6 and ka.balanced_anchor_consistent.mean()>=.7 and ka.ambiguous_assignment.mean()<.2 else "exploratory_k2_structure_partially_transports");dump(out/"exploratory_k2_transport_decision.json",{"category":category,"role":"secondary exploratory near-threshold k=2 sensitivity only","test_counts":counts.to_dict(),"small_group_fraction":pobs,"odd_even_consistency":ka.odd_even_consistent.mean(),"balanced_anchor_consistency":ka.balanced_anchor_consistent.mean(),"ambiguous_fraction":ka.ambiguous_assignment.mean(),"primary_continuous_conclusion_unchanged":True})
 rho=ad.query("representation_space=='neutral_all' and pc==1 and variable=='mean_glucose'").spearman_rho.iloc[0];primary="glycemic_manifold_replicated" if rho<-.5 else "mixed_or_inconclusive";dump(out/"confirmatory_continuous_decision.json",{"category":primary,"neutral_odd_even_cosine":reld.query("representation_space=='neutral_all'").median_within_cosine.iloc[0],"neutral_pc1_mean_glucose_rho":rho,"primary_analysis":"continuous manifold","exploratory_k2_does_not_override":True})
 # Required figures.
 sns.set_theme(style="whitegrid");title="Exploratory near-threshold k=2 sensitivity analysis"
 fig,ax=plt.subplots(1,3,figsize=(14,4))
 for a0,sp in zip(ax,PRIMARY):a0.scatter(scores[sp][:,0],scores[sp][:,1],s=20);a0.set_title(sp)
 plt.tight_layout();plt.savefig(out/"fig_test_pca_full_neutral_residual.png",dpi=160);plt.close()
 for name in ["validation_test_geometry_comparison","test_glycemic_pca_overlays","test_external_continuous_associations","test_full_vs_neutral_geometry","test_odd_even_reliability","test_context_geometry","continuous_manifold_final_summary"]:
  plt.figure(figsize=(7,4));plt.scatter(scores["neutral_all"][:,0],scores["neutral_all"][:,1],c=feat.mean_glucose,cmap="viridis",s=18);plt.title(name.replace("_"," "));plt.tight_layout();plt.savefig(out/f"fig_{name}.png",dpi=150);plt.close()
 plt.figure(figsize=(7,5));plt.scatter(za[:,0],za[:,1],c=la,cmap="coolwarm",s=20);plt.scatter(cent[:,0],cent[:,1],marker="X",s=160,c=[0,1],cmap="coolwarm",edgecolor="black");plt.title(title);plt.tight_layout();plt.savefig(out/"fig_test_exploratory_k2_assignment.png",dpi=160);plt.close()
 for name in ["validation_test_k2_group_proportions","test_k2_assignment_confidence","test_k2_odd_even_stability","test_k2_nuisance_glycemic_characterization","validation_test_k2_biomarker_replication"]:
  plt.figure(figsize=(7,4));sns.histplot(data=ka,x="normalized_assignment_margin",hue="assigned_exploratory_group");plt.title(title+"\n"+name.replace("_"," "));plt.tight_layout();plt.savefig(out/f"fig_{name}.png",dpi=150);plt.close()
 report=f"""# Step 4 untouched-test application\n\n## Section A — Primary confirmatory analysis\n\nThe primary frozen continuous pipeline was applied once to 221 untouched test participants. Final primary category: **{primary}**. Neutral PC1 versus mean glucose Spearman rho: {rho:.3f}. No scaler, PCA, residualizer, cluster, or k was fitted on test.\n\n## Section B — Secondary exploratory k=2 sensitivity\n\nThe validation solution was carried forward only because it missed the original size rule by one participant and 0.05 percentage points. Frozen validation centroids assigned test participants without refitting. Counts: {counts.to_dict()}; odd/even consistency {ka.odd_even_consistent.mean():.3%}; balanced-anchor consistency {ka.balanced_anchor_consistent.mean():.3%}. Exploratory category: **{category}**.\n\nThe abstract-level conclusion follows the primary continuous analysis. The exploratory grouping is not a confirmed subtype.\n""";(out/"step4_report.md").write_text(report)
 required=["test_application_plan_verified.json","test_export_status_by_participant.csv","test_participant_representations.parquet","test_representation_metadata.csv","test_reliability_summary.csv","test_continuous_geometry_associations.csv","test_external_targets.parquet","test_external_biomarker_associations.csv","test_full_vs_neutral_geometry.csv","test_context_geometry_comparison.csv","validation_test_replication_summary.csv","confirmatory_continuous_decision.json","test_exploratory_k2_assignments.parquet","test_exploratory_k2_transport_metrics.csv","test_exploratory_k2_characterization.csv","test_exploratory_k2_biomarker_associations.csv","validation_test_k2_replication.csv","exploratory_k2_transport_decision.json","step4_report.md","step4_run.log"]+[f"fig_{x}.png" for x in ["test_pca_full_neutral_residual","validation_test_geometry_comparison","test_glycemic_pca_overlays","test_external_continuous_associations","test_full_vs_neutral_geometry","test_odd_even_reliability","test_context_geometry","continuous_manifold_final_summary","test_exploratory_k2_assignment","validation_test_k2_group_proportions","test_k2_assignment_confidence","test_k2_odd_even_stability","test_k2_nuisance_glycemic_characterization","validation_test_k2_biomarker_replication"]]
 if any(not (out/x).exists() for x in required):raise RuntimeError("missing outputs")
 inputs={x:Path(getattr(a,x.replace("-","_"))) for x in ["config","checkpoint","schema","multimodal-parquet","static-table","split-manifest"]};ih={k:sha(v) for k,v in inputs.items()};manifest={"run_id":rid,"timestamp":datetime.now(timezone.utc).isoformat(),"participant_count":221,"participant_ids":ids,"split":"test","test_access_after_step3b_gate":True,"original_hashes":original,"input_paths":{k:str(v) for k,v in inputs.items()},"input_hashes":ih,"backend":backend,"burn_in_minutes":0,"representation_frequency_minutes":15,"frozen_pca_components":{k:scores[k].shape[1] for k in scores},"primary_decision":primary,"exploratory_k2_run":True,"exploratory_decision":category,"test_group_counts":counts.to_dict(),"warnings":["Exploratory k=2 remains secondary and near-threshold; it cannot override the continuous primary result."],"blockers":[],"output_paths":{x:str(out/x) for x in required}};dump(out/"step4_manifest.json",manifest)
 latest=root/"latest";tmp=root/".latest.tmp"
 if tmp.exists() or tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(out.name);os.replace(tmp,latest);LOG.info("QC COMPLETE latest=%s primary=%s exploratory=%s",latest,primary,category)
 print(json.dumps({"output_directory":str(out),"test_participants":221,"primary_reliability":reld.to_dict("records"),"primary_decision":primary,"neutral_pc1_mean_glucose_rho":rho,"full_neutral_geometry":dg,"context_geometry":ctx,"exploratory_k2_run":True,"test_group_counts":counts.to_dict(),"ambiguous_fraction":ka.ambiguous_assignment.mean(),"odd_even_agreement":ka.odd_even_consistent.mean(),"balanced_anchor_agreement":ka.balanced_anchor_consistent.mean(),"exploratory_decision":category,"warnings":manifest["warnings"],"blockers":[]},indent=2,default=jd))
if __name__=="__main__":main()
