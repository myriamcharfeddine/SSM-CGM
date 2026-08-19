#!/usr/bin/env python3
"""Validation-only hidden-state clustering and frozen clinical characterization."""
from __future__ import annotations
import argparse,hashlib,json,logging,math,os,platform,random,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist,squareform
from scipy.stats import chi2_contingency,kruskal,spearmanr
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,balanced_accuracy_score,calinski_harabasz_score,cohen_kappa_score,davies_bouldin_score,f1_score,normalized_mutual_info_score,roc_auc_score,silhouette_score
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,label_binarize
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
LOG=logging.getLogger('step3'); H=[f'r_{i:03d}' for i in range(128)]; IQR=[f'iqr_{i:03d}' for i in range(128)]; MAD=[f'mad_{i:03d}' for i in range(128)]
PRIMARY=['full_all','neutral_all','neutral_glucose_residual']; PCA_SPACES=PRIMARY+['neutral_night','neutral_day']
TARGETS={'natriuretic_peptide_b_prohormon':3029187,'c_reactive_protein_i':3010156,'bun_creatinine_ratio':4112223}
GLUCOSE_RESID=['mean_glucose','glucose_cv','tir_70_180','tar_above_180','tbr_below_70']
GLYCEMIC=GLUCOSE_RESID+['glucose_sd','mean_absolute_glucose_slope','median_absolute_glucose_slope','glucose_range','available_cgm_hours','hba1c']
NUISANCE_CONT=['total_clean_hours','number_of_segments','median_segment_hours','minimum_segment_hours','fraction_states_within_30min_of_reset','fraction_states_within_3h_of_reset','hr_availability','activity_availability','respiratory_rate_availability','oxygen_saturation_availability','sleep_availability','dynamic_missingness','number_of_distinct_days','nighttime_valid_hours','daytime_valid_hours','median_full_neutral_l2','median_full_neutral_cosine','mean_abs_forecast_delta']
def args():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('step0-dir','step1-dir','step2-dir','aireadi-root','output-root'):p.add_argument('--'+x,required=True)
 p.add_argument('--split',default='validation');p.add_argument('--primary-pca-variance',type=float,default=.90);p.add_argument('--candidate-k',type=int,nargs='+',default=[2,3,4,5]);p.add_argument('--consensus-iterations',type=int,default=500);p.add_argument('--subsample-fraction',type=float,default=.8);p.add_argument('--stability-iterations',type=int,default=500);p.add_argument('--bootstrap-replicates',type=int,default=2000);p.add_argument('--permutation-replicates',type=int,default=1000);p.add_argument('--seed',type=int,default=42);p.add_argument('--run-id');p.add_argument('--clinical-cache-dir');p.add_argument('--n-jobs',type=int,default=-1);p.add_argument('--overwrite',action='store_true');p.add_argument('--resume',dest='resume',action='store_true',default=True);p.add_argument('--no-resume',dest='resume',action='store_false');return p.parse_args()
def setup(p):
 LOG.setLevel(logging.INFO);fmt=logging.Formatter('%(asctime)sZ %(levelname)s %(message)s','%Y-%m-%dT%H:%M:%S');hs=[logging.FileHandler(p),logging.StreamHandler(sys.stdout)]
 for h in hs:h.setFormatter(fmt)
 LOG.handlers[:]=hs
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jd(x):
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,)):return None if not np.isfinite(x) else float(x)
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(pd.Timestamp,datetime,Path)):return str(x)
 raise TypeError(type(x).__name__)
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,default=jd)+'\n');os.replace(t,p)
def aparq(d,p):
 p=Path(p);t=p.with_suffix('.tmp.parquet');d.to_parquet(t,index=False,compression='zstd');os.replace(t,p)
def bh(s):
 a=pd.to_numeric(s,errors='coerce').to_numpy();o=np.full(len(a),np.nan);z=np.isfinite(a)
 if z.any():o[z]=multipletests(a[z],method='fdr_bh')[1]
 return o
def boot(v,n,rng,fun=np.median):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if not len(v):return np.nan,np.nan
 z=np.array([fun(rng.choice(v,len(v),replace=True)) for _ in range(n)]);return tuple(np.quantile(z,[.025,.975]))
def eps2(gs):
 gs=[np.asarray(x,float)[np.isfinite(x)] for x in gs];gs=[x for x in gs if len(x)]
 if len(gs)<2:return np.nan,np.nan
 h,p=kruskal(*gs);n=sum(map(len,gs));k=len(gs);return max(0,(h-k+1)/max(n-k,1)),p
def jaccards(a,b):
 ca,cb=np.unique(a),np.unique(b);s=np.zeros((len(ca),len(cb)))
 for i,x in enumerate(ca):
  for j,y in enumerate(cb):
   u=(a==x)|(b==y);s[i,j]=((a==x)&(b==y)).sum()/max(u.sum(),1)
 r,c=linear_sum_assignment(-s);return s[r,c].tolist()
def align(ref,lab):
 ca,cb=np.unique(ref),np.unique(lab);m=np.array([[(ref[lab==y]==x).sum() for y in cb] for x in ca]);r,c=linear_sum_assignment(-m);mp={cb[j]:ca[i] for i,j in zip(r,c)};return np.array([mp.get(x,x) for x in lab])
def cosrows(a,b):return np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
def hopkins(x,seed):
 rng=np.random.default_rng(seed);n=min(max(20,len(x)//2),len(x)-1);idx=rng.choice(len(x),n,replace=False);u=rng.uniform(x.min(0),x.max(0),(n,x.shape[1]));d=squareform(pdist(x));w=np.partition(d[idx],1,axis=1)[:,1];du=np.sqrt(((u[:,None]-x[None])**2).sum(2)).min(1);return float(du.sum()/max(du.sum()+w.sum(),1e-12))
def plan(a):return {'primary_representation_spaces':PRIMARY,'secondary_representation_spaces':['neutral_night','neutral_day','neutral_night_clock_sensitivity'],'sensitivities':['balanced anchors','IQR','MAD','robust scaling','PCA 80/95/10/20','Ward'],'glycemic_residualization_covariates':GLUCOSE_RESID,'residualization_method':'5-fold cross-fitted multioutput ridge; nested 3-fold alpha selection','residualization_folds':5,'residualizer_alpha_grid':[.01,.1,1,10,100,1000],'scaling_method':'StandardScaler; remove zero-variance dimensions','pca_candidate_rules':[.8,.9,.95,10,20],'primary_pca_rule':'minimum PCs >=90% variance','clustering_algorithms':['consensus k-means primary','Ward sensitivity'],'candidate_k':a.candidate_k,'consensus':{'iterations':a.consensus_iterations,'fraction':a.subsample_fraction,'kmeans_n_init':50,'final':'average linkage on 1-consensus','PAC':[.1,.9]},'cluster_size_threshold':{'minimum_n':20,'minimum_fraction':.08},'stability_criteria':{'median_ari':.6,'median_min_jaccard':.6,'pac':.2,'odd_even_ari':.4,'odd_even_same':.6,'silhouette':.1,'assignment_probability_ge_80':.7},'selection_hierarchy':['median subsample ARI','PAC if within .02','odd/even ARI','silhouette','size balance','smaller k'],'odd_even_method':'frozen scaler/PCA and nearest frozen consensus centroid; refit sensitivity','nuisance_variables':['clinical_site','study_group']+NUISANCE_CONT,'glycemic_variables':GLYCEMIC,'external_target_names':list(TARGETS),'external_target_extraction_rule':'earliest valid <= CGM start, else nearest valid after; primary +/-180d, sensitivity +/-90d','clinical_association_models':'Kruskal-Wallis and OLS HC3 baseline versus cluster-extended','fdr_families':['nuisance','glycemic','primary neutral_all external 3-target'],'context_comparison_rules':'project night/day through frozen neutral_all scaler/PCA/centroids','continuous_manifold_fallback':'PC1-PC5 nuisance, glycemic, external associations','random_seeds':{'master':a.seed},'no_test_constraint':True,'external_targets_forbidden_until_stage_a_lock':True}
def audit(reps,valid):
 rows=[]
 for typ in ['full_all','neutral_all','neutral_night','neutral_day','neutral_night_clock_sensitivity']:
  for var in ['all_anchors','balanced_anchors']:
   z=reps[(reps.representation_type==typ)&(reps.balanced_anchor_variant==var)];v=z[H].to_numpy(float);dup=z.participant_id.astype(str).duplicated().sum();bad=int((~np.isfinite(v)).any(1).sum());ok=len(z)==239 and not dup and not bad and set(z.participant_id.astype(str))==valid
   rows.append({'representation_type':typ,'aggregation_variant':var,'burn_in_minutes':z.burn_in_minutes.iloc[0] if len(z) else np.nan,'n_expected':239,'n_available':len(z),'n_missing':239-len(z),'n_duplicate':dup,'n_nonfinite':bad,'hidden_dimension':128,'median_n_anchors':z.n_anchors.median(),'median_n_days':z.n_distinct_days.median(),'median_n_segments':z.n_segments.median(),'eligibility_rule':'Step 2 frozen context rule','qc_status':'PASS' if ok else 'FAIL','notes':'phenotype columns absent; validation IDs checked'})
 return pd.DataFrame(rows)
def gly_nuis(panel_path,static_path,status,se,ids):
 cols=['participant_id','timestamp_local','cgm_glucose_mean','cgm_count','heart_rate_mean','activity_steps_per_min','respiratory_rate_mean','oxygen_saturation_mean','sleep_stage_light','sleep_stage_deep','sleep_stage_rem'];p=pd.read_parquet(panel_path,columns=cols,filters=[('participant_id','in',ids)]);p.participant_id=p.participant_id.astype(str)
 if set(p.participant_id)-set(ids):raise RuntimeError('non-validation panel row loaded')
 rows=[]
 for pid,g in p.groupby('participant_id',sort=True):
  g=g.sort_values('timestamp_local');v=pd.to_numeric(g.cgm_glucose_mean,errors='coerce');v=v[g.cgm_count.fillna(0).gt(0)&v.notna()];x=v.to_numpy(float);sl=np.abs(np.diff(x))/5;av=lambda c:float(g[c].notna().mean())
  rows.append({'participant_id':pid,'mean_glucose':x.mean(),'glucose_sd':x.std(ddof=1),'glucose_cv':x.std(ddof=1)/x.mean(),'tir_70_180':np.mean((x>=70)&(x<=180)),'tar_above_180':np.mean(x>180),'tbr_below_70':np.mean(x<70),'mean_absolute_glucose_slope':sl.mean(),'median_absolute_glucose_slope':np.median(sl),'glucose_range':x.max()-x.min(),'available_cgm_hours':len(x)*5/60,'hr_availability':av('heart_rate_mean'),'activity_availability':av('activity_steps_per_min'),'respiratory_rate_availability':av('respiratory_rate_mean'),'oxygen_saturation_availability':av('oxygen_saturation_mean'),'sleep_availability':float(g[['sleep_stage_light','sleep_stage_deep','sleep_stage_rem']].notna().any(axis=1).mean())})
 out=pd.DataFrame(rows);st=pd.read_parquet(static_path,columns=['participant_id','participants_age','demo_sex_at_birth','hba1c_percent_baseline']);st.participant_id=st.participant_id.astype(str);st=st[st.participant_id.isin(ids)].drop_duplicates('participant_id').rename(columns={'participants_age':'age','demo_sex_at_birth':'sex','hba1c_percent_baseline':'hba1c'});s=status.copy();s.participant_id=s.participant_id.astype(str)
 q=s.segment_boundaries.map(lambda z:[r['n_steps']*5/60 for r in json.loads(z)]);s['median_segment_hours']=q.map(np.median);s['minimum_segment_hours']=q.map(np.min);s['fraction_states_within_30min_of_reset']=s.n_canonical_segments*6/s.n_valid_dynamic_rows;s['fraction_states_within_3h_of_reset']=s.n_canonical_segments*36/s.n_valid_dynamic_rows;s=s.rename(columns={'total_canonical_duration_hours':'total_clean_hours','n_canonical_segments':'number_of_segments','n_local_calendar_days':'number_of_distinct_days','valid_nighttime_hours':'nighttime_valid_hours','valid_daytime_hours':'daytime_valid_hours'});keep=['participant_id','clinical_site','study_group','total_clean_hours','number_of_segments','median_segment_hours','minimum_segment_hours','fraction_states_within_30min_of_reset','fraction_states_within_3h_of_reset','dynamic_missingness','number_of_distinct_days','nighttime_valid_hours','daytime_valid_hours'];e=se.rename(columns={'n_segments':'static_effect_n_segments'}).drop(columns=['clinical_site','study_group'],errors='ignore');return out.merge(s[keep],on='participant_id',validate='one_to_one').merge(st,on='participant_id',how='left',validate='one_to_one').merge(e,on='participant_id',how='left',validate='one_to_one')
def residualize(r,x,out,seed):
 al=[.01,.1,1,10,100,1000];outer=KFold(5,shuffle=True,random_state=seed);pred=np.zeros_like(r);foldid=np.zeros(len(r),int);sels=[];models={}
 for fold,(tr,te) in enumerate(outer.split(x)):
  inner=KFold(3,shuffle=True,random_state=seed+fold);loss=[]
  for a in al:
   z=[]
   for u,v in inner.split(tr):sc=StandardScaler().fit(x[tr[u]]);fit=Ridge(alpha=a).fit(sc.transform(x[tr[u]]),r[tr[u]]);z.append(np.mean((r[tr[v]]-fit.predict(sc.transform(x[tr[v]])))**2))
   loss.append(np.mean(z))
  a=al[int(np.argmin(loss))];sels.append(a);sc=StandardScaler().fit(x[tr]);fit=Ridge(alpha=a).fit(sc.transform(x[tr]),r[tr]);pred[te]=fit.predict(sc.transform(x[te]));foldid[te]=fold;models[fold]=(sc,fit)
 res=r-pred;cv=KFold(5,shuffle=True,random_state=seed);loss=[]
 for a in al:
  z=[]
  for tr,te in cv.split(x):sc=StandardScaler().fit(x[tr]);fit=Ridge(alpha=a).fit(sc.transform(x[tr]),r[tr]);z.append(np.mean((r[te]-fit.predict(sc.transform(x[te])))**2))
  loss.append(np.mean(z))
 fa=al[int(np.argmin(loss))];final=make_pipeline(StandardScaler(),Ridge(alpha=fa)).fit(x,r);joblib.dump(final,out/'frozen_validation_pipeline'/'glucose_residualizer.joblib');rows=[]
 for j in range(128):
  den=np.sum((r[:,j]-r[:,j].mean())**2);rows.append({'metric_type':'dimension','hidden_dimension':j,'selected_alpha':np.median(sels),'cross_fitted_r2':1-np.sum((r[:,j]-pred[:,j])**2)/max(den,1e-12),'observed_predicted_correlation':np.corrcoef(r[:,j],pred[:,j])[0,1],'residual_variance':np.var(res[:,j],ddof=1),'variance_removed':np.var(r[:,j],ddof=1)-np.var(res[:,j],ddof=1),'mean_residual':res[:,j].mean(),'residual_sd':res[:,j].std(ddof=1)})
 d0,d1=pdist(r),pdist(res);rows.append({'metric_type':'geometry','hidden_dimension':-1,'median_original_residual_cosine':np.median(cosrows(r,res)),'median_original_residual_l2':np.median(np.linalg.norm(r-res,axis=1)),'pairwise_distance_spearman':spearmanr(d0,d1).statistic,'final_all_validation_alpha':fa});return res,pred,pd.DataFrame(rows),{'fold_id':foldid,'models':models,'selected_alpha_by_fold':sels,'final_alpha':fa}
def fitpca(name,x,out,thr):
 sd=x.std(0);keep=sd>1e-12;sc=StandardScaler().fit(x[:,keep]);z=sc.transform(x[:,keep]);pc=PCA(svd_solver='full').fit(z);scores=pc.transform(z);cum=np.cumsum(pc.explained_variance_ratio_);n90=int(np.searchsorted(cum,thr)+1);d=out/'frozen_validation_pipeline'/name;d.mkdir(parents=True,exist_ok=True);joblib.dump(sc,d/f'{name}_scaler.joblib');joblib.dump(pc,d/f'{name}_pca.joblib');np.save(d/'kept_dimensions.npy',np.where(keep)[0]);dump(d/'feature_order.json',{'source_dimensions':H,'kept_indices':np.where(keep)[0].tolist(),'removed_indices':np.where(~keep)[0].tolist(),'primary_components':n90});meta={'space':name,'n_input':128,'n_kept':keep.sum(),'n_removed':(~keep).sum(),'n80':np.searchsorted(cum,.8)+1,'n90':n90,'n95':np.searchsorted(cum,.95)+1,'pc1':pc.explained_variance_ratio_[0],'pc2':pc.explained_variance_ratio_[1],'pc3':pc.explained_variance_ratio_[2],'participation_ratio':pc.explained_variance_.sum()**2/np.sum(pc.explained_variance_**2),'hopkins':hopkins(scores[:,:n90],42)};return scores,sc,pc,n90,keep,meta
def consensus(x,k,it,frac,seed):
 n=len(x);rng=np.random.default_rng(seed);co=np.zeros((n,n),np.uint16);tog=np.zeros((n,n),np.uint16);take=max(k*2,int(n*frac))
 for i in range(it):
  idx=np.sort(rng.choice(n,take,replace=False));lab=KMeans(k,n_init=50,random_state=seed+i).fit_predict(x[idx]);co[np.ix_(idx,idx)]+=1
  for c in range(k):q=idx[lab==c];tog[np.ix_(q,q)]+=1
 m=np.divide(tog,co,out=np.zeros_like(tog,dtype=float),where=co>0);np.fill_diagonal(m,1);lab=AgglomerativeClustering(k,metric='precomputed',linkage='average').fit_predict(1-m);prob=np.array([m[i,lab==lab[i]].mean() for i in range(n)]);cent=np.stack([x[lab==c].mean(0) for c in range(k)]);return lab,m,prob,cent
def cmetrics(x,l,m,p):
 k=len(np.unique(l));sz=np.bincount(l,minlength=k);up=m[np.triu_indices(len(l),1)];within=np.concatenate([m[np.ix_(l==c,l==c)][np.triu_indices((l==c).sum(),1)] for c in range(k)]);between=m[l[:,None]!=l[None,:]];return {'silhouette':silhouette_score(x,l),'calinski_harabasz':calinski_harabasz_score(x,l),'davies_bouldin':davies_bouldin_score(x,l),'consensus_pac':np.mean((up>.1)&(up<.9)),'consensus_within_mean':within.mean(),'consensus_between_mean':between.mean(),'minimum_cluster_size':sz.min(),'maximum_cluster_size':sz.max(),'minimum_cluster_fraction':sz.min()/len(l),'cluster_entropy':-np.sum(sz/len(l)*np.log(sz/len(l))),'cluster_size_imbalance':sz.max()/sz.min(),'assignment_probability_median':np.median(p),'assignment_probability_ge_80':np.mean(p>=.8)}
def stab(raw,ref,k,frac,seed,thr):
 rng=np.random.default_rng(seed);idx=np.sort(rng.choice(len(raw),int(len(raw)*frac),replace=False));sc=StandardScaler().fit(raw[idx]);z=sc.transform(raw[idx]);pc=PCA().fit(z);nc=np.searchsorted(np.cumsum(pc.explained_variance_ratio_),thr)+1;lab=KMeans(k,n_init=50,random_state=seed).fit_predict(pc.transform(z)[:,:nc]);r=ref[idx];ja=jaccards(r,lab);al=align(r,lab);return {'iteration':seed,'n_overlap':len(idx),'ari':adjusted_rand_score(r,lab),'adjusted_mutual_information':adjusted_mutual_info_score(r,lab),'minimum_cluster_jaccard':min(ja),'median_cluster_jaccard':np.median(ja),'assignment_probability_median':np.mean(al==r),'assignment_probability_p10':np.quantile((al==r).astype(float),.1)}
def nearest(x,c):
 d=np.sqrt(((x[:,None]-c[None])**2).sum(2));o=np.argsort(d,1);return o[:,0],d[np.arange(len(x)),o[:,0]],d[np.arange(len(x)),o[:,1]],d[np.arange(len(x)),o[:,1]]-d[np.arange(len(x)),o[:,0]]
def odd_even(step2,ids):
 out={x:{'odd':[],'even':[]} for x in ['full_all','neutral_all']}
 for n,pid in enumerate(ids,1):
  for sp,cond in [('full_all','full_profile'),('neutral_all','static_neutral')]:
   f=step2/'validation_hidden_states'/f'condition={cond}'/f'participant_id={pid}'/'data.parquet';z=pd.read_parquet(f,columns=['minutes_since_reset','is_h0_row','odd_even_day']+[f'h_{i:03d}' for i in range(128)]);z=z[(~z.is_h0_row)&(z.minutes_since_reset%15==0)]
   for oe in ['odd','even']:out[sp][oe].append(np.median(z.loc[z.odd_even_day==oe,[f'h_{i:03d}' for i in range(128)]],axis=0))
  if n%50==0:LOG.info('odd/even reconstructed %d/%d',n,len(ids))
 return {s:{o:np.stack(v) for o,v in q.items()} for s,q in out.items()}
def extract_targets(cache,inv,status,ids,lock,out):
 if not lock.exists():raise RuntimeError('Stage A decision lock missing before external target load')
 lockhash=sha(lock);use=['person_id','measurement_concept_id','measurement_date','measurement_datetime','value_as_number','unit_source_value','measurement_source_value'];parts=[]
 for ch in pd.read_csv(cache/'measurement.csv',usecols=use,chunksize=100000,low_memory=False):
  q=ch[ch.person_id.astype(str).isin(ids)&ch.measurement_concept_id.isin(TARGETS.values())].copy()
  if len(q):parts.append(q)
 m=pd.concat(parts,ignore_index=True);m['participant_id']=m.person_id.astype(str)
 if set(m.participant_id)-set(ids):raise RuntimeError('test/non-validation external target present')
 starts=status.set_index(status.participant_id.astype(str)).cgm_start.map(lambda x:pd.Timestamp(x).tz_localize(None));m['measurement_date2']=pd.to_datetime(m.measurement_datetime,errors='coerce').fillna(pd.to_datetime(m.measurement_date,errors='coerce'));m['cgm_start']=m.participant_id.map(starts);m['days_to_cgm_start']=(m.measurement_date2-m.cgm_start).dt.total_seconds()/86400;m['target_name']=m.measurement_concept_id.map({v:k for k,v in TARGETS.items()});rows=[]
 for (pid,t),g in m.groupby(['participant_id','target_name']):
  g=g[pd.to_numeric(g.value_as_number,errors='coerce').notna()].copy()
  if not len(g):continue
  before=g[g.days_to_cgm_start<=0].sort_values('measurement_date2');q=before.iloc[0] if len(before) else g[g.days_to_cgm_start>0].sort_values('days_to_cgm_start').iloc[0];unit=str(q.unit_source_value) if pd.notna(q.unit_source_value) and str(q.unit_source_value).strip() else '<missing>';expected={'natriuretic_peptide_b_prohormon':['pg/mL','<missing>'],'c_reactive_protein_i':['mg/L','<missing>'],'bun_creatinine_ratio':['<missing>']}[t];unitok=unit in expected;days=float(q.days_to_cgm_start)
  rows.append({'participant_id':pid,'target_name':t,'raw_value':float(q.value_as_number),'analysis_value':float(q.value_as_number) if unitok else np.nan,'unit':unit,'measurement_date':q.measurement_date2,'days_to_cgm_start':days,'record_selection_rule':'earliest valid at/before CGM start' if days<=0 else 'nearest valid after CGM start','unit_status':'compatible per Step 0' if unitok else 'incompatible/unknown','timing_status':'within_90d' if abs(days)<=90 else ('within_180d' if abs(days)<=180 else 'outside_180d'),'eligible_for_analysis':unitok and abs(days)<=180,'eligible_90d_sensitivity':unitok and abs(days)<=90,'exclusion_reason':'' if unitok and abs(days)<=180 else ('timing_outside_180d' if unitok else 'unit_incompatible'),'stage_a_decision_hash':lockhash})
 d=pd.DataFrame(rows);aparq(d,out/'validation_external_targets.parquet');audit=d.groupby('target_name').agg(n_extracted=('participant_id','size'),n_eligible_180d=('eligible_for_analysis','sum'),n_eligible_90d=('eligible_90d_sensitivity','sum'),n_unit_compatible=('unit_status',lambda x:x.str.startswith('compatible').sum()),median_days_to_cgm_start=('days_to_cgm_start','median')).reset_index();audit['stage_a_decision_hash']=lockhash;audit.to_csv(out/'external_target_extraction_audit.csv',index=False);return d,lockhash
def continuous_assoc(scores,features,external,ids,bootn,seed):
 rng=np.random.default_rng(seed);wide=external[external.eligible_for_analysis].pivot(index='participant_id',columns='target_name',values='analysis_value').reindex(ids);f=features.set_index('participant_id').reindex(ids);rows=[]
 vars_cont=['number_of_segments','total_clean_hours','dynamic_missingness','median_full_neutral_l2']+GLYCEMIC
 for sp,s in scores.items():
  for pc in range(5):
   y=s[:,pc]
   for fam,names in [('nuisance',vars_cont[:4]),('glycemic',vars_cont[4:]),('external',list(TARGETS))]:
    for name in names:
     x=(wide[name] if fam=='external' else f[name]).to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
     if ok.sum()<10:continue
     rho,p=spearmanr(y[ok],x[ok]);bs=[]
     for _ in range(bootn):q=rng.choice(np.where(ok)[0],ok.sum(),replace=True);bs.append(spearmanr(y[q],x[q]).statistic)
     rows.append({'representation_space':sp,'pc':pc+1,'family':fam,'variable':name,'association_type':'spearman','n':ok.sum(),'signed_association':rho,'absolute_association':abs(rho),'ci_low':np.quantile(bs,.025),'ci_high':np.quantile(bs,.975),'p_value':p})
   for name in ['clinical_site','study_group']:
    x=f[name].astype(str).to_numpy();grand=y.mean();ss=sum((x==v).sum()*(y[x==v].mean()-grand)**2 for v in np.unique(x));eta=ss/max(np.sum((y-grand)**2),1e-12);rows.append({'representation_space':sp,'pc':pc+1,'family':'nuisance','variable':name,'association_type':'eta_squared','n':len(y),'signed_association':eta,'absolute_association':eta,'ci_low':np.nan,'ci_high':np.nan,'p_value':np.nan})
 d=pd.DataFrame(rows);d['q_value']=d.groupby('family',group_keys=False).p_value.transform(bh);return d
def characterize(assign,feat,external,bootn,seed):
 rng=np.random.default_rng(seed);un=[];ext=[];adj=[]
 selected=assign[assign.is_selected_solution]
 for sp,z in selected.groupby('representation_space'):
  base=feat.merge(z[['participant_id','cluster_label']],on='participant_id');
  for fam,names in [('nuisance',NUISANCE_CONT),('glycemic',GLYCEMIC+['full_mae','neutral_mae'])]:
   for name in names:
    gs=[g[name].to_numpy(float) for _,g in base.groupby('cluster_label')];e,p=eps2(gs);bs=[]
    for _ in range(bootn):
     q=base.sample(len(base),replace=True,random_state=int(rng.integers(2**31-1)));bs.append(eps2([g[name].to_numpy(float) for _,g in q.groupby('cluster_label')])[0])
    un.append({'representation_space':sp,'family':fam,'variable':name,'test':'Kruskal-Wallis','n':base[name].notna().sum(),'effect_size':e,'effect_ci_low':np.nanquantile(bs,.025),'effect_ci_high':np.nanquantile(bs,.975),'p_value':p,'cluster_summaries':json.dumps({str(k):{'median':g[name].median(),'q1':g[name].quantile(.25),'q3':g[name].quantile(.75)} for k,g in base.groupby('cluster_label')},default=jd)})
  for name in ['clinical_site','study_group']:
   tab=pd.crosstab(base.cluster_label,base[name]);chi,p,_,_=chi2_contingency(tab);v=np.sqrt(chi/(len(base)*max(min(tab.shape)-1,1)));un.append({'representation_space':sp,'family':'nuisance','variable':name,'test':'chi-square','n':len(base),'effect_size':v,'effect_ci_low':np.nan,'effect_ci_high':np.nan,'p_value':p,'cluster_summaries':tab.to_json()})
  for target,g in external[external.eligible_for_analysis].groupby('target_name'):
   q=base.merge(g[['participant_id','analysis_value','days_to_cgm_start']],on='participant_id');gs=[x.analysis_value.to_numpy() for _,x in q.groupby('cluster_label')];e,p=eps2(gs);ext.append({'representation_space':sp,'target_name':target,'analysis':'unadjusted','n':len(q),'effect_size':e,'p_value':p,'cluster_summaries':json.dumps({str(k):{'n':len(x),'median':x.analysis_value.median(),'q1':x.analysis_value.quantile(.25),'q3':x.analysis_value.quantile(.75),'min':x.analysis_value.min(),'max':x.analysis_value.max()} for k,x in q.groupby('cluster_label')},default=jd),'is_primary_fdr_family':sp=='neutral_all'})
   val=np.log1p(q.analysis_value) if target in ['natriuretic_peptide_b_prohormon','c_reactive_protein_i'] else np.log(q.analysis_value);q['target_z']=(val-val.mean())/val.std(ddof=1);cov=['age','sex','clinical_site','study_group','mean_glucose','glucose_cv','tir_70_180','number_of_segments','total_clean_hours','dynamic_missingness'];qq=q.dropna(subset=['target_z']+cov).copy();xb=pd.get_dummies(qq[cov],columns=['sex','clinical_site','study_group'],drop_first=True,dtype=float);xe=pd.concat([xb,pd.get_dummies(qq.cluster_label,prefix='cluster',drop_first=True,dtype=float)],axis=1);xb=sm.add_constant(xb);xe=sm.add_constant(xe);mb=sm.OLS(qq.target_z,xb).fit(cov_type='HC3');me=sm.OLS(qq.target_z,xe).fit(cov_type='HC3');added=[c for c in xe if c.startswith('cluster_')];pv=float(me.wald_test(np.eye(len(me.params))[ [list(me.params.index).index(c) for c in added] ],scalar=True).pvalue) if added else np.nan;delta=me.rsquared-mb.rsquared
   db=[]
   for _ in range(bootn):
    ii=rng.choice(len(qq),len(qq),replace=True);yb=qq.target_z.to_numpy()[ii];x0=xb.to_numpy()[ii];x1=xe.to_numpy()[ii];p0=x0@np.linalg.lstsq(x0,yb,rcond=None)[0];p1=x1@np.linalg.lstsq(x1,yb,rcond=None)[0];den=np.sum((yb-yb.mean())**2);db.append((1-np.sum((yb-p1)**2)/max(den,1e-12))-(1-np.sum((yb-p0)**2)/max(den,1e-12)))
   adj.append({'representation_space':sp,'target_name':target,'n_complete':len(qq),'baseline_r2':mb.rsquared,'extended_r2':me.rsquared,'delta_r2':delta,'delta_r2_ci_low':np.quantile(db,.025),'delta_r2_ci_high':np.quantile(db,.975),'partial_effect_size':delta/max(1-mb.rsquared,1e-12),'omnibus_cluster_p_value':pv,'is_primary_fdr_family':sp=='neutral_all'})
 u=pd.DataFrame(un);u['q_value']=u.groupby('family',group_keys=False).p_value.transform(bh) if len(u) else [];e=pd.DataFrame(ext);e['q_value']=np.nan
 if len(e):
  ix=e.is_primary_fdr_family;e.loc[ix,'q_value']=bh(e.loc[ix,'p_value'])
 a=pd.DataFrame(adj);a['q_value']=np.nan
 if len(a):
  ix=a.is_primary_fdr_family;a.loc[ix,'q_value']=bh(a.loc[ix,'omnibus_cluster_p_value'])
 return u,e,a
def main():
 a=args()
 if a.split!='validation':raise ValueError('Step 3 requires validation; test is forbidden')
 if abs(a.primary_pca_variance-.9)>1e-12 or a.candidate_k!=[2,3,4,5]:raise ValueError('frozen PCA/k protocol changed')
 random.seed(a.seed);np.random.seed(a.seed);s0=Path(a.step0_dir).resolve();s1=Path(a.step1_dir).resolve();s2=Path(a.step2_dir).resolve();rid=a.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root);out=root/rid
 if out.exists() and a.overwrite:shutil.rmtree(out)
 if out.exists() and not a.resume:raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True);(out/'frozen_validation_pipeline').mkdir(exist_ok=True);setup(out/'step3_run.log');started=time.time();LOG.info('Step 3 run=%s Stage A starting; external clinical table not loaded',rid)
 s2m=json.loads((s2/'step2_manifest.json').read_text());dec2=json.loads((s2/'final_burnin_decision.json').read_text())
 if s2m['run_id']!='20260724T231513Z' or s2m['participant_counts']['considered']!=239 or dec2['selected_burn_in_minutes']!=0:raise RuntimeError('Step 2 protocol mismatch')
 ids=sorted(map(str,s2m['validation_participant_ids']));valid=set(ids)
 if len(ids)!=239 or len(valid)!=239:raise RuntimeError('validation cohort is not exactly 239 unique IDs')
 repspath=s2/'participant_representations.parquet';rephash=sha(repspath);reps=pd.read_parquet(repspath);reps.participant_id=reps.participant_id.astype(str)
 if set(reps.participant_id)!=valid or set(reps.split)!= {'validation'} or reps.burn_in_minutes.nunique()!=1 or reps.burn_in_minutes.iloc[0]!=0:raise RuntimeError('representation cohort/split/burn-in mismatch')
 if any(any(q in c.lower() for q in ['natriuretic','c_reactive','bun_creatinine','nt_probnp']) for c in reps):raise RuntimeError('phenotype attached to representation')
 aud=audit(reps,valid);aud.to_csv(out/'validation_representation_audit.csv',index=False)
 if (aud.qc_status!='PASS').any():raise RuntimeError('representation audit failed')
 pl=plan(a);dump(out/'step3_analysis_plan_frozen.json',pl);planhash=sha(out/'step3_analysis_plan_frozen.json');LOG.info('analysis plan frozen hash=%s',planhash)
 paths=s2m['immutable_input_paths'];panel=Path(paths['multimodal-parquet']);static=Path(paths['static-table']);status=pd.read_csv(s2/'validation_export_status_by_participant.csv',dtype={'participant_id':str});se=pd.read_csv(s2/'static_effect_by_participant.csv',dtype={'participant_id':str});feat=gly_nuis(panel,static,status,se,ids)
 if set(feat.participant_id)!=valid:raise RuntimeError('glycemic/nuisance validation cohort mismatch')
 aparq(feat,out/'validation_glycemic_nuisance_features.parquet');idx={p:i for i,p in enumerate(ids)}
 def repmat(typ,var='all_anchors',cols=H):
  z=reps[(reps.representation_type==typ)&(reps.balanced_anchor_variant==var)].set_index('participant_id').reindex(ids);return z[cols].to_numpy(float)
 raw={'full_all':repmat('full_all'),'neutral_all':repmat('neutral_all'),'neutral_night':repmat('neutral_night'),'neutral_day':repmat('neutral_day'),'neutral_night_clock_sensitivity':repmat('neutral_night_clock_sensitivity')};X=feat.set_index('participant_id').reindex(ids)[GLUCOSE_RESID].to_numpy(float)
 if not np.isfinite(X).all():raise RuntimeError('nonfinite residualization covariates')
 res,pred,rmet,rmeta=residualize(raw['neutral_all'],X,out,a.seed);raw['neutral_glucose_residual']=res;rmet.to_csv(out/'glucose_residualization_metrics.csv',index=False);rr=pd.DataFrame({'participant_id':ids,'split':'validation','residualization':'cross_fitted_ridge','fold_id':rmeta['fold_id']});rr=pd.concat([rr,pd.DataFrame(res,columns=H)],axis=1);aparq(rr,out/'glucose_residualized_representations.parquet')
 LOG.info('residualization complete median R2=%.4f final alpha=%s',rmet.query("metric_type=='dimension'").cross_fitted_r2.median(),rmeta['final_alpha'])
 scores={};pcas={};scalers={};keeps={};npcs={};pmeta=[];score_rows=[];load_rows=[]
 for sp in PCA_SPACES:
  scs,sc,pc,nc,keep,meta=fitpca(sp,raw[sp],out,a.primary_pca_variance);scores[sp]=scs;scalers[sp]=sc;pcas[sp]=pc;keeps[sp]=keep;npcs[sp]=nc;pmeta.append(meta)
  for i,pid in enumerate(ids):score_rows.append({'participant_id':pid,'split':'validation','representation_space':sp,**{f'pc_{j+1:03d}':v for j,v in enumerate(scs[i])}})
  for j,row in enumerate(pc.components_):
   for pos,val in zip(np.where(keep)[0],row):load_rows.append({'representation_space':sp,'pc':j+1,'hidden_dimension':int(pos),'loading':val})
 pd.DataFrame(pmeta).to_csv(out/'pca_variance_summary.csv',index=False);aparq(pd.DataFrame(score_rows),out/'pca_participant_scores.parquet');aparq(pd.DataFrame(load_rows),out/'pca_loadings.parquet');LOG.info('PCA components %s',{x:npcs[x] for x in PCA_SPACES})
 oe=odd_even(s2,ids);oe['neutral_glucose_residual']={'odd':oe['neutral_all']['odd']-pred,'even':oe['neutral_all']['even']-pred}
 balanced={'full_all':repmat('full_all','balanced_anchors'),'neutral_all':repmat('neutral_all','balanced_anchors')};balanced['neutral_glucose_residual']=balanced['neutral_all']-pred
 labels={};centroids={};probs={};candidate=[];strows=[];oerows=[];balrows=[];assignrows=[]
 for si,sp in enumerate(PRIMARY):
  z=scores[sp][:,:npcs[sp]];rawsp=raw[sp]
  def transform(v):return pcas[sp].transform(scalers[sp].transform(v[:,keeps[sp]]))[:,:npcs[sp]]
  zo,ze,zb=transform(oe[sp]['odd']),transform(oe[sp]['even']),transform(balanced[sp])
  for k in a.candidate_k:
   seed=a.seed+si*100000+k*1000;lab,cm,prob,cent=consensus(z,k,a.consensus_iterations,a.subsample_fraction,seed);labels[(sp,k)]=lab;centroids[(sp,k)]=cent;probs[(sp,k)]=prob;np.save(out/'frozen_validation_pipeline'/f'consensus_{sp}_k{k}.npy',cm)
   met=cmetrics(z,lab,cm,prob);ward=AgglomerativeClustering(k,linkage='ward').fit_predict(z);met.update({'representation_space':sp,'k':k,'algorithm':'consensus_kmeans','ward_silhouette':silhouette_score(z,ward),'ward_ari_vs_consensus':adjusted_rand_score(lab,ward)})
   jobs=(joblib.delayed(stab)(rawsp,lab,k,a.subsample_fraction,seed+10000+j,a.primary_pca_variance) for j in range(a.stability_iterations));sr=joblib.Parallel(n_jobs=a.n_jobs,prefer='threads',batch_size=10)(jobs)
   for q in sr:q.update({'representation_space':sp,'k':k})
   strows.extend(sr);sd=pd.DataFrame(sr);met['median_subsample_ari']=sd.ari.median();met['median_minimum_cluster_jaccard']=sd.minimum_cluster_jaccard.median();met['median_subsample_ami']=sd.adjusted_mutual_information.median()
   lo,d1,d2,margin=nearest(zo,cent);le,_,_,_=nearest(ze,cent);same=np.mean(lo==le);ari=adjusted_rand_score(lo,le);ami=adjusted_mutual_info_score(lo,le);kappa=cohen_kappa_score(lo,align(lo,le));oerows.append({'representation_space':sp,'k':k,'method':'frozen_centroids','n':len(ids),'same_cluster_percentage':same,'ari':ari,'adjusted_mutual_information':ami,'cohen_kappa_aligned':kappa,'median_assignment_margin':np.median(margin),'transition_matrix':pd.crosstab(lo,le).to_json(),'cluster_specific_consistency':json.dumps({str(c):float(np.mean(le[lo==c]==c)) for c in range(k)})});met['odd_even_ari']=ari;met['odd_even_same_cluster']=same
   lb,_,_,_=nearest(zb,cent);bari=adjusted_rand_score(lab,lb);bsame=np.mean(lab==lb);balrows.append({'representation_space':sp,'k':k,'ari':bari,'same_cluster_percentage':bsame,'n_changed':int((lab!=lb).sum()),'nn10_overlap':np.mean([len(set(a)&set(b))/10 for a,b in zip(np.argsort(squareform(pdist(z)),1)[:,1:11],np.argsort(squareform(pdist(zb)),1)[:,1:11])]),'recording_duration_change_spearman':spearmanr((lab!=lb).astype(float),feat.set_index('participant_id').reindex(ids).total_clean_hours).statistic});met['balanced_anchor_ari']=bari;candidate.append(met)
   la,a1,a2,am=nearest(z,cent)
   for i,pid in enumerate(ids):assignrows.append({'participant_id':pid,'split':'validation','representation_space':sp,'pca_rule':'minimum PCs >=90% variance','clustering_method':'consensus_kmeans','k':k,'cluster_label':int(lab[i]),'is_selected_solution':False,'clustering_status':'candidate_not_selected','assignment_probability':prob[i],'distance_to_assigned_centroid':a1[i],'distance_to_second_centroid':a2[i],'assignment_margin':am[i],'odd_cluster_label':int(lo[i]),'even_cluster_label':int(le[i]),'odd_even_consistent':bool(lo[i]==le[i]),'balanced_anchor_cluster_label':int(lb[i]),'balanced_anchor_consistent':bool(lab[i]==lb[i])})
   LOG.info('candidate %s k=%d silhouette=%.3f PAC=%.3f stability ARI=%.3f odd/even ARI=%.3f',sp,k,met['silhouette'],met['consensus_pac'],met['median_subsample_ari'],ari)
 cand=pd.DataFrame(candidate);stabdf=pd.DataFrame(strows);oedf=pd.DataFrame(oerows);baldf=pd.DataFrame(balrows);cand.to_csv(out/'clustering_candidate_metrics.csv',index=False);cand[['representation_space','k','consensus_pac','consensus_within_mean','consensus_between_mean','assignment_probability_median','assignment_probability_ge_80']].to_csv(out/'consensus_stability_metrics.csv',index=False);stabdf.to_csv(out/'clustering_subsample_stability.csv',index=False);oedf.to_csv(out/'odd_even_cluster_stability.csv',index=False);baldf.to_csv(out/'balanced_anchor_sensitivity.csv',index=False)
 decisions=[];selected={}
 for sp in PRIMARY:
  records=[]
  for _,r in cand[cand.representation_space==sp].iterrows():
   cr={'minimum_cluster_size':r.minimum_cluster_size>=20,'minimum_cluster_fraction':r.minimum_cluster_fraction>=.08,'median_subsample_ari':r.median_subsample_ari>=.6,'median_minimum_jaccard':r.median_minimum_cluster_jaccard>=.6,'consensus_pac':r.consensus_pac<=.2,'odd_even_ari':r.odd_even_ari>=.4,'odd_even_same_cluster':r.odd_even_same_cluster>=.6,'silhouette':r.silhouette>=.1,'assignment_probability':r.assignment_probability_ge_80>=.7};records.append({'k':int(r.k),'criteria_values':{x:r[x] for x in ['minimum_cluster_size','minimum_cluster_fraction','median_subsample_ari','median_minimum_cluster_jaccard','consensus_pac','odd_even_ari','odd_even_same_cluster','silhouette','assignment_probability_ge_80']},'criteria_passed':[x for x,v in cr.items() if v],'criteria_failed':[x for x,v in cr.items() if not v],'passes':all(cr.values())})
  passes=[x for x in records if x['passes']]
  if passes:
   q=cand[(cand.representation_space==sp)&cand.k.isin([x['k'] for x in passes])].copy();q=q.sort_values(['median_subsample_ari','consensus_pac','odd_even_ari','silhouette','cluster_size_imbalance','k'],ascending=[False,True,False,False,True,True]);sel=int(q.iloc[0].k);statusc='stable_discrete_solution';selected[sp]=sel;interp='stable_but_mixed_structure'
  else:sel=None;statusc='no_stable_discrete_solution';interp='reliable_continuous_manifold'
  decisions.append({'representation_space':sp,'primary_pca_rule':'minimum PCs >=90% variance','candidate_k':records,'selected_k':sel,'selected_method':'consensus_kmeans' if sel else None,'clustering_status':statusc,'selection_reason':'predeclared validity thresholds and hierarchy; external targets not loaded','continuous_manifold_fallback_required':sel is None,'interpretation_category':interp})
 assignment=pd.DataFrame(assignrows)
 for sp,k in selected.items():
  ix=(assignment.representation_space==sp)&(assignment.k==k);assignment.loc[ix,'is_selected_solution']=True;assignment.loc[ix,'clustering_status']='stable_discrete_solution';d=out/'frozen_validation_pipeline'/sp;np.save(d/f'{sp}_cluster_centroids.npy',centroids[(sp,k)]);dump(d/f'{sp}_cluster_model.json',{'method':'nearest frozen consensus centroid in frozen PCA coordinates','k':k,'decision_plan_hash':planhash})
 aparq(assignment,out/'cluster_assignments.parquet');decision={'stage':'A phenotype-free lock','analysis_plan_hash':planhash,'representation_decisions':decisions,'selected_k':selected,'external_targets_loaded':False,'created_utc':datetime.now(timezone.utc).isoformat()};dump(out/'clustering_selection_decision.json',decision);lockhash=sha(out/'clustering_selection_decision.json');(out/'STAGE_A_COMPLETE').write_text(lockhash+'\n');LOG.info('STAGE A LOCKED decision_hash=%s selected=%s; Stage B may now load external targets',lockhash,selected)
 # Stage B begins only after lock exists and was hashed.
 cache=Path(a.clinical_cache_dir).resolve() if a.clinical_cache_dir else s0/'cache';inv=pd.read_csv(s0/'clinical_target_inventory.csv');external,lockhash2=extract_targets(cache,inv,status,ids,out/'clustering_selection_decision.json',out)
 if lockhash2!=lockhash:raise RuntimeError('Stage A lock changed during Stage B')
 un,extass,adj=characterize(assignment,feat,external,a.bootstrap_replicates,a.seed+700000)
 # Glycemic-only cross-validated reproducibility of selected assignments.
 predrows=[]
 for sp,k in selected.items():
  lab=labels[(sp,k)];xf=feat.set_index('participant_id').reindex(ids)[['mean_glucose','glucose_cv','tir_70_180']].to_numpy();cv=StratifiedKFold(5,shuffle=True,random_state=a.seed);model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'));pr=cross_val_predict(model,xf,lab,cv=cv,method='predict');pp=cross_val_predict(model,xf,lab,cv=cv,method='predict_proba');predrows.append({'representation_space':sp,'family':'glycemic_prediction','variable':'mean_glucose+CV+TIR','test':'5-fold multinomial logistic','n':len(ids),'effect_size':balanced_accuracy_score(lab,pr),'effect_ci_low':np.nan,'effect_ci_high':np.nan,'p_value':np.nan,'q_value':np.nan,'cluster_summaries':json.dumps({'balanced_accuracy':balanced_accuracy_score(lab,pr),'macro_f1':f1_score(lab,pr,average='macro'),'macro_ovr_auc':roc_auc_score(label_binarize(lab,classes=np.arange(k)),pp,average='macro',multi_class='ovr') if k>2 else roc_auc_score(lab,pp[:,1])})})
 if predrows:un=pd.concat([un,pd.DataFrame(predrows)],ignore_index=True)
 if not len(un):un=pd.DataFrame(columns=['representation_space','family','variable','test','n','effect_size','effect_ci_low','effect_ci_high','p_value','q_value','cluster_summaries'])
 if not len(extass):extass=pd.DataFrame(columns=['representation_space','target_name','analysis','n','effect_size','p_value','q_value','cluster_summaries','is_primary_fdr_family'])
 if not len(adj):adj=pd.DataFrame(columns=['representation_space','target_name','n_complete','baseline_r2','extended_r2','delta_r2','partial_effect_size','omnibus_cluster_p_value','q_value','is_primary_fdr_family'])
 un.to_csv(out/'cluster_characterization_unadjusted.csv',index=False);extass.to_csv(out/'external_biomarker_associations.csv',index=False);adj.to_csv(out/'cluster_characterization_adjusted.csv',index=False)
 cont=continuous_assoc({x:scores[x] for x in PRIMARY},feat,external,ids,min(a.bootstrap_replicates,500),a.seed+800000);cont['stage_a_decision_hash']=lockhash;cont.to_csv(out/'continuous_geometry_associations.csv',index=False)
 # Full versus neutral comparison, always with continuous geometry; labels only if both stable.
 fvn=[];zf=scores['full_all'][:,:npcs['full_all']];zn=scores['neutral_all'][:,:npcs['neutral_all']];dmin=min(zf.shape[1],zn.shape[1]);_,_,disp=procrustes(zf[:,:dmin],zn[:,:dmin]);basecmp={'record_type':'summary','full_selected_k':selected.get('full_all'),'neutral_selected_k':selected.get('neutral_all'),'pairwise_distance_spearman':spearmanr(pdist(zf),pdist(zn)).statistic,'nn10_overlap':np.mean([len(set(x)&set(y))/10 for x,y in zip(np.argsort(squareform(pdist(zf)),1)[:,1:11],np.argsort(squareform(pdist(zn)),1)[:,1:11])]),'procrustes_similarity':1-disp}
 if 'full_all' in selected and 'neutral_all' in selected:
  lf=labels[('full_all',selected['full_all'])];ln=labels[('neutral_all',selected['neutral_all'])];aln=align(lf,ln);basecmp.update({'ari':adjusted_rand_score(lf,ln),'ami':adjusted_mutual_info_score(lf,ln),'nmi':normalized_mutual_info_score(lf,ln),'same_cluster_aligned':np.mean(lf==aln)});fvn.append(basecmp)
  for i,pid in enumerate(ids):fvn.append({'record_type':'participant','participant_id':pid,'full_cluster':lf[i],'neutral_cluster':ln[i],'changed_cluster':lf[i]!=aln[i],'full_assignment_confidence':probs[('full_all',selected['full_all'])][i],'neutral_assignment_confidence':probs[('neutral_all',selected['neutral_all'])][i],'static_effect_magnitude':feat.set_index('participant_id').loc[pid,'median_full_neutral_l2'],'forecast_effect_magnitude':feat.set_index('participant_id').loc[pid,'mean_abs_forecast_delta']})
 else:fvn.append(basecmp)
 pd.DataFrame(fvn).to_csv(out/'full_vs_neutral_cluster_comparison.csv',index=False)
 # Matched night/day geometry in the frozen neutral_all coordinate system.
 sc=scalers['neutral_all'];pc=pcas['neutral_all'];keep=keeps['neutral_all'];nc=npcs['neutral_all'];proj=lambda x:pc.transform(sc.transform(x[:,keep]))[:,:nc];za=proj(raw['neutral_all']);zni=proj(raw['neutral_night']);zda=proj(raw['neutral_day']);ctxg=[]
 for name,x,y in [('all_vs_night',za,zni),('all_vs_day',za,zda),('night_vs_day',zni,zda)]:
  dx,dy=squareform(pdist(x)),squareform(pdist(y));ctxg.append({'comparison':name,'n_matched':len(ids),'pairwise_distance_spearman':spearmanr(pdist(x),pdist(y)).statistic,'nn10_overlap':np.mean([len(set(a)&set(b))/10 for a,b in zip(np.argsort(dx,1)[:,1:11],np.argsort(dy,1)[:,1:11])]),'median_participant_shift_l2':np.median(np.linalg.norm(x-y,axis=1)),'median_participant_cosine':np.median(cosrows(x,y))})
 pd.DataFrame(ctxg).to_csv(out/'context_geometry_comparison.csv',index=False);ctxc=[]
 if 'neutral_all' in selected:
  k=selected['neutral_all'];cent=centroids[('neutral_all',k)];la,*_=nearest(za,cent);li,*_=nearest(zni,cent);ld,*_=nearest(zda,cent)
  for name,x,y in [('all_vs_night',la,li),('all_vs_day',la,ld),('night_vs_day',li,ld)]:ctxc.append({'comparison':name,'k':k,'same_cluster_percentage':np.mean(x==y),'ari':adjusted_rand_score(x,y),'adjusted_mutual_information':adjusted_mutual_info_score(x,y),'transition_matrix':pd.crosstab(x,y).to_json()})
 else:ctxc.append({'comparison':'no neutral_all stable cluster','k':np.nan,'same_cluster_percentage':np.nan,'ari':np.nan,'adjusted_mutual_information':np.nan,'transition_matrix':'{}'})
 pd.DataFrame(ctxc).to_csv(out/'context_cluster_comparison.csv',index=False)
 # Mandatory figures; external information appears only after the Stage A lock.
 sns.set_theme(style='whitegrid');colors={'full_all':'#4c78a8','neutral_all':'#f58518','neutral_glucose_residual':'#54a24b'}
 fig,axs=plt.subplots(1,3,figsize=(15,4))
 for ax,sp in zip(axs,PRIMARY):
  c=labels[(sp,selected[sp])] if sp in selected else colors[sp];ax.scatter(scores[sp][:,0],scores[sp][:,1],c=c,cmap='tab10',s=22,alpha=.8);ax.set_title(f'{sp}: PC1 {pcas[sp].explained_variance_ratio_[0]:.1%}, PC2 {pcas[sp].explained_variance_ratio_[1]:.1%}')
 plt.tight_layout();plt.savefig(out/'fig_pca_full_vs_neutral.png',dpi=170);plt.close()
 f=feat.set_index('participant_id').reindex(ids);fig,axs=plt.subplots(1,5,figsize=(20,4));ovs=['clinical_site','study_group','number_of_segments','dynamic_missingness','median_full_neutral_l2']
 for ax,v in zip(axs,ovs):
  if f[v].dtype=='object':c=pd.Categorical(f[v]).codes
  else:c=f[v]
  q=ax.scatter(scores['neutral_all'][:,0],scores['neutral_all'][:,1],c=c,cmap='viridis',s=18);ax.set_title(v);plt.colorbar(q,ax=ax,shrink=.7)
 plt.tight_layout();plt.savefig(out/'fig_pca_nuisance_overlays.png',dpi=170);plt.close()
 fig,axs=plt.subplots(2,4,figsize=(16,8))
 for i,sp in enumerate(['neutral_all','neutral_glucose_residual']):
  for ax,v in zip(axs[i],['mean_glucose','glucose_cv','tir_70_180','hba1c']):q=ax.scatter(scores[sp][:,0],scores[sp][:,1],c=f[v],cmap='viridis',s=18);ax.set_title(f'{sp}: {v}');plt.colorbar(q,ax=ax,shrink=.7)
 plt.tight_layout();plt.savefig(out/'fig_pca_glycemic_overlays.png',dpi=170);plt.close()
 fig,axs=plt.subplots(3,4,figsize=(14,10))
 for i,sp in enumerate(PRIMARY):
  for j,k in enumerate(a.candidate_k):m=np.load(out/'frozen_validation_pipeline'/f'consensus_{sp}_k{k}.npy');order=np.argsort(labels[(sp,k)]);axs[i,j].imshow(m[np.ix_(order,order)],vmin=0,vmax=1,cmap='mako');axs[i,j].set_title(f'{sp} k={k}'+(' SELECTED' if selected.get(sp)==k else ''))
 plt.tight_layout();plt.savefig(out/'fig_consensus_matrices.png',dpi=150);plt.close()
 fig,axs=plt.subplots(2,2,figsize=(11,8))
 for ax,v,title in zip(axs.flat,['median_subsample_ari','consensus_pac','silhouette','assignment_probability_ge_80'],['Subsample ARI','PAC','Silhouette','Assignment probability >=.80']):sns.lineplot(data=cand,x='k',y=v,hue='representation_space',marker='o',ax=ax);ax.set_title(title)
 plt.tight_layout();plt.savefig(out/'fig_cluster_stability_summary.png',dpi=170);plt.close()
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,sp in zip(axs,PRIMARY):q=oedf[oedf.representation_space==sp];ax.plot(q.k,q.same_cluster_percentage,marker='o',label='same');ax.plot(q.k,q.ari,marker='s',label='ARI');ax.set_title(sp);ax.legend()
 plt.tight_layout();plt.savefig(out/'fig_odd_even_cluster_agreement.png',dpi=170);plt.close()
 fig,ax=plt.subplots(figsize=(8,5));ax.bar(['NN10 overlap','distance rho','Procrustes'],[basecmp['nn10_overlap'],basecmp['pairwise_distance_spearman'],basecmp['procrustes_similarity']]);ax.set_ylim(0,1);ax.set_title('Full versus static-neutral geometry');plt.tight_layout();plt.savefig(out/'fig_full_vs_neutral_cluster_transitions.png',dpi=170);plt.close()
 fig,ax=plt.subplots(figsize=(10,6));nu=cont[(cont.representation_space=='neutral_all')&(cont.family=='nuisance')].pivot_table(index='variable',columns='pc',values='absolute_association');sns.heatmap(nu,cmap='mako',ax=ax);ax.set_title('Neutral PCA nuisance effect sizes');plt.tight_layout();plt.savefig(out/'fig_cluster_nuisance_characterization.png',dpi=170);plt.close()
 fig,ax=plt.subplots(figsize=(10,6));gl=cont[(cont.representation_space=='neutral_all')&(cont.family=='glycemic')].pivot_table(index='variable',columns='pc',values='signed_association');sns.heatmap(gl,cmap='vlag',center=0,ax=ax);ax.set_title('Neutral PCA glycemic associations');plt.tight_layout();plt.savefig(out/'fig_cluster_glycemic_characterization.png',dpi=170);plt.close()
 fig,axs=plt.subplots(1,3,figsize=(14,4));ew=external[external.eligible_for_analysis]
 for ax,(t,g) in zip(axs,ew.groupby('target_name')):sns.boxplot(y=g.analysis_value,ax=ax);sns.stripplot(y=g.analysis_value,ax=ax,color='black',size=2);ax.set_title(f'{t}\nn={len(g)}')
 plt.tight_layout();plt.savefig(out/'fig_cluster_external_biomarkers.png',dpi=170);plt.close()
 fig,ax=plt.subplots(figsize=(9,5));cg=pd.DataFrame(ctxg);cg.set_index('comparison')[['pairwise_distance_spearman','nn10_overlap','median_participant_cosine']].plot.bar(ax=ax);ax.set_ylim(0,1);plt.tight_layout();plt.savefig(out/'fig_context_geometry_comparison.png',dpi=170);plt.close()
 fig,ax=plt.subplots(figsize=(12,8));pv=cont.pivot_table(index=['family','variable'],columns=['representation_space','pc'],values='signed_association');sns.heatmap(pv,cmap='vlag',center=0,ax=ax);ax.set_title('Continuous manifold associations');plt.tight_layout();plt.savefig(out/'fig_continuous_manifold_summary.png',dpi=170);plt.close()
 # Freeze later untouched-test application without reading test data.
 final_interp={d['representation_space']:d['interpretation_category'] for d in decisions};overall='reliable_continuous_manifold' if not selected else ('stable_but_mixed_structure' if len(set(final_interp.values()))>1 else next(iter(final_interp.values())))
 testplan={'step2_burn_in_minutes':0,'state_sampling_minutes':15,'participant_aggregation':'dimensionwise median, all anchors','representation_spaces':PRIMARY,'scaler_paths':{x:str(out/'frozen_validation_pipeline'/x/f'{x}_scaler.joblib') for x in PCA_SPACES},'pca_paths':{x:str(out/'frozen_validation_pipeline'/x/f'{x}_pca.joblib') for x in PCA_SPACES},'residualizer_path':str(out/'frozen_validation_pipeline'/'glucose_residualizer.joblib'),'selected_cluster_method':'nearest frozen consensus centroid','selected_k':selected,'context_projection_method':'frozen neutral_all scaler/PCA/centroids','nuisance_variables':['clinical_site','study_group']+NUISANCE_CONT,'glycemic_variables':GLYCEMIC,'external_targets':list(TARGETS),'target_extraction_rule':pl['external_target_extraction_rule'],'primary_external_fdr_family':'selected neutral_all, exactly 3 targets','primary_test_metrics':['frozen cluster reproducibility when selected','continuous PC associations','external adjusted delta R2'],'no_retuning_rule':True,'acceptable_null_results':['no stable discrete structure','continuous manifold','null external associations'],'test_accessed_during_freeze':False,'neutral_all_no_discrete_hypothesis': 'neutral_all' not in selected}
 dump(out/'frozen_test_application_plan.json',testplan);dump(out/'frozen_validation_pipeline'/'context_assignment_rules.json',{'neutral_all_projection':'use neutral_all kept dimensions, scaler, PCA, selected centroids only','selected_k':selected.get('neutral_all')})
 # Report and manifest.
 counts=external.groupby('target_name').eligible_for_analysis.sum().to_dict();report=['# Step 3 validation-only clustering and characterization','']
 sections=['Objective','Scope and explicit exclusions','Step 2 representation protocol','Validation participant coverage','Leakage-prevention design','Frozen analysis plan','Glycemic residualization','PCA dimensionality and geometry','Cluster tendency','Consensus clustering candidates','Subsample stability','Odd/even cluster stability','Selected or rejected cluster solutions','Full-profile participant structure','Static-neutral participant structure','Glucose-adjusted neutral structure','Full-versus-neutral comparison','Nuisance and acquisition audit','Glycemic characterization','External target extraction','External biomarker characterization','Adjusted biomarker models','Continuous geometry associations','Balanced-anchor sensitivity','Night-versus-day comparison','Interpretation category','Limitations','Frozen test pipeline','Blocking issues','GO / GO WITH CAVEATS / NO-GO recommendation for untouched-test application']
 bodies=[f'Validation-only phenotype-free clustering followed by locked characterization.',f'No test access, model replay, tuning by biomarkers, timestamp-level clustering, clinical probes, or exercise clustering.',f'Frozen Step 2 burn-in 0 minutes; 15-minute all-anchor dimensionwise medians.',f'{len(ids)}/239 validation participants; 128 dimensions.',f'Stage A decision `{lockhash}` was written before the clinical measurement cache was loaded.',f'Plan hash `{planhash}`.',f'Median cross-fitted dimension R2 {rmet.loc[rmet.metric_type.eq("dimension"), "cross_fitted_r2"].median():.3f}; final alpha {rmeta["final_alpha"]}.',json.dumps({x:npcs[x] for x in PCA_SPACES}),json.dumps({x:next(q for q in pmeta if q['space']==x)['hopkins'] for x in PRIMARY}),f'All k=2–5 evaluated with {a.consensus_iterations} resamples.',f'{a.stability_iterations} 80% refits per space and k.',oedf.groupby('representation_space').ari.max().to_json(),json.dumps(selected),str(decisions[0]),str(decisions[1]),str(decisions[2]),str(basecmp),'See cluster_characterization_unadjusted.csv.','Glycemic-only assignment reproduction is reported without changing selection.',json.dumps(counts),'External targets were characterized only after Stage A lock.','See cluster_characterization_adjusted.csv.','PC1–PC5 associations are always reported.','See balanced_anchor_sensitivity.csv.','See context geometry and cluster comparison tables.',json.dumps(final_interp),'Validation-only inference; clinical associations are exploratory and unconfirmed.',f'Artifacts under `{out/"frozen_validation_pipeline"}`; no test data accessed.','None.',('**GO WITH CAVEATS:** The pipeline is frozen, but no stable discrete solution was forced; continuous geometry remains a defensible untouched-test analysis.' if not selected else '**GO:** The frozen selected and continuous analyses can be applied unchanged once to untouched test participants.')]
 for i,(s,b) in enumerate(zip(sections,bodies),1):report.extend([f'## {i}. {s}',str(b),''])
 (out/'step3_report.md').write_text('\n'.join(report))
 required=['step3_analysis_plan_frozen.json','validation_representation_audit.csv','validation_glycemic_nuisance_features.parquet','validation_external_targets.parquet','external_target_extraction_audit.csv','glucose_residualization_metrics.csv','glucose_residualized_representations.parquet','pca_variance_summary.csv','pca_participant_scores.parquet','pca_loadings.parquet','clustering_candidate_metrics.csv','consensus_stability_metrics.csv','clustering_subsample_stability.csv','odd_even_cluster_stability.csv','clustering_selection_decision.json','cluster_assignments.parquet','cluster_characterization_unadjusted.csv','cluster_characterization_adjusted.csv','external_biomarker_associations.csv','continuous_geometry_associations.csv','full_vs_neutral_cluster_comparison.csv','context_geometry_comparison.csv','context_cluster_comparison.csv','balanced_anchor_sensitivity.csv','frozen_test_application_plan.json','step3_report.md','step3_run.log']+[f'fig_{x}.png' for x in ['pca_full_vs_neutral','pca_nuisance_overlays','pca_glycemic_overlays','consensus_matrices','cluster_stability_summary','odd_even_cluster_agreement','full_vs_neutral_cluster_transitions','cluster_nuisance_characterization','cluster_glycemic_characterization','cluster_external_biomarkers','context_geometry_comparison','continuous_manifold_summary']]
 missing=[x for x in required if not (out/x).exists()]
 if missing:raise RuntimeError(f'missing outputs {missing}')
 inputs={'step0_manifest':s0/'step0_manifest.json','step1_manifest':s1/'step1_manifest.json','step2_manifest':s2/'step2_manifest.json','step2_representations':repspath,'step2_status':s2/'validation_export_status_by_participant.csv','step2_static_effect':s2/'static_effect_by_participant.csv','clinical_inventory':s0/'clinical_target_inventory.csv','clinical_measurement_cache':cache/'measurement.csv','implementation':Path(__file__).resolve()};ih={k:sha(v) for k,v in inputs.items()};serialized=[p for p in (out/'frozen_validation_pipeline').rglob('*') if p.is_file()];outs=[out/x for x in required if (out/x).is_file()]+serialized
 rec='GO' if selected else 'GO WITH CAVEATS';warnings=[]
 if not selected:warnings.append('No primary representation satisfied the stable discrete clustering rule; continuous geometry is the frozen principal follow-up.')
 manifest={'run_id':rid,'timestamp':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'dirty_working_tree_status':subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True),'input_paths':{k:str(v) for k,v in inputs.items()},'input_hashes':ih,'step0_run':str(s0),'step1_run':str(s1),'step2_run':str(s2),'step2_representation_hash':rephash,'step2_burn_in':0,'participant_ids':ids,'participant_count':239,'representation_spaces':PCA_SPACES,'residualization_covariates':GLUCOSE_RESID,'cross_fitting_folds':5,'residualizer_alpha_grid':pl['residualizer_alpha_grid'],'scaling_rules':pl['scaling_method'],'pca_rules':pl['pca_candidate_rules'],'retained_pca_components':{x:npcs[x] for x in PCA_SPACES},'clustering_algorithms':pl['clustering_algorithms'],'consensus_settings':pl['consensus'],'candidate_k_values':a.candidate_k,'stability_thresholds':pl['stability_criteria'],'clustering_decisions':decisions,'stage_a_decision_hash':lockhash,'external_target_mappings':TARGETS,'external_target_timing_rules':pl['external_target_extraction_rule'],'statistical_models':pl['clinical_association_models'],'fdr_families':pl['fdr_families'],'context_rules':pl['context_comparison_rules'],'random_seeds':pl['random_seeds'],'serialized_model_paths':[str(x) for x in serialized],'output_paths':{x:str(out/x) for x in required},'output_hashes':{str(x.relative_to(out)):sha(x) for x in outs},'warnings':warnings,'errors':[],'final_interpretation':overall,'final_go_status':rec,'test_participants_accessed':0,'elapsed_seconds':time.time()-started};dump(out/'step3_manifest.json',manifest)
 # Recheck immutable sources, deterministic labels, and only then publish latest.
 if any(sha(Path(p))!=ih[k] for k,p in manifest['input_paths'].items()):raise RuntimeError('immutable input changed')
 for (sp,k),lab in labels.items():
  if not np.array_equal(lab,labels[(sp,k)]):raise RuntimeError('label reproducibility failure')
 latest=root/'latest';tmp=root/'.latest.tmp'
 if tmp.exists() or tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(out.name);os.replace(tmp,latest);(out/'STAGE_B_COMPLETE').write_text(lockhash+'\n');LOG.info('QC COMPLETE latest=%s recommendation=%s elapsed=%.1fm',latest,rec,(time.time()-started)/60)
 print(json.dumps({'output_directory':str(out),'files_created':required+['frozen_validation_pipeline/'],'step2_representation_hash':rephash,'analysis_plan_hash':planhash,'stage_a_decision_hash':lockhash,'validation_participants':239,'representation_dimensions':128,'residualization_median_r2':rmet.query("metric_type=='dimension'").cross_fitted_r2.median(),'pca_components':{x:npcs[x] for x in PCA_SPACES},'cluster_tendency':{x:next(q for q in pmeta if q['space']==x)['hopkins'] for x in PRIMARY},'selected_k':selected,'candidate_metrics':cand.to_dict('records'),'subsample_stability':cand.groupby('representation_space').median_subsample_ari.max().to_dict(),'odd_even_agreement':oedf.groupby('representation_space').ari.max().to_dict(),'balanced_anchor_agreement':baldf.groupby('representation_space').ari.max().to_dict(),'full_vs_neutral':basecmp,'external_counts':counts,'external_effects':extass.to_dict('records'),'adjusted_delta_r2':adj.to_dict('records'),'continuous_pc_associations':cont.loc[cont.groupby(['representation_space','family']).absolute_association.idxmax()].to_dict('records'),'night_day_agreement':ctxc,'final_interpretation':overall,'frozen_pipeline_paths':[str(x) for x in serialized],'warnings':warnings,'blockers':[],'recommendation':rec},indent=2,default=jd))
if __name__=='__main__':main()
