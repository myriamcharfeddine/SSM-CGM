"""Extract cumulative, non-leaking dynamic summaries for neighbor-transition analysis.

Uses the same clean 5-minute segments as the approved Phase 4 extraction. For
each segment and hour, only rows from elapsed time 0 through that hour are used;
segment summaries are then averaged per participant to match state aggregation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ssmcgm.analysis.within_subtype_config import DATASET, STUDY1_ROOT, STUDY2_ROOT
from ssmcgm.data.aireadi import AireadiSchema, prepare_aireadi_panel

ROOT=STUDY2_ROOT/"neighbor_transition_drivers"
HOURS=[6,12,24,48]
BIN_MINUTES=5
DYNAMIC_COLUMNS=[
 "participant_id","timestamp_local","cgm_glucose_mean","cgm_count",
 "calories_total","calories_per_min",
 "heart_rate_mean","heart_rate_std","heart_rate_min","heart_rate_max","heart_rate_count","heart_rate_device_availability",
 "respiratory_rate_mean","respiratory_rate_std","respiratory_rate_min","respiratory_rate_max","respiratory_rate_count","respiratory_rate_device_availability",
 "oxygen_saturation_mean","oxygen_saturation_std","oxygen_saturation_min","oxygen_saturation_max","oxygen_saturation_count","oxygen_saturation_device_availability",
 "stress_level_mean","stress_level_std","stress_level_min","stress_level_max","stress_level_count","stress_level_device_availability",
 "sleep_stage_awake","sleep_stage_light","sleep_stage_deep","sleep_stage_rem","sleep_stage_unknown","sleep_transitions","sleep_efficiency",
 "activity_stage_walking","activity_stage_sedentary","activity_stage_generic","activity_stage_running","activity_steps_per_min","activity_transitions","activity_intensity_score",
]
DOMAINS={
 "cgm_level":["cgm_mean","cgm_median","cgm_min","cgm_max","cgm_time_in_range","cgm_time_above_180","cgm_time_below_70"],
 "cgm_variability":["cgm_sd","cgm_cv","cgm_iqr","cgm_range","cgm_masd"],
 "cgm_dynamics":["cgm_mean_slope","cgm_mean_abs_slope","cgm_max_positive_slope","cgm_max_negative_slope","cgm_rising_excursions","cgm_falling_excursions","cgm_lag1_autocorrelation","cgm_area_above_segment_baseline","cgm_area_below_segment_baseline"],
 "hr_respiration":["heart_rate_mean_summary","heart_rate_sd_summary","respiratory_rate_mean_summary","respiratory_rate_sd_summary"],
 "stress_other_wearables":["spo2_mean_summary","stress_mean_summary","stress_sd_summary"],
 "activity":["total_steps","active_minutes","walking_proportion","sedentary_proportion","generic_activity_proportion","running_proportion","activity_intensity_mean","activity_intensity_sd","calories_total_summary","exercise_burden_minutes"],
 "sleep":["sleep_awake_proportion","sleep_light_proportion","sleep_deep_proportion","sleep_rem_proportion","sleep_unknown_proportion","sleep_transition_count","sleep_efficiency_mean","sleep_continuity"],
}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def write_json(path,value): path.write_text(json.dumps(value,indent=2)+"\n")
def vals(frame,column): return pd.to_numeric(frame[column],errors="coerce").dropna().to_numpy(float)
def safe_mean(x): return float(np.mean(x)) if len(x) else np.nan
def safe_std(x): return float(np.std(x,ddof=1)) if len(x)>1 else (0.0 if len(x)==1 else np.nan)
def transitions(flags):
 x=np.asarray(flags,dtype=bool)
 return int(np.sum(x & ~np.r_[False,x[:-1]])) if len(x) else 0

def segment_features(seg):
 g=vals(seg,"cgm_glucose_mean")
 out={}
 if len(g):
  q1,med,q3=np.percentile(g,[25,50,75]); sd=safe_std(g); mean=float(g.mean())
  out.update(cgm_mean=mean,cgm_median=float(med),cgm_min=float(g.min()),cgm_max=float(g.max()),
   cgm_time_in_range=float(np.mean((g>=70)&(g<=180))),cgm_time_above_180=float(np.mean(g>180)),
   cgm_time_below_70=float(np.mean(g<70)),cgm_sd=sd,cgm_cv=float(sd/mean) if mean else np.nan,
   cgm_iqr=float(q3-q1),cgm_range=float(g.max()-g.min()))
 else:
  for key in DOMAINS["cgm_level"]+DOMAINS["cgm_variability"]: out[key]=np.nan
  out["cgm_masd"]=np.nan
 series=pd.to_numeric(seg["cgm_glucose_mean"],errors="coerce").to_numpy(float)
 valid=np.isfinite(series)
 diffs=np.diff(series); pairvalid=valid[1:]&valid[:-1]; diffs=diffs[pairvalid]; slopes=diffs/BIN_MINUTES
 out["cgm_masd"]=safe_mean(np.abs(diffs))
 out["cgm_mean_slope"]=safe_mean(slopes); out["cgm_mean_abs_slope"]=safe_mean(np.abs(slopes))
 out["cgm_max_positive_slope"]=float(np.max(slopes)) if len(slopes) else np.nan
 out["cgm_max_negative_slope"]=float(np.min(slopes)) if len(slopes) else np.nan
 out["cgm_rising_excursions"]=transitions(slopes>=1.0); out["cgm_falling_excursions"]=transitions(slopes<=-1.0)
 if pairvalid.sum()>2:
  x=series[:-1][pairvalid]; y=series[1:][pairvalid]
  out["cgm_lag1_autocorrelation"]=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else np.nan
 else: out["cgm_lag1_autocorrelation"]=np.nan
 baseline=series[np.where(valid)[0][0]] if valid.any() else np.nan
 delta=series[valid]-baseline if valid.any() else np.array([])
 out["cgm_area_above_segment_baseline"]=float(np.maximum(delta,0).sum()*BIN_MINUTES/60) if len(delta) else np.nan
 out["cgm_area_below_segment_baseline"]=float(np.maximum(-delta,0).sum()*BIN_MINUTES/60) if len(delta) else np.nan
 for src,prefix in [("heart_rate_mean","heart_rate"),("respiratory_rate_mean","respiratory_rate")]:
  x=vals(seg,src); out[f"{prefix}_mean_summary"]=safe_mean(x); out[f"{prefix}_sd_summary"]=safe_std(x)
 x=vals(seg,"oxygen_saturation_mean"); out["spo2_mean_summary"]=safe_mean(x)
 x=vals(seg,"stress_level_mean"); out["stress_mean_summary"]=safe_mean(x); out["stress_sd_summary"]=safe_std(x)
 steps=pd.to_numeric(seg.activity_steps_per_min,errors="coerce").fillna(0).to_numpy(float)
 intensity=pd.to_numeric(seg.activity_intensity_score,errors="coerce").to_numpy(float)
 out["total_steps"]=float(np.sum(steps)*BIN_MINUTES)
 out["active_minutes"]=float(np.sum((steps>0)|((np.nan_to_num(intensity,nan=0))>0))*BIN_MINUTES)
 for col,name in [("activity_stage_walking","walking_proportion"),("activity_stage_sedentary","sedentary_proportion"),("activity_stage_generic","generic_activity_proportion"),("activity_stage_running","running_proportion")]: out[name]=safe_mean(vals(seg,col))
 out["activity_intensity_mean"]=safe_mean(intensity[np.isfinite(intensity)]); out["activity_intensity_sd"]=safe_std(intensity[np.isfinite(intensity)])
 out["calories_total_summary"]=float(np.nansum(pd.to_numeric(seg.calories_total,errors="coerce").to_numpy(float)))
 out["exercise_burden_minutes"]=float(np.sum(np.nan_to_num(intensity,nan=0)>=2)*BIN_MINUTES)
 sleep_cols=[("sleep_stage_awake","sleep_awake_proportion"),("sleep_stage_light","sleep_light_proportion"),("sleep_stage_deep","sleep_deep_proportion"),("sleep_stage_rem","sleep_rem_proportion"),("sleep_stage_unknown","sleep_unknown_proportion")]
 for col,name in sleep_cols: out[name]=safe_mean(vals(seg,col))
 tr=vals(seg,"sleep_transitions"); eff=vals(seg,"sleep_efficiency")
 out["sleep_transition_count"]=float(np.sum(tr)) if len(tr) else np.nan; out["sleep_efficiency_mean"]=safe_mean(eff)
 out["sleep_continuity"]=float(1/(1+np.sum(tr))) if len(tr) else np.nan
 return out

def participant_hour_features(group,hour):
 parts=[]; raw_n=0
 for _,seg in group.groupby("segment_id",sort=True):
  win=seg[seg.time_idx<=hour*60//BIN_MINUTES]
  parts.append(segment_features(win)); raw_n += int(win["_target_observed_raw"].sum())
 keys=sorted({k for p in parts for k in p})
 row={k:float(np.nanmean([p.get(k,np.nan) for p in parts])) if np.isfinite([p.get(k,np.nan) for p in parts]).any() else np.nan for k in keys}
 durations=group.groupby("segment_id").time_idx.max().to_numpy(float)*BIN_MINUTES/60
 row.update(hour=hour,valid_observation_count=raw_n,valid_observations_per_segment_mean=raw_n/len(parts),
  qualifying_segment_count=len(parts),available_streaming_duration_hours_mean=float(np.mean(durations)),
  available_streaming_duration_hours_min=float(np.min(durations)),available_streaming_duration_hours_max=float(np.max(durations)))
 return row

def main():
 ROOT.mkdir(parents=True,exist_ok=True); started=time.time()
 # Schema names are read separately because pandas returns no columns above.
 import pyarrow.parquet as pq
 available=set(pq.read_schema(DATASET).names); columns=[c for c in DYNAMIC_COLUMNS if c in available]
 missing=sorted(set(DYNAMIC_COLUMNS)-set(columns))
 data=pd.read_parquet(DATASET,columns=columns); schema=AireadiSchema(); panel=prepare_aireadi_panel(data,schema)
 h0=pd.read_parquet(STUDY1_ROOT/"step2/h0_matrix.parquet",columns=["participant_id","split"]); cohort=set(h0.participant_id.astype(str))
 panel=panel[panel.participant_id.astype(str).isin(cohort)].copy(); panel.participant_id=panel.participant_id.astype(str)
 counts=panel.groupby("participant_id").segment_id.nunique()
 if len(counts)!=len(cohort) or counts.min()<1: raise RuntimeError("Clean-stream cohort mismatch")
 rows=[]
 for i,(pid,group) in enumerate(panel.groupby("participant_id",sort=True),1):
  for hour in HOURS: rows.append({"participant_id":pid,**participant_hour_features(group,hour)})
  if i%100==0: print(f"Processed {i}/{len(cohort)} participants",flush=True)
 out=pd.DataFrame(rows); out.to_parquet(ROOT/"participant_dynamic_features.parquet",index=False)
 feature_cols=[c for c in out.columns if c not in ("participant_id","hour")]
 coverage=[]
 for hour,g in out.groupby("hour"):
  for feature in feature_cols: coverage.append({"hour":int(hour),"feature":feature,"domain":next((d for d,fs in DOMAINS.items() if feature in fs),"matching_metadata"),"participant_n":len(g),"finite_n":int(np.isfinite(g[feature]).sum()),"coverage_fraction":float(np.isfinite(g[feature]).mean())})
 pd.DataFrame(coverage).to_csv(ROOT/"dynamic_feature_coverage.csv",index=False)
 report={"created_at":now(),"hours":HOURS,"participant_count":len(cohort),"clean_segment_count":int(len(panel[["participant_id","segment_id"]].drop_duplicates())),"feature_domains":DOMAINS,"missing_requested_columns":missing,"meal_proxy":"Unavailable: no observed meal annotations; calories are wearable energy expenditure and were not relabeled as meals.","rising_falling_excursion_definition":"Onset of a contiguous run with absolute slope at least 1 mg/dL/min.","sleep_continuity_definition":"1/(1 + observed sleep-stage transition count) within the cumulative window, averaged across segments.","segment_aggregation":"Compute each cumulative feature within every clean segment, then average across a participant's qualifying segments.","future_data_used":False,"elapsed_seconds":time.time()-started}
 write_json(ROOT/"dynamic_feature_extraction_report.json",report)
 files=[ROOT/"participant_dynamic_features.parquet",ROOT/"dynamic_feature_coverage.csv",ROOT/"dynamic_feature_extraction_report.json"]
 write_json(ROOT/"dynamic_extraction_hashes.json",{p.name:sha(p) for p in files})
 print("Dynamic feature extraction complete; no future observations were used.")

if __name__=="__main__": main()
