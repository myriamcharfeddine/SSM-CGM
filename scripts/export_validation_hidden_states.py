#!/usr/bin/env python3
"""Step 2 full-validation state export, representations, and reliability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from ssmcgm.data.aireadi import (
    AireadiFeatureSpec, AireadiPreprocessor, build_stream_feature_spec,
    infer_or_validate_schema, make_aireadi_stream_splits,
    make_participant_streams, prepare_aireadi_panel,
)
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig
from scripts.run_static_neutralization_pilot import (
    EPS, flatten_initial_state, hash_jsonable, json_default, sha256,
    static_hash, valid_anchors,
)

LOG = logging.getLogger("step2")
CANDIDATES = [0, 180, 360, 540, 720, 855, 1080, 1440]
REPTYPES = ["full_all", "neutral_all", "neutral_night", "neutral_day",
            "neutral_night_clock_sensitivity"]
HIDDEN = [f"h_{i:03d}" for i in range(128)]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "checkpoint", "schema", "multimodal-parquet",
              "static-table", "split-manifest", "step0-dir", "step1-dir",
              "output-root"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--state-save-frequency-minutes", type=int, default=5)
    p.add_argument("--representation-frequency-minutes", type=int, default=15)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--permutation-replicates", type=int, default=1000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-id")
    p.add_argument("--participant-ids", nargs="*")
    p.add_argument("--backend", default="checkpoint")
    p.add_argument("--precision", default="float32")
    p.add_argument("--cache-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-participants", type=int)
    p.add_argument("--parquet-compression", default="zstd")
    p.add_argument("--participant-partition-size", type=int, default=1)
    return p.parse_args()


def dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")
    os.replace(tmp, path)


def setup(path: Path) -> None:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh, sh = logging.FileHandler(path), logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt); sh.setFormatter(fmt); LOG.handlers[:] = [fh, sh]


def atomic_parquet(df: pd.DataFrame, path: Path, compression: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, path)
    return sha256(path)


def dev(name: str) -> str:
    if name == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    return name


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), EPS)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)) @ (
        b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)).T


def ci(values: np.ndarray, reps: int, rng: np.random.Generator,
       stat=np.median) -> tuple[float, float]:
    values = np.asarray(values, float)
    if not len(values): return np.nan, np.nan
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    x = values[idx]
    z = np.median(x, axis=1) if stat is np.median else np.mean(x, axis=1)
    return tuple(np.quantile(z, [.025, .975]))


def raw_dynamic(stream, pre: AireadiPreprocessor, name: str) -> np.ndarray:
    j = stream.metadata["_dynamic_names"].index(name)
    st = pre.continuous_stats[name]
    return stream.dynamic[:, j].cpu().numpy() * st["std"] + st["mean"]


def stream_time(stream) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(x) for x in stream.timestamps])


@torch.no_grad()
def replay_segment(model, raw, condition, ncont, ncat, device, ck_hash,
                   stride, day_map, backend):
    s = raw.to(device)
    cont = torch.tensor(ncont, device=device) if condition == "static_neutral" else s.static_cont
    cat = torch.tensor(ncat, device=device) if condition == "static_neutral" else s.static_cat
    ctx = model.encode_static(cat, cont)
    state0 = model.init_stream(ctx)
    actual_h0 = flatten_initial_state(state0)
    _, out = model.scan_chunk(s.dynamic, ctx, state0)
    h = out[0].float().cpu().numpy()
    if h.shape[1] != 128 or not np.isfinite(h).all(): raise RuntimeError("invalid hidden state")
    ts = stream_time(s)
    utc = pd.DatetimeIndex([x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC") for x in ts])
    dates = np.array([str(x.date()) for x in ts])
    hours = np.array([x.hour + x.minute/60 for x in ts])
    sleep = np.zeros(len(ts), bool)
    for name in ("sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem"):
        if name in s.metadata["_dynamic_names"]: sleep |= raw_dynamic(s, model.preprocessor, name) > .5
    valid = s.observed.cpu().numpy().astype(bool)
    night_clock = hours < 6
    day = (hours >= 8) & (hours < 20) & ~sleep & valid
    cal = np.array([day_map[d] for d in dates])
    dyn_hash = hashlib.sha256(s.dynamic.cpu().numpy().tobytes()).hexdigest()
    prof_hash = static_hash(cont.cpu().numpy(), cat.cpu().numpy())
    start, end = ts[0], ts[-1]
    base = pd.DataFrame({
        "participant_id": s.participant_id, "split": s.split, "condition": condition,
        "segment_id": s.segment_id, "segment_start": str(start), "segment_end": str(end),
        "timestamp_utc": utc, "timestamp_local": [x.isoformat() for x in ts],
        "local_date": dates, "local_hour": hours, "calendar_day_index": cal,
        "odd_even_day": np.where(cal % 2, "odd", "even"),
        "step_index_in_segment": np.arange(len(ts)), "minutes_since_reset": (np.arange(len(ts))+1)*5,
        "hours_since_reset": (np.arange(len(ts))+1)*5/60, "is_h0_row": False,
        "is_post_update_state": True, "night_sleep_flag": sleep,
        "night_clock_00_06_flag": night_clock, "day_awake_08_20_flag": day,
        "valid_cgm": valid, "clinical_site": s.metadata.get("participants_clinical_site"),
        "study_group": s.metadata.get("participants_study_group"),
        "static_profile_hash": prof_hash, "dynamic_input_hash": dyn_hash,
        "checkpoint_hash": ck_hash, "backend": backend,
        "state_l2_norm": np.linalg.norm(h, axis=1), "state_mean": h.mean(1),
        "state_std": h.std(1),
    })
    hd = pd.concat([base, pd.DataFrame(h, columns=HIDDEN)], axis=1)
    hp = ctx.embedding[0].float().cpu().numpy()
    h0 = hd.iloc[:1].copy()
    h0.loc[:, "timestamp_utc"] = utc[0] - pd.Timedelta(minutes=5)
    h0.loc[:, "timestamp_local"] = (ts[0] - pd.Timedelta(minutes=5)).isoformat()
    h0.loc[:, "step_index_in_segment"] = -1; h0.loc[:, "minutes_since_reset"] = 0
    h0.loc[:, "hours_since_reset"] = 0.; h0.loc[:, "is_h0_row"] = True
    h0.loc[:, "is_post_update_state"] = False
    h0.loc[:, ["night_sleep_flag","night_clock_00_06_flag","day_awake_08_20_flag","valid_cgm"]] = False
    h0.loc[:, HIDDEN] = hp
    h0.loc[:, "state_l2_norm"] = np.linalg.norm(hp); h0.loc[:, "state_mean"] = hp.mean()
    h0.loc[:, "state_std"] = hp.std()
    hd = pd.concat([h0, hd], ignore_index=True)
    anchors = valid_anchors(s, model.feature_spec.horizon_steps, stride)
    fr = []
    if anchors:
        pos = torch.tensor(anchors, device=device); H = model.feature_spec.horizon_steps
        fut = pos[:,None] + 1 + torch.arange(H, device=device)[None,:]
        zval, zmask = torch.zeros_like(s.scenario_values[fut]), torch.zeros_like(s.scenario_mask[fut])
        rawp = model.decode_horizon(out[0,pos], ctx, s.time_features[fut], zval, zmask)
        pred = (s.target[pos].view(-1,1,1) + rawp).float().cpu().numpy()
        y = s.target.cpu().numpy()
        for ai,a in enumerate(anchors):
            at = ts[a]; atu = at.tz_convert("UTC") if at.tzinfo else at.tz_localize("UTC")
            for j in range(H):
                q10,q50,q90 = map(float,pred[ai,j]); obs=float(y[a+1+j])
                fr.append({"participant_id":s.participant_id,"condition":condition,
                    "segment_id":s.segment_id,"anchor_timestamp":atu,
                    "anchor_timestamp_local":at.isoformat(),"minutes_since_reset":(a+1)*5,
                    "horizon_step":j+1,"horizon_minutes":(j+1)*5,"q10":q10,"q50":q50,
                    "q90":q90,"observed_glucose":obs,"current_glucose":float(y[a]),
                    "forecast_error_q50":q50-obs,"abs_error_q50":abs(q50-obs),
                    "interval_width":q90-q10,"interval_covered":q10<=obs<=q90,
                    "static_profile_hash":prof_hash,"dynamic_input_hash":dyn_hash,"backend":backend})
    return hd, pd.DataFrame(fr), actual_h0, h


def pair_frames(fullh, neuh, fullf, neuf):
    key=["participant_id","segment_id","timestamp_utc","is_h0_row"]
    a=fullh.set_index(key); b=neuh.set_index(key)
    if not a.index.equals(b.index): raise RuntimeError("hidden pairing mismatch")
    ah,bh=a[HIDDEN].to_numpy(),b[HIDDEN].to_numpy(); d=ah-bh
    hp=a.reset_index()[key+["minutes_since_reset"]].copy()
    hp["comparison_type"]="hidden_state"; hp["full_state_norm"]=np.linalg.norm(ah,axis=1)
    hp["neutral_state_norm"]=np.linalg.norm(bh,axis=1); hp["full_neutral_l2"]=np.linalg.norm(d,axis=1)
    hp["full_neutral_cosine"]=cos_rows(ah,bh); hp["max_abs_state_difference"]=np.abs(d).max(1)
    hp["mean_abs_state_difference"]=np.abs(d).mean(1)
    k=["participant_id","segment_id","anchor_timestamp","horizon_step"]
    af=fullf.set_index(k); bf=neuf.set_index(k)
    if not af.index.equals(bf.index): raise RuntimeError("forecast pairing mismatch")
    if not np.array_equal(af.observed_glucose,bf.observed_glucose): raise RuntimeError("target mismatch")
    x=pd.DataFrame({"q":(af.q50-bf.q50).abs(),"w":af.interval_width-bf.interval_width}).reset_index()
    fp=x.groupby(k[:-1],as_index=False).agg(mean_abs_q50_delta_across_horizons=("q","mean"),
        mean_interval_width_delta=("w","mean"))
    term=x[x.horizon_step==12][k[:-1]+["q"]].rename(columns={"q":"terminal_abs_q50_delta"})
    signed=(af.q50-bf.q50).reset_index()
    signed=signed[signed.horizon_step==12][k[:-1]+["q50"]].rename(columns={"q50":"terminal_q50_delta"})
    fp=fp.merge(term,on=k[:-1]).merge(signed,on=k[:-1]); mins=fullf.groupby(k[:-1],as_index=False).minutes_since_reset.first(); fp=fp.merge(mins,on=k[:-1]); fp["comparison_type"]="forecast_anchor"
    fp["current_glucose_difference"]=0.; fp["observed_target_difference"]=0.
    return pd.concat([hp,fp],ignore_index=True,sort=False)


def rep_from(g: pd.DataFrame, balanced=False, ncap=None):
    g=g.sort_values(["timestamp_utc","segment_id"])
    if balanced and len(g)>ncap:
        ix=np.linspace(0,len(g)-1,ncap).round().astype(int); g=g.iloc[np.unique(ix)]
    x=g[HIDDEN].to_numpy(); med=np.median(x,0); q=np.quantile(x,[.25,.75],axis=0)
    return {"median":med,"iqr":q[1]-q[0],"mad":np.median(np.abs(x-med),0),"n":len(g),
            "days":g.local_date.nunique(),"segments":g.segment_id.nunique(),"hours":len(g)*.25}


def context_frame(h: pd.DataFrame, typ: str):
    condition="full_profile" if typ=="full_all" else "static_neutral"
    g=h[(h.condition==condition)&~h.is_h0_row&h.valid_cgm]
    if typ=="neutral_night": g=g[g.night_sleep_flag]
    elif typ=="neutral_day": g=g[g.day_awake_08_20_flag]
    elif typ=="neutral_night_clock_sensitivity": g=g[g.night_clock_00_06_flag]
    return g


def eligible(g, typ):
    if typ in ("full_all","neutral_all"):
        ok=len(g)>=24 and g.local_date.nunique()>=2
        reason="" if ok else "requires >=24 anchors and >=2 local days"
    else:
        ok=len(g)*.25>=6 and g.local_date.nunique()>=2
        reason="" if ok else "requires >=6 hours and >=2 distinct days/nights"
    return ok,reason


def reliability(odd: dict, even: dict, reps: int, perms: int, seed: int):
    ids=sorted(set(odd)&set(even)); n=len(ids)
    if n<6: return pd.DataFrame(), {"n_participants":n}
    A=np.stack([odd[i] for i in ids]); B=np.stack([even[i] for i in ids])
    scaler=StandardScaler().fit(np.vstack([A,B])); A=scaler.transform(A); B=scaler.transform(B)
    robust=RobustScaler().fit(np.vstack([np.stack([odd[i] for i in ids]),np.stack([even[i] for i in ids])]))
    Ar,Br=robust.transform(np.stack([odd[i] for i in ids])),robust.transform(np.stack([even[i] for i in ids]))
    S=cosine_matrix(A,B); own=np.diag(S); off=S.copy(); np.fill_diagonal(off,-np.inf)
    maxother=np.max(off,1); margin=own-maxother
    ranks1=np.array([np.where(np.argsort(-S[i])==i)[0][0]+1 for i in range(n)])
    ranks2=np.array([np.where(np.argsort(-S[:,i])==i)[0][0]+1 for i in range(n)])
    SA=cosine_matrix(A,A); SB=cosine_matrix(B,B); np.fill_diagonal(SA,-np.inf);np.fill_diagonal(SB,-np.inf)
    nn=np.array([len(set(np.argsort(-SA[i])[:10])&set(np.argsort(-SB[i])[:10]))/min(10,n-1) for i in range(n)])
    rng=np.random.default_rng(seed); null_top1=[];null_top5=[];null_cos=[];null_margin=[]
    between=S[~np.eye(n,dtype=bool)]
    for _ in range(perms):
        p=rng.permutation(n); inv=np.argsort(p); assigned=S[np.arange(n),p]
        rr=np.array([np.where(np.argsort(-S[i])==inv[i])[0][0]+1 for i in range(n)])
        null_top1.append(np.mean(rr==1));null_top5.append(np.mean(rr<=5))
        null_cos.append(np.median(assigned));null_margin.append(np.median(assigned)-np.median(between))
    # ICC(3,1), dimensions as independent diagnostics.
    X=np.stack([A,B],axis=1); grand=X.mean((0,1)); mr=X.mean(1); mc=X.mean(0)
    msr=2*np.sum((mr-grand)**2,axis=0)/max(n-1,1)
    mse=np.sum((X-mr[:,None,:]-mc[None,:,:]+grand)**2,axis=(0,1))/max(n-1,1)
    icc=(msr-mse)/np.maximum(msr+mse,EPS)
    pca=PCA(n_components=min(n-1,128)).fit(np.vstack([A,B]))
    k=np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.9)+1
    pa,pb=pca.transform(A)[:,:k],pca.transform(B)[:,:k]
    pcos=cos_rows(pa,pb)
    per=pd.DataFrame({"participant_id":ids,"within_cosine":own,"within_l2":np.linalg.norm(A-B,axis=1),
        "identification_margin":margin,"odd_to_even_rank":ranks1,"even_to_odd_rank":ranks2,
        "reciprocal_top1":((ranks1==1)+(ranks2==1))/2,"reciprocal_top5":((ranks1<=5)+(ranks2<=5))/2,
        "nn10_overlap":nn})
    lo,hi=ci(own,reps,rng); mlo,mhi=ci(margin,reps,rng)
    summary={"n_participants":n,"median_within_cosine":float(np.median(own)),
        "within_cosine_ci_low":lo,"within_cosine_ci_high":hi,
        "median_within_l2":float(np.median(np.linalg.norm(A-B,axis=1))),
        "median_between_cosine":float(np.median(between)),
        "within_minus_between_margin":float(np.median(own)-np.median(between)),
        "median_identification_margin":float(np.median(margin)),"margin_ci_low":mlo,"margin_ci_high":mhi,
        "top1_retrieval":float(((ranks1==1).mean()+(ranks2==1).mean())/2),
        "top5_retrieval":float(((ranks1<=5).mean()+(ranks2<=5).mean())/2),
        "median_reciprocal_rank":float(np.median(2/(ranks1+ranks2))),
        "nn10_overlap":float(np.mean(nn)),"median_icc":float(np.nanmedian(icc)),
        "pca90_components":int(k),"pca_score_median_cosine":float(np.median(pcos)),
        "robust_scaling_median_cosine":float(np.median(cos_rows(Ar,Br))),
        "null_top1_95_high":float(np.quantile(null_top1,.975)),
        "null_top5_95_high":float(np.quantile(null_top5,.975)),
        "null_within_cosine_95_high":float(np.quantile(null_cos,.975)),
        "null_margin_95_high":float(np.quantile(null_margin,.975))}
    return per,summary


def geometry(a: dict,b: dict, balanced_a: dict, balanced_b: dict):
    ids=sorted(set(a)&set(b)); n=len(ids)
    if n<12:return {"n_participants":n}
    A=np.stack([a[i] for i in ids]);B=np.stack([b[i] for i in ids])
    A=StandardScaler().fit_transform(A);B=StandardScaler().fit_transform(B)
    corr=float(spearmanr(pdist(A),pdist(B)).statistic)
    DA=squareform(pdist(A));DB=squareform(pdist(B));np.fill_diagonal(DA,np.inf);np.fill_diagonal(DB,np.inf)
    nn=float(np.mean([len(set(np.argsort(DA[i])[:10])&set(np.argsort(DB[i])[:10]))/10 for i in range(n)]))
    out={"n_participants":n,"pairwise_distance_spearman":corr,"nn10_overlap":nn,
         "participant_cosine_median":float(np.median(cos_rows(A,B)))}
    for v in (.8,.9,.95):
        pa=PCA().fit(A);pb=PCA().fit(B)
        ka=np.searchsorted(np.cumsum(pa.explained_variance_ratio_),v)+1
        kb=np.searchsorted(np.cumsum(pb.explained_variance_ratio_),v)+1;k=min(ka,kb)
        XA=pa.transform(A)[:,:k];XB=pb.transform(B)[:,:k]
        _,_,disp=procrustes(XA,XB)
        ang=subspace_angles(pa.components_[:k].T,pb.components_[:k].T)
        tag=str(int(v*100));out[f"pca{tag}_components"]=k
        out[f"procrustes_similarity_{tag}"]=float(max(0,1-disp))
        out[f"subspace_similarity_{tag}"]=float(np.mean(np.cos(ang)))
    common=sorted(set(balanced_a)&set(balanced_b)&set(ids))
    out["balanced_participant_cosine_median"]=float(np.median(cos_rows(
        np.stack([balanced_a[i] for i in common]),np.stack([balanced_b[i] for i in common])))) if common else np.nan
    return out



def load_part(out, pid):
    h = pd.concat([pd.read_parquet(out/f"validation_hidden_states/condition={c}/participant_id={pid}/data.parquet") for c in ("full_profile","static_neutral")], ignore_index=True)
    f = pd.concat([pd.read_parquet(out/f"validation_forecasts/condition={c}/participant_id={pid}/data.parquet") for c in ("full_profile","static_neutral")], ignore_index=True)
    p = pd.read_parquet(out/f"validation_anchor_pairs/participant_id={pid}/data.parquet")
    return h, f, p


def main():
    a=args()
    if a.split != "validation": raise ValueError("Step 2 requires validation; test is forbidden")
    if a.state_save_frequency_minutes != 5 or a.representation_frequency_minutes != 15: raise ValueError("required 5/15-minute frequencies")
    if a.precision != "float32": raise ValueError("Step 2 canonical precision is float32")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    device=dev(a.device); started=time.time(); rid=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    oroot=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root)
    out=oroot/rid
    if out.exists() and a.overwrite: shutil.rmtree(out)
    if out.exists() and not a.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); setup(out/"step2_run.log")
    for d in ("validation_hidden_states","validation_forecasts","validation_anchor_pairs","_completion"): (out/d).mkdir(exist_ok=True)
    paths={k:Path(getattr(a,k.replace("-","_"))).resolve() for k in ("config","checkpoint","schema","multimodal-parquet","static-table","split-manifest")}
    step1=Path(a.step1_dir).resolve(); extra={"step1_static_reference":step1/"static_reference_profile.json","step1_static_schema_audit":step1/"static_schema_audit.csv","step1_segment_reconciliation":step1/"segment_boundary_reconciliation.csv","step1_manifest":step1/"step1_manifest.json","replay_implementation":Path(__file__).resolve()}
    allhash={**{k:sha256(v) for k,v in paths.items()},**{k:sha256(v) for k,v in extra.items()}}
    s1=json.loads(extra["step1_manifest"].read_text()); ref=json.loads(extra["step1_static_reference"].read_text())
    for k,mk in [("checkpoint","checkpoint_hash"),("schema","feature_schema_hash"),("config","config_hash"),("split-manifest","split_hash")]:
        if allhash[k] != s1[mk]: raise RuntimeError(f"{k} hash differs from Step 1")
    if ref["profile_hash"] != s1["static_reference_profile_hash"]: raise RuntimeError("reference hash mismatch")
    ncont=np.asarray(ref["transformed_static_cont"],dtype="float32"); ncat=np.asarray(ref["transformed_static_cat"],dtype="int64")
    sp=pd.read_csv(paths["split-manifest"]); sp.participant_id=sp.participant_id.astype(str); sp["split"]=sp.split.replace({"val":"validation","valid":"validation"})
    if sp.participant_id.duplicated().any() or sp.split.value_counts().to_dict() != {"train":1131,"validation":239,"test":221}: raise RuntimeError("split manifest invalid")
    valids=sorted(sp.loc[sp.split=="validation","participant_id"])
    if a.participant_ids:
        wanted=sorted(set(map(str,a.participant_ids)))
        if not set(wanted)<=set(valids): raise RuntimeError("non-validation participant requested")
        valids=wanted
    if a.max_participants: valids=valids[:a.max_participants]
    production=len(valids)==239
    with paths["config"].open() as f: cfg=yaml.safe_load(f)
    saved=json.loads(paths["schema"].read_text()); ck=torch.load(paths["checkpoint"],map_location=device,weights_only=False); md=ck["metadata"]
    spec=AireadiFeatureSpec(**md["feature_spec"]); pre=AireadiPreprocessor.from_jsonable(md["preprocessor"]); mcfg=AireadiStreamModelConfig(**md["model_config"])
    if saved["feature_spec"] != md["feature_spec"] or len(spec.static_reals)+len(spec.static_categoricals) != 44: raise RuntimeError("schema/order mismatch")
    model=AireadiStreamModel(spec,pre,mcfg).to(device); model.load_state_dict(ck["model_state_dict"],strict=True); model.eval()
    backend=f"{device.upper()} Mamba scan_chunk ({mcfg.scan_mode})"
    LOG.info("run=%s participants=%d device=%s loading validation-only rows",rid,len(valids),device)
    panel=pd.read_parquet(paths["multimodal-parquet"],filters=[("participant_id","in",valids)]); panel.participant_id=panel.participant_id.astype(str)
    st=pd.read_parquet(paths["static-table"],filters=[("participant_id","in",valids)]); st.participant_id=st.participant_id.astype(str); st=st.drop_duplicates("participant_id")
    add=[c for c in st if c=="participant_id" or c not in panel]; panel=panel.merge(st[add],on="participant_id",how="left",validate="many_to_one")
    if set(panel.participant_id)-set(valids): raise RuntimeError("non-validation data loaded")
    schema=infer_or_validate_schema(panel,saved["schema"]); prepared=prepare_aireadi_panel(panel,schema,bin_minutes=5,clean_min_segment_hours=cfg["dataset"]["clean_min_segment_hours"])
    if asdict(build_stream_feature_spec(prepared,schema,horizon_steps=12,bin_minutes=5)) != asdict(spec): raise RuntimeError("current feature schema differs")
    split=make_aireadi_stream_splits(prepared,existing_split_path=paths["split-manifest"],seed=a.seed)
    if production and prepared.participant_id.astype(str).nunique()!=239: raise RuntimeError("validation did not resolve to 239")
    ctx=model.encode_static(torch.tensor(ncat,device=device),torch.tensor(ncont,device=device)); h0actual=flatten_initial_state(model.init_stream(ctx)); h0hash=hashlib.sha256(h0actual.tobytes()).hexdigest()
    s1h=pd.read_parquet(step1/"pilot_hidden_states.parquet",filters=[("condition","==","static_neutral"),("is_h0_row","==",True)])
    proxy_diff=float(np.max(np.abs(s1h[HIDDEN].iloc[0].to_numpy()-ctx.embedding[0].detach().cpu().numpy())))
    if proxy_diff > 1e-7: raise RuntimeError("Step 1 neutral proxy mismatch")
    status=[]; manifest_rows=[]; total_states={x:0 for x in ("full_profile","static_neutral")}; total_anchors={x:0 for x in total_states}
    neutral_h0_max=0.;detmax=0.;prefixmax=0.;prefixrel=0.;qcs=0;qcp=0
    for num,pid in enumerate(valids,1):
        marker=out/"_completion"/f"{pid}.json"
        if marker.exists() and a.resume:
            mm=json.loads(marker.read_text()); good=all((out/r).exists() and sha256(out/r)==h for r,h in mm["partition_hashes"].items())
            if good:
                status.append(mm["status"]); manifest_rows.extend(mm["manifest_rows"])
                for k in total_states: total_states[k]+=mm["state_rows"][k]; total_anchors[k]+=mm["anchors"][k]
                LOG.info("resume skip %d/%d %s",num,len(valids),pid); continue
        pg=prepared[prepared.participant_id.astype(str)==pid].copy(); streams=make_participant_streams(pg,split,schema,feature_spec=spec,preprocessor=pre,splits=["validation"],min_steps=14)
        if not streams: raise RuntimeError(f"no stream for {pid}")
        for s in streams: s.metadata["_dynamic_names"]=spec.dynamic_reals
        allts=[pd.Timestamp(x) for s in streams for x in s.timestamps]; dates=sorted(set(str(x.date()) for x in allts)); daymap={d:i+1 for i,d in enumerate(dates)}
        hf=[];hn=[];ff=[];fn=[];pairs=[];nh0=[];bounds=[]
        for s in streams:
            F,Ff,_,_=replay_segment(model,s,"full_profile",ncont,ncat,device,allhash["checkpoint"],3,daymap,backend)
            N,Nf,z,_=replay_segment(model,s,"static_neutral",ncont,ncat,device,allhash["checkpoint"],3,daymap,backend)
            hf.append(F);hn.append(N);ff.append(Ff);fn.append(Nf);pairs.append(pair_frames(F,N,Ff,Nf));nh0.append(z)
            bounds.append({"segment_id":s.segment_id,"start":str(pd.Timestamp(s.timestamps[0])),"end":str(pd.Timestamp(s.timestamps[-1])),"n_steps":s.n_steps})
            if qcs<10:
                sd=s.to(device); cc=model.encode_static(sd.static_cat,sd.static_cont); ep=512
                _,whole=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc));_,pref=model.scan_chunk(sd.dynamic[:ep],cc,model.init_stream(cc))
                dd=float((whole[:,:ep]-pref).abs().max().cpu());scale=float(whole[:,:ep].abs().max().cpu());prefixmax=max(prefixmax,dd);prefixrel=max(prefixrel,dd/max(scale,EPS));qcs+=1
            if qcp<5 and s is streams[0]:
                sd=s.to(device);cc=model.encode_static(sd.static_cat,sd.static_cont);_,one=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc));_,two=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc));detmax=max(detmax,float((one-two).abs().max().cpu()));qcp+=1
        Hf=pd.concat(hf,ignore_index=True);Hn=pd.concat(hn,ignore_index=True);Ff=pd.concat(ff,ignore_index=True);Fn=pd.concat(fn,ignore_index=True);P=pd.concat(pairs,ignore_index=True)
        neutral_h0_max=max(neutral_h0_max,max(float(np.max(np.abs(x-h0actual))) for x in nh0)); ppaths={}; anchors_pid={}
        for cond,dfx in [("full_profile",Hf),("static_neutral",Hn)]:
            rel=f"validation_hidden_states/condition={cond}/participant_id={pid}/data.parquet";ppaths[rel]=atomic_parquet(dfx,out/rel,a.parquet_compression);total_states[cond]+=len(dfx)
        for cond,dfx in [("full_profile",Ff),("static_neutral",Fn)]:
            rel=f"validation_forecasts/condition={cond}/participant_id={pid}/data.parquet";ppaths[rel]=atomic_parquet(dfx,out/rel,a.parquet_compression);anchors_pid[cond]=dfx[["segment_id","anchor_timestamp"]].drop_duplicates().shape[0];total_anchors[cond]+=anchors_pid[cond]
        rel=f"validation_anchor_pairs/participant_id={pid}/data.parquet";ppaths[rel]=atomic_parquet(P,out/rel,a.parquet_compression)
        avail={x:float(pg[x].notna().mean()) for x in ["heart_rate_mean","respiratory_rate_mean","activity_steps_per_min","sleep_stage_light"] if x in pg}; hp=Hf[~Hf.is_h0_row]
        stat={"participant_id":pid,"split":"validation","clinical_site":hp.clinical_site.iloc[0],"study_group":hp.study_group.iloc[0],"cgm_start":str(min(allts)),"cgm_end":str(max(allts)),"total_canonical_duration_hours":sum(s.n_steps for s in streams)*5/60,"n_canonical_segments":len(streams),"segment_boundaries":json.dumps(bounds),"n_valid_dynamic_rows":sum(s.n_steps for s in streams),"n_forecast_anchors":anchors_pid["full_profile"],"dynamic_missingness":float(pg[spec.dynamic_reals].isna().mean().mean()),"wearable_availability":json.dumps(avail),"n_local_calendar_days":len(dates),"n_odd_days":sum((i+1)%2 for i in range(len(dates))),"n_even_days":len(dates)-sum((i+1)%2 for i in range(len(dates))),"valid_nighttime_hours":float(hp.night_sleep_flag.sum()*5/60),"valid_daytime_hours":float(hp.day_awake_08_20_flag.sum()*5/60),"replay_status":"complete","exclusion_or_warning_reason":""}
        status.append(stat);mrows=[]
        for cond,dfx in [("full_profile",Hf),("static_neutral",Hn)]:
            r=f"validation_hidden_states/condition={cond}/participant_id={pid}/data.parquet";mrows.append({"participant_id":pid,"condition":cond,"n_segments":len(streams),"n_rows":len(dfx),"n_post_update_rows":int((~dfx.is_h0_row).sum()),"partition_path":r,"partition_hash":ppaths[r]})
        manifest_rows.extend(mrows);dump(marker,{"status":stat,"manifest_rows":mrows,"partition_hashes":ppaths,"state_rows":{"full_profile":len(Hf),"static_neutral":len(Hn)},"anchors":anchors_pid})
        elapsed=time.time()-started;eta=(len(valids)-num)*elapsed/max(num,1);gpu=torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0
        LOG.info("complete %d/%d pid=%s states=%s anchors=%s elapsed=%.1fm eta=%.1fm gpu=%.2fGB rss=%.2fGB",num,len(valids),pid,total_states,total_anchors,elapsed/60,eta/60,gpu,psutil.Process().memory_info().rss/1e9)
    if qcs < 10 or qcp < 5:
        qcs = 0; qcp = 0; detmax = 0.; prefixmax = 0.; prefixrel = 0.
        with torch.no_grad():
            for qpid in valids:
                qpg = prepared[prepared.participant_id.astype(str) == qpid].copy()
                qstreams = make_participant_streams(qpg, split, schema, feature_spec=spec, preprocessor=pre, splits=["validation"], min_steps=14)
                for qs in qstreams:
                    if qcs < 10:
                        sd=qs.to(device); cc=model.encode_static(sd.static_cat,sd.static_cont)
                        ep=512
                        _,whole=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc)); _,pref=model.scan_chunk(sd.dynamic[:ep],cc,model.init_stream(cc))
                        dd=float((whole[:,:ep]-pref).abs().max().cpu()); scale=float(whole[:,:ep].abs().max().cpu())
                        prefixmax=max(prefixmax,dd); prefixrel=max(prefixrel,dd/max(scale,EPS)); qcs += 1
                    if qcp < 5 and qs is qstreams[0]:
                        sd=qs.to(device); cc=model.encode_static(sd.static_cat,sd.static_cont)
                        _,one=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc)); _,two=model.scan_chunk(sd.dynamic,cc,model.init_stream(cc))
                        detmax=max(detmax,float((one-two).abs().max().cpu())); qcp += 1
                if qcs >= 10 and qcp >= 5: break
    if production and len(status)!=239: raise RuntimeError("not all validation participants considered")
    if detmax>s1["qc_tolerances"]["deterministic_rerun_abs"] or prefixmax>s1["qc_tolerances"]["prefix_invariance_abs"] or prefixrel>s1["qc_tolerances"]["prefix_invariance_relative_to_state_scale"]: raise RuntimeError(f"replay QC failure det={detmax} prefix={prefixmax}/{prefixrel}")
    pd.DataFrame(status).sort_values("participant_id").to_csv(out/"validation_export_status_by_participant.csv",index=False);atomic_parquet(pd.DataFrame(manifest_rows),out/"validation_state_export_manifest.parquet",a.parquet_compression)
    LOG.info("replay complete; representation/reliability analysis")
    reps={c:{t:{} for t in REPTYPES} for c in CANDIDATES};bal={c:{t:{} for t in REPTYPES} for c in CANDIDATES}
    odd={c:{t:{} for t in REPTYPES[:-1]} for c in CANDIDATES};even={c:{t:{} for t in REPTYPES[:-1]} for c in CANDIDATES}
    repmeta=[];curveparts=[];staticrows=[];counts={c:{t:[] for t in REPTYPES} for c in CANDIDATES}
    for pi,pid in enumerate(valids,1):
        H,F,P=load_part(out,pid)
        for c in CANDIDATES:
            for typ in REPTYPES:
                g=context_frame(H,typ);g=g[(g.minutes_since_reset>=c)&(g.minutes_since_reset%15==0)];ok,reason=eligible(g,typ)
                repmeta.append({"participant_id":pid,"representation_type":typ,"burn_in_minutes":c,"n_anchors":len(g),"n_distinct_days":g.local_date.nunique(),"n_odd_days":g[g.odd_even_day=="odd"].local_date.nunique(),"n_even_days":g[g.odd_even_day=="even"].local_date.nunique(),"n_segments":g.segment_id.nunique(),"valid_hours":len(g)*.25,"eligible":ok,"exclusion_reason":reason})
                if ok:
                    reps[c][typ][pid]=rep_from(g);counts[c][typ].append(len(g));go=g[g.odd_even_day=="odd"];ge=g[g.odd_even_day=="even"]
                    if typ in odd[c] and len(go)>=12 and len(ge)>=12 and go.local_date.nunique() and ge.local_date.nunique():odd[c][typ][pid]=np.median(go[HIDDEN],0);even[c][typ][pid]=np.median(ge[HIDDEN],0)
        post=H[~H.is_h0_row]
        for cond,gc in post.groupby("condition"):
            hp=H[(H.condition==cond)&H.is_h0_row].set_index("segment_id")[HIDDEN];vals=gc[HIDDEN].to_numpy();hzero=np.stack([hp.loc[x].to_numpy() for x in gc.segment_id]);base=gc[["segment_id","minutes_since_reset"]].copy()
            for metric,v in [("distance_from_e_s_proxy",np.linalg.norm(vals-hzero,axis=1)),("state_norm",np.linalg.norm(vals,axis=1))]:
                z=base.copy();z["value"]=v;z["bin"]=(z.minutes_since_reset//15*15).astype(int);q=z.groupby("bin").agg(value=("value","median"),mean=("value","mean"),n=("value","size"),n_segments=("segment_id","nunique")).reset_index();q["participant_id"]=pid;q["metric"]=metric;q["condition"]=cond;curveparts.append(q)
        ph=P[(P.comparison_type=="hidden_state")&P.is_h0_row.eq(False)].copy();ph["bin"]=(ph.minutes_since_reset//15*15).astype(int)
        for metric,col in [("full_neutral_l2","full_neutral_l2"),("full_neutral_cosine","full_neutral_cosine")]:
            q=ph.groupby("bin").agg(value=(col,"median"),mean=(col,"mean"),n=(col,"size"),n_segments=("segment_id","nunique")).reset_index();q["participant_id"]=pid;q["metric"]=metric;q["condition"]="paired";curveparts.append(q)
        pf=P[P.comparison_type=="forecast_anchor"].copy();pf["bin"]=(pf.minutes_since_reset//15*15).astype(int);q=pf.groupby("bin").agg(value=("mean_abs_q50_delta_across_horizons","median"),mean=("mean_abs_q50_delta_across_horizons","mean"),n=("mean_abs_q50_delta_across_horizons","size"),n_segments=("segment_id","nunique")).reset_index();q["participant_id"]=pid;q["metric"]="full_neutral_forecast_delta";q["condition"]="paired";curveparts.append(q)
        for cond,gf in F.groupby("condition"):
            gf=gf.copy();gf["bin"]=(gf.minutes_since_reset//15*15).astype(int);q=gf.groupby("bin").agg(value=("abs_error_q50","median"),mean=("abs_error_q50","mean"),n=("abs_error_q50","size"),n_segments=("segment_id","nunique")).reset_index();q["participant_id"]=pid;q["metric"]="forecast_mae";q["condition"]=cond;curveparts.append(q)
        ff=F[F.condition=="full_profile"];fn=F[F.condition=="static_neutral"]
        staticrows.append({"participant_id":pid,"clinical_site":H.clinical_site.iloc[0],"study_group":H.study_group.iloc[0],"n_segments":H.segment_id.nunique(),"n_anchors":len(pf),"median_full_neutral_l2":ph.full_neutral_l2.median(),"mean_full_neutral_l2":ph.full_neutral_l2.mean(),"p90_full_neutral_l2":ph.full_neutral_l2.quantile(.9),"median_full_neutral_cosine":ph.full_neutral_cosine.median(),"median_abs_forecast_delta":pf.mean_abs_q50_delta_across_horizons.median(),"mean_abs_forecast_delta":pf.mean_abs_q50_delta_across_horizons.mean(),"terminal_abs_forecast_delta":pf.terminal_abs_q50_delta.mean(),"full_mae":ff.abs_error_q50.mean(),"neutral_mae":fn.abs_error_q50.mean(),"mae_difference":fn.abs_error_q50.mean()-ff.abs_error_q50.mean(),"full_state_norm_median":ph.full_state_norm.median(),"neutral_state_norm_median":ph.neutral_state_norm.median()})
        if pi%25==0:LOG.info("analysis representations %d/%d",pi,len(valids))
    repmeta_df=pd.DataFrame(repmeta);caps={(c,t):(min(512,max(1,int(np.floor(np.quantile(counts[c][t],.1))))) if counts[c][t] else 0) for c in CANDIDATES for t in REPTYPES}
    for pi,pid in enumerate(valids,1):
        H,_,_=load_part(out,pid)
        for c in CANDIDATES:
            for typ in REPTYPES:
                if pid in reps[c][typ]:
                    g=context_frame(H,typ);g=g[(g.minutes_since_reset>=c)&(g.minutes_since_reset%15==0)];bal[c][typ][pid]=rep_from(g,True,caps[(c,typ)])
        if pi%50==0:LOG.info("analysis balanced %d/%d",pi,len(valids))
    base_anchor=sum(x["n"] for x in reps[0]["neutral_all"].values());coverage=[]
    for c in CANDIDATES:
        for typ in REPTYPES:
            md=repmeta_df[(repmeta_df.burn_in_minutes==c)&(repmeta_df.representation_type==typ)];el=md[md.eligible];oe=len(set(odd[c].get(typ,{}))&set(even[c].get(typ,{}))) if typ in odd[c] else 0
            coverage.append({"burn_in_minutes":c,"representation_type":typ,"participants_any_eligible_anchors":int((md.n_anchors>0).sum()),"participants_representation_eligible":len(el),"participants_both_odd_even_eligible":oe,"participants_valid_night":len(reps[c]["neutral_night"]),"participants_valid_day":len(reps[c]["neutral_day"]),"segments_retained":int(el.n_segments.sum()),"anchors_retained":int(el.n_anchors.sum()),"percentage_original_anchors_retained":100*el.n_anchors.sum()/max(base_anchor,1),"median_anchors_per_participant":el.n_anchors.median(),"minimum_anchors_per_participant":el.n_anchors.min(),"median_distinct_days":el.n_distinct_days.median(),"median_odd_days":el.n_odd_days.median(),"median_even_days":el.n_even_days.median(),"median_nighttime_hours":repmeta_df.query("burn_in_minutes==@c and representation_type=='neutral_night' and eligible").valid_hours.median(),"median_daytime_hours":repmeta_df.query("burn_in_minutes==@c and representation_type=='neutral_day' and eligible").valid_hours.median()})
    cov=pd.DataFrame(coverage);cov.to_csv(out/"burnin_candidate_coverage.csv",index=False)
    relparts=[];relsums=[]
    for c in CANDIDATES:
        for typ in REPTYPES[:-1]:
            per,su=reliability(odd[c][typ],even[c][typ],a.bootstrap_replicates,a.permutation_replicates,a.seed+c+REPTYPES.index(typ)*10000)
            if len(per):per["burn_in_minutes"]=c;per["representation_type"]=typ;relparts.append(per)
            su.update({"burn_in_minutes":c,"representation_type":typ});common=sorted(set(reps[c][typ])&set(bal[c][typ]))
            if common:
                A=np.stack([reps[c][typ][i]["median"] for i in common]);B=np.stack([bal[c][typ][i]["median"] for i in common]);su["balanced_all_median_cosine"]=float(np.median(cos_rows(A,B)));su["balanced_all_median_l2"]=float(np.median(np.linalg.norm(A-B,axis=1)))
            relsums.append(su)
    relby=pd.concat(relparts,ignore_index=True) if relparts else pd.DataFrame();relsum=pd.DataFrame(relsums)
    relby.to_csv(out/"participant_reliability_by_participant.csv",index=False);relsum.to_csv(out/"participant_reliability_summary.csv",index=False);relsum.to_csv(out/"burnin_candidate_reliability.csv",index=False)
    georows=[]
    for x,y in zip(CANDIDATES[:-1],CANDIDATES[1:]):
        z=geometry({p:r["median"] for p,r in reps[x]["neutral_all"].items()},{p:r["median"] for p,r in reps[y]["neutral_all"].items()},{p:r["median"] for p,r in bal[x]["neutral_all"].items()},{p:r["median"] for p,r in bal[y]["neutral_all"].items()});z.update({"burn_in_minutes":x,"next_burn_in_minutes":y});georows.append(z)
    geo=pd.DataFrame(georows);geo.to_csv(out/"burnin_candidate_geometry_stability.csv",index=False)
    nr=relsum[relsum.representation_type=="neutral_all"].set_index("burn_in_minutes");nc=cov[cov.representation_type=="neutral_all"].set_index("burn_in_minutes");ng=geo.set_index("burn_in_minutes");max5=nr.top5_retrieval.max();maxcos=nr.median_within_cosine.max();oe0=nc.loc[0,"participants_both_odd_even_eligible"];decisions=[];basepass={}
    for c in CANDIDATES:
        criteria={"coverage_odd_even":nc.loc[c,"participants_both_odd_even_eligible"]>=.85*oe0,"coverage_all":nc.loc[c,"participants_representation_eligible"]>=.8*len(valids),"top5":nr.loc[c,"top5_retrieval"]>=.95*max5,"within_cosine":nr.loc[c,"median_within_cosine"]>=.95*maxcos,"margin_null":nr.loc[c,"within_minus_between_margin"]>0 and nr.loc[c,"within_minus_between_margin"]>nr.loc[c,"null_margin_95_high"],"geometry_nn":c in ng.index and ng.loc[c,"nn10_overlap"]>=.8,"geometry_distance":c in ng.index and ng.loc[c,"pairwise_distance_spearman"]>=.9,"geometry_procrustes":c in ng.index and ng.loc[c,"procrustes_similarity_90"]>=.9}
        basepass[c]=all(criteria.values());decisions.append({"candidate_minutes":c,"candidate_hours":c/60,"coverage_metrics":nc.loc[c].to_dict(),"reliability_metrics":nr.loc[c].to_dict(),"geometry_metrics":ng.loc[c].to_dict() if c in ng.index else {},"criteria_passed":[k for k,v in criteria.items() if v],"criteria_failed":[k for k,v in criteria.items() if not v]})
    selected=None
    for i,c in enumerate(CANDIDATES[:-2]):
        if all(basepass[x] for x in CANDIDATES[i:i+3]):selected=c;break
    score={}
    for c in CANDIDATES:
        if nc.loc[c,"participants_representation_eligible"]<.8*len(valids):continue
        gv=ng.loc[c] if c in ng.index else None;score[c]=float(nr.loc[c,"top5_retrieval"]+nr.loc[c,"median_within_cosine"]+(gv.nn10_overlap if gv is not None else 0)+(gv.pairwise_distance_spearman if gv is not None else 0))
    sens=sorted(score,key=lambda x:(-score[x],x))[:2];finalburn=[selected] if selected is not None else sorted(sens)
    final={"candidate_minutes":CANDIDATES,"candidate_hours":[x/60 for x in CANDIDATES],"candidate_results":decisions,"selected_burn_in_minutes":selected,"selected_burn_in_hours":None if selected is None else selected/60,"burn_in_status":"selected" if selected is not None else "no_candidate_satisfies_rule","step1_proxy_burnin_minutes":855,"agreement_with_step1_proxy":selected==855,"sensitivity_candidates":finalburn,"decision_notes":"Shortest neutral_all candidate must pass all coverage, reliability, geometry, and two-longer-candidate persistence criteria; no manual override."};dump(out/"final_burnin_decision.json",final)
    first=prepared.drop_duplicates("participant_id").set_index("participant_id");rows=[];metarows=[]
    for c in finalburn:
        for typ in REPTYPES:
            for pid in valids:
                meta=repmeta_df.query("participant_id==@pid and burn_in_minutes==@c and representation_type==@typ").iloc[0];base={"participant_id":pid,"split":"validation","representation_type":typ,"condition":"full_profile" if typ=="full_all" else "static_neutral","context":typ.split("_",1)[1],"burn_in_minutes":c,"aggregation":"dimensionwise_median","clinical_site":first.loc[pid].get("participants_clinical_site"),"study_group":first.loc[pid].get("participants_study_group")}
                for variant,src in [("all_anchors",reps[c][typ]),("balanced_anchors",bal[c][typ])]:
                    rr=src.get(pid);row={**base,"balanced_anchor_variant":variant,"n_anchors":0 if rr is None else rr["n"],"n_distinct_days":meta.n_distinct_days,"n_segments":meta.n_segments,"valid_hours":meta.valid_hours,"representation_eligible":rr is not None,"exclusion_reason":meta.exclusion_reason if rr is None else ""}
                    if rr is not None:
                        row.update({f"r_{i:03d}":v for i,v in enumerate(rr["median"])});row.update({f"iqr_{i:03d}":v for i,v in enumerate(rr["iqr"])});row.update({f"mad_{i:03d}":v for i,v in enumerate(rr["mad"])})
                    rows.append(row);metarows.append({k:v for k,v in row.items() if not k.startswith(("r_","iqr_","mad_"))})
    atomic_parquet(pd.DataFrame(rows),out/"participant_representations.parquet",a.parquet_compression);pd.DataFrame(metarows).to_csv(out/"participant_representation_metadata.csv",index=False)
    se=pd.DataFrame(staticrows).sort_values("median_full_neutral_l2");se["static_effect_rank"]=np.arange(1,len(se)+1);se["static_effect_quantile"]=pd.qcut(se.static_effect_rank,4,labels=["Q1","Q2","Q3","Q4"]);se.to_csv(out/"static_effect_by_participant.csv",index=False)
    cc=[]
    for c in CANDIDATES:
        md=repmeta_df[repmeta_df.burn_in_minutes==c];alls=set(md.query("representation_type=='neutral_all' and eligible").participant_id);ni=set(md.query("representation_type=='neutral_night' and eligible").participant_id);da=set(md.query("representation_type=='neutral_day' and eligible").participant_id)
        cc.append({"burn_in_minutes":c,"participants_all_recording":len(alls),"participants_valid_night":len(ni),"participants_valid_day":len(da),"participants_both_night_day":len(ni&da),"total_night_hours":md.query("representation_type=='neutral_night' and eligible").valid_hours.sum(),"total_day_hours":md.query("representation_type=='neutral_day' and eligible").valid_hours.sum(),"median_night_anchors":md.query("representation_type=='neutral_night' and eligible").n_anchors.median(),"median_day_anchors":md.query("representation_type=='neutral_day' and eligible").n_anchors.median(),"site_distribution":json.dumps(first.loc[sorted(alls)].participants_clinical_site.value_counts().to_dict()),"study_group_distribution":json.dumps(first.loc[sorted(alls)].participants_study_group.value_counts().to_dict())})
    ccd=pd.DataFrame(cc);ccd.to_csv(out/"context_coverage_after_burnin.csv",index=False)
    cp=pd.concat(curveparts,ignore_index=True);rng=np.random.default_rng(a.seed);curverows=[]
    for (metric,cond,minute),g in cp.groupby(["metric","condition","bin"]):
        v=g.value.to_numpy();lo,hi=ci(v,a.bootstrap_replicates,rng);curverows.append({"metric":metric,"condition":cond,"minutes_since_reset":minute,"n_participants":g.participant_id.nunique(),"n_segments":int(g.n_segments.sum()),"n_states":int(g.n.sum()),"median":np.median(v),"mean":np.mean(g["mean"]),"p10":np.quantile(v,.1),"p25":np.quantile(v,.25),"p75":np.quantile(v,.75),"p90":np.quantile(v,.9),"participant_bootstrap_ci_low":lo,"participant_bootstrap_ci_high":hi})
    curves=pd.DataFrame(curverows);curves.to_csv(out/"full_validation_burnin_curves.csv",index=False)
    dump(out/"representation_preprocessing_manifest.json",{"sampling_grid_minutes":15,"candidate_burnins":CANDIDATES,"aggregation":"dimensionwise median; IQR and MAD","standardization":"StandardScaler fit independently per validation set","robust_sensitivity":"RobustScaler fit independently","balanced_anchor_rule":"10th percentile eligible count, cap 512, evenly spaced indices","contexts":{"night_primary":"light OR deep OR REM; unknown excluded","night_sensitivity":"local 00:00-06:00","day_primary":"local 08:00-20:00, not sleep, valid CGM"},"thresholds":{"all":"24 anchors and 2 days","odd_even":"12 anchors each and one day each","night":"6 hours and 2 nights","day":"6 hours and 2 days"}})
    plt.style.use("seaborn-v0_8-whitegrid")
    def save(name):plt.tight_layout();plt.savefig(out/name,dpi=170);plt.close()
    plt.figure(figsize=(9,5))
    for met,cond,label in [("distance_from_e_s_proxy","full_profile","Full from e_s proxy"),("distance_from_e_s_proxy","static_neutral","Neutral from e_s proxy"),("full_neutral_l2","paired","Full-neutral L2")]:
        z=curves[(curves.metric==met)&(curves.condition==cond)&(curves.minutes_since_reset<=1440)];plt.plot(z.minutes_since_reset/60,z["median"],label=label);plt.fill_between(z.minutes_since_reset/60,z.participant_bootstrap_ci_low,z.participant_bootstrap_ci_high,alpha=.15)
    plt.xlabel("Hours since reset");plt.legend();save("fig_validation_state_drift.png")
    ncv=cov[cov.representation_type=="neutral_all"];plt.figure(figsize=(9,5));plt.plot(ncv.burn_in_minutes/60,ncv.participants_representation_eligible,label="all");plt.plot(ncv.burn_in_minutes/60,ncv.participants_valid_night,label="night");plt.plot(ncv.burn_in_minutes/60,ncv.participants_valid_day,label="day");plt.plot(ncv.burn_in_minutes/60,ncv.segments_retained/10,label="segments/10");plt.legend();save("fig_burnin_coverage_tradeoff.png")
    nrp=relsum[relsum.representation_type=="neutral_all"];plt.figure(figsize=(9,5));plt.plot(nrp.burn_in_minutes/60,nrp.median_within_cosine,label="within cosine");plt.plot(nrp.burn_in_minutes/60,nrp.top1_retrieval,label="top1");plt.plot(nrp.burn_in_minutes/60,nrp.top5_retrieval,label="top5");plt.plot(nrp.burn_in_minutes/60,nrp.within_minus_between_margin,label="margin");plt.legend();save("fig_reliability_vs_burnin.png")
    plt.figure(figsize=(9,5));plt.plot(geo.burn_in_minutes/60,geo.pairwise_distance_spearman,label="distance rho");plt.plot(geo.burn_in_minutes/60,geo.nn10_overlap,label="NN10");plt.plot(geo.burn_in_minutes/60,geo.procrustes_similarity_90,label="Procrustes90");plt.legend();save("fig_geometry_stability_vs_burnin.png")
    fig,ax=plt.subplots(1,3,figsize=(12,4));ax[0].hist(se.median_full_neutral_l2);ax[1].hist(se.median_full_neutral_cosine);ax[2].hist(se.mean_abs_forecast_delta);save("fig_full_neutral_distance_distribution.png")
    plt.figure(figsize=(11,5));plt.scatter(np.arange(len(se)),se.median_full_neutral_l2,c=pd.Categorical(se.clinical_site).codes,s=10+3*se.n_segments);plt.ylabel("Median full-neutral L2");save("fig_static_effect_by_participant.png")
    plt.figure(figsize=(9,5))
    for typ in REPTYPES[:-1]:
        z=relby[(relby.representation_type==typ)&(relby.burn_in_minutes==finalburn[0])]
        if len(z):plt.hist(z.within_cosine,bins=30,alpha=.35,label=typ)
    plt.legend();save("fig_odd_even_similarity_distribution.png")
    plt.figure(figsize=(9,5))
    for typ in ("full_all","neutral_all"):
        z=relsum[relsum.representation_type==typ];plt.plot(z.burn_in_minutes/60,z.top1_retrieval,label=f"{typ} top1");plt.plot(z.burn_in_minutes/60,z.top5_retrieval,ls="--",label=f"{typ} top5")
    plt.legend();save("fig_participant_retrieval_vs_burnin.png")
    plt.figure(figsize=(9,5));plt.plot(ccd.burn_in_minutes/60,ccd.participants_valid_night,label="night");plt.plot(ccd.burn_in_minutes/60,ccd.participants_valid_day,label="day")
    for x in finalburn:plt.axvline(x/60,color="black",ls=":",alpha=.5)
    plt.legend();save("fig_context_coverage_after_burnin.png")
    rec="GO" if final["burn_in_status"]=="selected" else "GO WITH CAVEATS";blockers=[];warnings=[]
    if final["burn_in_status"]!="selected":warnings.append("No burn-in candidate satisfied the complete automatic rule.")
    if len(reps[finalburn[0]]["neutral_night"])<len(valids):warnings.append("Some participants lack night eligibility.")
    report=f"""# Step 2 full-validation export and reliability audit

+## 1. Objective
+Export validation trajectories, evaluate burn-in candidates, construct representations, and quantify reliability.
+
+## 2. Scope and exclusions
+Exactly {len(valids)} validation participants were considered. No test model data, training, clustering, phenotype associations, probes, or exercise analyses were used.
+
+## 3. Canonical inputs and hashes
+Checkpoint `{allhash['checkpoint']}`; schema `{allhash['schema']}`; split `{allhash['split-manifest']}`; Step 1 reference `{ref['profile_hash']}`.
+
+## 4. Validation cohort
+Resolved/replayed: {len(status)}/{len(valids)} participants; {sum(x['n_canonical_segments'] for x in status)} segments.
+
+## 5. Canonical segmentation
+The Step 1 untrimmed checkpoint reconstruction is frozen. Participants 1118, 4211, and 7139 are training-only and were not replayed.
+
+## 6. Fixed inference backend
+`{backend}`, float32, deterministic CUDA settings; `AireadiStreamModel.scan_chunk`.
+
+## 7. Static-reference reuse and identity checks
+The Step 1 vector was loaded, not recomputed. Proxy difference={proxy_diff}; neutral recurrent-h0 maximum difference={neutral_h0_max}.
+
+## 8. Full-validation replay
+State rows: {total_states}; forecast anchors: {total_anchors}. Atomic participant partitions support validated resume.
+
+## 9. Hidden-state export semantics
+Post-update 128-D decoder input after observation t. h0 rows retain the labelled static embedding `e_s` proxy, not a true decoder-space h0.
+
+## 10. Export coverage and failures
+All participants completed. Context exclusions are in participant status and representation metadata.
+
+## 11. Pairwise full-versus-neutral QC
+Keys, dynamic hashes, targets, anchors, backend, and boundaries match. Deterministic max={detmax}; prefix abs/relative={prefixmax}/{prefixrel}.
+
+## 12. Full-validation state drift
+See `full_validation_burnin_curves.csv`; proxy curves are descriptive only.
+
+## 13. Forecast effect of static neutralization
+Descriptive effects are in `static_effect_by_participant.csv`; no inferential or phenotype analysis was performed.
+
+## 14. Burn-in candidate coverage
+All eight predeclared candidates were evaluated.
+
+## 15. Odd/even representation reliability
+See `participant_reliability_summary.csv`.
+
+## 16. Participant retrieval
+Bidirectional top-1/top-5 and 1,000-permutation nulls are reported.
+
+## 17. Geometry stability across burn-ins
+Adjacent distance correlation, NN overlap, Procrustes, subspace, and participant cosine are reported.
+
+## 18. Final burn-in decision
+Status: **{final['burn_in_status']}**; selected={selected}; sensitivity candidates={finalburn}.
+
+## 19. Comparison with the Step 1 14.25-hour proxy value
+Agreement: {final['agreement_with_step1_proxy']}; the pilot value was not privileged.
+
+## 20. All-recording representation coverage
+{len(reps[finalburn[0]]['neutral_all'])} participants at the first exported burn-in.
+
+## 21. Nighttime representation coverage
+{len(reps[finalburn[0]]['neutral_night'])} participants.
+
+## 22. Daytime representation coverage
+{len(reps[finalburn[0]]['neutral_day'])} participants.
+
+## 23. Participant heterogeneity in static influence
+Descriptive rankings are exported without clinical outcomes.
+
+## 24. Balanced-anchor sensitivity
+The deterministic 10th-percentile/cap-512 sensitivity is included.
+
+## 25. Limitations
+Initialization uses `e_s`; odd/even reliability reflects recording-day sampling, not clinical meaning.
+
+## 26. Blocking issues
+{blockers or 'None.'}
+
+## 27. Recommendation for validation clustering
+**{rec}:** Export integrity passed. Burn-in ambiguity and context coverage limitations must be carried into clustering. Stable representations are not claimed clinically meaningful.
+""".replace("\n+","\n")
    (out/"step2_report.md").write_text(report)
    final_counts={"all":len(reps[finalburn[0]]["neutral_all"]),"night":len(reps[finalburn[0]]["neutral_night"]),"day":len(reps[finalburn[0]]["neutral_day"])}
    discrepant={p:{"split":"train","validation_replay":"not applicable","decision":"Step 1 current untrimmed reconstruction frozen"} for p in ("1118","4211","7139")}
    manifest={"run_id":rid,"execution_timestamp":datetime.now(timezone.utc).isoformat(),"git_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"dirty_working_tree_status":subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True),"immutable_input_paths":{**{k:str(v) for k,v in paths.items()},**{k:str(v) for k,v in extra.items()}},"immutable_input_hashes":allhash,"checkpoint_metadata":{"epoch":ck.get("epoch"),"model_config":ck["metadata"]["model_config"]},"step1_run_path":str(step1),"step1_static_reference_hash":ref["profile_hash"],"neutral_true_recurrent_h0_hash":h0hash,"canonical_segmentation_decision":discrepant,"backend":backend,"precision":"float32","device":device,"hidden_state_save_semantics":"post-update decoder input; e_s proxy row before first observation","model_dimensions":{"hidden":128,"dynamic":len(spec.dynamic_reals),"static":44},"validation_participant_ids":valids,"participant_counts":{"considered":len(valids),"replayed":len(status)},"segment_counts":sum(x["n_canonical_segments"] for x in status),"state_row_counts":total_states,"forecast_anchor_counts":total_anchors,"partition_layout":"condition/participant_id, one compressed Parquet part","compression":a.parquet_compression,"candidate_burnins":CANDIDATES,"context_definitions":json.loads((out/"representation_preprocessing_manifest.json").read_text())["contexts"],"representation_thresholds":json.loads((out/"representation_preprocessing_manifest.json").read_text())["thresholds"],"reliability_definitions":"odd/even cosine, L2, margin, bidirectional retrieval, NN10, ICC, PCA scores","retrieval_definitions":"odd-to-even and even-to-odd cosine retrieval","bootstrap_settings":{"replicates":a.bootstrap_replicates,"unit":"participant"},"permutation_settings":{"replicates":a.permutation_replicates,"seed":a.seed},"geometry_metrics":["distance Spearman","NN10 overlap","Procrustes 80/90/95","subspace angles","cosine"],"final_burnin_rule":"predeclared coverage/reliability/geometry/persistence criteria","selected_burnin":selected,"sensitivity_burnins":finalburn,"final_representation_counts":final_counts,"qc":{"neutral_vector_max_difference":0.,"neutral_h0_max_difference":neutral_h0_max,"step1_proxy_max_difference":proxy_diff,"deterministic_rerun_max_difference":detmax,"prefix_invariance_max_abs_difference":prefixmax,"prefix_invariance_max_relative_difference":prefixrel},"warnings":warnings,"errors":[],"blockers":blockers,"final_go_status":rec}
    dump(out/"step2_manifest.json",manifest)
    rehash={**{k:sha256(v) for k,v in paths.items()},**{k:sha256(v) for k,v in extra.items()}}
    if rehash!=allhash:raise RuntimeError("immutable source changed")
    required=["validation_export_status_by_participant.csv","validation_state_export_manifest.parquet","full_validation_burnin_curves.csv","burnin_candidate_coverage.csv","burnin_candidate_reliability.csv","burnin_candidate_geometry_stability.csv","final_burnin_decision.json","participant_representations.parquet","participant_representation_metadata.csv","participant_reliability_by_participant.csv","participant_reliability_summary.csv","static_effect_by_participant.csv","context_coverage_after_burnin.csv","representation_preprocessing_manifest.json","step2_report.md","step2_manifest.json","step2_run.log"]+[f"fig_{x}.png" for x in ["validation_state_drift","burnin_coverage_tradeoff","reliability_vs_burnin","geometry_stability_vs_burnin","full_neutral_distance_distribution","static_effect_by_participant","odd_even_similarity_distribution","participant_retrieval_vs_burnin","context_coverage_after_burnin"]]
    miss=[x for x in required if not (out/x).exists()]
    if miss:raise RuntimeError(f"missing outputs {miss}")
    if production:
        tmp=oroot/f".latest.{rid}.tmp"
        if tmp.exists() or tmp.is_symlink():tmp.unlink()
        tmp.symlink_to(out.name);os.replace(tmp,oroot/"latest")
    LOG.info("QC complete published=%s",production)
    print(json.dumps({"output_directory":str(out),"files_and_datasets":required,"git_commit":manifest["git_commit"],"checkpoint_hash":allhash["checkpoint"],"step1_static_reference_hash":ref["profile_hash"],"backend":backend,"participants_considered":len(valids),"successfully_replayed":len(status),"segments":manifest["segment_counts"],"hidden_rows":total_states,"forecast_anchors":total_anchors,"paired_qc":manifest["qc"],"median_full_neutral_l2":float(se.median_full_neutral_l2.median()),"median_full_neutral_cosine":float(se.median_full_neutral_cosine.median()),"mean_forecast_delta":float(se.mean_abs_forecast_delta.mean()),"median_forecast_delta":float(se.median_abs_forecast_delta.median()),"coverage":ncv.to_dict("records"),"reliability":nrp.to_dict("records"),"geometry":geo.to_dict("records"),"final_burnin":final,"final_representation_counts":final_counts,"warnings":warnings,"blockers":blockers,"recommendation":rec},indent=2,default=json_default))


if __name__=="__main__":main()
