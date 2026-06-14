#!/usr/bin/env python
"""Block 5 - optional decoder-adapter ablation (feature-as-input, shape-aware).

ABLATION ONLY. An additive per-horizon residual ``r(C, M, h)`` fits each horizon
independently and cannot represent a meal, which is a shaped rise-then-fall
curve. This adapter instead fits the meal feature as an input covariate to a
single joint multi-horizon corrector, so the correction is a shaped 12-step
trajectory:

    yhat[:, h] = q50[:, h] + adapter(base_anchor, meal_anchor)[h]

It is compared on the same gate endpoints against A' and against the additive
residual form. It never changes the headline unless it passes the coded gate;
this script only reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_meal_feature_evaluation as RUN     # noqa: E402  (module-level helpers)
from ssmcgm.meal_transfer import online_evaluation as OE   # noqa: E402
from ssmcgm.meal_transfer import endpoints as EP           # noqa: E402

OUT_DIR = RUN.OUT_DIR


def _anchor_design(df: pd.DataFrame, base: list[str], meal: list[str]):
    """One row per (participant, anchor_ds): anchor features + 12-step residual."""
    keep = ["participant_id", "anchor_ds", "horizon_step", "q50", "target"] + \
           [c for c in base + meal if c not in ("horizon_step",)]
    sub = df[keep].copy()
    feat_cols = [c for c in base + meal if c != "horizon_step"]
    anchors = sub.drop_duplicates(["participant_id", "anchor_ds"]).set_index(
        ["participant_id", "anchor_ds"])
    X = anchors[feat_cols]
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.astype(float)
    # residual matrix: rows=anchors, cols=horizon 1..12
    piv = sub.pivot_table(index=["participant_id", "anchor_ds"], columns="horizon_step",
                          values="target", aggfunc="first")
    q50p = sub.pivot_table(index=["participant_id", "anchor_ds"], columns="horizon_step",
                           values="q50", aggfunc="first")
    horizons = sorted(c for c in piv.columns)
    R = (piv[horizons] - q50p[horizons])
    common = X.index.intersection(R.index)
    return X.loc[common], R.loc[common], q50p.loc[common, horizons], horizons


def _fit_adapter(Xtr, Rtr, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    valid = np.isfinite(Rtr.to_numpy()).all(axis=1)
    base = HistGradientBoostingRegressor(
        max_iter=cfg.get("max_iter", 200), learning_rate=cfg.get("learning_rate", 0.04),
        max_depth=cfg.get("max_depth", 4), l2_regularization=cfg.get("l2_regularization", 10.0),
        random_state=cfg.get("random_state", 42))
    model = MultiOutputRegressor(base)
    model.fit(Xtr.to_numpy(dtype=np.float64)[valid], Rtr.to_numpy(dtype=np.float64)[valid])
    return model


def _apply_adapter(model, X, q50p, horizons) -> pd.DataFrame:
    corr = model.predict(X.to_numpy(dtype=np.float64))
    pred = q50p.to_numpy(dtype=np.float64) + corr
    return pd.DataFrame(pred, index=X.index, columns=horizons)


def _expand_to_rows(test: pd.DataFrame, pred_wide: pd.DataFrame) -> np.ndarray:
    """Map a wide (anchor x horizon) prediction back to the long test rows."""
    long = pred_wide.stack().rename("pred_adapter").reset_index()
    long.columns = ["participant_id", "anchor_ds", "horizon_step", "pred_adapter"]
    m = test.merge(long, on=["participant_id", "anchor_ds", "horizon_step"], how="left")
    return m["pred_adapter"].to_numpy(dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/meal_transfer/decoder_adapter.yaml"))
    ap.add_argument("--max-participants", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg = cfg["adapter"]
    setup = acfg["setup"]
    base = RUN.SLC_BASE if acfg.get("base", "slc") == "slc" else RUN.CONTEXT_BASE
    meal = RUN.MEAL_SETUPS[setup]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    states = pd.read_parquet(RUN.ONLINE_DIR / "online_meal_states.parquet")
    sizes = pd.read_parquet(RUN.ONLINE_DIR / "online_response_size_predictions.parquet")
    mask = pd.read_parquet(RUN.ONLINE_DIR / "independent_meal_eval_mask.parquet")
    val_a = pd.read_parquet(RUN.MEAL_DIR / "_aligned_validation.parquet")
    tst_a = pd.read_parquet(RUN.MEAL_DIR / "_aligned_test.parquet")
    keep_union = (set(val_a["participant_id"].astype(str)) | set(tst_a["participant_id"].astype(str)))
    for f in (states, sizes, mask):
        f["participant_id"] = f["participant_id"].astype(str)
    states = states[states["participant_id"].isin(keep_union)].copy()
    sizes = sizes[sizes["participant_id"].isin(keep_union)].copy()
    mask = mask[mask["participant_id"].isin(keep_union)].copy()

    train = OE.attach_online_frame(val_a, states, sizes, mask)
    test = OE.attach_online_frame(tst_a, states, sizes, mask)
    extra = RUN._build_extra_anchor(False, "causal", keep_union)
    cols = ["dy15", "dy30", "dy60", "level", "hour_sin", "hour_cos"]
    train = RUN._attach_extra(train, extra, cols)
    test = RUN._attach_extra(test, extra, cols)
    non_insulin = test["med_insulin"].fillna(0).to_numpy() == 0

    # additive residual (current form)
    ab = OE.AblationConfig()
    add_model = RUN._fit(train, base, meal, ab)
    add_pred = RUN._apply(test, add_model, base, meal)
    aprime_model = RUN._fit(train, base, [], ab)
    aprime_pred = RUN._apply(test, aprime_model, base, [])

    # shape-aware adapter (feature as input, joint horizons)
    Xtr, Rtr, _q, _h = _anchor_design(train, base, meal)
    Xte, _Rte, q50te, horizons = _anchor_design(test, base, meal)
    adapter = _fit_adapter(Xtr, Rtr, acfg)
    pred_wide = _apply_adapter(adapter, Xte, q50te, horizons)
    adapter_pred = _expand_to_rows(test, pred_wide)
    # fall back to additive where the adapter could not map a row
    adapter_pred = np.where(np.isfinite(adapter_pred), adapter_pred, add_pred)

    rows = []
    for name, pred in [("Aprime_baseline", aprime_pred),
                       (f"{setup}_additive_residual", add_pred),
                       (f"{setup}_feature_as_input_adapter", adapter_pred)]:
        rows.append({"setup": name, "cohort": "non_insulin",
                     **EP.forecast_metrics(test[non_insulin], pred[non_insulin])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "decoder_adapter_ablation.csv", index=False)

    # does the adapter beat A' on >=2 gate endpoints? (informational gate echo)
    base_m = out[out.setup == "Aprime_baseline"].iloc[0]
    adp_m = out[out.setup.str.endswith("adapter")].iloc[0]
    improved = []
    for ep, lower in EP.GATE_ENDPOINTS.items():
        d = (base_m[ep] - adp_m[ep]) if lower else (adp_m[ep] - base_m[ep])
        if np.isfinite(d) and d > 0:
            improved.append(ep)
    decision = {"feature_as_input": bool(acfg.get("feature_as_input", True)),
                "setup": setup, "adapter_improved_endpoints_vs_Aprime": improved,
                "n_improved": len(improved), "passes_min2": len(improved) >= 2,
                "allow_headline_change": bool(cfg["run"].get("allow_headline_change", False)),
                "note": "ablation only; never changes headline unless the full coded gate selects it"}
    (OUT_DIR / "decoder_adapter_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    print("\n=== DECODER ADAPTER ABLATION (non-insulin) ===")
    show = list(EP.GATE_ENDPOINTS) + ["MAE_60min"]
    print("setup | " + " | ".join(show))
    for _, r in out.iterrows():
        print(f"{r.setup} | " + " | ".join(RUN._fmt(r[c]) for c in show))
    print("\nadapter vs A' improved endpoints:", improved, "-> passes_min2:", len(improved) >= 2)
    print("ablation only; headline unchanged (allow_headline_change=False)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
