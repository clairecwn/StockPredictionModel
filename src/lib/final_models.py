"""Held-out (split conformal) calibration of the final 10%/90% quantile models used by the live board.

Why: the final models are fit on all events, so their own errors are in-sample and too small, and the old conformal
correction was stored in % units while the quantile models predict in noise-scaled units (return targets), so it was
almost ignored live. Here, for each target:
  1. the 10% and 90% quantile models are refit on events up to CAL_YEARS (+ the purge gap) before the last event;
  2. the conformal correction c is the finite-sample 80% quantile of max(lo - y, y - hi) on the last CAL_YEARS,
     in the model's own units, so the live interval is (lo - c, hi + c) * noise scale.
The point model is untouched. Coverage on the calibration window is ~80% by construction; the honest walk-forward
coverage is the one in the 17_train_risk_models report.
Run: LIVE_FEATS=1 python src/lib/final_models.py   (also called at the end of 17_train_risk_models.py)
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from config import MODELS
from modeling import ev, PURGE, TARGETS, to_model_target

CAL_YEARS = 2

def calibrate(t):
    p = MODELS / f"{t}.joblib"
    d = joblib.load(p)
    X = ev[d["feats"]].astype(float); y = to_model_target(t).astype(float); ok = y.notna()
    last = ev.day0[ok].max()
    cal = ok & (ev.day0 > last - pd.DateOffset(years=CAL_YEARS))
    fit = ok & (ev.day0 < last - pd.DateOffset(years=CAL_YEARS) - PURGE)
    qlo = clone(d["q_lo"]).fit(X[fit], y[fit]); qhi = clone(d["q_hi"]).fit(X[fit], y[fit])
    lo, hi = qlo.predict(X[cal]), qhi.predict(X[cal])
    e = np.maximum(lo - y[cal].values, y[cal].values - hi)
    n = len(e); c = float(np.quantile(e, min(1.0, np.ceil((n + 1) * 0.8) / n)))
    d.update(q_lo=qlo, q_hi=qhi, cqr_model=c, version=3,
             cal_window=[str(ev.day0[cal].min().date()), str(last.date())], cal_n=int(n))
    joblib.dump(d, p)
    return {"target": t, "fit_n": int(fit.sum()), "cal_n": n, "c_model_units": round(c, 4),
            "raw_coverage_before_c": round(float(((y[cal] >= lo) & (y[cal] <= hi)).mean()), 3)}

if __name__ == "__main__":
    rows = [calibrate(t) for t in TARGETS if (MODELS / f"{t}.joblib").exists()]
    print(pd.DataFrame(rows).to_string(index=False))
