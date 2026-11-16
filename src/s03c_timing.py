"""
Step 3c: buy-timing layer. Answers "buy now, wait for a lower entry, or avoid?" and tests it honestly.

Run after s03_train.py:  python src/s03c_timing.py
Needs ar_d2..ar_d20 columns in events.parquet (re-run s02_events.py if missing).

Models (purged walk-forward, same folds as s03):
  P(late_dip)  = probability that the lowest point of days 2-20 comes after day 7 AND is at least 3% below day-1 close
  drift_1_20   = out-of-sample prediction already produced by s03 (re-computed here for the same rows)

Rule:  predicted drift_1_20 < -1%            -> AVOID
       P(late_dip) > threshold               -> WAIT  (enter at close of day 7)
       otherwise                             -> BUY NOW (enter at close of day 1)
Evaluation: realised abnormal return from entry to day 20, 0.20% cost, out of sample, vs always-buy-now and always-wait.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from config import PROC, RESULTS, MODELS
from modeling import ev, FEATS, FOLDS, PURGE, clf
X = ev[FEATS].astype(float)
oos = pd.read_parquet(PROC / "oos_predictions.parquet")      # out-of-sample predictions saved by s03_train.py

path_cols = [f"ar_d{k}" for k in range(2, 21)]
assert all(c in ev for c in path_cols), "run python src/s02_events.py first (adds the daily path columns)"
P = ev[path_cols].astype(float).values                     # P[:, k-2] = abnormal return on day k
cum = np.cumsum(P, axis=1)                                 # cumulative from close of day 1
ok = np.isfinite(cum).all(axis=1)
trough_day = np.where(ok, cum.argmin(axis=1) + 2, np.nan)
trough = np.where(ok, cum.min(axis=1), np.nan)
y_late = pd.Series(np.where(ok, ((trough_day > 7) & (trough < -0.03)).astype(float), np.nan), index=ev.index)

# ---- walk-forward probability of a late dip
prob = pd.Series(np.nan, index=ev.index)
for a, b in FOLDS:
    te = (ev.day0.dt.year >= a) & (ev.day0.dt.year <= b) & y_late.notna()
    if not te.any(): continue
    tr = (ev.day0 < ev.day0[te].min() - PURGE) & y_late.notna()
    if tr.sum() < 1000: continue
    m = clf().fit(X[tr], y_late[tr].astype(int))
    prob[te] = m.predict_proba(X[te])[:, 1]

from sklearn.metrics import roc_auc_score, log_loss
msk = prob.notna()
base_rate = y_late[msk].mean()
auc = roc_auc_score(y_late[msk], prob[msk])
ll, ll0 = log_loss(y_late[msk], prob[msk]), log_loss(y_late[msk], np.full(msk.sum(), base_rate))
out = ["# Buy-timing layer", "",
       f"Events with a full 20-day path and an out-of-sample prediction: {int(msk.sum()):,}",
       f"Base rate of a 'late dip' (bottom after day 7 and at least 3% down): {base_rate:.1%}",
       f"Out-of-sample AUC for predicting a late dip: {auc:.3f}  (0.5 = coin flip)",
       f"Log loss {ll:.4f} vs always-predict-base-rate {ll0:.4f}  (lower is better)", ""]

# ---- strategy comparison: return from entry close to day 20, net of 0.20%
def ret_from(entry_day):          # entry at close of entry_day; return = sum of ar on days entry_day+1..20
    start = entry_day + 1 - 2     # column index for day entry_day+1
    return np.where(ok, cum[:, -1] - (cum[:, start - 1] if start > 0 else 0.0), np.nan)
r_now, r_wait = ret_from(1), ret_from(7)
d20 = oos["drift_1_20"]
rows = []
for thr in (0.3, 0.4, 0.5, 0.6):
    for period, pm in (("2006-2026", msk), ("2021-2026", msk & (ev.day0.dt.year >= 2021))):
        avoid = d20 < -0.01
        wait = (~avoid) & (prob > thr)
        buy = (~avoid) & (~wait)
        rule = np.where(avoid, 0.0, np.where(wait, r_wait, r_now)) - np.where(avoid, 0.0, 0.002)
        sel = pm.values & ok
        rows.append(dict(wait_threshold=thr, period=period, n=int(sel.sum()),
                         share_avoid=float(avoid[sel].mean()), share_wait=float(wait[sel].mean()), share_buy=float(buy[sel].mean()),
                         always_buy_now=np.nanmean(r_now[sel]) - 0.002, always_wait_to_day7=np.nanmean(r_wait[sel]) - 0.002,
                         rule=np.nanmean(rule[sel]),
                         rule_vs_buy_now_tstat=float(np.nanmean((rule - (r_now - 0.002))[sel]) / (np.nanstd((rule - (r_now - 0.002))[sel]) / np.sqrt(sel.sum())))))
tab = pd.DataFrame(rows)
out += ["## Mean 20-day abnormal return by entry rule (net of 0.20% cost; avoid = 0 return)",
        "rule_vs_buy_now_tstat > 2 means the rule beats simply buying at the close of day 1.", "",
        tab.round(4).to_string(index=False), ""]

# ---- when the dip comes, conditional on the model's view (for the tracker's wording)
q = pd.qcut(prob[msk], 5, labels=False)
cond = pd.DataFrame({"prob_quintile": q, "late_dip_rate": y_late[msk], "mean_trough_pct": trough[msk.values] * 100,
                     "median_trough_day": trough_day[msk.values]}).groupby("prob_quintile").agg(
    late_dip_rate=("late_dip_rate", "mean"), mean_trough_pct=("mean_trough_pct", "mean"), median_trough_day=("median_trough_day", "median"))
out += ["## Realised outcome by predicted late-dip probability quintile (0 = lowest)", cond.round(3).to_string(), ""]

# ---- final classifier on all data
m = clf().fit(X[y_late.notna()], y_late[y_late.notna()].astype(int))
joblib.dump({"model": m, "feats": FEATS, "base_rate": base_rate, "auc_oos": auc}, MODELS / "late_dip.joblib")
(RESULTS / "timing.md").write_text("\n".join(out))
print("\n".join(out)); print("saved models/late_dip.joblib and results/timing.md")
