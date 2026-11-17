"""Experiment 2: meta-labeling for the Buy/Wait call (Lopez de Prado 2018, ch. 3.6).

The primary model (19_entry_engine, expected-value rule) says BUY on day k when its forecast of the next 5-day move is above a
threshold, otherwise WAIT. Its calls are right only slightly more than half the time. A secondary model learns, from the
primary model's OUT-OF-SAMPLE calls in earlier years, when a call tends to be right:
    label  correct = 1 if (BUY and the next 5-day move was up) or (WAIT and it was down)
    inputs the primary forecasts (and how far from the threshold), the day, the path so far, volatility, size, calendar
Walk-forward: the meta model for 2011-15 learns from 2006-10 calls, for 2016-20 from 2006-15, for 2021-25 from 2006-20,
always with a 120-day gap. Thresholds are chosen on 2011-15 and reported on 2016-25 (untouched).
Tests (2016-25):
  A. AUC of the meta probability for 'call correct', against the simplest alternative: distance of the forecast from the
     threshold. Monthly paired t-test on the difference.
  B. Hit rate by meta-confidence quintile.
  C. Two decision rules scored by entry regret (as in 19_entry_engine), month-clustered t vs buying on day 1 and vs the plain EV rule,
     Holm-adjusted: (1) only act on BUY calls the meta model trusts; (2) skip WAIT calls it does not trust (buy instead).
If the meta model passes, a final version is saved for the board: models/<live>/meta_entry.joblib.
Needs 19_entry_engine run after this change (it saves entry_oos_rows.parquet). Run: LIVE_FEATS=1 python src/experiments/02_meta_labeling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from config import RESULTS, MODELS, ROOT, LIVE

OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
R = pd.read_parquet(RESULTS / "entry_oos_rows.parquet"); Z = np.load(RESULTS / "entry_oos_events.npz", allow_pickle=True)
cum, trough, year, evok, best_f, K = Z["cum"], Z["trough"], Z["year"], Z["evok"], float(Z["best_f"]), int(Z["KMAX"])
day0 = pd.to_datetime(Z["day0"]); NE = len(year); month = day0.to_period("M").astype(str).values
LOG = []
def say(s=""): print(s, flush=True); LOG.append(str(s))

R["yr"] = year[R.ev.values]; R["d0"] = day0[R.ev.values]
R["buy"] = (R.p_fwd > best_f).astype(np.int8)
R["dist"] = (R.p_fwd - best_f).abs()
R["correct"] = np.where(R.fwd_z.isna() | R.p_fwd.isna(), np.nan, ((R.fwd_z > 0) == (R.buy == 1)).astype(float))
META = [c for c in R.columns if c not in ("ev", "fwd_z", "yr", "d0", "correct")]
say(f"rows with an out-of-sample call and a known outcome: {int(R.correct.notna().sum()):,}; primary hit rate {R.correct.mean():.3f}")

def model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1)
rng = np.random.default_rng(0)
meta = np.full(len(R), np.nan)
for a, b in ((2011, 2015), (2016, 2020), (2021, 2026)):
    te = np.where((R.yr >= a) & (R.yr <= b) & R.p_fwd.notna())[0]
    tr = np.where((R.d0 < R.d0.values[te].min() - pd.Timedelta(days=120)) & R.correct.notna() & (R.yr >= 2006))[0]
    if len(tr) > 1_500_000: tr = np.sort(rng.choice(tr, 1_500_000, replace=False))
    meta[te] = model().fit(R.iloc[tr][META], R.correct.iloc[tr].astype(int)).predict_proba(R.iloc[te][META])[:, 1]
    say(f"  meta fold {a}-{b}: trained on {len(tr):,} earlier calls")
R["meta"] = meta

# ---- A. does the meta model know when a call is right, beyond 'how far from the threshold'?
late = (R.yr >= 2016) & R.correct.notna() & R.meta.notna()
y = R.correct[late].astype(int)
auc_m, auc_d = roc_auc_score(y, R.meta[late]), roc_auc_score(y, R.dist[late])
mm = pd.DataFrame({"y": y, "m": R.meta[late], "d": R.dist[late], "mo": month[R.ev[late].values]})
dd = mm.groupby("mo").apply(lambda g: roc_auc_score(g.y, g.m) - roc_auc_score(g.y, g.d) if g.y.nunique() == 2 and len(g) > 200 else np.nan).dropna()
tA = dd.mean() / (dd.std() / np.sqrt(len(dd)))
say(f"\nA. AUC for 'call correct', 2016-25: meta {auc_m:.4f} vs distance-from-threshold {auc_d:.4f}; monthly t = {tA:.2f}")

# ---- B. hit rate by meta quintile
q = pd.qcut(R.meta[late].rank(method="first"), 5, labels=False) + 1
B = pd.DataFrame({"q": q, "hit": y}).groupby("q").hit.agg(["mean", "count"]).round(3)
say("B. primary hit rate by meta-confidence quintile (5 = most trusted), 2016-25:"); say(B.T.to_string())

# ---- C. decision rules scored by entry regret
M = R.meta.values.reshape(NE, K); F = R.p_fwd.values.reshape(NE, K)
early_e, late_e = evok & (year >= 2011) & (year <= 2015), evok & (year >= 2016)
def regret(ek): return cum[np.arange(NE), ek - 1] - trough
def first(cond): return np.where(cond.any(1), cond.argmax(1) + 1, K)
def tclust(d, msk):
    s = pd.Series(d[msk]).groupby(month[msk]).mean(); return s.mean() / (s.std() / np.sqrt(len(s)))
base1, ev_rule = regret(np.ones(NE, int)), regret(first(F > best_f))
grid = np.nanquantile(M[early_e], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
def pick(rule):
    vals = [np.nanmean(regret(rule(c))[early_e]) for c in grid]; return grid[int(np.nanargmin(vals))]
rules = {"trusted BUY calls only": lambda c: first((F > best_f) & (M >= c)),
         "skip untrusted WAIT calls": lambda c: first((F > best_f) | ((F <= best_f) & (M < c)))}
rows = [{"rule": "plain EV rule (19_entry_engine)", "threshold": np.nan, "regret_2016_25": 100 * ev_rule[late_e].mean(),
         "t_vs_day1": tclust(ev_rule - base1, late_e), "t_vs_ev_rule": 0.0}]
for name, rule in rules.items():
    c = pick(rule); rg = regret(rule(c))
    rows.append({"rule": name, "threshold": c, "regret_2016_25": 100 * rg[late_e].mean(),
                 "t_vs_day1": tclust(rg - base1, late_e), "t_vs_ev_rule": tclust(rg - ev_rule, late_e)})
T = pd.DataFrame(rows)
T["p_vs_ev_rule"] = norm.cdf(T.t_vs_ev_rule)                     # one-sided: lower regret than the EV rule
o = T.index[1:]; p = T.loc[o, "p_vs_ev_rule"].sort_values(); m_ = len(p)
T.loc[p.index, "p_holm"] = np.minimum(1, np.maximum.accumulate(p.values * (m_ - np.arange(m_))))
say(f"\nC. entry regret on 2016-25 (buy day 1 = {100 * base1[late_e].mean():.3f}%; lower is better)"); say(T.round(4).to_string(index=False))

passed = (tA > 2) and (B["mean"].iloc[-1] - B["mean"].iloc[0] > 0.02)
say(f"\nverdict: {'PASS' if passed else 'FAIL'} (needs monthly t > 2 against the simple alternative and a clear hit-rate gap between quintiles)")
if passed:
    ok = R.correct.notna() & (R.yr >= 2006)
    idx = np.where(ok)[0]
    if len(idx) > 2_000_000: idx = np.sort(rng.choice(idx, 2_000_000, replace=False))
    fin = model().fit(R.iloc[idx][META], R.correct.iloc[idx].astype(int))
    cuts = np.nanquantile(R.meta[late], [1 / 3, 2 / 3])
    joblib.dump({"model": fin, "feats": META, "best_f": best_f, "cuts": [float(x) for x in cuts],
                 "hit_by_tercile": [float(x) for x in pd.Series(y.values).groupby(np.searchsorted(cuts, R.meta[late].values)).mean()]},
                MODELS / "meta_entry.joblib")
    say(f"saved {MODELS / 'meta_entry.joblib'}")
(OUT / ("02_meta_labeling" + ("_live" if LIVE else "") + ".md")).write_text("# Experiment 2: meta-labeling\n\n```\n" + "\n".join(LOG) + "\n```\n")
