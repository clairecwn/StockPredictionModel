"""Experiment 7: is the drift that survives in high-friction stocks real and tradeable after costs?

06_frictions_volatility_insiders found the out-of-sample drift forecast ranks stocks better where arbitrage is costly, with a gross long-short spread
of about 2.5% over 60 days in the most frictional fifth. Three things could make that an illusion, and each is checked:
  1. Overlap. A 60-day return overlaps the next two months, so monthly t-statistics are too large. Here every t uses
     Newey-West standard errors with lags covering the holding period.
  2. Costs. Each leg pays a round trip of the quoted spread (clipped to 0.05%-2%) plus 0.10% for price impact, the
     project's standard cost; the short leg also pays a borrow fee (1% a year in fifths 1-3, 3% in fifth 4, 5% in fifth 5,
     a conservative guess since hard-to-trade stocks are often hard to borrow).
  3. Time. Results are split into 2006-2015, 2016-2025 and 2021-2025, because many anomalies faded after publication.
A long-only version (buy the top fifth, compared with the average stock in the same friction fifth) is also reported,
since a small account usually cannot short.
Run: python src/experiments/07_frictions_after_costs.py      (live features: LIVE_FEATS=1)
Output: results/experiments/07_frictions_after_costs[_live].md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, from_model_pred

OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
month = ev.day0.dt.to_period("M")
pr = lambda c, sign=1: (sign * ev[c].astype(float)).groupby(month).rank(pct=True)
parts = [pr("amihud"), pr("idio_vol_pre"), pr("log_mcap", -1), pr("spread")]
if "analyst_cov" in ev and not LIVE:
    parts.append(pr("analyst_cov", -1))
fr = pd.concat(parts, axis=1).mean(axis=1)
Q = fr.groupby(month).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
COST = ev.spread.astype(float).clip(0.0005, 0.02).fillna(0.002) + 0.001
BORROW = {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.03, 5: 0.05}
H = {"drift_1_20": 19, "drift_1_60": 59}

def nw_t(x, lags):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 12: return np.nan
    e = x - x.mean(); v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)

rows = []
for t, h in H.items():
    y = to_model_target(t).astype(float); p = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        p[te] = _fit(gbm(n_estimators=300, learning_rate=0.04), ev.loc[tr, FEATS].astype(float), y[tr], wts(tr, ev.day0)).predict(ev.loc[te, FEATS].astype(float))
    p = from_model_pred(t, p)
    D = pd.DataFrame({"p": p, "r": ev[t].astype(float), "q": Q, "m": month, "c": COST, "yr": ev.day0.dt.year}).dropna()
    lags = int(np.ceil(h / 21)) + 1
    for k in range(1, 6):
        g = D[D.q == k]
        recs = []
        for _, x in g.groupby("m"):
            if len(x) < 50:
                continue
            hi, lo = x[x.p >= x.p.quantile(0.8)], x[x.p <= x.p.quantile(0.2)]
            gross = hi.r.mean() - lo.r.mean()
            net = gross - hi.c.mean() - lo.c.mean() - BORROW[k] * (h + 1) / 252
            lo_only = hi.r.mean() - x.r.mean() - hi.c.mean()
            recs.append({"gross": gross, "net": net, "long_only_net": lo_only, "yr": int(x.yr.iloc[0])})
        M = pd.DataFrame(recs).dropna()
        for per, (a, b) in {"2006-2015": (2006, 2015), "2016-2025": (2016, 2025), "2021-2025": (2021, 2025)}.items():
            S = M[(M.yr >= a) & (M.yr <= b)]
            rows.append({"target": t, "friction_fifth": k, "period": per, "months": len(S),
                         "gross_pct": 100 * S.gross.mean(), "net_pct": 100 * S.net.mean(), "t_net_nw": nw_t(S.net, lags),
                         "long_only_net_pct": 100 * S.long_only_net.mean(), "t_long_only_nw": nw_t(S.long_only_net, lags)})
    print(t, "done", flush=True)
T = pd.DataFrame(rows)
txt = ("# Experiment 7: friction fifths after overlap, costs and time (" + ("live" if LIVE else "research") + " features)\n\n"
       "Returns are per holding period (20 or 60 trading days), in %, factor-adjusted. t uses Newey-West errors.\n"
       "A result counts only if net_pct > 0 with t_net_nw > 2 in BOTH 2016-2025 and 2021-2025.\n\n```\n"
       + T.round(3).to_string(index=False) + "\n```\n")
(OUT / ("07_frictions_after_costs" + ("_live" if LIVE else "") + ".md")).write_text(txt); print("\n" + txt)
