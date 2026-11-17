"""Experiment 14: how do the live risk models compare with the standard textbook forecasts on the same events and folds?
Volatility: the HAR model (Corsi 2009) is the usual benchmark: tomorrow's volatility from last day's, last week's and last
month's. Here it is written in ratio form, so it predicts the same target as the live model (volatility over days 2 to 20
relative to the stock's normal level) from the day 0-1 shock, the last 5 days and the last 20 days. A simpler benchmark,
persistence, assumes the next month looks like the last one.
Volume: persistence of abnormal volume (the week before the event and days 0-1), as a linear model.
Trough: a linear model on the same live features (the README's ridge column), plus a volatility-only rule.
Every forecast is out of sample on the same purged walk-forward folds. The gain of the live model over each benchmark is
tested with a monthly paired t-test of rank correlations.
Run: LIVE_FEATS=1 python src/x11_benchmarks.py   (quick: EXP_TEST=1)   Output: results/experiments/x11_benchmarks.md
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from config import ROOT, LIVE
from modeling import ev, FEATS, gbm, ridge, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
month = ev.day0.dt.to_period("M")
P = dict(n_estimators=60) if TEST else {}
ev["shock_1d"] = np.log(ev.ar_day0_sd.abs().clip(lower=0.05))          # size of the day-0 move in sd, logged
ols = lambda: make_pipeline(SimpleImputer(strategy="median"), LinearRegression())

def oos(t, make, cols, weighted=True):
    y = to_model_target(t).astype(float); X = ev[cols].astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(make(), X.loc[tr], y[tr], wts(tr, ev.day0) if weighted else None).predict(X.loc[te])
    return from_model_pred(t, pred)
def mrho(y, p):
    ok = y.notna() & p.notna()
    return pd.DataFrame({"y": y[ok], "p": p[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.y, g.p).correlation if len(g) >= 30 else np.nan).dropna()

BENCH = {
    "vol_ratio_2_20": {"persistence (last 20 days)": ("rule", "lv_20"),
                       "HAR (day, week, month)": ("ols", ["shock_1d", "lv_5", "lv_20"]),
                       "HAR + long-run level": ("ols", ["shock_1d", "lv_5", "lv_20", "lv_long"])},
    "abn_volume_2_20": {"persistence (week before)": ("rule", "pre_abvol_5"),
                        "linear on recent volume": ("ols", ["pre_abvol_5", "abn_volume_0_1", "turn_chg"])},
    "trough_2_20": {"volatility only": ("rule", "vol_pre"), "ridge, same features": ("ridge", None)},
}
if TEST: BENCH = {"vol_ratio_2_20": BENCH["vol_ratio_2_20"]}

rows = []
for t, bench in BENCH.items():
    y = target_relative(t).astype(float)
    live = oos(t, lambda: gbm(**P), FEATS); ml = mrho(y, live); ok = y.notna() & live.notna()
    r_live = spearmanr(y[ok], live[ok]).correlation; print(f"{t}: live model {r_live:.4f}", flush=True)
    for name, (kind, cols) in bench.items():
        if kind == "rule":
            p = ev[cols].astype(float).where(ok)
            if t == "trough_2_20": p = -p                                  # more volatile stocks fall further
        elif kind == "ridge":
            p = oos(t, ridge, FEATS)
        else:
            p = oos(t, ols, [c for c in cols if c in ev], weighted=False)
        mb = mrho(y, p); d = (ml - mb).dropna(); okb = ok & p.notna()
        rows.append({"target": t, "benchmark": name, "benchmark_rho": spearmanr(y[okb], p[okb]).correlation, "live_rho": r_live,
                     "monthly_gain": d.mean(), "t": d.mean() / (d.std() / np.sqrt(len(d))),
                     "gain_2021_on": d[d.index.year >= 2021].mean(), "share_months_better": (d > 0).mean()})
        print(f"   vs {name:28s} {rows[-1]['benchmark_rho']:.4f}  (live better in {rows[-1]['share_months_better']:.0%} of months, t = {rows[-1]['t']:.1f})", flush=True)
T = pd.DataFrame(rows)
txt = ("# Experiment 14: live risk models against standard benchmarks (" + ("live" if LIVE else "research") + " features)\n\n"
       "Rank correlations are pooled over all test years; the t-test is on the monthly difference (live minus benchmark).\n\n```\n"
       + T.round(4).to_string(index=False) + "\n```\n")
(OUT / ("x11_benchmarks" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
