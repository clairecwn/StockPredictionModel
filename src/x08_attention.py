"""Experiment 11: does investor attention (Wikipedia page views, s02h) add out-of-sample skill?

Page views start in July 2015, so most training years have no attention data. Two tests, as for the customer links:
  A. ablation (base vs base + attention) on the purged walk-forward folds, scored month by month, Holm-adjusted;
  B. for events from 2017 on, whether each attention feature lines up with the base model's out-of-sample error.
The economic question is whether high attention means faster pricing (less drift after the news, DellaVigna and
Pollet 2009) or overreaction that later reverses (Da, Engelberg and Gao 2011).
Run: python src/x08_attention.py      Output: results/experiments/x08_attention.md
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from config import PROC, ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
a = pd.read_parquet(PROC / "wiki_attention.parquet").drop_duplicates(["permno", "day0"])
a["permno"] = a.permno.astype(ev.permno.dtype); a["day0"] = pd.to_datetime(a.day0)
new = ev[["permno", "day0"]].merge(a, on=["permno", "day0"], how="left"); new.index = ev.index
COLS = ["att_base", "att_trend", "att_spike"]
print("coverage, all events:", ", ".join(f"{c} {new[c].notna().mean():.0%}" for c in COLS),
      "| from 2016:", f"{new.att_base[ev.day0.dt.year >= 2016].notna().mean():.0%}")
X = pd.concat([ev[FEATS].astype(float), new[COLS].astype(float)], axis=1)
month = ev.day0.dt.to_period("M")
P = dict(n_estimators=300, learning_rate=0.04)

def oos(t, cols):
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(gbm(**P), X.loc[tr, cols], y[tr], wts(tr, ev.day0)).predict(X.loc[te, cols])
    return from_model_pred(t, pred)
def mrho(x, y, mask, min_n=30):
    ok = x.notna() & y.notna() & mask
    return pd.DataFrame({"x": x[ok], "y": y[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.x, g.y).correlation if len(g) >= min_n and g.x.nunique() > 1 else np.nan).dropna()
tstat = lambda s: s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 else np.nan

rowsA, rowsB = [], []
late = ev.day0.dt.year >= 2017
for t in ["vol_ratio_2_20", "abn_volume_2_20", "trough_2_20", "drift_1_5", "drift_1_20", "drift_1_60"]:
    y = target_relative(t).astype(float)
    base = oos(t, FEATS); withA = oos(t, FEATS + COLS)
    allm = pd.Series(True, index=ev.index)
    d = (mrho(withA, y, late) - mrho(base, y, late)).dropna()
    rowsA.append({"target": t, "base_2017_on": spearmanr(y[late & base.notna() & y.notna()], base[late & base.notna() & y.notna()]).correlation,
                  "with_attention": spearmanr(y[late & withA.notna() & y.notna()], withA[late & withA.notna() & y.notna()]).correlation,
                  "monthly_gain": d.mean(), "t": tstat(d), "share_months_better": (d > 0).mean()})
    err = y - base
    for c in COLS:
        r = mrho(new[c], err, late)
        rowsB.append({"target": t, "feature": c, "months": len(r), "mean_rho": r.mean(), "t": tstat(r)})
    print(f"{t}: gain t = {rowsA[-1]['t']:.2f}", flush=True)
def holm(T, two):
    T["p_raw"] = 2 * (1 - norm.cdf(T.t.abs())) if two else 1 - norm.cdf(T.t)
    o = T.p_raw.sort_values().index; k = len(T)
    T.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_raw[o].values * (k - np.arange(k)))); return T
A, B = holm(pd.DataFrame(rowsA), False), holm(pd.DataFrame(rowsB), True)
txt = ("# Experiment 11: Wikipedia attention (" + ("live" if LIVE else "research") + " features as base), scored on 2017 on\n\n"
       "## A. Ablation\n\n```\n" + A.round(4).to_string(index=False) + "\n```\n\n"
       "## B. Does the feature line up with the base model's error?\n\n```\n" + B.round(4).to_string(index=False) + "\n```\n")
(OUT / ("x08_attention" + ("_live" if LIVE else "") + ".md")).write_text(txt); print("\n" + txt)
