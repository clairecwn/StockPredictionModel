"""Experiment 12: do price patterns (s02i) add out-of-sample skill? Same method as x01/x05/x06: purged walk-forward,
monthly paired t-test of rank correlations, Holm-adjusted across all target x group tests, gain must hold from 2021 on.
Run: python src/x09_price_patterns.py   (quick: EXP_TEST=1)     Output: results/experiments/x09_price_patterns.md
"""
import os, warnings; warnings.filterwarnings("ignore")
os.environ["EXP_NO_PP"] = "1"           # base = the model before experiment 12
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from config import PROC, ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_parquet(PROC / "price_patterns.parquet").drop_duplicates(["permno", "day0"])
x["permno"] = x.permno.astype(ev.permno.dtype); x["day0"] = pd.to_datetime(x.day0)
new = ev[["permno", "day0"]].merge(x, on=["permno", "day0"], how="left"); new.index = ev.index
GROUPS = {"overnight_intraday": ["intra_01", "ovn_01", "ovn_share"], "lottery": ["max_20", "min_20", "skew_60"],
          "reversal": ["rev_5"], "seasonality": ["seas_5y"]}
GROUPS["all_new"] = sum(GROUPS.values(), [])
print("coverage:", ", ".join(f"{c} {new[c].notna().mean():.0%}" for c in GROUPS["all_new"]), flush=True)
X = pd.concat([ev[FEATS].astype(float), new[GROUPS["all_new"]].astype(float)], axis=1)
month = ev.day0.dt.to_period("M")
P = dict(n_estimators=40 if TEST else 300, learning_rate=0.04)
TARGETS = ["drift_1_20", "vol_ratio_2_20"] if TEST else ["drift_1_5", "drift_1_20", "drift_1_60", "vol_ratio_2_20", "abn_volume_2_20", "trough_2_20"]

def oos(t, cols):
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(gbm(**P), X.loc[tr, cols], y[tr], wts(tr, ev.day0)).predict(X.loc[te, cols])
    return from_model_pred(t, pred)
def mrho(y, p):
    ok = y.notna() & p.notna()
    return pd.DataFrame({"y": y[ok], "p": p[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.y, g.p).correlation if len(g) >= 30 else np.nan).dropna()

rows = []
for t in TARGETS:
    y = target_relative(t).astype(float)
    base = oos(t, FEATS); mb = mrho(y, base); ok = y.notna() & base.notna()
    rb = spearmanr(y[ok], base[ok]).correlation; print(f"{t}: base {rb:.4f}", flush=True)
    for g, cols in GROUPS.items():
        p = oos(t, FEATS + cols); d = (mrho(y, p) - mb).dropna()
        rows.append({"target": t, "group": g, "base": rb, "with_group": spearmanr(y[ok], p[ok]).correlation,
                     "monthly_gain": d.mean(), "t": d.mean() / (d.std() / np.sqrt(len(d))),
                     "gain_2021_on": d[d.index.year >= 2021].mean(), "share_months_better": (d > 0).mean()})
        print(f"   + {g:20s} {rows[-1]['with_group']:.4f} (t = {rows[-1]['t']:.2f})", flush=True)
T = pd.DataFrame(rows)
T["p_raw"] = 1 - norm.cdf(T.t)
o = T.p_raw.sort_values().index; k = len(T)
T.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_raw[o].values * (k - np.arange(k))))
txt = ("# Experiment 12: price patterns (" + ("live" if LIVE else "research") + " features as base)\n\n"
       "A group is kept only if p_holm < 0.05 and the gain also holds from 2021 on.\n\n```\n" + T.round(4).to_string(index=False) + "\n```\n")
(OUT / ("x09_price_patterns" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
