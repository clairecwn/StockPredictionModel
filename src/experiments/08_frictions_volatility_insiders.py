"""Experiments 8-10, scored the same way as experiments 1 and 6 (purged walk-forward, monthly paired t, Holm).

8. Volatility mean reversion. vol_ratio_2_20 divides future volatility by vol_pre, the volatility of the last 60 days.
   When those 60 days were unusually calm or wild relative to the stock's normal level, the ratio should revert.
   Trees handle ratios badly, so the gaps are given directly (all rebuildable live from existing features):
     lv_long   log(idio_vol_pre / vol_pre)     long-run (days -250..-30) vs recent volatility
     lv_5      log(yz_vol_5 / vol_pre)         last week vs the 60-day level
     lv_20     log(yz_vol_20 / vol_pre)
     lv_range  log(yz_vol_60 / vol_pre)        range-based vs close-to-close volatility
9. Limits to arbitrage (Shleifer and Vishny 1997). Drift should survive only where trading it away is costly. A friction
   score is the average monthly percentile of Amihud illiquidity, idiosyncratic volatility, small size, low analyst
   coverage and wide spreads. Within each friction fifth we report (a) how well the out-of-sample drift forecast ranks
   stocks, month by month, and (b) whether the day 0-1 reaction continues (underreaction) or reverses, and (c) the
   gross long-short spread of the forecast against a round-trip cost of one bid-ask spread.
10. Insider trading (needs data/processed/insider_features.parquet from 08_insider_features): routine vs opportunistic insiders
   (Cohen, Malloy and Pomorski 2012), purchases and sales filed by the close of day 1.
Run: python src/experiments/08_frictions_volatility_insiders.py     (quick: EXP_TEST=1; skip parts: X06_SKIP=frictions,insider)
Output: results/experiments/08_frictions_volatility_insiders.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
os.environ["EXP_NO_VOLGAP"] = "1"       # base = the model before experiment 8
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from config import PROC, ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
SKIP = set(os.environ.get("X06_SKIP", "").split(","))
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
P = dict(n_estimators=40 if TEST else 300, learning_rate=0.04)
month = ev.day0.dt.to_period("M")
lg = lambda a, b: np.log(ev[a].clip(lower=1e-4) / ev[b].clip(lower=1e-4))
new = pd.DataFrame({"lv_long": lg("idio_vol_pre", "vol_pre"), "lv_5": lg("yz_vol_5", "vol_pre"),
                    "lv_20": lg("yz_vol_20", "vol_pre"), "lv_range": lg("yz_vol_60", "vol_pre")}, index=ev.index)
GROUPS = {"vol_gaps": list(new.columns)}
INS = PROC / "insider_features.parquet"
if INS.exists() and "insider" not in SKIP:
    x = pd.read_parquet(INS).drop_duplicates(["permno", "day0"])
    x["permno"] = x.permno.astype(ev.permno.dtype); x["day0"] = pd.to_datetime(x.day0)
    xi = ev[["permno", "day0"]].merge(x, on=["permno", "day0"], how="left"); xi.index = ev.index
    icols = [c for c in x.columns if c not in ("permno", "day0")]
    new = pd.concat([new, xi[icols]], axis=1)
    GROUPS["insider_all"] = icols
    GROUPS["insider_opportunistic"] = [c for c in icols if c.startswith("opp_")]
    print("insider coverage:", ", ".join(f"{c} {new[c].notna().mean():.0%}" for c in icols))
X = pd.concat([ev[FEATS].astype(float), new.astype(float)], axis=1)

def oos(t, cols):
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(gbm(**P), X.loc[tr, cols], y[tr], wts(tr, ev.day0)).predict(X.loc[te, cols])
    return from_model_pred(t, pred)
def mrho(a, b, mask=None, min_n=30):
    ok = a.notna() & b.notna() & (mask if mask is not None else True)
    return pd.DataFrame({"a": a[ok], "b": b[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.a, g.b).correlation if len(g) >= min_n and g.a.nunique() > 1 else np.nan).dropna()
tstat = lambda s: s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 else np.nan
def holm(T, two_sided=False):
    T["p_raw"] = 2 * (1 - norm.cdf(T.t.abs())) if two_sided else 1 - norm.cdf(T.t)
    o = T.p_raw.sort_values().index; k = len(T)
    T.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_raw[o].values * (k - np.arange(k))))
    return T

TARGETS = ["vol_ratio_2_20", "trough_2_20"] if TEST else ["vol_ratio_2_20", "abn_volume_2_20", "trough_2_20", "drift_1_5", "drift_1_20", "drift_1_60"]
rows, base_pred = [], {}
for t in TARGETS:
    y = target_relative(t).astype(float)
    base = oos(t, FEATS); base_pred[t] = base; mb = mrho(y, base); ok = y.notna() & base.notna()
    rb = spearmanr(y[ok], base[ok]).correlation; print(f"{t}: base {rb:.4f}", flush=True)
    for g, cols in GROUPS.items():
        if g == "vol_gaps" and not t.startswith(("vol_", "trough", "abn_")):
            continue
        p = oos(t, FEATS + cols); d = (mrho(y, p) - mb).dropna()
        rows.append({"target": t, "group": g, "base": rb, "with_group": spearmanr(y[ok], p[ok]).correlation,
                     "monthly_gain": d.mean(), "t": tstat(d), "gain_2021_on": d[d.index.year >= 2021].mean(),
                     "share_months_better": (d > 0).mean()})
        print(f"   + {g:22s} {rows[-1]['with_group']:.4f} (t = {rows[-1]['t']:.2f})", flush=True)
A = holm(pd.DataFrame(rows))
txt = "# Experiments 8-10 (" + ("live" if LIVE else "research") + " features as base)\n\n## Ablation\n\n```\n" + A.round(4).to_string(index=False) + "\n```\n"

if "frictions" not in SKIP and not TEST:
    pr = lambda c, sign=1: (sign * ev[c].astype(float)).groupby(month).rank(pct=True)
    fr = pd.concat([pr("amihud"), pr("idio_vol_pre"), pr("log_mcap", -1),
                    pr("analyst_cov", -1) if "analyst_cov" in ev else None, pr("spread")], axis=1).mean(axis=1)
    q = fr.groupby(month).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
    B = []
    for t in ["drift_1_5", "drift_1_20", "drift_1_60"]:
        y = target_relative(t).astype(float); raw = ev[t].astype(float); p = base_pred.get(t)
        for k in range(1, 6):
            m = q == k
            r_fc = mrho(y, p, m) if p is not None else pd.Series(dtype=float)
            r_cont = mrho(y, ev.car_0_1.astype(float), m)
            # gross long-short: top minus bottom fifth of the forecast within this friction fifth, each month
            d = pd.DataFrame({"p": p[m], "r": raw[m], "m": month[m], "c": ev.spread[m]}).dropna()
            ls = d.groupby("m").apply(lambda g: g.r[g.p >= g.p.quantile(0.8)].mean() - g.r[g.p <= g.p.quantile(0.2)].mean() if len(g) >= 50 else np.nan).dropna()
            B.append({"target": t, "friction_fifth": k, "forecast_rho": r_fc.mean(), "t": tstat(r_fc),
                      "continuation_rho": r_cont.mean(), "t_cont": tstat(r_cont),
                      "long_short_gross_pct": 100 * ls.mean(), "t_ls": tstat(ls), "median_spread_pct": 100 * d.c.median()})
    Bt = holm(pd.DataFrame(B), two_sided=True)
    txt += ("\n## Limits to arbitrage (1 = least friction, 5 = most)\n\nforecast_rho: monthly rank correlation of the out-of-sample drift "
            "forecast within the fifth (Holm, two-sided). continuation_rho: does the day 0-1 move continue (+) or reverse (-)? "
            "long_short_gross_pct: top minus bottom fifth of the forecast, before costs; compare with median_spread_pct, roughly "
            "the cost of one round trip.\n\n```\n" + Bt.round(4).to_string(index=False) + "\n```\n")
(OUT / ("08_frictions_volatility_insiders" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt)
print("\n" + txt)
