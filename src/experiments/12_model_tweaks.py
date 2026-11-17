"""Experiment 12 (one overnight batch, three questions), same method as experiments 5 to 11: purged walk-forward folds,
monthly paired t-test of rank correlations against the current live model, Holm correction across every test in the
batch, and a change is kept only if p_holm < 0.05 and the gain also holds from 2021 on.

A. Averaging and tuning. (1) Average five copies of the model trained with different random seeds; each copy sees a
   different random subset of rows and features, so averaging removes some noise. (2) Pick the model settings inside
   each training window (the last three training years act as a validation set), never using the test years.
B. Volatility forecast into the drop models. How far a stock falls is roughly its volatility times the square root of
   time, so a sharper volatility forecast should sharpen the drop forecast. The volatility and volume forecasts are made
   year by year, each from a model trained only on earlier years, and added as inputs to the drop and drift models.
C. Did the group move? The share of the day 0-1 move that the stock's closest peers (its 10 most correlated stocks over
   the year before) also made. Industry-wide news may fade or persist differently from firm-specific news. If it helps,
   the live board would use sector funds instead, which needs its own parity check.

Run: LIVE_FEATS=1 caffeinate -i python src/experiments/12_model_tweaks.py      (quick check first: EXP_TEST=1)
Output: results/experiments/12_model_tweaks_live.md   (progress is printed as it goes; about 2 to 3 hours)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from config import ROOT, LIVE
from modeling import ev, FEATS, GRID, PURGE, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); say = lambda m: print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)
d = ev.day0; month = d.dt.to_period("M")
SMALL = dict(n_estimators=60) if TEST else {}
SEEDS = [1, 2] if TEST else [1, 2, 3, 4, 5]
PEER = [c for c in ["peer_car_01", "own_vs_peer", "peer_pre20", "peer_event", "peer_corr"] if c in ev]
X = ev[sorted(set(FEATS) | set(PEER))].astype(float).copy()
RISK = ["vol_ratio_2_20", "abn_volume_2_20", "trough_2_20"]
if TEST: RISK = ["vol_ratio_2_20", "trough_2_20"]

def mrho(y, p):
    ok = y.notna() & p.notna()
    return pd.DataFrame({"y": y[ok], "p": p[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.y, g.p).correlation if len(g) >= 30 else np.nan).dropna()

def oos(t, cols, make=lambda: gbm(**SMALL), pick=None):
    """Walk-forward out-of-sample prediction. make() builds one model, or a list of models to average.
    pick(t, cols, tr) chooses settings inside the training window (B: tuning)."""
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(d, y):
        ms = pick(t, cols, tr) if pick else make()
        ms = ms if isinstance(ms, list) else [ms]
        pred[te] = np.mean([_fit(m, X.loc[tr, cols], y[tr], wts(tr, d)).predict(X.loc[te, cols]) for m in ms], axis=0)
    return from_model_pred(t, pred)

def inner_pick(t, cols, tr):
    y = to_model_target(t).astype(float); last = d[tr].max()
    va = tr & (d > last - pd.DateOffset(years=3)); itr = tr & (d < d[va].min() - PURGE)
    best, bs = {}, -9.0
    for g in [dict()] + GRID:
        m = _fit(gbm(**{**g, **SMALL}), X.loc[itr, cols], y[itr], wts(itr, d))
        r = spearmanr(y[va], m.predict(X.loc[va, cols])).correlation
        if r > bs: best, bs = g, r
    say(f"      {t}: settings picked for fold ending {last.year}: {best or 'default'} (validation {bs:.4f})")
    return gbm(**{**best, **SMALL})

rows, base = [], {}
def compare(t, name, p):
    y = target_relative(t).astype(float)
    if t not in base:
        base[t] = oos(t, FEATS); ok = y.notna() & base[t].notna()
        say(f"{t}: current live model {spearmanr(y[ok], base[t][ok]).correlation:.4f}")
    b = base[t]; dm = (mrho(y, p) - mrho(y, b)).dropna(); ok = y.notna() & b.notna() & p.notna()
    rows.append({"target": t, "change": name, "base": spearmanr(y[ok], b[ok]).correlation, "new": spearmanr(y[ok], p[ok]).correlation,
                 "monthly_gain": dm.mean(), "t": dm.mean() / (dm.std() / np.sqrt(len(dm))),
                 "gain_2021_on": dm[dm.index.year >= 2021].mean(), "share_months_better": (dm > 0).mean()})
    r = rows[-1]; say(f"   {name:34s} {r['new']:.4f}  (gain {r['monthly_gain']:+.4f}, t = {r['t']:.2f}, from 2021 {r['gain_2021_on']:+.4f})")

# A. averaging and tuning
say("A. averaging five seeds, and tuning inside each training window")
for t in RISK:
    compare(t, "A1 average of 5 seeds", oos(t, FEATS, make=lambda: [gbm(**{**SMALL, "random_state": s}) for s in SEEDS]))
    compare(t, "A2 settings tuned per window", oos(t, FEATS, pick=inner_pick))

# B. volatility forecast into the drop models
say("B. year-by-year volatility and volume forecasts, each from earlier years only")
def yearly(t):
    y = to_model_target(t).astype(float); p = pd.Series(np.nan, index=ev.index)
    for Y in sorted(d.dt.year.unique()):
        tr = (d < pd.Timestamp(f"{Y}-01-01") - PURGE) & y.notna()
        if tr.sum() < 5000: continue
        te = d.dt.year == Y
        p[te] = _fit(gbm(**SMALL), X.loc[tr, FEATS], y[tr], wts(tr, d)).predict(X.loc[te, FEATS])
    say(f"   {t}: forecasts made for {p.notna().mean():.0%} of events")
    return p
X["vol_hat"] = yearly("vol_ratio_2_20"); X["volu_hat"] = yearly("abn_volume_2_20")
X["vol_hat_x_term"] = X.vol_hat - X.get("lv_20", 0)          # expected change in volatility from the last month
for t in (["trough_2_20"] if TEST else ["trough_2_20", "drift_1_20"]):
    compare(t, "B volatility and volume forecasts", oos(t, FEATS + ["vol_hat", "volu_hat", "vol_hat_x_term"]))

# C. did the group move?
say(f"C. peer reaction ({', '.join(PEER)}); coverage " + ", ".join(f"{c} {ev[c].notna().mean():.0%}" for c in PEER))
if PEER:
    for t in RISK + ([] if TEST else ["drift_1_20"]):
        compare(t, "C peer reaction", oos(t, FEATS + PEER))

# verdicts
T = pd.DataFrame(rows)
T["p_raw"] = 1 - norm.cdf(T.t)
o = T.p_raw.sort_values().index; k = len(T)
T.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_raw[o].values * (k - np.arange(k))))
T["keep"] = (T.p_holm < 0.05) & (T.gain_2021_on > 0)
txt = ("# Experiment 12: averaging and tuning, volatility forecast into the drop models, peer reaction ("
       + ("live" if LIVE else "research") + " features)\n\nEach row compares one change with the current live model on the same folds. "
       "Kept only if p_holm < 0.05 and the gain holds from 2021 on.\n\n```\n" + T.round(4).to_string(index=False) + "\n```\n"
       + f"\nTotal time {(time.time() - T0) / 60:.0f} minutes.\n")
(OUT / ("12_model_tweaks" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
