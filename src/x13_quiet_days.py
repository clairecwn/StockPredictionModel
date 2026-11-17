"""Experiment 16: can Settle give a buy-or-wait lean for stocks WITHOUT recent company news?

The live board only makes a call after news. This test asks whether price and volume alone carry a usable signal on
quiet days (no event in the last 15 trading days, the same rule as the live board). Ideas from the literature:
  short-term reversal (Jegadeesh 1990; Lehmann 1990): last week's losers tend to bounce, mostly in less-traded stocks;
  short-term momentum in heavily traded stocks (Medhat and Schmeling 2022);
  nearness to the 52-week high (George and Hwang 2004); distance from the 50- and 200-day averages; RSI;
  volatility and volume changes; the market's own recent move and volatility.

Sample: S&P-sized and smaller US stocks above $5 and above the smallest 20% by value, one decision day every 5 trading
days, 2001 to 2026, quiet days only. Features use only data up to that day's close.
Targets: (1) abnormal return over the next 5 trading days, in units of the stock's own volatility (the same target as the
live buy-or-wait rule: buy if the forecast is above 0, otherwise wait); (2) the deepest abnormal drop over the next 20
trading days (for a quiet-stock expected low).
Method: LightGBM, walk-forward over 2006-2010 / 2011-2015 / 2016-2020 / 2021-2026, each fold trained only on earlier
years with a 40-day gap; monthly rank correlations with t-statistics; simple one-rule baselines for comparison.
Kept only if the gain is clear, holds from 2021 on, and beats the simple reversal rule.

Run: caffeinate -i python src/x13_quiet_days.py      (quick check first: EXP_TEST=1)
Output: results/experiments/x13_quiet_days.md        (about 30-60 minutes; uses data/processed/crsp_daily.parquet)
"""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
from config import PROC, ROOT

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); say = lambda m: print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

# ---------------------------------------------------------------- daily panel
cols = ["permno", "date", "prc", "cap", "ret", "vol", "shrout", "mkt"]
d = pd.read_parquet(PROC / "crsp_daily.parquet", columns=cols)
d = d[d.date >= "2000-01-01"]
if TEST:
    keep = d.permno.drop_duplicates().sample(400, random_state=0)
    d = d[d.permno.isin(keep) & (d.date >= "2012-01-01")]
d = d.sort_values(["permno", "date"]).reset_index(drop=True)
for c in ["prc", "cap", "ret", "vol", "shrout", "mkt"]: d[c] = d[c].astype("float32")
d["prc"] = d.prc.abs()
say(f"daily rows: {len(d):,}, stocks: {d.permno.nunique():,}")

d["ar"] = (d.ret - d.mkt).astype("float32")
d["lr"] = np.log1p(d.ret.fillna(0).clip(-0.95, 10)).astype("float64")
d["lpx"] = d.groupby("permno", sort=False).lr.cumsum()
d["px"] = np.exp(d.lpx - d.groupby("permno", sort=False).lpx.transform("max")).astype("float32")   # split-proof price index
d["up"], d["dn"] = d.ret.clip(lower=0), (-d.ret).clip(lower=0)
d["turn"] = (d.vol / d.shrout.replace(0, np.nan)).astype("float32")
d["cs"] = d.groupby("permno", sort=False).ar.transform(lambda x: x.fillna(0).cumsum()).astype("float64")
g = d.groupby("permno", sort=False)
roll = lambda s, w, f, mp=None: getattr(g[s].rolling(w, min_periods=mp or w), f)().reset_index(level=0, drop=True)
F = pd.DataFrame(index=d.index)
F["ar_1"] = d.ar
F["rev_5"] = roll("ar", 5, "sum")
F["ar_20"] = roll("ar", 20, "sum")
F["ar_60"] = roll("ar", 60, "sum")
F["mom_250_20"] = (g.lpx.shift(20) - g.lpx.shift(250)).astype("float32")                 # 12-1 month momentum
F["hi_52"] = (d.px / roll("px", 250, "max", 120) - 1).astype("float32")                  # nearness to the 52-week high
F["lo_52"] = (d.px / roll("px", 250, "min", 120) - 1).astype("float32")
F["ma50_gap"] = (d.px / roll("px", 50, "mean") - 1).astype("float32")
F["ma200_gap"] = (d.px / roll("px", 200, "mean", 150) - 1).astype("float32")
au, ad = roll("up", 14, "mean"), roll("dn", 14, "mean")
F["rsi_14"] = (100 - 100 / (1 + au / ad.replace(0, np.nan))).astype("float32")
sd5, sd20, sd60, sd250 = roll("ret", 5, "std"), roll("ret", 20, "std"), roll("ret", 60, "std"), roll("ret", 250, "std", 150)
F["lv_5_60"] = np.log(sd5 / sd60).astype("float32")
F["lv_20_250"] = np.log(sd20 / sd250).astype("float32")
F["vol_60"] = (sd60 * np.sqrt(252)).astype("float32")
F["turn_chg"] = np.log(roll("turn", 20, "mean") / roll("turn", 250, "mean", 120)).astype("float32")
F["abvol_5"] = np.log(roll("vol", 5, "mean") / roll("vol", 60, "mean")).astype("float32")
F["max_20"] = roll("ret", 20, "max"); F["min_20"] = roll("ret", 20, "min")
F["log_cap"] = np.log(d.cap.clip(lower=1)).astype("float32")
F["log_turn_lvl"] = np.log(roll("turn", 60, "mean").clip(lower=1e-6)).astype("float32")
m = d.groupby("date").mkt.first().sort_index()
mk = pd.DataFrame({"mkt_5": m.rolling(5).sum(), "mkt_20": m.rolling(20).sum(), "mkt_vol20": m.rolling(20).std() * np.sqrt(252)})
F = F.join(d[["date"]].join(mk, on="date").drop(columns="date"))
FEATS = list(F.columns)
say(f"{len(FEATS)} features built")

# ---------------------------------------------------------------- targets (future, used only for scoring)
sdd = (sd60).clip(lower=0.003)
fwd5 = g.cs.shift(-5) - d.cs                                      # abnormal return over the next 5 trading days
d["y_fwd"] = (fwd5 / (sdd * np.sqrt(5))).astype("float32")
low = pd.Series(np.inf, index=d.index)
for k in range(1, 21):                                             # deepest point of the abnormal path over the next 20 days
    low = np.minimum(low, g.cs.shift(-k) - d.cs)
low[g.cs.shift(-20).isna()] = np.nan
d["y_dip"] = (low.clip(upper=0) / (sdd * np.sqrt(20))).astype("float32")
d["raw_fwd5"] = fwd5.astype("float32")
del low

# ---------------------------------------------------------------- quiet decision days
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"]).drop_duplicates()
ev["permno"] = ev.permno.astype(d.permno.dtype); ev["e"] = 1
d = d.merge(ev.rename(columns={"day0": "date"}), on=["permno", "date"], how="left"); d["e"] = d.e.fillna(0)
d["recent_event"] = d.groupby("permno", sort=False).e.transform(lambda x: x.rolling(16, min_periods=1).max())
d["i"] = d.groupby("permno", sort=False).cumcount()
d["cap_pct"] = d.groupby("date").cap.rank(pct=True)
ok = (d.recent_event == 0) & (d.prc >= 5) & (d.cap_pct >= 0.2) & (d.i >= 250) & (d.i % 5 == 0) & d.y_fwd.notna() & (d.date >= "2001-01-01")
S = pd.concat([d.loc[ok, ["permno", "date", "y_fwd", "y_dip", "raw_fwd5", "cap_pct"]], F.loc[ok, FEATS]], axis=1).reset_index(drop=True)
del d, F
S["month"] = S.date.dt.to_period("M")
say(f"quiet decision days: {len(S):,} ({S.permno.nunique():,} stocks)")

# ---------------------------------------------------------------- walk-forward
FOLDS = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)] if not TEST else [(2017, 2019), (2020, 2026)]
P = dict(objective="huber", alpha=1.0, n_estimators=80 if TEST else 500, learning_rate=0.03, num_leaves=31, min_child_samples=500,
         subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1)
def oos(target):
    pred = pd.Series(np.nan, index=S.index)
    for a, b in FOLDS:
        te = (S.date.dt.year >= a) & (S.date.dt.year <= b)
        tr = (S.date < pd.Timestamp(f"{a}-01-01") - pd.Timedelta(days=40)) & S[target].notna()
        if tr.sum() < 20000 or not te.any(): continue
        m = lgb.LGBMRegressor(**P).fit(S.loc[tr, FEATS], S.loc[tr, target].clip(-6, 6))
        pred[te] = m.predict(S.loc[te, FEATS]); say(f"   {target}: fold {a}-{b} trained on {tr.sum():,} days")
    return pred
def monthly_ic(y, p):
    ok = y.notna() & p.notna()
    ic = pd.DataFrame({"y": y[ok], "p": p[ok], "m": S.month[ok]}).groupby("m").apply(lambda x: spearmanr(x.y, x.p).correlation if len(x) > 50 else np.nan).dropna()
    return ic
def summarize(name, y, p, same_as=None):
    if same_as is not None: p = p.where(same_as.notna())           # score simple rules on the same test months as the model
    ic = monthly_ic(y, p)
    late = ic[ic.index.year >= 2021]
    return {"forecast": name, "IC": ic.mean(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic))), "IC_2021_on": late.mean(),
            "months_positive": (ic > 0).mean(), "months": len(ic)}

rows, out = [], []
say("Target 1: next-5-day move (buy or wait)")
p_fwd = oos("y_fwd")
rows.append(summarize("model", S.y_fwd, p_fwd))
for nm, rule in {"reversal: minus last 5 days": -S.rev_5, "reversal: minus today": -S.ar_1, "nearness to 52-week high": S.hi_52,
                 "low RSI (30 minus RSI)": 30 - S.rsi_14, "12-1 month momentum": S.mom_250_20, "above 50-day average": S.ma50_gap}.items():
    rows.append(summarize(nm, S.y_fwd, rule, p_fwd))
T1 = pd.DataFrame(rows)
ok = p_fwd.notna()
X = S[ok].assign(p=p_fwd[ok], buy=p_fwd[ok] > 0)
mt = X.groupby(["month", "buy"]).raw_fwd5.mean().unstack()
spread = (mt[True] - mt[False]).dropna()
late = spread[spread.index.year >= 2021]
X["dec"] = pd.qcut(X.p, 10, labels=False)
dec = X.groupby("dec").agg(pred=("p", "mean"), realised_vol_units=("y_fwd", "mean"), up_share=("y_fwd", lambda z: (z > 0).mean()),
                           raw_5day_pct=("raw_fwd5", lambda z: 100 * z.mean())).round(4)
say("Target 2: deepest drop over 20 days")
p_dip = oos("y_dip")
rows2 = [summarize("model", S.y_dip, p_dip), summarize("volatility rising (lv_5_60)", S.y_dip, -S.lv_5_60, p_dip),
         summarize("recent swings (max_20 minus min_20)", S.y_dip, -(S.max_20 - S.min_20), p_dip)]
T2 = pd.DataFrame(rows2)

# ---------------------------------------------------------------- the simple rule the live page can use today
# Last week's move against the market, in units of the stock's own volatility. Cut points are fixed in advance (not fitted).
# Large stocks only (top 20% by market value, roughly the size of the S&P 500), since that is what the live board covers.
L = S[S.cap_pct >= 0.8].copy()
L["z"] = L.rev_5 / (L.vol_60 / np.sqrt(252) * np.sqrt(5)).clip(lower=0.003 * np.sqrt(5))
CUT = [-1.0, -0.33, 0.33, 1.0]
NAMES = ["fell a lot more than the market", "fell a little more than the market", "moved with the market",
         "rose a little more than the market", "rose a lot more than the market"]
L["bin"] = np.searchsorted(CUT, L.z.values, side="right")
def tab(x):
    return x.groupby("bin").agg(n=("y_fwd", "size"), up_share=("y_fwd", lambda z: (z > 0).mean()), mean_sd=("y_fwd", "mean"),
                                mean_5day_pct=("raw_fwd5", lambda z: 100 * z.mean()))
RA, RL = tab(L), tab(L[L.date.dt.year >= 2021])
rule = {"about": "Quiet stocks (no company news in 15 trading days), large US stocks, 2001 onward. z = last 5 days' return minus the market's, "
                 "divided by the stock's usual 5-day swing. Outcome: return minus the market's over the next 5 trading days.",
        "cut": CUT, "bins": []}
for b in range(5):
    a, l = RA.loc[b], RL.loc[b] if b in RL.index else None
    holds = l is not None and np.sign(l.mean_sd) == np.sign(a.mean_sd)
    rule["bins"].append({"when": NAMES[b], "n": int(a.n), "up_share": round(float(a.up_share), 4), "mean_sd": round(float(a.mean_sd), 4),
                         "mean_5day_pct": round(float(a.mean_5day_pct), 3), "up_share_2021_on": None if l is None else round(float(l.up_share), 4),
                         "holds_2021_on": bool(holds), "call": "BUY" if a.mean_sd > 0 else "WAIT"})
if not TEST:
    import json; (ROOT / "models" / "live" / "quiet_rule.json").write_text(json.dumps(rule, indent=1))
RULE_TXT = pd.DataFrame(rule["bins"]).to_string(index=False)

# ---------------------------------------------------------------- live dip model: features that free daily prices can rebuild
# Turnover needs shares outstanding and size needs market value, which the live board does not fetch, so the live model uses
# trading volume changes instead and leaves size out. Its own walk-forward score is printed so the drop can be checked.
S["vol_chg"] = S.turn_chg                                       # shares outstanding barely move, so this equals log(volume 20d / 250d)
LIVE_DIP = [c for c in FEATS if c not in ("turn_chg", "log_cap", "log_turn_lvl")] + ["vol_chg"]
FEATS_ALL = FEATS; FEATS = LIVE_DIP
p_dip_live = oos("y_dip"); live_row = summarize("live model (price and volume only)", S.y_dip, p_dip_live)
FEATS = FEATS_ALL
T2 = pd.concat([T2, pd.DataFrame([live_row])], ignore_index=True)
if not TEST:
    import joblib
    ok = S.y_dip.notna()
    mdl = lgb.LGBMRegressor(**P).fit(S.loc[ok, LIVE_DIP], S.loc[ok, "y_dip"].clip(-6, 6))
    joblib.dump({"model": mdl, "features": LIVE_DIP, "ic": float(live_row["IC"]), "ic_2021_on": float(live_row["IC_2021_on"]),
                 "about": "Deepest drop against the market over the next 20 trading days, in units of sd60*sqrt(20), for stocks without news."},
                ROOT / "models" / "live" / "quiet_dip.joblib")
    say("saved models/live/quiet_dip.joblib")

txt = f"""# Experiment 16: a buy-or-wait lean for stocks without recent news

{len(S):,} quiet decision days (no company event in the last 15 trading days), {S.permno.nunique():,} stocks, test years
{FOLDS[0][0]}-{FOLDS[-1][1]}. IC = monthly rank correlation between forecast and outcome; t uses monthly values.

## 1. Next-5-day move, the buy-or-wait target

```
{T1.round(4).to_string(index=False)}
```

Buy days (forecast above 0) minus wait days, average abnormal return over the next 5 days:
all test months {100 * spread.mean():.3f}% (t = {spread.mean() / (spread.std() / np.sqrt(len(spread))):.2f}, {len(spread)} months),
from 2021 {100 * late.mean():.3f}% (t = {late.mean() / (late.std() / np.sqrt(max(len(late), 2))):.2f}).
Share of quiet days the model would call Buy: {100 * X.buy.mean():.1f}%.

By tenth of the forecast (0 = strongest wait, 9 = strongest buy):

```
{dec.to_string()}
```

## 2. Deepest drop over the next 20 days, in units of each stock's volatility

```
{T2.round(4).to_string(index=False)}
```

## 3. The simple rule for the live page (large stocks, last week's move against the market)

```
{RULE_TXT}
```

Saved to models/live/quiet_rule.json (full run only). The page uses the call in each row, and shows the up share as the past odds.

Total time {(time.time() - T0) / 60:.0f} minutes.
"""
(OUT / ("x13_quiet_days" + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
