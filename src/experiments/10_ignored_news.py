"""Experiment 10: does company news that the market ignores at first matter later? And does the tone of a headline
(bullish or bearish words) tell us anything?

A headline about a firm (FNSPID, 2009-2023) is "quiet" when the stock barely reacted on the news day and the day
after: abnormal return within 2 standard deviations and volume under twice normal on both days. Example: a report that
a former CEO may return, which moves nothing on the day. For each quiet-news firm-day we measure what happens over
trading days 2-21 and compare it with the same firm's ordinary days (no headline within 3 days) in the same month,
so the comparison is within firm and within month:
  shock_20   a large move (|abnormal return| > 3 sd or volume > 3x normal) on any day 2-21
  vol_ratio  volatility on days 2-21 / volatility in the year before
  abvol      volume on days 2-21 / volume in the year before (log)
  drift      abnormal return on days 2-21
Results are given by news type and by tone. Tone counts bullish and bearish finance words in the headline
(a short list in the spirit of Loughran and McDonald 2011). t-statistics are clustered by month.
Run: python src/experiments/10_ignored_news.py      Output: results/experiments/10_ignored_news.md and results/experiments/10_ignored_news.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import re, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from config import PROC, ROOT, find_file
from taxonomy import classify_headline

OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
BULL = re.compile(r"\b(beat|beats|tops|raises?|raised|upgrade[sd]?|record|surges?|soars?|jumps?|rall(y|ies)|outperform\w*|"
                  r"strong|growth|wins?|approv\w*|bullish|buyback|breakthrough|expands?|partnership|boost\w*)\b", re.I)
BEAR = re.compile(r"\b(miss|misses|missed|cuts?|lowers?|downgrade[sd]?|plunges?|slumps?|falls?|drops?|sinks?|tumbles?|weak\w*|"
                  r"lawsuit|sued|probe|investigation|recall\w*|warn\w*|layoffs?|bearish|halt\w*|delay\w*|resign\w*|fraud)\b", re.I)

# headlines -> firm-days
hl = pd.read_parquet(PROC / "news_headlines.parquet", columns=["date", "ticker", "headline"])
hl = hl[(hl.date.dt.year >= 2009) & (hl.date.dt.year <= 2023)]
hl["ticker"] = hl.ticker.str.upper()
hl["day"] = np.where(hl.date.dt.hour >= 16, hl.date.dt.normalize() + pd.Timedelta(days=1), hl.date.dt.normalize())
hl["day"] = pd.to_datetime(hl.day)
parts = []
for ch in pd.read_csv(find_file("crsp_daily"), usecols=["PERMNO", "Ticker", "DlyCalDt"], dtype={"Ticker": str}, chunksize=2_000_000):
    ch = ch.dropna(subset=["Ticker"]); ch["Ticker"] = ch.Ticker.str.upper()
    parts.append(ch.groupby(["PERMNO", "Ticker"]).DlyCalDt.agg(["min", "max"]).reset_index())
tk = pd.concat(parts).groupby(["PERMNO", "Ticker"]).agg({"min": "min", "max": "max"}).reset_index()
tk.columns = ["permno", "ticker", "lo", "hi"]; tk["lo"], tk["hi"] = pd.to_datetime(tk.lo), pd.to_datetime(tk.hi)
hl = hl.merge(tk, on="ticker"); hl = hl[(hl.day >= hl.lo) & (hl.day <= hl.hi)]
hl["cls"] = hl.headline.map(classify_headline)
hl["tone"] = hl.headline.str.count(BULL) - hl.headline.str.count(BEAR)
MOVE = re.compile(r"\b(surges?|soars?|jumps?|rall(y|ies)|plunges?|slumps?|falls?|drops?|sinks?|tumbles?|gains?|rises?|climbs?|"
                  r"slides?|dips?|spikes?|shares|stock|stocks|up|down|higher|lower)\b", re.I)
hl["tone_np"] = (hl.headline.str.count(BULL) - hl.headline.str.count(BEAR)).where(~hl.headline.str.contains(MOVE), np.nan)
prio = ["accounting_fraud", "credit_distress", "legal_regulatory", "activist_short", "leadership", "politics_policy",
        "ma_strategy", "operational", "analyst", "earnings_guidance", "capital", "other"]
hl["p"] = hl.cls.map({c: i for i, c in enumerate(prio)}).fillna(len(prio))
fd = hl.sort_values("p").groupby(["permno", "day"]).agg(cls=("cls", "first"), tone=("tone", "sum"), tone_np=("tone_np", "sum", ), n=("headline", "size")).reset_index()
print(f"firm-days with headlines: {len(fd):,}; classified (not 'other'): {(fd.cls != 'other').mean():.0%}", flush=True)
del hl

# daily z-scores and forward outcomes per firm
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "mkt", "vol", "prc", "cap"],
                  filters=[("date", ">=", pd.Timestamp("2007-06-01")), ("date", "<=", pd.Timestamp("2024-03-31"))]).to_pandas()
D["permno"] = D.permno.astype(np.int64)
D = D[D.permno.isin(set(fd.permno))].sort_values(["permno", "date"]).reset_index(drop=True)
D["ar"] = D.ret - D.mkt
g = D.groupby("permno", sort=False)
D["sd"] = g.ar.transform(lambda x: x.shift(1).rolling(250, min_periods=120).std())
D["v0"] = g.vol.transform(lambda x: x.shift(1).rolling(250, min_periods=120).mean())
D["z"], D["vr"] = D.ar / D.sd, D.vol / D.v0
D["shock"] = ((D.z.abs() > 3) | (D.vr > 3)).astype(float)
def fwd(col, how):                                               # over rows +2..+21 within the firm
    s = g[col].transform(lambda x: getattr(x[::-1].rolling(20, min_periods=15), how)()[::-1].shift(-2))
    return s
D["shock_20"] = fwd("shock", "max")
D["vol_ratio"] = fwd("ar", "std") / D.sd
D["abvol"] = np.log(fwd("vol", "mean") / D.v0).replace([np.inf, -np.inf], np.nan)
D["drift"] = fwd("ar", "sum")
D["z1"], D["vr1"] = g.z.shift(-1), g.vr.shift(-1)
D["rev5"] = g.ar.transform(lambda x: x.shift(1).rolling(5).sum())
D["ar01"] = D.ar + g.ar.shift(-1)
D = D.dropna(subset=["sd", "v0", "shock_20"])
print(f"firm-days with outcomes: {len(D):,}", flush=True)

# label days
D = D.merge(fd.rename(columns={"day": "date"}), on=["permno", "date"], how="left")
has = D.cls.notna()
near = has.copy()
for k in (1, 2, 3):                                             # any headline within 3 trading days either side
    near = near | D.groupby("permno").cls.shift(k).notna() | D.groupby("permno").cls.shift(-k).notna()
quiet_move = (D.z.abs() < 2) & (D.vr < 2) & (D.z1.abs() < 2) & (D.vr1 < 2)
D["kind"] = np.where(has & (D.cls != "other") & quiet_move, "quiet_news", np.where(~near, "no_news", np.where(has & ~quiet_move, "loud_news", "skip")))
D = D[D.kind.isin(["quiet_news", "no_news", "loud_news"])]
D["fm"] = D.permno.astype(str) + "_" + D.date.dt.to_period("M").astype(str)
D["month"] = D.date.dt.to_period("M")
base = D[D.kind == "no_news"].groupby("fm")[["shock_20", "vol_ratio", "abvol", "drift"]].mean()
Q = D[D.kind != "no_news"].join(base, on="fm", rsuffix="_base").dropna(subset=["shock_20_base"])
for c in ["shock_20", "vol_ratio", "abvol", "drift"]:
    Q[f"d_{c}"] = Q[c] - Q[f"{c}_base"]
def tm(x, m):                                                   # mean and month-clustered t
    a = pd.Series(x.values, index=m.values).groupby(level=0).mean()
    return x.mean(), a.mean() / (a.std() / np.sqrt(len(a))) if len(a) > 5 else np.nan

rows = []
for (kind, cls), G in [((k, "all"), Q[Q.kind == k]) for k in ("quiet_news", "loud_news")] + \
                      [(("quiet_news", c), Q[(Q.kind == "quiet_news") & (Q.cls == c)]) for c in prio[:-1]]:
    if len(G) < 300: continue
    r = {"kind": kind, "news_type": cls, "n": len(G), "p_shock_20": G.shock_20.mean(), "p_shock_20_ordinary": G.shock_20_base.mean()}
    for c in ["shock_20", "vol_ratio", "abvol", "drift"]:
        r[f"diff_{c}"], r[f"t_{c}"] = tm(G[f"d_{c}"], G.month)
    rows.append(r)
T = pd.DataFrame(rows)
qq = Q[Q.kind == "quiet_news"].copy()
qq["tone_group"] = np.where(qq.tone > 0, "bullish words", np.where(qq.tone < 0, "bearish words", "neutral"))
tone_rows = []
for tg, G in qq.groupby("tone_group"):
    r = {"tone": tg, "n": len(G)}
    for c in ["drift", "shock_20", "vol_ratio"]:
        r[f"diff_{c}"], r[f"t_{c}"] = tm(G[f"d_{c}"], G.month)
    tone_rows.append(r)
TT = pd.DataFrame(tone_rows)
bb = qq[qq.tone != 0]
spread = pd.Series(np.sign(bb.tone).values * bb.d_drift.values, index=bb.month.values).groupby(level=0).mean()
yr = qq.date.dt.year
cut = qq.groupby(yr).cap.transform(lambda c: c.quantile(0.2))
qq["tradeable"] = (qq.prc.abs() >= 5) & (qq.cap >= cut)
def fm(G, xcol):                                                 # Fama-MacBeth: drift on sign(tone) with price controls
    out = []
    for m_, h in G.groupby("month"):
        h = h.dropna(subset=["drift", xcol, "ar01", "rev5"])
        if len(h) < 30 or h[xcol].nunique() < 2: continue
        X = np.column_stack([np.ones(len(h)), np.sign(h[xcol]), h.ar01.clip(-.5, .5), h.rev5.clip(-.5, .5)])
        out.append(np.linalg.lstsq(X, h.drift.clip(-1, 1).values, rcond=None)[0][1])
    o = pd.Series(out); return o.mean(), o.mean() / (o.std() / np.sqrt(len(o))), len(o)
R = []
for name, G, col in [("all quiet news", qq, "tone"), ("tradeable stocks only", qq[qq.tradeable], "tone"),
                     ("no price-movement words", qq[qq.tone_np.notna()], "tone_np"),
                     ("tradeable, no price words", qq[qq.tradeable & qq.tone_np.notna()], "tone_np"),
                     ("2016-2023, tradeable", qq[qq.tradeable & (yr >= 2016)], "tone")]:
    b, t, n = fm(G[G[col] != 0], col)
    R.append({"sample": name, "events_with_tone": int((G[col] != 0).sum()), "drift_per_tone_sign_pct": 100 * b, "t": t, "months": n})
RB = pd.DataFrame(R)
txt_rb = ("\n## Robustness of the tone result\n\nFama-MacBeth slope of the day 2-21 drift on the sign of the headline tone, controlling for the "
          "day 0-1 move and the prior five days' return, so it is not short-term reversal of the move the words describe.\n\n```\n"
          + RB.round(4).to_string(index=False) + "\n```\n")
txt = ("# Experiment 10: quiet news and headline tone (FNSPID headlines 2009-2023)\n\n"
       "Differences are against the same firm's ordinary days in the same month. drift is in return units over days 2-21.\n\n"
       "## By kind and news type\n\n```\n" + T.round(4).to_string(index=False) + "\n```\n\n## Quiet news by tone\n\n```\n"
       + TT.round(4).to_string(index=False) + f"\n```\n\nbullish-minus-bearish drift, signed by tone: {spread.mean():.4%} per 20 days "
       f"(t = {spread.mean() / (spread.std() / np.sqrt(len(spread))):.2f}, {len(spread)} months)\n")
txt += txt_rb
(OUT / "10_ignored_news.md").write_text(txt); print(txt)
json.dump({"by_type": T[T.kind == "quiet_news"].set_index("news_type")[["n", "p_shock_20", "p_shock_20_ordinary", "t_shock_20"]].round(4).to_dict("index")},
          open(ROOT / "results" / "experiments" / "10_ignored_news.json", "w"), indent=1)
print("saved results/experiments/10_ignored_news.json")
