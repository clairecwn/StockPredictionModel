"""
Step 19: entry engine. Re-asks "buy now or wait?" at the close of every day after an event, and is scored by
how far above the eventual 20-day low you ended up buying (entry regret).

Why this design (see results/timing.md for the old layer it replaces):
  The old layer made one decision at day 1. The day of the low behaves close to a random walk, so a single
  early guess cannot work. Here the decision is re-made daily with what is known that evening: how far the
  stock has fallen so far, how long ago the low was, whether selling pressure is fading (volume, close
  location, short-sale ratio), market fear (VIX), upcoming earnings, follow-on events. That is how desks
  time entries after a shock (triple-barrier labels, Lopez de Prado; liquidity-provision reversal, Nagel).

Run (after 17_train_risk_models.py and 11_short_volume.py):  python src/pipeline/19_entry_engine.py
Quick test on a few hundred firms:                    ENTRY_TEST=1 python src/pipeline/19_entry_engine.py

Label (one row per event per decision day k = 1..15):
  wait_better = 1 if, within the next H=5 trading days, the stock's abnormal price first falls by at least
                b = max(1%, 0.75 * daily idio sd * sqrt(5)) before it rises by b. Otherwise 0.
Also fitted: further_dip = how much lower than today's close the stock goes before day 20 (for the tracker text).
Rules tested (buy at the first day the condition holds; if never, buy at day 15):
  P(next move is down) < threshold;  expected further dip < x%;  expected next-5-day move > y sd (expected-value rule).
Benchmarks: buy day 1/2/3/5/10/15, random day, first up-day after a down-day, spreading purchases evenly.
Threshold is chosen on 2006-2015 test years only, then reported untouched on 2016-2026.
All numbers are abnormal (factor-adjusted) returns, out of sample (purged walk-forward, same folds as 17_train_risk_models).
Research only. Not investment advice.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import pyarrow.parquet as pq
from config import PROC, PUBLIC, RESULTS, MODELS, find_file, LIVE
from modeling import ev, BASE_FEATS, FOLDS, PURGE, lgb

TEST = os.environ.get("ENTRY_TEST") == "1"
KMAX, H, BAR_MULT, PRE = 15, 5, 0.75, 60          # decision days 1..KMAX, label horizon, barrier width, pre-event window
OUT = RESULTS / "entry"; OUT.mkdir(parents=True, exist_ok=True)
EPOCH = np.datetime64("1990-01-01")
LOG = []
def say(s=""):
    print(s); LOG.append(str(s))
def section(n, title):
    say(""); say("=" * 70); say(f"{n}. {title}"); say("=" * 70)
def daynum(x):
    return ((pd.to_datetime(x).values.astype("datetime64[D]") - EPOCH).astype("int64")).astype(np.int64)

# events with a full 20-day path
path_cols = [f"ar_d{k}" for k in range(2, 21)]
E = ev.copy()
E = E[np.isfinite(E[path_cols].astype(float)).all(axis=1)].copy()
if TEST:
    keep = np.sort(E.permno.unique())[:300]
    E = E[E.permno.isin(keep)].copy()
E = E.reset_index(drop=True)
NE = len(E)
permnos = np.sort(E.permno.unique()).astype(np.int64)

section(1, "Inputs")
say(f"events with a full 20-day path (tradeable universe): {NE:,}   firms: {len(permnos):,}" + ("   [TEST MODE]" if TEST else ""))

# daily data: CRSP returns + CRSP OHLC + FINRA short volume
t = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "vol", "shrout", "mkt"],
                  filters=[("permno", "in", [int(p) for p in permnos])])
D = t.to_pandas()
D["dn"] = daynum(D.date); D = D.drop(columns="date")
D["key"] = D.permno.astype(np.int64) * 100000 + D.dn
D = D.sort_values("key").reset_index(drop=True)
say(f"CRSP daily rows for these firms: {len(D):,}")

f_ohlc = find_file("crsp_ohlc")
assert f_ohlc is not None, "data/raw/wrds/crsp_ohlc.csv.gz missing"
chunks = []
for ch in pd.read_csv(f_ohlc, usecols=["PERMNO", "DlyCalDt", "Ticker", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose"],
                      chunksize=2_000_000, dtype={"Ticker": "string"}):
    ch = ch[ch.PERMNO.isin(permnos)]
    chunks.append(ch)
O = pd.concat(chunks, ignore_index=True); del chunks
O["key"] = O.PERMNO.astype(np.int64) * 100000 + daynum(O.DlyCalDt)
O = O.drop_duplicates("key").sort_values("key").reset_index(drop=True)
pos = np.searchsorted(O.key.values, D.key.values).clip(0, len(O) - 1)
hit = O.key.values[pos] == D.key.values
for a, b in (("DlyOpen", "open"), ("DlyHigh", "high"), ("DlyLow", "low"), ("DlyClose", "close")):
    D[b] = np.where(hit, pd.to_numeric(O[a], errors="coerce").values[pos], np.nan).astype("float32")
tick = pd.Series(np.where(hit, O.Ticker.astype(object).values[pos], None))
say(f"OHLC matched to {hit.mean():.1%} of CRSP daily rows")
del O

fin_f = PUBLIC / "finra_shortvol.parquet"
D["sr"] = np.float32(np.nan)
if fin_f.exists():
    tickers = sorted(set(t for t in tick.dropna().unique()))
    F = pq.read_table(fin_f, columns=["date", "symbol", "short_vol", "total_vol"],
                      filters=[("symbol", "in", tickers)]).to_pandas()
    F = F[F.total_vol > 0]
    code = {s: i for i, s in enumerate(tickers)}
    F["key"] = F.symbol.map(code).astype(np.int64) * 100000 + daynum(F.date)
    F = F.drop_duplicates("key").sort_values("key")
    dcode = tick.map(code)
    dkey = np.where(dcode.notna(), dcode.fillna(0).astype(np.int64) * 100000 + D.dn.values, -1)
    p2 = np.searchsorted(F.key.values, dkey).clip(0, len(F) - 1)
    ok2 = (F.key.values[p2] == dkey) & (dkey >= 0)
    D["sr"] = np.where(ok2, (F.short_vol.values / F.total_vol.values)[p2], np.nan).astype("float32")
    say(f"FINRA short-sale ratio matched to {ok2.mean():.1%} of daily rows "
        f"({np.isfinite(D.sr[D.dn >= daynum(['2009-08-03'])[0]]).mean():.1%} of rows from Aug 2009, when FINRA files start)")
    del F
else:
    say("FINRA file not found; short-sale features will be empty (run python src/pipeline/11_short_volume.py)")

# align each event to its trading-day window
k0 = E.permno.astype(np.int64).values * 100000 + daynum(E.day0)
g0 = np.searchsorted(D.key.values, k0)
offs = np.arange(-PRE, 21)                                   # trading-day offsets -60..20
G = g0[:, None] + offs[None, :]
inb = (G >= 0) & (G < len(D))
Gc = G.clip(0, len(D) - 1)
same = inb & (D.permno.values[Gc] == E.permno.values[:, None])
def win(col):
    a = D[col].values[Gc].astype("float32")              # float32 halves memory; precision is ample for returns
    a[~same] = np.nan
    return a
def at(arr, off):                                            # column for trading-day offset off
    return arr[:, off + PRE]
ret, mkt, vol = win("ret"), win("mkt"), win("vol")
opn, hig, low, cls_, sr, shr = win("open"), win("high"), win("low"), win("close"), win("sr"), win("shrout")
dnw = win("dn")
day_gap = at(dnw, 0) - daynum(E.day0)
ar = E[path_cols].astype(float).values                       # ar[:, j] = abnormal return on day j+2
_x, _y = ar[:, 0], at(ret, 2) - at(mkt, 2); _m = np.isfinite(_x) & np.isfinite(_y)
chk = np.corrcoef(_x[_m], _y[_m])[0, 1] if _m.sum() > 10 else np.nan
say(f"alignment check: corr(ar_d2, raw day-2 return minus market) = {chk:.2f} (should be high, ~0.8+); "
    f"event day matched exactly for {np.mean(day_gap == 0):.1%}")

# outside data: earnings dates, short interest, VIX, follow-on events
gv = pd.to_numeric(E.gvkey, errors="coerce").fillna(-1).astype(np.int64).values
date_k = np.stack([at(dnw, k) for k in range(1, KMAX + 1)], axis=1)   # calendar day number of decision day k

rdq_f = find_file("comp_rdq")
days_to_earn = np.full((NE, KMAX), np.nan); earn_passed = np.full((NE, KMAX), np.nan)
if rdq_f is not None:
    R = pd.read_csv(rdq_f, usecols=["gvkey", "rdq"]).dropna()
    R["key"] = R.gvkey.astype(np.int64) * 100000 + daynum(R.rdq)
    rk = np.unique(R.key.values)
    for j in range(KMAX):
        q = gv * 100000 + np.nan_to_num(date_k[:, j], nan=0).astype(np.int64)
        i = np.searchsorted(rk, q, side="left").clip(0, len(rk) - 1)        # next report on/after day k
        nxt_ok = (rk[i] // 100000 == gv) & (rk[i] >= q)
        actual = np.where(nxt_ok, (rk[i] % 100000) - (q % 100000), np.nan)
        # firms confirm report dates only ~2-4 weeks ahead: use the actual date only if it is within 30 days,
        # otherwise the expected date (last report + 91 days), which an investor could compute that day
        ip0 = (np.searchsorted(rk, q, side="right") - 1).clip(0, len(rk) - 1)
        prev_ok0 = (rk[ip0] // 100000 == gv) & (rk[ip0] <= q)
        expected = np.where(prev_ok0, np.maximum((rk[ip0] % 100000) + 91 - (q % 100000), 0), np.nan)
        days_to_earn[:, j] = np.where(np.isfinite(actual) & (actual <= 30), actual, expected)
        ip = (np.searchsorted(rk, q, side="right") - 1).clip(0, len(rk) - 1)  # last report on/before day k
        prv_ok = (rk[ip] // 100000 == gv) & (rk[ip] <= q)
        earn_passed[:, j] = np.where(prv_ok, ((rk[ip] % 100000) > daynum(E.day0)).astype(float), 0.0)
    days_to_earn = np.clip(days_to_earn, 0, 120)
    days_to_earn[~np.isfinite(date_k)] = np.nan

si_f = find_file("comp_shortint")
si_ratio = np.full((NE, KMAX), np.nan)
if si_f is not None:
    S = pd.read_csv(si_f, usecols=["gvkey", "datadate", "shortint"]).dropna()
    S = S.groupby(["gvkey", "datadate"], as_index=False).shortint.max()
    S["key"] = S.gvkey.astype(np.int64) * 100000 + daynum(S.datadate) + 10   # published ~8 business days after settlement
    S = S.sort_values("key")
    sk, sv = S.key.values, S.shortint.values   # raw shares, matched to shares outstanding on the same day
    for j in range(KMAX):
        q = gv * 100000 + np.nan_to_num(date_k[:, j], nan=0).astype(np.int64)
        i = (np.searchsorted(sk, q, side="right") - 1).clip(0, len(sk) - 1)
        okk = (sk[i] // 100000 == gv) & (sk[i] <= q) & ((q - sk[i]) < 60)
        si_ratio[:, j] = np.where(okk, sv[i] / (at(shr, j + 1) * 1000), np.nan)
    si_ratio = np.clip(si_ratio, 0, 1)

vix_k = np.full((NE, KMAX), np.nan); vix0 = np.full(NE, np.nan)
fr_f = PUBLIC / "fred_daily.csv"
if fr_f.exists():
    fr = pd.read_csv(fr_f, parse_dates=["date"]).sort_values("date")
    fr["vix"] = fr.vix.ffill()
    fd, fv = daynum(fr.date), fr.vix.values
    def vix_at(dn):
        i = (np.searchsorted(fd, np.nan_to_num(dn, nan=0).astype(np.int64), side="right") - 1).clip(0, len(fd) - 1)
        return np.where(np.isfinite(dn), fv[i], np.nan)
    vix0 = vix_at(at(dnw, 0))
    for j in range(KMAX):
        vix_k[:, j] = vix_at(date_k[:, j])

allev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"])
ak = np.unique(allev.permno.astype(np.int64).values * 100000 + daynum(allev.day0))
i = np.searchsorted(ak, k0, side="right").clip(0, len(ak) - 1)
next_ev = np.where((ak[i] // 100000) == E.permno.values, ak[i] % 100000, np.inf)

# the path so far and labels
cum = np.concatenate([np.zeros((NE, 1)), np.cumsum(ar, axis=1)], axis=1)   # cum[:, k-1] = abnormal return day 1 close -> day k close
trough = cum.min(axis=1)
sdd = (E.idio_vol_pre.values / np.sqrt(252)).clip(0.003)                 # daily idiosyncratic sd (same as modeling.SD_DAY)
bar = np.maximum(0.01, BAR_MULT * sdd * np.sqrt(H))
pre_vol_med = np.nanmedian(vol[:, :PRE - 10], axis=1)                      # days -60..-11
pre_sr = np.nanmean(sr[:, PRE - 30:PRE - 5], axis=1)                       # days -30..-6

rows = {}
def put(name, arr):
    rows[name] = np.asarray(arr, dtype="float32").reshape(-1)
K = np.arange(1, KMAX + 1)
ck = cum[:, :KMAX]                                                          # level at close of day k
runlow = np.minimum.accumulate(cum, axis=1)[:, :KMAX]
argl = np.stack([np.argmin(cum[:, :k], axis=1) + 1 for k in K], axis=1)
sq = np.sqrt(K)[None, :]
put("k", np.broadcast_to(K, (NE, KMAX)))
put("cum_k", ck); put("cum_k_z", ck / (sdd[:, None] * sq))
put("above_low", ck - runlow); put("above_low_z", (ck - runlow) / (sdd[:, None] * sq))
put("days_since_low", K[None, :] - argl)
put("car01_plus_cum", E.car_0_1.values[:, None] + ck)
ar_k = np.full((NE, KMAX), np.nan); ar_k[:, 1:] = ar[:, :KMAX - 1]
put("ar_k", ar_k); put("ar_k_z", ar_k / sdd[:, None])
last3 = np.full((NE, KMAX), np.nan)
for j in range(KMAX):
    last3[:, j] = ck[:, j] - (cum[:, j - 3] if j >= 3 else 0.0)
put("ar_last3_z", last3 / (sdd[:, None] * np.sqrt(3)))
pv = np.full((NE, KMAX), np.nan)
for j in range(2, KMAX):
    pv[:, j] = ar[:, :j].std(axis=1)
put("path_vol_ratio", pv / sdd[:, None])
mk = np.stack([at(mkt, k) for k in K], axis=1)
mcum = np.cumsum(np.nan_to_num(np.stack([at(mkt, k) if k >= 2 else np.zeros(NE) for k in K], axis=1)), axis=1)
put("mkt_k", mk); put("mkt_cum_k", mcum)
vk = np.stack([at(vol, k) for k in K], axis=1)
put("vol_rel_k", np.log1p(vk) - np.log1p(pre_vol_med)[:, None])
put("vol_decay_k", np.log1p(vk) - np.log1p(at(vol, 0))[:, None])
v3 = np.stack([np.nanmean(np.stack([at(vol, max(k - d, 0)) for d in range(3)], 1), 1) for k in K], axis=1)
put("vol_rel3", np.log1p(v3) - np.log1p(pre_vol_med)[:, None])
hk, lk, ckp, ok_ = (np.stack([at(a, k) for k in K], axis=1) for a in (hig, low, cls_, opn))
rng = hk - lk
put("clv_k", np.where(rng > 0, (ckp - lk) / np.where(rng > 0, rng, 1), 0.5))
put("range_k_z", (rng / ckp) / sdd[:, None])
prevc = np.stack([at(cls_, k - 1) for k in K], axis=1)
put("gap_k", np.clip(ok_ / prevc - 1, -0.5, 0.5))
# FINRA posts each day's file in the evening, after the close, so the close of day k can only use day k-1
srk = np.stack([at(sr, k - 1) for k in K], axis=1)
sr3 = np.stack([np.nanmean(np.stack([at(sr, max(k - 1 - d, 0)) for d in range(3)], 1), 1) for k in K], axis=1)
# deviations from the stock's own pre-event level only: the raw level jumps when FINRA added venues (pre-2018 files)
put("sr_dev_k", srk - pre_sr[:, None]); put("sr3_dev", sr3 - pre_sr[:, None])
put("sr_day0_dev", np.broadcast_to((at(sr, 0) - pre_sr)[:, None], (NE, KMAX)))
put("vix_k", vix_k); put("vix_chg_k", vix_k - vix0[:, None])
put("days_to_earn", days_to_earn); put("earn_passed", earn_passed)
put("si_ratio", si_ratio)
dtc = si_ratio * (np.stack([at(shr, k) for k in K], axis=1) * 1000) / np.maximum(pre_vol_med, 1)[:, None]
put("days_to_cover", np.clip(dtc, 0, 60))
put("follow_on", (next_ev[:, None] <= date_k).astype(float))
DYN = list(rows.keys())
if LIVE:   # not available live for free: Compustat short interest, I/B/E/S report dates, our event database
    DYN = [c for c in DYN if c not in ("si_ratio", "days_to_cover", "days_to_earn", "earn_passed", "follow_on")]

# labels
fwd = np.full((NE, KMAX), np.nan)
# "today is the low": close of day k is within 0.25 daily sd of the lowest close over days 1..20
is_low = (cum[:, :KMAX] <= (trough + 0.25 * sdd)[:, None]).astype(float)
# random-walk baseline: share of events whose low fell on day k (the arcsine U-shape), per k
rw_base = is_low.mean(axis=0)
lab = np.full((NE, KMAX), np.nan); lab_any = np.full((NE, KMAX), np.nan); dip = np.full((NE, KMAX), np.nan)
for j in range(KMAX):
    f = cum[:, j + 1:j + 1 + H] - cum[:, [j]]
    dn_hit = f <= -bar[:, None]; up_hit = f >= bar[:, None]
    first_dn = np.where(dn_hit.any(1), dn_hit.argmax(1), 99); first_up = np.where(up_hit.any(1), up_hit.argmax(1), 99)
    lab_any[:, j] = ((first_dn < 99) & (first_dn < first_up)).astype(float)
    lab[:, j] = np.where(first_dn < first_up, 1.0, np.where(first_up < first_dn, 0.0, np.nan))   # NaN = neither hit
    dip[:, j] = np.maximum(0, cum[:, j] - cum[:, j + 1:].min(axis=1))
    fwd[:, j] = cum[:, min(j + H, 19)] - cum[:, j]                  # abnormal move over the next H days

# days until the 20-day low, seen from the close of day k: 0 = the low is already in (on or before day k)
LOW_DAY = np.argmin(cum, axis=1) + 1                                        # day 1..20 of the lowest close
TTL_EDGES = [0, 2, 5, 10]                                                  # buckets: 0 | 1-2 | 3-5 | 6-10 | 11+
TTL_LABELS = ["already in", "1-2 days", "3-5 days", "6-10 days", "11+ days"]
rem = np.maximum(LOW_DAY[:, None] - K[None, :], 0)
ttl = np.digitize(rem, TTL_EDGES, right=True)                              # 0..4
# 17_train_risk_models predictions are left out: they are functions of these same features, and are blank before 2006,
# which the first fold would have to learn from.
STAT = [c for c in BASE_FEATS if c in E]
FEATS = STAT + DYN
P = pd.DataFrame({c: np.repeat(E[c].astype("float32").values, KMAX) for c in STAT})
for c in DYN:
    P[c] = rows[c]
P["ev"] = np.repeat(np.arange(NE), KMAX)
P["y"] = lab.reshape(-1); P["y_any"] = lab_any.reshape(-1); P["dip"] = dip.reshape(-1)
P["is_low"] = is_low.reshape(-1); P["rw_base"] = np.tile(rw_base, NE)
P["ttl"] = ttl.reshape(-1)
P["fwd_z"] = (fwd / (sdd[:, None] * np.sqrt(H))).reshape(-1)
for c in FEATS:                                                   # infinities (e.g. zero volume) -> missing
    v = P[c].values; v[~np.isfinite(v)] = np.nan
P["dip_s"] = (dip / (sdd[:, None] * np.sqrt(20 - K)[None, :])).reshape(-1)
day0 = E.day0.values
year = pd.DatetimeIndex(day0).year.values

# EDA
section(2, "Decision panel (one row per event per decision day)")
say(f"rows: {len(P):,} = {NE:,} events x {KMAX} decision days;  features: {len(FEATS)} ({len(STAT)} event-level, {len(DYN)} daily)")
say(f"barrier b = {BAR_MULT} x daily idio sd x sqrt({H}); look-ahead {H} trading days")
say(f"  share of rows where price falls by b before rising by b (old symmetric label): {P.y_any.mean():.1%}")
say(f"  share where neither barrier is hit (flat): {P.y.isna().mean():.1%}")
say(f"  label used, 'waiting pays' = fell first, among rows where a barrier was hit: {P.y.mean():.1%}")
say("  why: the old label rewarded volatility (jumpy stocks hit any barrier more), not direction. "
    "This one only scores the model on which way the next move goes.")
say(f"median barrier size: {np.median(bar):.2%} of price")

section(3, "Base rate of 'waiting pays' (fell first | a barrier was hit) by decision day k")
br = P.groupby("k").agg(base_rate=("y", "mean"), mean_further_dip_pct=("dip", lambda s: 100 * s.mean())).round(3)
say(br.T.to_string())

section(4, "Base rate by size of the day 0-1 shock (car_0_1)")
sb = pd.cut(np.repeat(E.car_0_1.values, KMAX), [-9, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 9],
            labels=["<-10%", "-10..-5", "-5..-2", "-2..2", "2..5", "5..10", ">10%"])
tab = pd.DataFrame({"shock": sb, "y": P.y, "k": P.k})
say(tab[tab.k <= 5].groupby("shock").y.agg(["mean", "count"]).round(3).rename(columns={"mean": "rate_days1to5"}).to_string())

section(5, "Missing values in the daily features (share of rows)")
miss = P[DYN].isna().mean().sort_values(ascending=False)
say((miss[miss > 0] * 100).round(1).to_string())
say("FINRA short-sale data starts Aug 2009 and Compustat short interest Jul 2006; earlier rows are blank by design.")

section(6, "Single-feature AUC for 'waiting pays' (0.5 = no information; ranked by distance from 0.5)")
from sklearn.metrics import roc_auc_score, log_loss
aucs = {}
for c in DYN + [c for c in ("car_0_1", "idio_vol_pre") if c in P]:
    m = P[c].notna() & P.y.notna()
    if m.sum() > 1000 and P.loc[m, c].nunique() > 1:
        aucs[c] = roc_auc_score(P.y[m], P[c][m])
au = pd.Series(aucs); au = au.reindex((au - 0.5).abs().sort_values(ascending=False).index)
say(au.head(20).round(3).to_string())

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(br.index, br.base_rate * 100, marker="o")
    ax.set_title("Share of days where waiting would have paid, by day after the event")
    ax.set_xlabel("Decision day k (trading days after the event)"); ax.set_ylabel("% of events")
    fig.tight_layout(); fig.savefig(OUT / "01_waiting_pays_by_day.png", dpi=120); plt.close(fig)
except Exception as e:
    say(f"(chart skipped: {e})")

# models
def make_clf():
    if lgb is None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=3000, l2_regularization=50)
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=3000,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50, verbose=-1)
def make_mc():
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=3000,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50, verbose=-1)
NC = len(TTL_LABELS)
def mc_proba(m, Xs):
    out = np.zeros((len(Xs), NC)); out[:, m.classes_.astype(int)] = m.predict_proba(Xs); return out
def make_logit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
def make_reg():
    if lgb is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=1000)
    return lgb.LGBMRegressor(objective="huber", alpha=1.0, n_estimators=400, learning_rate=0.03, num_leaves=31,
                             min_child_samples=1000, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1)
CAP = 300_000 if TEST else 1_500_000
rng_ = np.random.default_rng(0)
def cap(idx):
    return idx if len(idx) <= CAP else np.sort(rng_.choice(idx, CAP, replace=False))

p_wait = np.full(len(P), np.nan); p_dip = np.full(len(P), np.nan); placebo_auc = None
p_logit = np.full(len(P), np.nan); p_fwd = np.full(len(P), np.nan); p_low = np.full(len(P), np.nan)
p_ttl = np.full((len(P), NC), np.nan); b_ttl = np.full((len(P), NC), np.nan)
row_year = np.repeat(year, KMAX); row_day = np.repeat(day0, KMAX)
X = P[FEATS]
for fi, (a, b) in enumerate(FOLDS):
    te = np.where((row_year >= a) & (row_year <= b))[0]
    if len(te) == 0: continue
    tr = np.where(row_day < (row_day[te].min() - PURGE.to_timedelta64()))[0]
    if len(tr) < 5000 or P.y.iloc[tr].nunique() < 2: continue
    trd = cap(tr); tr = cap(tr[P.y.iloc[tr].notna().values])
    m = make_clf().fit(X.iloc[tr], P.y.iloc[tr].astype(int))
    p_wait[te] = m.predict_proba(X.iloc[te])[:, 1]
    r = make_reg().fit(X.iloc[trd], P.dip_s.iloc[trd])
    p_dip[te] = r.predict(X.iloc[te])
    p_logit[te] = make_logit().fit(X.iloc[tr], P.y.iloc[tr].astype(int)).predict_proba(X.iloc[te])[:, 1]
    p_fwd[te] = make_reg().fit(X.iloc[trd], P.fwd_z.iloc[trd]).predict(X.iloc[te])
    p_low[te] = make_clf().fit(X.iloc[trd], P.is_low.iloc[trd].astype(int)).predict_proba(X.iloc[te])[:, 1]
    p_ttl[te] = mc_proba(make_mc().fit(X.iloc[trd], P.ttl.iloc[trd].astype(int)), X.iloc[te])
    bk_ = pd.crosstab(P.k.iloc[trd], P.ttl.iloc[trd], normalize="index").reindex(columns=range(NC), fill_value=0)   # baseline: day number only
    b_ttl[te] = bk_.reindex(P.k.iloc[te]).values
    if fi == len(FOLDS) - 1 or (TEST and placebo_auc is None):
        ys = P.y.iloc[tr].values.copy(); rng_.shuffle(ys)
        pm = make_clf().fit(X.iloc[tr], ys.astype(int))
        tl = te[P.y.iloc[te].notna().values]
        placebo_auc = roc_auc_score(P.y.iloc[tl], pm.predict_proba(X.iloc[tl])[:, 1])
    say(f"  fold {a}-{b}: trained on {len(tr):,} rows, tested on {len(te):,}")

ok = np.isfinite(p_wait) & P.y.notna().values
section(7, "Out-of-sample skill of the daily 'wait or buy' model")
yv = P.y.values
auc = roc_auc_score(yv[ok], p_wait[ok])
base = yv[ok].mean()
say(f"direction AUC {auc:.3f}   (shuffled-label placebo: "
    + (f"{placebo_auc:.3f})" if placebo_auc is not None else "n/a)"))
say("note: the old late-dip AUC (0.570) answered a different question, so it is not directly comparable.")
okl = ok & np.isfinite(p_logit)
say(f"linear benchmark (logistic regression) direction AUC {roc_auc_score(yv[okl], p_logit[okl]):.3f}  "
    "(close to the boosted model = the signal is simple; far below = boosting finds real interactions)")
say(f"log loss {log_loss(yv[ok], p_wait[ok]):.4f} vs base rate {log_loss(yv[ok], np.full(ok.sum(), base)):.4f}")
byk = pd.DataFrame({"k": P.k[ok], "y": yv[ok], "p": p_wait[ok]}).groupby("k").apply(
    lambda g: pd.Series({"auc": roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan, "base_rate": g.y.mean()}))
say("AUC by decision day:"); say(byk.T.round(3).to_string())
cal = pd.DataFrame({"p": p_wait[ok], "y": yv[ok]})
cal["bin"] = pd.qcut(cal.p, 10, labels=False, duplicates="drop")
say("Calibration (predicted vs realised, by decile of prediction):")
say(cal.groupby("bin").agg(predicted=("p", "mean"), realised=("y", "mean"), n=("y", "size")).round(3).to_string())
dip_pct = p_dip * np.repeat(sdd, KMAX) * np.tile(np.sqrt(20 - K), NE)       # back to % of price
okd = np.isfinite(dip_pct)
from scipy.stats import spearmanr
say(f"further-dip model: spearman {spearmanr(dip_pct[okd], P.dip.values[okd]).correlation:.3f} between predicted and realised further dip")
wk = pd.DataFrame({"k": P.k.values[okd], "p": p_dip[okd], "r": P.dip_s.values[okd]})
say("  part of that is mechanical (fewer days left = smaller dip; volatile stocks dip more). Honest skill = "
    f"volatility-scaled dip, within the same decision day: spearman {wk.groupby('k').apply(lambda g: spearmanr(g.p, g.r).correlation).mean():.3f}")
okf = np.isfinite(p_fwd)
fk = pd.DataFrame({"k": P.k.values[okf], "p": p_fwd[okf], "r": P.fwd_z.values[okf]})
say(f"next-{H}-day move model (direction and size, volatility-scaled), within decision day: "
    f"spearman {fk.groupby('k').apply(lambda g: spearmanr(g.p, g.r).correlation).mean():.3f}")
dq = pd.qcut(dip_pct[okd], 5, labels=False, duplicates="drop")
say("realised further dip (% below that day's close, by day 20) by predicted quintile:")
say(pd.DataFrame({"q": dq, "pred": dip_pct[okd] * 100, "real": P.dip.values[okd] * 100}).groupby("q").mean().round(2).to_string())

section("7b", "How likely is today the low? (calibrated probability vs the random-walk baseline)")
yl = P.is_low.values
at_low = P.above_low.values <= 0.25 * np.repeat(sdd, KMAX)   # today is within 0.25 daily sd of the lowest close so far (same cutoff as the label)
okL = np.isfinite(p_low) & at_low
say("Only days where the stock is AT its lowest close since the event are scored. On any other day the answer is")
say("known to be 'no', and counting those would inflate the result (unconditional AUC ~0.86, which is not skill).")
say(f"days at a new low: {at_low.mean():.1%} of event-days.  Of those, the share that hold as the 20-day low: {yl[okL].mean():.1%}")
# baseline that knows only the day number, among at-low days
rwb = pd.Series(yl[okL]).groupby(P.k.values[okL]).transform("mean").values
say(f"AUC {roc_auc_score(yl[okL], p_low[okL]):.3f} vs day-number-only baseline {roc_auc_score(yl[okL], rwb):.3f}  "
    f"(logistic benchmark {roc_auc_score(yl[okL], make_logit().fit(X[at_low & np.isfinite(p_low) & (row_year <= 2015)], yl[at_low & np.isfinite(p_low) & (row_year <= 2015)].astype(int)).predict_proba(X[okL])[:, 1]) if (row_year[okL] > 2015).any() else float('nan'):.3f} fitted on <=2015, scored on all at-low days)")
cl = pd.DataFrame({"p": p_low[okL], "y": yl[okL], "base": rwb, "k": P.k.values[okL]})
cl["bin"] = pd.qcut(cl.p, 10, labels=False, duplicates="drop")
say("calibration by decile of predicted P(this low holds): predicted vs realised vs day-number baseline")
say(cl.groupby("bin").agg(predicted=("p", "mean"), realised=("y", "mean"), baseline=("base", "mean"), n=("y", "size")).round(3).to_string())
top = cl[cl.p >= cl.p.quantile(0.9)]
say(f"top 10% most confident days: model says {top.p.mean():.1%}, realised {top.y.mean():.1%}, "
    f"baseline for those same days {top.base.mean():.1%}  -> lift x{top.y.mean() / max(top.base.mean(), 1e-6):.2f}")
say("reading: a probability, not a call. The tracker shows it as 'stock is at a new low; chance this low holds: X%'.")
# by day number: how often a new low on day k holds, and what the model adds
bk = cl.groupby("k").agg(n=("y", "size"), holds=("y", "mean"))
bk["auc_within_day"] = cl.groupby("k").apply(lambda g: roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan)
say("by day of the new low:"); say(bk.T.round(3).to_string())

section("7c", "Where the signal lives: direction AUC and dip-depth skill by subgroup (2006-2026, out of sample)")
grp_rows = []
ev_size = pd.qcut(E.log_mcap.values, 3, labels=["small", "mid", "large"]).astype(str)
ev_shock = pd.cut(E.car_0_1.values, [-9, -0.10, -0.03, 0.03, 0.10, 9], labels=["<-10%", "-10..-3", "-3..3", "3..10", ">10%"]).astype(str)
ev_cls = E.cls.values
for name, arr in (("size", ev_size), ("shock", ev_shock), ("class", ev_cls)):
    rowg = np.repeat(arr, KMAX)
    for g in pd.unique(arr):
        m = ok & (rowg == g); md = okd & (rowg == g)
        if m.sum() < 2000: continue
        dsk = pd.DataFrame({"k": P.k.values[md], "p": p_dip[md], "r": P.dip_s.values[md]}).groupby("k").apply(
            lambda t: spearmanr(t.p, t.r).correlation).mean()
        grp_rows.append({"split": name, "group": g, "n_rows": int(m.sum()), "direction_auc": roc_auc_score(yv[m], p_wait[m]),
                         "dip_depth_spearman": dsk, "low_holds_auc": roc_auc_score(yl[okL & (rowg == g)], p_low[okL & (rowg == g)]) if (okL & (rowg == g)).sum() > 500 else np.nan})
say(pd.DataFrame(grp_rows).round(3).to_string(index=False))
say("the literature (Lopez-Lira & Tang 2024; PEAD studies) finds predictability concentrated in small caps and after negative news; check whether we see the same.")

TTL_LL, TTL_T = (np.nan, np.nan), np.nan
try:                                    # a reporting error here must not stop the overnight retrain
    section("7d", "How many more trading days until the 20-day low? (probabilities, out of sample)")
    okT = np.isfinite(p_ttl).all(axis=1)
    yT = P.ttl.values[okT].astype(int)
    def ll(pr, y): return -np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-6, 1)))
    say("buckets: " + ", ".join(TTL_LABELS) + ". 'already in' = the lowest close of days 1-20 has already happened.")
    say(f"log loss: model {ll(p_ttl[okT], yT):.4f}  vs baseline that knows only the day number {ll(b_ttl[okT], yT):.4f}  (lower is better)")
    late = okT & (row_year >= 2016); yL = P.ttl.values[late].astype(int)
    TTL_LL = (ll(p_ttl[late], yL), ll(b_ttl[late], yL))
    say(f"2016-2026 only: model {TTL_LL[0]:.4f} vs baseline {TTL_LL[1]:.4f}")
    mon = pd.Series(row_day[late]).dt.to_period("M").values
    dd = pd.DataFrame({"m": mon, "d": -np.log(np.clip(b_ttl[late][np.arange(len(yL)), yL], 1e-6, 1)) + np.log(np.clip(p_ttl[late][np.arange(len(yL)), yL], 1e-6, 1))}).groupby("m").d.mean()
    TTL_T = dd.mean() / (dd.std() / np.sqrt(len(dd)))
    say(f"monthly gain in log-likelihood over the baseline, 2016-2026: {dd.mean():.4f} (t = {TTL_T:.2f}, {(dd > 0).mean():.0%} of months better)")
    say(f"most likely bucket is right: model {np.mean(p_ttl[okT].argmax(1) == yT):.1%}, baseline {np.mean(b_ttl[okT].argmax(1) == yT):.1%}")
    cT = pd.DataFrame({"p": p_ttl[okT][:, 0], "y": (yT == 0).astype(float)}); cT["bin"] = pd.qcut(cT.p, 10, labels=False, duplicates="drop")
    say("calibration of P(low already in), by decile of prediction:")
    say(cT.groupby("bin").agg(predicted=("p", "mean"), realised=("y", "mean"), n=("y", "size")).round(3).to_string())
    cal_all = pd.DataFrame({f"pred_{l}": p_ttl[okT][:, i] for i, l in enumerate(TTL_LABELS)}).mean().round(3)
    say("average predicted vs realised share per bucket:"); say(pd.DataFrame({"predicted": cal_all.values,
        "realised": [np.mean(yT == i) for i in range(NC)]}, index=TTL_LABELS).round(3).to_string())
except Exception as e:
    say(f"(days-to-low report failed: {e})")

# entry simulation, scored by regret
PW = p_wait.reshape(NE, KMAX); PD = dip_pct.reshape(NE, KMAX); PF = p_fwd.reshape(NE, KMAX)
months = pd.DatetimeIndex(day0).to_period("M").astype(str).values
def tclust(d, msk):               # events in the same month move together, so t-stats are computed on monthly averages
    s = pd.Series(d[msk]).groupby(months[msk]).mean()
    return s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else 0.0
def first_above(M, thr):
    a = M > thr
    return np.where(a.any(1), a.argmax(1) + 1, KMAX)
evok = np.isfinite(PW).all(axis=1)
r60 = E.drift_1_60.values
def score(entry_k):                       # entry_k in 1..KMAX (array over events)
    lvl = cum[np.arange(NE), entry_k - 1]
    return lvl - trough, cum[:, 19] - lvl, r60 - lvl
def first_below(M, thr):
    below = M < thr
    return np.where(below.any(1), below.argmax(1) + 1, KMAX)
strategies = {f"buy day {d}": np.full(NE, d) for d in (1, 2, 3, 5, 10, 15)}
strategies["random day 1-15 (placebo)"] = rng_.integers(1, KMAX + 1, NE)
upd = np.full(NE, KMAX)
for j in range(1, KMAX):
    hitj = (ar[:, j - 1] > 0) & (ar[:, j - 2] < 0 if j >= 2 else E.car_0_1.values < 0) & (upd == KMAX)
    upd = np.where(hitj, j + 1, upd)
strategies["first up-day after a down-day (simple rule)"] = upd
early = evok & (year <= 2015)
late = evok & (year >= 2016)
def _masks():
    return evok & (year <= 2015), evok & (year >= 2016)
best_thr, best_val = None, np.inf
for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
    v = score(first_below(PW, thr))[0][early].mean()
    if v < best_val: best_thr, best_val = thr, v
best_dip, best_dv = None, np.inf
for x in (0.01, 0.015, 0.02, 0.03, 0.04):
    v = score(first_below(PD, x))[0][early].mean()
    if v < best_dv: best_dip, best_dv = x, v
strategies[f"ENGINE: buy when P(waiting pays) < {best_thr:.2f}"] = first_below(PW, best_thr)
strategies[f"ENGINE: buy when expected further dip < {best_dip:.1%}"] = first_below(PD, best_dip)
best_f, best_fv = None, np.inf
for x in (-0.3, -0.2, -0.1, 0.0, 0.1):
    v = score(first_above(PF, x))[0][early].mean()
    if v < best_fv: best_f, best_fv = x, v
strategies[f"ENGINE: buy when expected next-{H}-day move > {best_f:+.1f} sd"] = first_above(PF, best_f)
# out-of-sample calls saved for the meta-labeling experiment (02_meta_labeling.py); results/ is git-ignored
if not TEST:
    _keep = [c for c in ("k", "fwd_z", "cum_k_z", "above_low_z", "days_since_low", "vol_rel_k", "vol_decay_k", "range_k_z",
                         "path_vol_ratio", "clv_k", "gap_k", "vix_k", "ar_last3_z") if c in P]
    _stat = [c for c in ("car_0_1", "ar_day0_sd", "abn_volume_0_1", "idio_vol_pre", "log_mcap", "mom_60", "dist_52w_high",
                         "exp_earn_in_win", "days_to_exp_earn", "earn_react_hist") if c in E]
    MO = P[_keep].astype("float32").copy()
    MO["p_fwd"], MO["p_wait"], MO["p_dip"] = p_fwd.astype("float32"), p_wait.astype("float32"), p_dip.astype("float32")
    MO["ev"] = np.repeat(np.arange(NE), KMAX).astype("int32")
    for c in _stat: MO[c] = np.repeat(E[c].values.astype("float32"), KMAX)
    MO["cls"] = np.repeat(E.cls.astype("category").cat.codes.values.astype("int16"), KMAX)
    MO.to_parquet(RESULTS / "entry_oos_rows.parquet")
    np.savez_compressed(RESULTS / "entry_oos_events.npz", cum=cum[:, :20].astype("float32"), trough=trough.astype("float32"),
                        year=year, day0=day0.astype("datetime64[D]"), evok=evok, best_f=best_f, KMAX=KMAX)
    say(f"saved out-of-sample calls for meta-labeling -> {RESULTS / 'entry_oos_rows.parquet'}")
# MDP: optimal stopping by fitted Q-iteration
# The buy/wait choice is a finite-horizon Markov decision problem (SC3000 Lectures 3-6; Ernst, Geurts & Wehenkel 2005).
# State x_k = everything known at the close of day k (the same features as above). Actions: BUY now (cost 0 relative to
# today's price) or WAIT one day. Must buy by day KMAX. Cost-to-go, measured in entry price relative to today:
#   V_KMAX(x) = 0
#   Q_k(x)    = E[ (level_{k+1} - level_k) + V_{k+1}(x_{k+1}) | x_k = x ]      (Bellman equation)
#   V_k(x)    = min(0, Q_k(x))                                                 (buy now costs 0)
# Q_k is fitted backward from day KMAX-1 to day 1 with gradient boosting. To stop the regression from grading its own
# homework, V_{k+1} on training rows comes from cross-fitting (model fitted on one half of firms, predicted on the other).
# Policy: buy at the first day with Q_k >= 0, i.e. when waiting is no longer expected to lower the entry price.
# No threshold is tuned: the rule comes from the Bellman equation itself.
q_fqi = np.full((NE, KMAX), np.nan)
FOLDS_MDP = [] if LIVE else FOLDS                 # the MDP is a second opinion; skipped when training the live models
half = (E.permno.values % 2).astype(int)
ev_day = pd.DatetimeIndex(day0)
CAP_FQI = 150_000 if TEST else 300_000
def _cap_ev(ix):
    return ix if len(ix) <= CAP_FQI else np.sort(rng_.choice(ix, CAP_FQI, replace=False))
for fi, (a_, b_) in enumerate(FOLDS_MDP):
    te_ev = np.where((year >= a_) & (year <= b_))[0]
    if len(te_ev) == 0: continue
    tr_ev = np.where(ev_day < (ev_day[te_ev].min() - PURGE))[0]
    if len(tr_ev) < 2000: continue
    v_tr = np.zeros(len(tr_ev))
    for k in range(KMAX - 1, 0, -1):
        rows_tr, rows_te = tr_ev * KMAX + (k - 1), te_ev * KMAX + (k - 1)
        tgt = (cum[tr_ev, k] - cum[tr_ev, k - 1]) + v_tr
        sub = _cap_ev(np.arange(len(tr_ev)))
        q_tr = np.zeros(len(tr_ev))
        for h in (0, 1):                                   # cross-fitting by firm halves
            fit_i = sub[half[tr_ev[sub]] == h]
            pred_i = np.where(half[tr_ev] != h)[0]
            mh = make_reg().fit(X.iloc[rows_tr[fit_i]], tgt[fit_i])
            q_tr[pred_i] = mh.predict(X.iloc[rows_tr[pred_i]])
        m_full = make_reg().fit(X.iloc[rows_tr[sub]], tgt[sub])
        q_fqi[te_ev, k - 1] = m_full.predict(X.iloc[rows_te])
        v_tr = np.minimum(0.0, q_tr)
    q_fqi[te_ev, KMAX - 1] = 0.0
    say(f"  fitted Q-iteration fold {a_}-{b_}: {len(tr_ev):,} training events, {KMAX - 1} backward steps")
def first_q_nonneg(Q):
    ok_ = Q >= 0
    return np.where(ok_.any(1), ok_.argmax(1) + 1, KMAX)
if not LIVE:
    evok = evok & np.isfinite(q_fqi).all(axis=1)
if not LIVE:
    strategies["ENGINE: MDP optimal stopping (fitted Q-iteration, no tuning)"] = first_q_nonneg(q_fqi)
    say(f"MDP policy: mean entry day {strategies['ENGINE: MDP optimal stopping (fitted Q-iteration, no tuning)'][evok].mean():.2f}; "
        f"share buying on day 1: {(strategies['ENGINE: MDP optimal stopping (fitted Q-iteration, no tuning)'][evok] == 1).mean():.1%}")
dca_reg = np.mean([cum[:, j] - trough for j in range(KMAX)], axis=0)
early, late = _masks()

section(8, "Entry regret: how far above the 20-day low you bought (lower is better)")
say(f"Thresholds chosen using 2006-2015 only (P threshold {best_thr:.2f}, dip threshold {best_dip:.1%}); "
    "the 2016-2026 column is an untouched test.")
say("regret = abnormal price at entry minus lowest abnormal price on days 1-20, in %. "
    "ret_to_d20 / ret_to_d60 = abnormal return from entry to day 20 / 60 (checks that waiting does not just miss gains).")
base1 = score(strategies["buy day 1"])[0]
res = []
for name, ek in strategies.items():
    rg, r20, r6 = score(ek)
    row = {"strategy": name, "mean_entry_day": ek[evok].mean()}
    for lab_, msk in (("2006-15", early), ("2016-26", late)):
        d = (rg - base1)[msk]
        row[f"regret_{lab_}"] = 100 * rg[msk].mean()
        row[f"vs_day1_t_{lab_}"] = tclust(rg - base1, msk)
    row["ret_to_d20_2016-26"] = 100 * r20[late].mean(); row["ret_to_d60_2016-26"] = 100 * np.nanmean(r6[late])
    res.append(row)
res.append({"strategy": "spread evenly over days 1-15", "mean_entry_day": 8.0,
            "regret_2006-15": 100 * dca_reg[early].mean(), "regret_2016-26": 100 * dca_reg[late].mean()})
RT = pd.DataFrame(res)
from scipy.stats import norm as _norm
_p = pd.Series(1 - _norm.cdf(-RT["vs_day1_t_2016-26"].fillna(0).values), index=RT.index)   # one-sided: beats day 1 on the untouched years?
_o = _p.sort_values().index; _m = int(RT["vs_day1_t_2016-26"].notna().sum())
RT.loc[_o, "p_holm_2016-26"] = np.minimum(1, np.maximum.accumulate(_p[_o].values * (_m - np.arange(len(_o))).clip(1)))
say(RT.round(3).to_string(index=False))
say(f"p_holm_2016-26: one-sided p-value for beating 'buy day 1' on the untouched years, Holm-adjusted for testing {_m} strategies.")
say("vs_day1_t < -2 means the strategy buys significantly closer to the low than buying at day 1 "
    "(t-stats use monthly averages, because events in the same month are not independent).")

section(9, "Where it matters most: big negative shocks (day 0-1 abnormal return worse than -10%)")
big = E.car_0_1.values < -0.10
rows9 = []
for name in [n for n in strategies if n in ("buy day 1", "buy day 5") or n.startswith(("first up", "ENGINE"))]:
    rg, r20, _ = score(strategies[name])
    msk = evok & big & (year >= 2006)
    d = (rg - base1)[msk]
    rows9.append({"strategy": name, "n": int(msk.sum()), "regret": 100 * rg[msk].mean(), "ret_to_d20": 100 * r20[msk].mean(),
                  "vs_day1_t": tclust(rg - base1, msk)})
say(pd.DataFrame(rows9).round(3).to_string(index=False))

try:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    RT2 = RT.dropna(subset=["regret_2016-26"]).sort_values("regret_2016-26")
    ax.barh(RT2.strategy, RT2["regret_2016-26"])
    ax.set_title("Entry regret, 2016-2026 test years (lower = closer to the 20-day low)")
    ax.set_xlabel("Mean % above the 20-day low at entry"); fig.tight_layout()
    fig.savefig(OUT / "02_entry_regret_by_strategy.png", dpi=120); plt.close(fig)
except Exception as e:
    say(f"(chart skipped: {e})")

# final models on all rows, for the live tracker
if not TEST:
    allr = cap(np.where(P.y.notna())[0])
    mf = make_clf().fit(X.iloc[allr], P.y.iloc[allr].astype(int))
    alld = cap(np.arange(len(P)))
    rf = make_reg().fit(X.iloc[alld], P.dip_s.iloc[alld])
    ff = make_reg().fit(X.iloc[alld], P.fwd_z.iloc[alld])
    lf = make_clf().fit(X.iloc[alld], P.is_low.iloc[alld].astype(int))
    tf = make_mc().fit(X.iloc[alld], P.ttl.iloc[alld].astype(int))
    ttl_base = pd.crosstab(P.k, P.ttl, normalize="index").reindex(columns=range(NC), fill_value=0)
    joblib.dump({"clf": mf, "dip_reg": rf, "fwd_reg": ff, "fwd_threshold": best_f, "low_clf": lf, "rw_base": rw_base, "ttl_clf": tf, "ttl_labels": TTL_LABELS, "ttl_base": ttl_base.values.tolist(),
                 "ttl_logloss_2016": TTL_LL, "ttl_t": float(TTL_T), "feats": FEATS, "stat_feats": STAT, "dyn_feats": DYN, "threshold": best_thr,
                 "dip_threshold": best_dip, "H": H, "KMAX": KMAX, "bar_mult": BAR_MULT, "auc_oos": auc, "version": 1},
                MODELS / "entry_engine.joblib")
    say(""); say(f"saved {MODELS / 'entry_engine.joblib'}")
(RESULTS / ("entry_engine_TEST.md" if TEST else "entry_engine.md")).write_text(
    "# Entry engine (research only, not investment advice)\n\n```\n" + "\n".join(LOG) + "\n```\n")
say(f"saved {RESULTS / ('entry_engine_TEST.md' if TEST else 'entry_engine.md')} and charts in {RESULTS / 'entry'}")
