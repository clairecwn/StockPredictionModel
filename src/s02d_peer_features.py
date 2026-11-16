"""Step 2d: relational (peer-graph) and short-interest features, keyed by (permno, day0). Run after s02_events.py.

1. Peer graph (the cheap, honest version of graph neural networks such as RTGCN / HIST):
   for each year Y, every stock's 10 most correlated stocks are found from daily returns in year Y-1 only
   (known before any event in Y). Then for each event:
     peer_car_01   mean market-adjusted return of the 10 peers on days 0-1 (did the whole group move?)
     own_vs_peer   car_0_1 - peer_car_01 (the firm-specific part of the shock)
     peer_pre20    peers' mean market-adjusted return over days -20..-1 (group momentum)
     peer_event    share of peers that had their own event on day 0 or 1 (spillover / contagion)
     peer_corr     average correlation with the 10 peers (how connected the stock is)
2. Short interest (Compustat, twice a month): si_pre = short interest / shares outstanding, using the latest figure
   published at least 10 calendar days before day 0; si_chg = change vs the previous figure (shorts building up?).

Output: data/processed/peer_features.parquet
Quick check on a few years: PEER_TEST=1 python src/s02d_peer_features.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from config import PROC, find_file

TEST = os.environ.get("PEER_TEST") == "1"
K = 10
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "gvkey", "day0", "car_0_1"]).drop_duplicates(["permno", "day0"])
ev["permno"] = ev.permno.astype(np.int64)
years = sorted(ev.day0.dt.year.unique())
if TEST:
    years = years[-3:]
allev = ev[["permno", "day0"]].copy()
print(f"events: {len(ev):,}; years {years[0]}-{years[-1]}" + ("   [TEST MODE]" if TEST else ""))

out = []
for Y in years:
    t = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "mkt"],
                      filters=[("date", ">=", pd.Timestamp(f"{Y - 1}-01-01")), ("date", "<", pd.Timestamp(f"{Y + 1}-01-15"))]).to_pandas()
    t["ar"] = t.ret - t.mkt
    P = t.pivot_table(index="date", columns="permno", values="ar").sort_index()
    est = P[P.index.year == Y - 1]
    est = est.loc[:, est.notna().sum() >= 200]
    if est.shape[1] < 50:
        continue
    Z = (est - est.mean()) / est.std()
    Z = Z.fillna(0.0).values.astype(np.float32)
    C = (Z.T @ Z) / (Z.shape[0] - 1)
    np.fill_diagonal(C, -np.inf)
    nb = np.argpartition(-C, K, axis=1)[:, :K]
    cols = est.columns.values
    nb_corr = np.take_along_axis(C, nb, axis=1).mean(axis=1)
    pos = {p: i for i, p in enumerate(cols)}
    cur = P.loc[P.index >= pd.Timestamp(f"{Y}-01-01")]
    dates = cur.index.values
    A = cur.reindex(columns=cols).values                        # days x stocks (market-adjusted returns)
    cum = np.nancumsum(np.nan_to_num(A), axis=0)
    evd = allev[allev.day0.dt.year.isin([Y, Y + 1])]
    has_ev = np.zeros(A.shape, dtype=bool)
    di = np.searchsorted(dates, evd.day0.values); ci = evd.permno.map(pos)
    okk = ci.notna().values & (di < len(dates))
    has_ev[di[okk], ci[okk].astype(int).values] = True
    e = ev[ev.day0.dt.year == Y]
    i0 = np.searchsorted(dates, e.day0.values)
    ci = e.permno.map(pos)
    ok = ci.notna().values & (i0 + 1 < len(dates)) & (i0 >= 20)
    e = e[ok]; i0 = i0[ok]; ci = ci[ok].astype(int).values
    peers = nb[ci]                                               # events x K
    R0, R1 = i0[:, None], (i0 + 1)[:, None]
    r01 = np.nanmean(A[R0, peers] + A[R1, peers], axis=1)
    pre = (cum[(i0 - 1)[:, None], peers] - cum[(i0 - 21)[:, None], peers]).mean(axis=1)
    pev = (has_ev[R0, peers] | has_ev[R1, peers]).mean(axis=1)
    out.append(pd.DataFrame({"permno": e.permno.values, "day0": e.day0.values, "peer_car_01": r01,
                             "own_vs_peer": e.car_0_1.values - r01, "peer_pre20": pre, "peer_event": pev,
                             "peer_corr": nb_corr[ci]}))
    print(f"  {Y}: {est.shape[1]:,} stocks in the graph, {len(e):,} events matched", flush=True)
    del t, P, est, Z, C, A, cum, has_ev
F = pd.concat(out, ignore_index=True)

# ---------------------------------------------------------------- short interest
si_f = find_file("comp_shortint")
if si_f is not None:
    S = pd.read_csv(si_f, usecols=["gvkey", "datadate", "shortint"]).dropna()
    S = S.groupby(["gvkey", "datadate"], as_index=False).shortint.max()
    S["avail"] = pd.to_datetime(S.datadate) + pd.Timedelta(days=10)
    S = S.sort_values(["gvkey", "avail"])
    S["si_prev"] = S.groupby("gvkey").shortint.shift(1)
    sh = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "shrout"],
                       filters=[("date", ">=", pd.Timestamp(f"{years[0]}-01-01"))]).to_pandas()
    E2 = ev[["permno", "gvkey", "day0"]].copy()
    E2["gvkey"] = pd.to_numeric(E2.gvkey, errors="coerce")
    S["gvkey"] = pd.to_numeric(S.gvkey, errors="coerce").astype("float64")
    E2 = E2.dropna(subset=["gvkey"]).sort_values("day0")
    E2["gvkey"] = E2.gvkey.astype("float64")
    sh = sh[sh.permno.isin(set(E2.permno))]
    m = pd.merge_asof(E2, S[["gvkey", "avail", "shortint", "si_prev"]].sort_values("avail"), left_on="day0", right_on="avail",
                      by="gvkey", direction="backward", tolerance=pd.Timedelta(days=45))
    m = m.merge(sh.rename(columns={"date": "day0"}), on=["permno", "day0"], how="left")
    shares = m.shrout * 1000
    m["si_pre"] = (m.shortint / shares).clip(0, 1)
    m["si_chg"] = ((m.shortint - m.si_prev) / shares).clip(-0.5, 0.5)
    F = F.merge(m[["permno", "day0", "si_pre", "si_chg"]], on=["permno", "day0"], how="outer")
    print(f"short interest matched for {m.si_pre.notna().mean():.1%} of events (data starts Jul 2006)")

F = F.drop_duplicates(["permno", "day0"])
print("\n=== Peer and short-interest features (EDA) ===")
print(F.drop(columns=["permno", "day0"]).describe().T[["count", "mean", "50%", "min", "max"]].round(4).to_string())
chk = F.merge(ev[["permno", "day0", "car_0_1"]], on=["permno", "day0"])
print(f"\ncorr(car_0_1, peer_car_01) = {chk[['car_0_1', 'peer_car_01']].corr().iloc[0, 1]:.3f}  (positive = peers move with the firm)")
if not TEST:
    F.to_parquet(PROC / "peer_features.parquet", index=False)
    print(f"saved {len(F):,} rows -> data/processed/peer_features.parquet")
