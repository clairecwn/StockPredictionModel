"""Step 7 (experiments 6 and 7): spillover, liquidity and split features, keyed by (permno, day0). Run after 05_peer_features.

Everything below is known by the close of day 1, the same information cutoff as car_0_1 and the peer features.

1. Liquidity and trading activity (CRSP daily):
     turn_chg     log(turnover days -20..-1 / turnover days -250..-21)
     amihud_chg   log(Amihud illiquidity -20..-1 / same over -250..-21)
     spread_chg   relative bid-ask spread -20..-1 minus -250..-21
     zero_ret_60  share of zero-return days in -60..-1
     pre_abvol_5  log(volume -5..-1 / volume -250..-21), the pre-announcement run-up in trading
2. Stock splits, found where shares outstanding jump and the price moves by the inverse factor on the same day:
     split_250    forward split in days -250..1          rsplit_250  reverse split in days -250..1
     split_now    any split in days -5..1
3. Scandal contagion through the return-correlation peers (top 10 from the previous year, as in 05_peer_features):
     peer_scandal_30   number of peers with an accounting or legal event in days -21..-1
     peer_scandal_car  mean day 0-1 return of those peer events (0 if none)
4. News co-mention network (FNSPID, 2009-2023): two firms are linked when the same headline is tagged to both
   (headlines tagged to 2-5 firms). Partners are the top 10 by shared headlines in the previous calendar year.
     cm_n, cm_car_01, cm_pre20, cm_event, cm_scandal_30, and
     cm_conflict  co-mention headlines about lawsuits, accusations, disputes or criticism in calendar days -30..0
5. Customer-supplier links (Compustat customer segments, filings from mid-2020 on). A link is usable 90 days
   after the filing date (time for the 10-K to be public) and for 455 days after that.
     cust_n, cust_pre20, cust_car_01, cust_event   the firm's major customers (sales-weighted)
     supp_n, supp_pre20, supp_car_01               the firm's suppliers, when the firm is someone's customer

Output: data/processed/spillover_features.parquet          Quick check: SPILL_TEST=1 python src/pipeline/07_spillover_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from config import PROC, find_file

TEST = os.environ.get("SPILL_TEST") == "1"
K = 10
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0", "car_0_1", "has_accounting_fraud", "has_legal_regulatory"])
ev = ev.drop_duplicates(["permno", "day0"]).reset_index(drop=True)
ev["permno"] = ev.permno.astype(np.int64)
ev["scandal"] = ((ev.has_accounting_fraud.fillna(0) > 0) | (ev.has_legal_regulatory.fillna(0) > 0)).astype(bool)
years = sorted(ev.day0.dt.year.unique())
if TEST:
    years = [2016, 2022]
print(f"events: {len(ev):,}; scandal events {ev.scandal.sum():,}" + ("   [TEST MODE]" if TEST else ""))

# co-mention edges from FNSPID
CONFLICT = re.compile(r"\b(sue[sd]?|suing|lawsuit|accus\w*|slams?|blasts?|condemn\w*|criticiz\w*|feud|disputes?|boycott\w*|"
                      r"antitrust|infring\w*|spat|clash\w*|probe[sd]?|allegation\w*)\b", re.I)
hl = pd.read_parquet(PROC / "news_headlines.parquet", columns=["date", "ticker", "headline"])
hl = hl[hl.date.dt.year >= 2009]
hl["day"] = hl.date.dt.normalize(); hl["ticker"] = hl.ticker.str.upper()
if TEST:
    hl = hl[hl.date.dt.year.isin([2015, 2021])]
parts = []                                                  # ticker -> permno by date, read in chunks to save memory
for ch in pd.read_csv(find_file("crsp_daily"), usecols=["PERMNO", "Ticker", "DlyCalDt"], dtype={"Ticker": str}, chunksize=2_000_000):
    ch = ch.dropna(subset=["Ticker"]); ch["Ticker"] = ch.Ticker.str.upper()
    parts.append(ch.groupby(["PERMNO", "Ticker"]).DlyCalDt.agg(["min", "max"]).reset_index())
tk = pd.concat(parts).groupby(["PERMNO", "Ticker"]).agg({"min": "min", "max": "max"}).reset_index()
tk.columns = ["permno", "ticker", "min", "max"]; tk["min"] = pd.to_datetime(tk["min"]); tk["max"] = pd.to_datetime(tk["max"])
del parts
hl = hl.merge(tk, on="ticker")
hl = hl[(hl.day >= hl["min"]) & (hl.day <= hl["max"] + pd.Timedelta(days=5))][["day", "headline", "permno"]].drop_duplicates()
hl["gid"] = hl.groupby(["day", "headline"]).ngroup()
size = hl.groupby("gid").permno.transform("nunique")
hl = hl[(size >= 2) & (size <= 5)]
pairs = hl[["gid", "permno", "day", "headline"]].merge(hl[["gid", "permno"]].rename(columns={"permno": "b"}), on="gid")
pairs = pairs[pairs.permno != pairs.b].rename(columns={"permno": "a"})
pairs["year"] = pairs.day.dt.year
pairs["conflict"] = pairs.headline.str.contains(CONFLICT)
EDGES = pairs.groupby(["year", "a", "b"]).size().rename("w").reset_index()
conf = pairs[pairs.conflict].drop_duplicates(["gid", "a"])[["a", "day"]]
CONF_DAYS = conf.groupby("a").day.apply(lambda s: np.sort(s.values.astype("datetime64[D]").astype(np.int64)))
print(f"co-mention: {len(EDGES):,} firm-pair-years; {len(conf):,} conflict co-mentions")
del hl, pairs, tk

# customer-supplier edges
def norm_name(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    drop = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp", "holdings", "holding",
            "group", "the", "sa", "ag", "nv", "incorporated", "stores", "com", "de", "se"}
    return "".join(w for w in s.split() if w not in drop)

CUST = None
cf = find_file("comp_customer")
if cf is not None:
    c = pd.read_csv(cf, low_memory=False)
    c = c[(c.ctype == "COMPANY") & c.salecs.gt(0)]
    c = c[~c.cnms.str.contains(r"^\s*\d+\s+customers|not reported|^customer\s+[a-z0-9]\b", case=False, regex=True, na=True)]
    ids = pd.read_csv(find_file("compustat_ids"), usecols=["gvkey", "conm", "datadate"]).sort_values("datadate")
    ids["key"] = ids.conm.map(norm_name)
    name2gv = ids.drop_duplicates("key", keep="last").set_index("key").gvkey
    c["cust_gvkey"] = c.cnms.map(norm_name).map(name2gv)
    print(f"customer names matched to a listed company: {c.cust_gvkey.notna().mean():.1%} of {len(c):,} links")
    lk = pd.read_csv(find_file("ccm_link")); lk = lk[lk.linkprim.isin(["P", "C"])].dropna(subset=["permno"])
    lk["linkenddt"] = pd.to_datetime(lk.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-01-01"))
    gv2p = lk.sort_values("linkenddt").drop_duplicates("gvkey", keep="last").set_index("gvkey").permno.astype(np.int64)
    c["sup"] = c.gvkey.map(gv2p); c["cus"] = c.cust_gvkey.map(gv2p)
    c = c.dropna(subset=["sup", "cus"]); c = c[c.sup != c.cus]
    c["avail"] = pd.to_datetime(c.srcdate) + pd.Timedelta(days=90); c["end"] = c.avail + pd.Timedelta(days=455)
    CUST = c.groupby(["sup", "cus", "avail", "end"], as_index=False).salecs.max()
    CUST[["sup", "cus"]] = CUST[["sup", "cus"]].astype(np.int64)
    print(f"customer links usable: {len(CUST):,} ({CUST.sup.nunique():,} suppliers, {CUST.cus.nunique():,} customers)")

# helpers
def prefix(M):
    """Prefix sums with a leading zero row, NaN-aware: mean over rows s..e = (S[e+1]-S[s])/(N[e+1]-N[s])."""
    S = np.vstack([np.zeros((1, M.shape[1])), np.nancumsum(np.nan_to_num(M, nan=0.0), axis=0)])
    N = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(~np.isnan(M), axis=0)])
    return S, N
def wmean(SN, i0, ci, a, b, total=False):
    S, N = SN; s, e = i0 + a, i0 + b + 1
    num = S[e, ci] - S[s, ci]
    if total:
        return num
    den = N[e, ci] - N[s, ci]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)

def nbr_stats(e, i0, nbr, prefix_name, ARp, has_ev, scd, scd_car):
    """e: events (with row id 'r'); nbr: DataFrame [r, col, w]. Returns weighted neighbour features per event row."""
    if nbr.empty:
        return pd.DataFrame(index=e.index)
    ii = i0[nbr.r.values]; cc = nbr.col.values; w = nbr.w.values.astype(float)
    pre = wmean(ARp, ii, cc, -20, -1, total=True)
    car = wmean(ARp, ii, cc, 0, 1, total=True)
    evt = (has_ev[ii, cc] | has_ev[ii + 1, cc]).astype(float)
    d = pd.DataFrame({"r": nbr.r.values, "w": w, "pre": pre * w, "car": car * w, "evt": evt * w})
    if scd is not None:
        Ss, Sc = scd
        d["sc"] = Ss[ii, cc] - Ss[ii - 21, cc]
        d["scar"] = Sc[ii, cc] - Sc[ii - 21, cc]
    g = d.groupby("r").sum()
    out = pd.DataFrame(index=g.index)
    out[f"{prefix_name}_n"] = np.log1p(d.groupby("r").size())
    out[f"{prefix_name}_pre20"] = g.pre / g.w
    out[f"{prefix_name}_car_01"] = g.car / g.w
    out[f"{prefix_name}_event"] = g.evt / g.w
    if scd is not None:
        out[f"{prefix_name}_scandal_30"] = g.sc
        out[f"{prefix_name}_scandal_car"] = np.where(g.sc > 0, g.scar / g.sc.clip(lower=1), 0.0)
    return out

# main loop, one calendar year at a time
out = []
for Y in years:
    t = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "mkt", "prc", "vol", "shrout", "bid", "ask"],
                      filters=[("date", ">=", pd.Timestamp(f"{Y - 1}-01-01")), ("date", "<", pd.Timestamp(f"{Y + 1}-01-15"))]).to_pandas()
    t["permno"] = t.permno.astype(np.int64)
    t["ar"] = t.ret - t.mkt
    t["prc"] = t.prc.abs()
    t["turn"] = np.where(t.shrout > 0, t.vol / t.shrout, np.nan)
    t["illiq"] = np.where((t.vol > 0) & (t.prc > 0), t.ret.abs() / (t.prc * t.vol) * 1e6, np.nan)
    mid = (t.ask + t.bid) / 2
    t["spr"] = np.where((t.bid > 0) & (t.ask > t.bid), (t.ask - t.bid) / mid, np.nan)
    t["zero"] = np.where(t.ret.notna(), (t.ret == 0).astype(float), np.nan)
    t = t.drop_duplicates(["date", "permno"]).set_index(["date", "permno"]).sort_index()
    piv = lambda v: t[v].unstack()
    AR = piv("ar"); cols = AR.columns.values; dates = AR.index.values
    get = lambda v: piv(v).reindex(index=AR.index, columns=cols).values
    A = AR.values; TURN, ILL, SPR, ZERO, VOL, SH, PRC = (get(v) for v in ["turn", "illiq", "spr", "zero", "vol", "shrout", "prc"])
    pos = pd.Series(np.arange(len(cols)), index=cols)
    del t

    # splits: share count jumps and the price moves by roughly the inverse factor
    r_sh = SH[1:] / SH[:-1]; r_p = PRC[1:] / PRC[:-1]
    consistent = np.abs(np.log(r_sh) + np.log(r_p)) < 0.25
    fwd = np.vstack([np.zeros((1, A.shape[1]), bool), (r_sh >= 1.2) & consistent]).astype(float)
    rev = np.vstack([np.zeros((1, A.shape[1]), bool), (r_sh <= 0.8) & consistent]).astype(float)

    # events in this window as a day x stock grid, and the scandal grid with its day 0-1 return
    evw = ev[(ev.day0 >= pd.Timestamp(dates[0])) & (ev.day0 <= pd.Timestamp(dates[-1]))]
    di = np.searchsorted(dates, evw.day0.values); ci = evw.permno.map(pos)
    ok = ci.notna().values & (di < len(dates)); di, ci, evw = di[ok], ci[ok].astype(int).values, evw[ok]
    has_ev = np.zeros(A.shape, bool); has_ev[di, ci] = True
    sc_grid = np.zeros(A.shape); sc_car = np.zeros(A.shape)
    s = evw.scandal.values; sc_grid[di[s], ci[s]] = 1.0; sc_car[di[s], ci[s]] = evw.car_0_1.values[s]
    SCD = (np.vstack([np.zeros((1, A.shape[1])), np.cumsum(sc_grid, 0)]), np.vstack([np.zeros((1, A.shape[1])), np.cumsum(sc_car, 0)]))

    e = ev[ev.day0.dt.year == Y]
    i0 = np.searchsorted(dates, e.day0.values); ci = e.permno.map(pos)
    ok = ci.notna().values & (i0 + 1 < len(dates)) & (i0 >= 250)
    e = e[ok].reset_index(drop=True); i0 = i0[ok]; ci = ci[ok].astype(int).values
    e["r"] = np.arange(len(e))
    F = pd.DataFrame({"permno": e.permno.values, "day0": e.day0.values})

    # 1. liquidity and trading activity
    TP, IP, SP, ZP, VP = (prefix(M) for M in [TURN, ILL, SPR, ZERO, VOL])
    lg = lambda x: np.log(np.where(x > 0, x, np.nan))
    F["turn_chg"] = lg(wmean(TP, i0, ci, -20, -1)) - lg(wmean(TP, i0, ci, -250, -21))
    F["amihud_chg"] = lg(wmean(IP, i0, ci, -20, -1)) - lg(wmean(IP, i0, ci, -250, -21))
    F["spread_chg"] = wmean(SP, i0, ci, -20, -1) - wmean(SP, i0, ci, -250, -21)
    F["zero_ret_60"] = wmean(ZP, i0, ci, -60, -1)
    F["pre_abvol_5"] = lg(wmean(VP, i0, ci, -5, -1)) - lg(wmean(VP, i0, ci, -250, -21))
    # 2. splits
    FS, RS = (np.vstack([np.zeros((1, A.shape[1])), np.cumsum(M, 0)]) for M in (fwd, rev))
    F["split_250"] = (FS[i0 + 2, ci] - FS[i0 - 250, ci] > 0).astype(float)
    F["rsplit_250"] = (RS[i0 + 2, ci] - RS[i0 - 250, ci] > 0).astype(float)
    F["split_now"] = ((FS[i0 + 2, ci] - FS[i0 - 5, ci]) + (RS[i0 + 2, ci] - RS[i0 - 5, ci]) > 0).astype(float)

    ARp = prefix(A)
    # 3. scandal contagion through correlation peers (graph from year Y-1 only)
    est = AR[AR.index.year == Y - 1]; est = est.loc[:, est.notna().sum() >= 200]
    if est.shape[1] >= 50:
        Z = ((est - est.mean()) / est.std()).fillna(0.0).values.astype(np.float32)
        C = (Z.T @ Z) / (Z.shape[0] - 1); np.fill_diagonal(C, -np.inf)
        nb = np.argpartition(-C, K, axis=1)[:, :K]
        epos = pd.Series(np.arange(est.shape[1]), index=est.columns.values)
        ei = e.permno.map(epos); m = ei.notna().values
        rows = np.repeat(e.r.values[m], K)
        peer_perm = est.columns.values[nb[ei[m].astype(int).values].ravel()]
        nbr = pd.DataFrame({"r": rows, "col": pos[peer_perm].values, "w": 1.0})
        ps = nbr_stats(e, i0, nbr, "peer", ARp, has_ev, SCD, None)
        F = F.join(ps[["peer_scandal_30", "peer_scandal_car"]], how="left") if "peer_scandal_30" in ps else F
        del Z, C, nb

    # 4. news co-mention partners (previous year's headlines only)
    E = EDGES[EDGES.year == Y - 1]
    if len(E):
        E = E.sort_values("w", ascending=False).groupby("a").head(K)
        E = E[E.b.isin(pos.index)]
        nbr = e[["r", "permno"]].merge(E, left_on="permno", right_on="a")
        nbr = pd.DataFrame({"r": nbr.r.values, "col": pos[nbr.b.values].values, "w": nbr.w.values})
        cm = nbr_stats(e, i0, nbr, "cm", ARp, has_ev, SCD, None).drop(columns=["cm_scandal_car"], errors="ignore")
        F = F.join(cm, how="left")
        # conflict co-mentions in calendar days -30..0 (only while the headline data runs)
        d0 = e.day0.values.astype("datetime64[D]").astype(np.int64)
        F["cm_conflict"] = [np.searchsorted(CONF_DAYS[p], d + 1) - np.searchsorted(CONF_DAYS[p], d - 30) if p in CONF_DAYS.index else 0
                            for p, d in zip(e.permno.values, d0)]
        F.loc[e.day0 > pd.Timestamp("2023-12-31"), "cm_conflict"] = np.nan

    # 5. customers and suppliers
    if CUST is not None and Y >= CUST.avail.min().year:
        for side, me, other, name in [("cust", "sup", "cus", "cust"), ("supp", "cus", "sup", "supp")]:
            L = e[["r", "permno", "day0"]].merge(CUST, left_on="permno", right_on=me)
            L = L[(L.day0 >= L.avail) & (L.day0 < L.end) & L[other].isin(pos.index)]
            L = L.sort_values("salecs", ascending=False).drop_duplicates(["r", other])
            if len(L):
                nbr = pd.DataFrame({"r": L.r.values, "col": pos[L[other].values].values, "w": L.salecs.values})
                F = F.join(nbr_stats(e, i0, nbr, name, ARp, has_ev, None, None), how="left")

    out.append(F)
    print(f"  {Y}: {len(F):,} events" + "".join(f"  {k} {F[k].notna().mean():.0%}" for k in ["peer_scandal_30", "cm_n", "cust_n"] if k in F), flush=True)
    del A, TURN, ILL, SPR, ZERO, VOL, SH, PRC, AR

F = pd.concat(out, ignore_index=True).drop_duplicates(["permno", "day0"])
for k in ["cm_n", "cust_n", "supp_n"]:                     # a firm with no partners in a covered period gets 0, not missing
    if k in F:
        covered = (F.day0.dt.year.between(2010, 2023)) if k == "cm_n" else (F.day0 >= (CUST.avail.min() if CUST is not None else pd.Timestamp("2099-01-01")))
        F.loc[covered & F[k].isna(), k] = 0.0
print("\n=== Spillover, liquidity and split features (EDA) ===")
print(F.drop(columns=["permno", "day0"]).describe().T[["count", "mean", "50%", "min", "max"]].round(4).to_string())
print(f"\nforward splits flagged: {F.split_250.mean():.2%} of events; reverse splits: {F.rsplit_250.mean():.2%}")
if not TEST:
    F.to_parquet(PROC / "spillover_features.parquet", index=False)
    print(f"saved {len(F):,} rows -> data/processed/spillover_features.parquet")
