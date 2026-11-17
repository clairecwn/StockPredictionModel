"""Step 2: build the event table.
For every company event (Capital IQ Key Developments, or CEO departures from ExecuComp as fallback):
  1. find its first trading day (day 0) in CRSP
  2. fit a market model on days -250..-30 (each stock's normal relation to the market)
  3. compute abnormal returns, volume and volatility after the event   -> TARGETS
  4. compute what was known by the close of day 1                       -> FEATURES
Output: data/processed/events.parquet (one row per stock per event day)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import PROC, EST_WINDOW, FIRST_EVENT_YEAR
from wrds_io import load_crsp_daily, load_market, load_keydev, load_execucomp, load_ccm
from taxonomy import classify_keydev, is_scandal, CLASSES, PRIORITY
from events_wrds import build_raw_events

# 1. raw events with gvkey + announcement date + class ----------
EXTRA = ["ceo_involved", "exec_departure", "forced_exit", "gov_involved", "log_claim", "activist_stake", "restate_adverse", "rec_net",
         "eps_sue", "eps_surp_rel", "eps_beat", "est_dispersion", "n_analysts"]
# when an earnings report and analyst rating changes land on the same day, the report is the event (the ratings react to it)
PRIORITY = dict(PRIORITY); PRIORITY["earnings_guidance"] = PRIORITY["analyst"] - 0.5
kd = load_keydev()
if kd is not None:                                   # Capital IQ Key Developments (if your institution has it)
    print(f"Capital IQ key developments: {len(kd):,}")
    kd["cls"] = [classify_keydev(e, h) for e, h in zip(kd.eventtype, kd.get("headline", pd.Series("", index=kd.index)))]
    kd = kd[kd.cls.notna()]
    kd["scandal_word"] = kd.get("headline", pd.Series("", index=kd.index)).map(is_scandal)
    raw = kd[["gvkey", "ann_date", "cls", "scandal_word"]].copy()
else:                                                # Audit Analytics + I/B/E/S (NTU subscription)
    print("Building events from Audit Analytics + I/B/E/S:")
    raw, CUS_MAP = build_raw_events(return_maps=True)
for c in EXTRA:
    if c not in raw:
        raw[c] = np.nan
print(raw.cls.value_counts().to_string())
raw = raw[raw.ann_date.dt.year >= FIRST_EVENT_YEAR]

# 2. gvkey -> permno (valid on the event date) ----------
ccm = load_ccm()
raw = raw.merge(ccm[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
raw = raw[(raw.ann_date >= raw.linkdt) & (raw.ann_date <= raw.linkenddt)].drop(columns=["linkdt", "linkenddt"])
print(f"events linked to a CRSP stock: {len(raw):,}")

# 3. CRSP + market ----------
crsp = load_crsp_daily()
if "mkt" not in crsp or crsp.mkt.isna().all():      # market return: CRSP value-weighted index if downloaded, else Fama-French
    crsp = crsp.drop(columns=["mkt"], errors="ignore").merge(load_market(), on="date", how="inner")
crsp = crsp.dropna(subset=["mkt"])
# Fama-French 5 factors + momentum (daily, decimals). Used for the factor-model abnormal return (v2 benchmark).
from config import find_file
ff_f = find_file("ff5_umd_daily")
FACT = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
if ff_f is not None:
    ff = pd.read_csv(ff_f); ff.columns = [c.lower() for c in ff.columns]
    ff["date"] = pd.to_datetime(ff.date.astype(str))
    if ff.mktrf.abs().max() > 1: ff[FACT + ["rf"]] /= 100      # percent -> decimal
    crsp = crsp.merge(ff[["date", "rf"] + FACT], on="date", how="left")
    print(f"Fama-French 5 + momentum factors merged ({ff.date.min().date()} to {ff.date.max().date()})")
else:
    print("WARNING: ff5_umd_daily not found; falling back to market-only abnormal returns")
HAVE_FF = ff_f is not None
last_day = crsp.date.max()
print(f"CRSP+market rows: {len(crsp):,}, last day {last_day.date()}")

# 4. governance features from ExecuComp (CEO in office at the time) ----------
ex = load_execucomp().sort_values("year").drop_duplicates(["gvkey", "year"], keep="last")
# ExecuComp fiscal-year data is only public when the proxy statement is filed, ~4-6 months after the fiscal year ends.
# Stamping year Y data on Jan 1 of Y leaked up to 18 months of future information (e.g. a mid-year CEO change showed up
# as a short tenure before it happened). Now: year Y data becomes usable on Jul 1 of Y+1.
ex["asof"] = pd.to_datetime((ex.year.astype(int) + 1).astype(str) + "-07-01")
gov = ex[["gvkey", "asof", "becameceo", "age", "shrown_excl_opts_pct"]].sort_values("asof")

# 5. event study per stock ----------
def window_stats(a, i0, n):
    """a = dict of numpy arrays for one stock; i0 = index of day 0."""
    lo, hi = i0 + EST_WINDOW[0], i0 + EST_WINDOW[1]
    if lo < 0 or i0 + 1 >= n:
        return None
    r, m = a["ret"][lo:hi], a["mkt"][lo:hi]
    ok = np.isfinite(r) & np.isfinite(m)
    if ok.sum() < 120:
        return None
    beta, alpha = np.polyfit(m[ok], r[ok], 1)
    sd = np.std(r[ok] - alpha - beta * m[ok])
    # Abnormal return = return - beta*market, with NO alpha term. Subtracting the estimation-window alpha would
    # mechanically push post-event abnormal returns against past momentum (alpha does not persist), which looks like
    # predictability but is not tradeable. beta-hedged return is what a long/short position actually earns.
    if HAVE_FF:
        # Factor model on excess returns, estimated on days -250..-30:
        #   r - rf = a + b*MKTRF + s*SMB + h*HML + r*RMW + c*CMA + u*UMD + e
        # Loadings are shrunk (Blume-style: market beta toward 1, the others toward 0) because single-stock estimates are noisy.
        # AR = (r - rf) - sum(loading * factor). No intercept, for the same reason as before (alpha does not persist).
        F = np.column_stack([a[f][lo:hi] for f in FACT]); ex_r = r - a["rf"][lo:hi]
        okf = ok & np.isfinite(F).all(1) & np.isfinite(ex_r)
        if okf.sum() >= 120:
            Xf = np.column_stack([np.ones(okf.sum()), F[okf]])
            coef = np.linalg.lstsq(Xf, ex_r[okf], rcond=None)[0][1:]
            coef = 0.67 * coef + np.array([0.33, 0, 0, 0, 0, 0])
            Fall = np.column_stack([a[f] for f in FACT])
            ar = np.nan_to_num((a["ret"] - a["rf"]) - Fall @ coef)
            beta_ff = coef
        else:
            ar = np.nan_to_num(a["ret"] - beta * a["mkt"]); beta_ff = np.full(6, np.nan)
    else:
        ar = np.nan_to_num(a["ret"] - beta * a["mkt"]); beta_ff = np.full(6, np.nan)
    vol_est = np.nanmean(a["vol"][lo:hi]) or np.nan
    out = {}
    # --- features: known at close of day 1
    out["car_0_1"] = ar[i0] + ar[i0 + 1]
    out["ar_day0_sd"] = ar[i0] / sd if sd > 0 else np.nan
    out["abn_volume_0_1"] = np.nanmean(a["vol"][i0:i0 + 2]) / vol_est
    out["beta"] = beta
    for nm, v in zip(("ld_mkt", "ld_smb", "ld_hml", "ld_rmw", "ld_cma", "ld_umd"), beta_ff):
        out[nm] = v                                                    # factor loadings: what kind of stock this is
    out["prc_pre"] = a["prc"][i0 - 1]                                  # for the $5 price filter
    out["alpha_pre"] = alpha * 252                                     # past annualised alpha, a known feature
    out["idio_vol_pre"] = sd * np.sqrt(252)
    out["vol_pre"] = np.nanstd(a["ret"][i0 - 60:i0]) * np.sqrt(252)
    for k in (20, 60, 250):
        out[f"mom_{k}"] = np.nansum(a["ret"][max(0, i0 - k):i0])
    p = a["prc"]
    out["dist_52w_high"] = p[i0 - 1] / np.nanmax(p[max(0, i0 - 252):i0]) - 1
    dv = a["prc"][i0 - 60:i0] * a["vol"][i0 - 60:i0]
    out["amihud"] = np.nanmean(np.abs(a["ret"][i0 - 60:i0]) / np.where(dv > 0, dv, np.nan)) * 1e6
    out["log_dollar_vol"] = np.log1p(np.nanmean(dv))
    out["log_mcap"] = np.log1p(a["cap"][i0 - 1]) if np.isfinite(a["cap"][i0 - 1]) else np.nan
    if "bid" in a:
        out["spread"] = np.nanmean((a["ask"][i0 - 60:i0] - a["bid"][i0 - 60:i0]) / p[i0 - 60:i0])
    out["mkt_day0"] = a["mkt"][i0]
    out["mkt_pre5"] = np.nansum(a["mkt"][i0 - 5:i0])
    out["mkt_vol20"] = np.nanstd(a["mkt"][i0 - 20:i0]) * np.sqrt(252)
    # --- targets: what happens from the close of day 1 onward
    for h in (5, 20, 60):
        out[f"drift_1_{h}"] = ar[i0 + 2:i0 + h + 1].sum() if i0 + h < n else np.nan
    if i0 + 20 < n:
        path = np.cumsum(ar[i0 + 2:i0 + 21])
        out["trough_2_20"] = path.min()
        out["days_to_trough"] = int(path.argmin()) + 2
        for k in range(2, 21):                                   # daily abnormal-return path, used by the timing layer
            out[f"ar_d{k}"] = ar[i0 + k]
        out["vol_ratio_2_20"] = np.nanstd(a["ret"][i0 + 2:i0 + 21]) * np.sqrt(252) / out["vol_pre"] if out["vol_pre"] > 0 else np.nan
        out["abn_volume_2_20"] = np.nanmean(a["vol"][i0 + 2:i0 + 21]) / vol_est
    return out

cols = [c for c in ("ret", "mkt", "vol", "prc", "cap", "bid", "ask", "rf", "mktrf", "smb", "hml", "rmw", "cma", "umd") if c in crsp]
rows = []
raw = raw.sort_values(["permno", "ann_date"])
ev_by_permno = dict(tuple(raw.groupby("permno")))
for k, (permno, g) in enumerate(crsp.groupby("permno", sort=False)):
    evs = ev_by_permno.get(permno)
    if evs is None:
        continue
    dates = g.date.values
    a = {c: g[c].values.astype("float64") for c in cols}
    n = len(dates)
    evs = evs.assign(i0=np.searchsorted(dates, evs.ann_date.values))
    evs = evs[evs.i0 < n]
    # news intensity: number of events for this stock in the prior 30 calendar days
    t = evs.ann_date.values.astype("datetime64[D]").astype(np.int64)
    evs = evs.assign(news_30d=np.searchsorted(t, t, "left") - np.searchsorted(t, t - 30, "left"))
    for i0, grp in evs.groupby("i0"):                     # several events on one trading day -> one row
        s = window_stats(a, int(i0), n)
        if s is None:
            continue
        clss = set(grp.cls)
        s.update(permno=permno, gvkey=grp.gvkey.iloc[0], day0=pd.Timestamp(dates[i0]), ann_date=grp.ann_date.min(),
                 cls=min(clss, key=lambda c: PRIORITY.get(c, 99)), n_events=len(grp),
                 scandal_word=int(grp.scandal_word.max()), news_30d=int(grp.news_30d.max()))
        for c in CLASSES:
            s[f"has_{c}"] = int(c in clss)
        for c in EXTRA:
            v = pd.to_numeric(grp[c], errors="coerce")
            s[c] = v.sum() if c == "rec_net" else v.max()
        s["delisted_60d"] = int(i0 + 60 >= n and dates[-1] < np.datetime64(last_day) - np.timedelta64(5, "D"))
        rows.append(s)
    if k % 500 == 0:
        print(f"  {k:,} stocks processed, {len(rows):,} events", flush=True)

ev = pd.DataFrame(rows).sort_values("ann_date")
ev = pd.merge_asof(ev, gov, left_on="ann_date", right_on="asof", by="gvkey", direction="backward")
ev["ceo_tenure"] = (ev.ann_date - ev.becameceo).dt.days / 365.25
ev["ceo_age"], ev["ceo_own_pct"] = ev.age, ev.shrown_excl_opts_pct
ev["long_tenure_ceo"] = (ev.ceo_tenure >= 15).astype(int)
ev = ev.drop(columns=["asof", "becameceo", "age", "shrown_excl_opts_pct"])
# analyst coverage known before the event (latest monthly I/B/E/S count, at most 120 days old), for every event class
from events_wrds import coverage
cov = coverage(CUS_MAP) if "CUS_MAP" in globals() else None
if cov is not None:
    cov = cov.sort_values("cov_date"); cov["gvkey"] = cov.gvkey.astype(float)
    ev["gvkey"] = ev.gvkey.astype(float); ev = ev.sort_values("ann_date")
    ev = pd.merge_asof(ev, cov, left_on="ann_date", right_on="cov_date", by="gvkey", direction="backward",
                       allow_exact_matches=False, tolerance=pd.Timedelta(days=120)).drop(columns="cov_date")
    ev["analyst_cov"] = ev.analyst_cov.fillna(0)
    print(f"analyst coverage attached; share of events with >=1 analyst: {(ev.analyst_cov > 0).mean():.1%}")
ev = ev.reset_index(drop=True)
ev.to_parquet(PROC / "events.parquet", index=False)
print(f"events where the stock stopped trading within 60 days (delisting / merger): {ev.delisted_60d.sum():,}")
print(f"\nsaved {len(ev):,} events ({ev.day0.min().date()} to {ev.day0.max().date()}) -> data/processed/events.parquet")
print("\nMean abnormal return by event class (car_0_1 = days 0-1 reaction; drift_1_k = days 2..k after acting at close of day 1):")
print(ev.groupby("cls")[["car_0_1", "drift_1_5", "drift_1_20", "drift_1_60"]].agg(["mean", "count"]).round(4).to_string())
