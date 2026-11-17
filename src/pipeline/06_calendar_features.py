"""Step 6 (experiment track): calendar, firm-memory and analyst-revision features, keyed by (permno, day0).

Every value is known at the close of day 1 at the latest.
A. Earnings calendar (Compustat report dates). The EXPECTED next report date uses past dates only:
   the report four quarters back + 364 days, else the last report + 91 days.
     exp_earn_in_win   1 if the expected report falls inside the holding window (2 to 29 calendar days ahead)
     days_to_exp_earn  calendar days to the expected report (0..120)
     days_since_earn   calendar days since the last report (0..120)
B. Earnings sensitivity: earn_react_hist = mean |car_0_1| over the firm's previous 8 earnings events.
C. Firm memory: averages over the firm's previous 8 events whose 20-day window had already closed (45+ days earlier):
     mem_logvr, mem_logvol, mem_trough_z   (log volatility ratio, log volume ratio, trough in units of expected noise)
     mem_n                                 how many such events exist (capped at 8)
D. Analyst revisions (I/B/E/S monthly consensus for the current fiscal year, last snapshot strictly before day 0):
     rev1_rel, rev3_rel  change in the mean estimate over 1 and 3 months / max(|earlier estimate|, 0.10)
     rev3_z              3-month change / max(dispersion, 0.01)
     dnum3, ddisp3       3-month change in the number of analysts and in dispersion / max(|estimate|, 0.10)
Output: data/processed/calendar_features.parquet        Run: python src/pipeline/06_calendar_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import PROC, find_file

ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "gvkey", "day0", "cls", "car_0_1", "idio_vol_pre",
                                                        "vol_ratio_2_20", "abn_volume_2_20", "trough_2_20"])
ev["day0"] = pd.to_datetime(ev.day0); n0 = len(ev)
out = ev[["permno", "day0"]].copy()
dn = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64)

# ---- A. earnings calendar
R = pd.read_csv(find_file("comp_rdq"), usecols=["gvkey", "rdq"]).dropna()
R["g"] = R.gvkey.astype(np.int64); R["d"] = dn(R.rdq)
R = R.drop_duplicates(["g", "d"]).sort_values(["g", "d"]); key = R.g.values * 100000 + R.d.values
gv = pd.to_numeric(ev.gvkey, errors="coerce").fillna(-1).astype(np.int64).values; d0 = dn(ev.day0)
ip = np.searchsorted(key, gv * 100000 + d0, side="right") - 1                       # last report on or before day 0
ok = (ip >= 0) & (key[ip.clip(0)] // 100000 == gv)
prev = np.where(ok, key[ip.clip(0)] % 100000, np.nan)
i4 = ip - 3; ok4 = ok & (i4 >= 0) & (key[i4.clip(0)] // 100000 == gv)
exp4 = np.where(ok4, key[i4.clip(0)] % 100000 + 364, np.nan)
exp = np.where(np.isfinite(exp4) & (exp4 > d0) & (exp4 - prev > 45) & (exp4 - prev < 150), exp4, prev + 91)
exp = np.where(exp <= d0, d0 + 1, exp)                                              # overdue: could come any day
out["days_to_exp_earn"] = np.clip(exp - d0, 0, 120); out["days_since_earn"] = np.clip(d0 - prev, 0, 120)
out["exp_earn_in_win"] = np.where(np.isfinite(exp), ((exp - d0 >= 2) & (exp - d0 <= 29)).astype(float), np.nan)

# ---- B and C. the firm's own history
ev["_i"] = np.arange(n0); ev = ev.sort_values(["permno", "day0"])
e = ev[ev.cls == "earnings_guidance"].copy()
e["earn_react_hist"] = e.groupby("permno").car_0_1.transform(lambda s: s.abs().shift(1).rolling(8, min_periods=2).mean())
last = e[["permno", "day0", "car_0_1"]].copy(); last["h"] = last.groupby("permno").car_0_1.transform(lambda s: s.abs().rolling(8, min_periods=2).mean())
last = last.rename(columns={"day0": "d_src"}).sort_values("d_src")
q = ev[["_i", "permno", "day0"]].sort_values("day0")
m = pd.merge_asof(q, last[["permno", "d_src", "h"]], left_on="day0", right_on="d_src", by="permno", allow_exact_matches=False)
out["earn_react_hist"] = m.set_index("_i").h.reindex(np.arange(n0)).values

sd = (ev.idio_vol_pre / np.sqrt(252)).clip(lower=0.003)
ev["lvr"] = np.log(ev.vol_ratio_2_20.clip(lower=0.05)); ev["lvo"] = np.log(ev.abn_volume_2_20.clip(lower=0.05))
ev["tz"] = (ev.trough_2_20 / (sd * np.sqrt(19))).clip(-10, 0)
g = ev.groupby("permno")
src = pd.DataFrame({"permno": ev.permno, "d_src": ev.day0 + pd.Timedelta(days=45),                # usable 45 days later
                    "mem_logvr": g.lvr.transform(lambda s: s.rolling(8, min_periods=1).mean()),
                    "mem_logvol": g.lvo.transform(lambda s: s.rolling(8, min_periods=1).mean()),
                    "mem_trough_z": g.tz.transform(lambda s: s.rolling(8, min_periods=1).mean()),
                    "mem_n": g.lvr.transform(lambda s: s.notna().rolling(8, min_periods=1).sum())}).dropna(subset=["mem_logvr"]).sort_values("d_src")
m = pd.merge_asof(q, src, left_on="day0", right_on="d_src", by="permno", allow_exact_matches=True).set_index("_i").reindex(np.arange(n0))
for c in ("mem_logvr", "mem_logvol", "mem_trough_z", "mem_n"):
    out[c] = m[c].values
out["mem_n"] = out.mem_n.fillna(0)

# ---- D. analyst revisions
sf, lf = find_file("ibes_summary"), find_file("ibes_crsp_link")
if sf is not None and lf is not None:
    S = pd.read_csv(sf, dtype={"TICKER": str}, usecols=["TICKER", "STATPERS", "FPI", "NUMEST", "MEANEST", "STDEV", "FPEDATS"])
    S.columns = S.columns.str.lower(); S = S[S.fpi.astype(str) == "1"]
    S["statpers"] = pd.to_datetime(S.statpers, errors="coerce")
    for c in ("numest", "meanest", "stdev"): S[c] = pd.to_numeric(S[c], errors="coerce")
    S = S.dropna(subset=["statpers", "meanest"]).sort_values(["ticker", "fpedats", "statpers"])
    gs = S.groupby(["ticker", "fpedats"])
    den = lambda x: np.maximum(x.abs(), 0.10)
    p1, p3 = gs.meanest.shift(1), gs.meanest.shift(3)
    S["rev1_rel"] = ((S.meanest - p1) / den(p1)).clip(-2, 2); S["rev3_rel"] = ((S.meanest - p3) / den(p3)).clip(-2, 2)
    S["rev3_z"] = ((S.meanest - p3) / np.maximum(S.stdev.fillna(0), 0.01)).clip(-20, 20)
    S["dnum3"] = S.numest - gs.numest.shift(3); S["ddisp3"] = ((S.stdev - gs.stdev.shift(3)) / den(S.meanest)).clip(-2, 2)
    L = pd.read_csv(lf, dtype={"TICKER": str}); L.columns = L.columns.str.lower()
    L["sdate"], L["edate"] = pd.to_datetime(L.sdate, errors="coerce"), pd.to_datetime(L.edate, errors="coerce")
    L = L[pd.to_numeric(L.score, errors="coerce") <= 2].dropna(subset=["permno"])
    j = ev[["_i", "permno", "day0"]].merge(L[["ticker", "permno", "sdate", "edate"]], on="permno")
    j = j[(j.day0 >= j.sdate) & (j.day0 <= j.edate)].drop_duplicates("_i").sort_values("day0")
    cols = ["rev1_rel", "rev3_rel", "rev3_z", "dnum3", "ddisp3"]
    m = pd.merge_asof(j, S[["ticker", "statpers"] + cols].sort_values("statpers"), left_on="day0", right_on="statpers", by="ticker",
                      allow_exact_matches=False, tolerance=pd.Timedelta(days=45)).set_index("_i").reindex(np.arange(n0))
    for c in cols: out[c] = m[c].values

out.to_parquet(PROC / "calendar_features.parquet")
print(f"saved {len(out):,} rows -> data/processed/calendar_features.parquet")
print("coverage (share of events with a value):"); print(out.drop(columns=["permno", "day0"]).notna().mean().round(3).to_string())
print("\nsanity: realised log volatility ratio by 'expected earnings inside the window'")
chk = pd.DataFrame({"in_win": out.exp_earn_in_win.values, "lvr": ev.sort_values("_i").lvr.values}).groupby("in_win").lvr.agg(["mean", "count"]).round(3)
print(chk.to_string())
