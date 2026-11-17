"""Step 10 (experiment 9): price-pattern features from CRSP daily data, all rebuildable live from Yahoo OHLC.

  intra_01, ovn_01   log return of days 0-1 earned during trading hours (open to close) and overnight (close to next
                     open, including the opening gap). Institutions trade mostly during the day and retail investors
                     at the open, and the two parts of a return behave differently afterwards (Lou, Polk and Skouras 2019).
  ovn_share          overnight part / (|overnight| + |intraday|), from -1 to 1
  max_20, min_20     largest and smallest daily return in days -20..-1. Lottery-like stocks with a high MAX tend to
                     do poorly afterwards (Bali, Cakici and Whitelaw 2011).
  skew_60            skewness of daily returns in days -60..-1
  rev_5              log return in days -5..-1 (short-term reversal, Jegadeesh 1990)
  seas_5y            average return in the same calendar month over the previous five years (at least two),
                     the return seasonality of Heston and Sadka (2008)
Output: data/processed/price_patterns.parquet          Run: python src/pipeline/10_price_pattern_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from config import PROC, find_file

ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"]).drop_duplicates().reset_index(drop=True)
ev["permno"] = ev.permno.astype(np.int64)
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"]).to_pandas()
D["permno"] = D.permno.astype(np.int64); D["ret"] = D.ret.astype("float32")
dn = lambda s: (pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64))
D["key"] = D.permno.values * 100000 + dn(D.date)
D = D.sort_values("key").reset_index(drop=True)
k0 = ev.permno.values * 100000 + dn(ev.day0)
g0 = np.searchsorted(D.key.values, k0)
okk = (g0 < len(D)) & (D.key.values[np.minimum(g0, len(D) - 1)] == k0)
print(f"events: {len(ev):,}, day 0 found in CRSP for {okk.mean():.1%}")
P, R = D.permno.values, D.ret.values.astype(float)
def win(a, b):                                                   # returns in days a..b around day 0 (NaN off the firm)
    idx = (g0[:, None] + np.arange(a, b + 1)[None, :]).clip(0, len(D) - 1)
    return np.where((P[idx] == ev.permno.values[:, None]) & okk[:, None], R[idx], np.nan)
pre60, pre20, pre5, d01 = win(-60, -1), win(-20, -1), win(-5, -1), win(0, 1)
F = pd.DataFrame({"permno": ev.permno, "day0": ev.day0})
F["max_20"], F["min_20"] = np.nanmax(pre20, 1), np.nanmin(pre20, 1)
m, sd = np.nanmean(pre60, 1, keepdims=True), np.nanstd(pre60, 1, keepdims=True)
F["skew_60"] = np.nanmean(((pre60 - m) / np.where(sd > 0, sd, np.nan)) ** 3, 1)
F["rev_5"] = np.nansum(np.log1p(np.clip(pre5, -0.99, None)), 1)

# overnight vs intraday on days 0 and 1 (CRSP open and close are unadjusted, but both are on the same day)
need = pd.DataFrame({"key": np.concatenate([D.key.values[np.minimum(g0, len(D) - 1)], D.key.values[np.minimum(g0 + 1, len(D) - 1)]])})
need = set(need.key.values)
parts = []
for ch in pd.read_csv(find_file("crsp_ohlc"), usecols=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose"], chunksize=2_000_000):
    ch["key"] = ch.PERMNO.astype(np.int64).values * 100000 + dn(ch.DlyCalDt)
    parts.append(ch.loc[ch.key.isin(need), ["key", "DlyOpen", "DlyClose"]])
O = pd.concat(parts).drop_duplicates("key").set_index("key")
intra = np.full((len(ev), 2), np.nan)
for j in (0, 1):
    kk = D.key.values[np.minimum(g0 + j, len(D) - 1)]
    o = O.reindex(kk)
    r = (o.DlyClose.abs() / o.DlyOpen.abs()).values
    intra[:, j] = np.where(np.isfinite(r) & (r > 0), np.log(r), np.nan)
tot = np.log1p(np.clip(d01, -0.99, None))
F["intra_01"] = np.nansum(intra, 1); F.loc[np.isnan(intra).all(1), "intra_01"] = np.nan
F["ovn_01"] = np.nansum(tot, 1) - F.intra_01
F["ovn_share"] = F.ovn_01 / (F.ovn_01.abs() + F.intra_01.abs()).replace(0, np.nan)
for c in ("intra_01", "ovn_01"):
    F[c] = F[c].clip(-1, 1)

# return seasonality: same calendar month, previous five years
D["ym"] = pd.to_datetime(D.date).dt.to_period("M")
M = D.assign(l=np.log1p(np.clip(D.ret.astype(float), -0.99, None))).groupby(["permno", "ym"]).l.sum().reset_index()
M["y"], M["m"] = M.ym.dt.year, M.ym.dt.month
Mi = M.set_index(["permno", "y", "m"]).l
ey, em = ev.day0.dt.year.values, ev.day0.dt.month.values
lags = np.column_stack([Mi.reindex(pd.MultiIndex.from_arrays([ev.permno.values, ey - L, em])).values for L in range(1, 6)])
cnt = np.isfinite(lags).sum(1)
F["seas_5y"] = np.where(cnt >= 2, np.nanmean(lags, 1), np.nan)
print(F.drop(columns=["permno", "day0"]).describe().T[["count", "mean", "50%", "min", "max"]].round(4).to_string())
F.to_parquet(PROC / "price_patterns.parquet", index=False)
print(f"saved {len(F):,} rows -> data/processed/price_patterns.parquet")
