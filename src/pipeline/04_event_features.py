"""Step 4: extra event-level features, keyed by (permno, day0). Run after 02_build_events.py.

1. Range-based volatility (Yang & Zhang 2000) from daily open/high/low/close, measured BEFORE the event
   (windows end on day -1), in a HAR layout (5, 20, 60 days; Corsi 2009): yz_vol_5 / yz_vol_20 / yz_vol_60 (annualised).
   Range estimators use the whole day's path, so they are several times more efficient than close-to-close volatility.
2. Event-window range shock: Parkinson range on days 0 and 1 relative to the pre-event yz_vol_20 (log ratio),
   i.e. how violent trading was inside the reaction days, not just where the price closed. Known at the close of day 1.
3. Investor attention (Hirshleifer, Lim & Teoh 2009; DellaVigna & Pollet 2009):
   n_same_day_earn = log(1 + number of quarterly earnings announcements by ALL firms on the same date)
   friday          = announced on a Friday
4. Earnings reaction vs surprise (original to this project; motivated by under/over-reaction theory):
   react_per_sue   = day 0-1 reaction in noise units divided by the size of the surprise in analyst-disagreement units
   sue_react_clash = 1 if the price moved against the surprise (beat but fell, or missed but rose)

Output: data/processed/extra_features.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import PROC, find_file

ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0", "ann_date", "car_0_1", "idio_vol_pre", "eps_sue"])
ev["permno"] = ev.permno.astype(np.int64)
import os
if os.environ.get("EXTRA_TEST") == "1":                      # quick check on 200 firms; output not saved
    ev = ev[ev.permno.isin(sorted(ev.permno.unique())[:200])].reset_index(drop=True)
permnos = set(ev.permno.unique())
print(f"events: {len(ev):,}")

# 1-2. Yang-Zhang / Parkinson from OHLC
chunks = []
for ch in pd.read_csv(find_file("crsp_ohlc"), usecols=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose"],
                      chunksize=2_000_000):
    chunks.append(ch[ch.PERMNO.isin(permnos)])
d = pd.concat(chunks, ignore_index=True); del chunks
d = d.rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyOpen": "o", "DlyHigh": "h", "DlyLow": "l", "DlyClose": "c"})  # by name: usecols keeps file order
d["date"] = pd.to_datetime(d.date)
for k in ("o", "h", "l", "c"):
    d[k] = pd.to_numeric(d[k], errors="coerce").abs()          # CRSP stores bid/ask midpoints as negative prices
d = d.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"]).reset_index(drop=True)
bad = (d[["o", "h", "l", "c"]] <= 0).any(axis=1) | (d.h < d.l) | d[["o", "h", "l", "c"]].isna().any(axis=1)
d.loc[bad, ["o", "h", "l", "c"]] = np.nan
pc = d.groupby("permno").c.shift(1)
d["ov"] = np.log(d.o / pc)                                      # overnight
d["oc"] = np.log(d.c / d.o)                                     # open to close
d["rs"] = np.log(d.h / d.c) * np.log(d.h / d.o) + np.log(d.l / d.c) * np.log(d.l / d.o)   # Rogers-Satchell
d["pk"] = np.log(d.h / d.l) ** 2 / (4 * np.log(2))              # Parkinson daily variance
for x in ("ov", "oc", "rs", "pk"):
    d.loc[d[x].abs() > 1, x] = np.nan                           # bad prints / splits not adjusted in OHLC

g = d.groupby("permno")
def yz(n):
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    mp = max(3, int(0.75 * n))
    v_o = g.ov.transform(lambda s: s.rolling(n, min_periods=mp).var())
    v_c = g.oc.transform(lambda s: s.rolling(n, min_periods=mp).var())
    m_rs = g.rs.transform(lambda s: s.rolling(n, min_periods=mp).mean())
    return np.sqrt((v_o + k * v_c + (1 - k) * m_rs).clip(lower=0) * 252)
for n in (5, 20, 60):
    d[f"yz_vol_{n}"] = yz(n)
    d[f"yz_vol_{n}"] = d.groupby("permno")[f"yz_vol_{n}"].shift(1)       # window ends on day -1 when looked up at day 0
d["pk_next"] = d.groupby("permno").pk.shift(-1)                       # day 1 range, known at the close of day 1
d["range_shock_01"] = np.log(np.sqrt((d.pk + d.pk_next) / 2 * 252).clip(lower=0.01) / d.yz_vol_20.clip(lower=0.02))
d["yz_term"] = np.log(d.yz_vol_5.clip(lower=0.01) / d.yz_vol_60.clip(lower=0.01))   # short vs long vol (HAR slope)
f = ev[["permno", "day0"]].merge(d[["permno", "date", "yz_vol_5", "yz_vol_20", "yz_vol_60", "yz_term", "range_shock_01"]],
                                 left_on=["permno", "day0"], right_on=["permno", "date"], how="left").drop(columns="date")
print(f"range-based volatility matched for {f.yz_vol_20.notna().mean():.1%} of events")
del d, g

# 3. attention
a = pd.read_csv(find_file("ibes_actuals_q"), usecols=lambda c: c.upper() in {"ANNDATS", "PDICITY"})
a.columns = [c.lower() for c in a.columns]
a = a[a.pdicity.eq("QTR")] if "pdicity" in a else a
cnt = pd.to_datetime(a.anndats, errors="coerce").dt.normalize().value_counts()
ad = ev.ann_date.dt.normalize()
f["n_same_day_earn"] = np.log1p(ad.map(cnt).fillna(0).values)
f["friday"] = (ad.dt.weekday == 4).astype(int).values

# 4. reaction vs surprise
car_z = ev.car_0_1 / ((ev.idio_vol_pre / np.sqrt(252)).clip(lower=0.003) * np.sqrt(2))
f["react_per_sue"] = np.where(ev.eps_sue.notna(), (car_z / (ev.eps_sue.abs() + 1) * np.sign(ev.eps_sue).replace(0, 1)).clip(-10, 10), np.nan)
f["sue_react_clash"] = np.where(ev.eps_sue.notna() & (ev.eps_sue.abs() > 0.5),
                                (np.sign(ev.eps_sue) * np.sign(ev.car_0_1) < 0).astype(float), np.nan)

f = f.drop_duplicates(["permno", "day0"])
if os.environ.get("EXTRA_TEST") != "1":
    f.to_parquet(PROC / "extra_features.parquet", index=False)
print("\n=== Extra features (EDA) ===")
print(f.drop(columns=["permno", "day0"]).describe().T[["count", "mean", "50%", "min", "max"]].round(3).to_string())
print(f"\nsaved {len(f):,} rows -> data/processed/extra_features.parquet")
