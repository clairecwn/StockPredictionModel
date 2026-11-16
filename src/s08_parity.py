"""Step 8: parity check. Rebuilds features for 1,500 historical events with the LIVE code (live_engine.event_features),
feeding it CRSP prices instead of Yahoo, and compares them with the values the models were trained on.
If a feature's rank correlation is low, the live model would be fed something different from what it learned.
Expected differences: training uses a Fama-French factor model for abnormal returns, live uses a market model,
so car_0_1 / ar_day0_sd / idio_vol_pre will match closely but not exactly.
Run: python src/s08_parity.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from config import PROC, RESULTS, find_file
from live_engine import event_features

ev = pd.read_parquet(PROC / "events.parquet")
ex = pd.read_parquet(PROC / "extra_features.parquet")
ev = ev.merge(ex, on=["permno", "day0"], how="left")
import os
ev = ev[(ev.day0.dt.year >= 2012) & ev.drift_1_20.notna()].sample(150 if os.environ.get("PARITY_TEST") == "1" else 1500, random_state=0)
perm = [int(p) for p in ev.permno.unique()]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "mkt", "vol"], filters=[("permno", "in", perm)]).to_pandas()
chunks = []
for ch in pd.read_csv(find_file("crsp_ohlc"), usecols=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose"], chunksize=2_000_000):
    chunks.append(ch[ch.PERMNO.isin(perm)])
O = pd.concat(chunks).rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyOpen": "o", "DlyHigh": "h", "DlyLow": "l", "DlyClose": "c"})
O["date"] = pd.to_datetime(O.date)
for k in "ohlc": O[k] = pd.to_numeric(O[k], errors="coerce").abs()
F = D.merge(O, on=["permno", "date"], how="left").sort_values(["permno", "date"])
frames = {p: g.set_index("date") for p, g in F.groupby("permno")}
COMPARE = ["car_0_1", "ar_day0_sd", "abn_volume_0_1", "beta", "idio_vol_pre", "vol_pre", "mom_20", "mom_60", "mom_250",
           "dist_52w_high", "amihud", "log_dollar_vol", "mkt_day0", "mkt_pre5", "mkt_vol20", "yz_vol_5", "yz_vol_20", "yz_vol_60",
           "yz_term", "range_shock_01"]
rows = []
for _, e in ev.iterrows():
    f = frames.get(e.permno)
    if f is None: continue
    i0 = f.index.searchsorted(e.day0)
    if i0 >= len(f) - 1 or f.index[i0] != e.day0: continue
    try:
        x = event_features(f, i0)
    except Exception:
        continue
    rows.append({**{f"live_{c}": x.get(c) for c in COMPARE}, **{f"train_{c}": e.get(c) for c in COMPARE}})
R = pd.DataFrame(rows)
res = []
for c in COMPARE:
    a, b = R[f"live_{c}"].astype(float), R[f"train_{c}"].astype(float); m = a.notna() & b.notna()
    res.append({"feature": c, "n": int(m.sum()), "rank_corr": spearmanr(a[m], b[m])[0] if m.sum() > 20 else np.nan,
                "median_abs_diff": float((a[m] - b[m]).abs().median()), "median_train_value": float(b[m].median())})
T = pd.DataFrame(res).round(4)
txt = f"# Parity check: live feature builder vs training values ({len(R):,} events)\n\n" + T.to_string(index=False) + \
      "\n\nrank_corr above ~0.95 = the live model sees what it was trained on. Lower values are listed in the README as limitations."
print(txt); (RESULTS / "parity.md").write_text(txt)
