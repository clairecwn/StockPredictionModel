"""Sizing check by volatility bucket (2021-2026): does each tier's downside bound hold for high-volatility stocks
like the watchlist (TSLA, NVDA, INTC), not just on average? Uses the live sizing models exactly as the board does."""
import os, warnings; warnings.filterwarnings("ignore")
os.environ["LIVE_FEATS"] = "1"
import numpy as np, pandas as pd, joblib, pyarrow.parquet as pq
from config import PROC, MODELS, RESULTS
from modeling import ev
sz = joblib.load(MODELS / "sizing.joblib")
E = ev[ev.day0 >= "2021-01-01"].copy()
perm = [int(p) for p in E.permno.unique()]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"],
                  filters=[("permno", "in", perm), ("date", ">=", pd.Timestamp("2020-12-01"))]).to_pandas()
D["key"] = D.permno.astype(np.int64) * 100000 + (D.date.values.astype("datetime64[D]").astype(np.int64) - 7300)
D = D.sort_values("key").reset_index(drop=True)
k0 = E.permno.astype(np.int64).values * 100000 + (E.day0.values.astype("datetime64[D]").astype(np.int64) - 7300)
g0 = np.searchsorted(D.key.values, k0)
idx = (g0[:, None] + np.arange(2, 21)[None, :]).clip(0, len(D) - 1)
same = D.permno.values[idx] == E.permno.values[:, None]
R = np.where(same & np.isfinite(D.ret.values[idx]), D.ret.values[idx], 0.0)
path = np.cumprod(1 + np.clip(R, -0.95, 3), axis=1) - 1
ok = same[:, 0] & (D.key.values[g0.clip(0, len(D) - 1)] == k0)
E = E[ok].copy(); E["trough"] = np.minimum(0, path[ok].min(axis=1))
E = E[E.day0 <= E.day0.max() - pd.Timedelta(days=35)]
E = E[E.day0 >= pd.Timestamp(sz.get("cal_start", "2021-01-01"))]               # the final model never saw these years
X = E[sz["feats"]].astype(float)
E["vol_q"] = pd.qcut(E.idio_vol_pre, 5, labels=["q1 calm", "q2", "q3", "q4", "q5 volatile"])
E["shock"] = np.where(E.car_0_1.abs() > 0.05, "|car01|>5%", "|car01|<=5%")
rows = []
for b in sorted({t["beta"] for t in sz["tiers"].values()}):
    E[f"bound_{b}"] = sz["q_models"][b].predict(X) - sz["aci_offset"][b]
    for grp in ["vol_q", "shock"]:
        for g, d in E.groupby(grp):
            rows.append({"beta": b, "split": grp, "group": g, "n": len(d), "breach": (d.trough < d[f"bound_{b}"]).mean(),
                         "median_bound_pct": 100 * d[f"bound_{b}"].median(), "realised_q_pct": 100 * d.trough.quantile(b)})
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
print(f"events {E.day0.min().date()} to {E.day0.max().date()} (held out from the final model's fit; its offset was set on this window,")
print("so the overall breach is ~beta by construction: the check is whether every volatility bucket stays near beta)")
