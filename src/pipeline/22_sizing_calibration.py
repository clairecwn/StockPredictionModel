"""Step 22: volatility-bucket (Mondrian) calibration of the live sizing bound.

One offset for all stocks left the most volatile fifth slightly over its promise (12% breaches vs 10%) and calm stocks
under it. Mondrian conformal prediction (Vovk et al. 2003) sets a separate offset inside each bucket of pre-event
idiosyncratic volatility, so the promise holds per bucket, not just on average.
  check: offsets set on 2024 only, breach scored on 2025 (honest);
  final: offsets set on the whole held-out window (2024 on), saved into models/live/sizing.joblib.
Needs 20_position_sizing.py first. Run: LIVE_FEATS=1 python src/pipeline/22_sizing_calibration.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("LIVE_FEATS", "1")
import numpy as np, pandas as pd, joblib, pyarrow.parquet as pq
from config import PROC, MODELS, RESULTS
from modeling import ev
NB = 5
sz = joblib.load(MODELS / "sizing.joblib")
E = ev[ev.day0 >= pd.Timestamp(sz["cal_start"])].copy()
perm = [int(p) for p in E.permno.unique()]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"],
                  filters=[("permno", "in", perm), ("date", ">=", pd.Timestamp(sz["cal_start"]) - pd.Timedelta(days=10))]).to_pandas()
D["key"] = D.permno.astype(np.int64) * 100000 + (D.date.values.astype("datetime64[D]").astype(np.int64) - 7300)
D = D.sort_values("key").reset_index(drop=True)
k0 = E.permno.astype(np.int64).values * 100000 + (E.day0.values.astype("datetime64[D]").astype(np.int64) - 7300)
g0 = np.searchsorted(D.key.values, k0)
idx = (g0[:, None] + np.arange(2, 21)[None, :]).clip(0, len(D) - 1)
same = D.permno.values[idx] == E.permno.values[:, None]
R = np.where(same & np.isfinite(D.ret.values[idx]), D.ret.values[idx], 0.0)
path = np.cumprod(1 + np.clip(R, -0.95, 3), axis=1) - 1
ok = same[:, 0] & (D.key.values[g0.clip(0, len(D) - 1)] == k0) & same[:, -1]          # full 20-day path known
E = E[ok].copy(); E["trough"] = np.minimum(0, path[ok].min(axis=1))
X = E[sz["feats"]].astype(float)
edges = np.quantile(E.idio_vol_pre.dropna(), np.linspace(0, 1, NB + 1)[1:-1])
E["b"] = np.searchsorted(edges, E.idio_vol_pre.fillna(E.idio_vol_pre.median()).values)
def offsets(m, beta):
    out = {}
    for b in range(NB):
        e = m.loc[m.b == b, "err"].values; n = len(e)
        out[b] = float(np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - beta)) / n)))
    return out
fit_y, test_y = E.day0.dt.year == E.day0.dt.year.min(), E.day0.dt.year > E.day0.dt.year.min()
mond, rows = {}, []
for beta, m in sz["q_models"].items():
    E["pred"] = m.predict(X); E["err"] = E.pred - E.trough                          # > offset means a breach
    o_chk = offsets(E[fit_y], beta)
    o_one = float(np.quantile(E.err[fit_y], 1 - beta))
    for b in range(NB):
        t = E[test_y & (E.b == b)]
        rows.append({"beta": beta, "vol_bucket": b + 1, "n_test": len(t),
                     "breach_one_offset": (t.err > o_one).mean(), "breach_per_bucket": (t.err > o_chk[b]).mean()})
    mond[beta] = offsets(E, beta)
T = pd.DataFrame(rows).round(3)
print(f"check: offsets set on {E.day0.dt.year.min()}, breaches scored on later events ({int(test_y.sum()):,})")
print(T.to_string(index=False))
T.to_csv(RESULTS / "mondrian_check.csv", index=False)
sz["mondrian_edges"] = [float(x) for x in edges]; sz["mondrian_offset"] = mond
joblib.dump(sz, MODELS / "sizing.joblib")
print("saved per-bucket offsets into", MODELS / "sizing.joblib")
