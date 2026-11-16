"""Experiment 4: a structural bad case from volatility decay, to tighten the downside bound.

1. Decay curve (Omori-type power law). After a shock, a stock's squared abnormal return on day k, in units of its
   normal daily variance, is modelled as
        m(k, s) = 1 + A * s * k^(-p),      s = log(1 + |day-0 move in sd|) + B * range_shock
   A, B, p are fitted by least squares on bucket averages from training years only (re-fitted for each fold).
2. Reflection principle. For a driftless random walk with total variance V over the holding period,
        P(min of the path < -x) = 2 P(end < -x),   so the beta-quantile of the worst point is  -z_{1-beta/2} sqrt(V),
   with V = sum over days 2..20 of [ sd^2 m(k, s) + beta_mkt^2 sigma_m^2 ],  sigma_m = VIX / (100 sqrt(252)).
   This gives each event a structural bad case, struct_q10 = -1.645 sqrt(V) (total return, as in s06).
3. Test. The 10% and 5% quantile models of the worst point (as in s06, live features) are fitted with and without the
   structural features, each with split-conformal calibration on the 2 years before the test block. On 2016-25:
   pinball loss (monthly paired t), breach rate overall and per volatility fifth, and median bound (tighter is better
   only if the breach rate holds). Pass: pinball improves with monthly t > 2 at both levels and the most volatile fifth
   is no further from its promise than before (within half a point).
Run: LIVE_FEATS=1 python src/x04_vol_decay.py      (quick check: EXP_TEST=1)
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.stats import norm
from config import PROC, ROOT, MODELS, LIVE
from modeling import ev, FEATS, PURGE, splits, gbm_quantile, _fit

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
LOG = []
def say(s=""): print(s, flush=True); LOG.append(str(s))

E = ev.copy()
if TEST: E = E[E.permno.isin(np.sort(E.permno.unique())[:400])]
# ---- total-return worst point over days 2..20 (same construction as s06_sizing)
perm = [int(p) for p in np.sort(E.permno.unique())]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"], filters=[("permno", "in", perm)]).to_pandas()
D["key"] = D.permno.astype(np.int64) * 100000 + (D.date.values.astype("datetime64[D]").astype(np.int64) - 7300)
D = D.sort_values("key").reset_index(drop=True)
k0 = E.permno.astype(np.int64).values * 100000 + (E.day0.values.astype("datetime64[D]").astype(np.int64) - 7300)
g0 = np.searchsorted(D.key.values, k0)
idx = (g0[:, None] + np.arange(2, 21)[None, :]).clip(0, len(D) - 1)
same = D.permno.values[idx] == E.permno.values[:, None]
R = np.where(same & np.isfinite(D.ret.values[idx]), D.ret.values[idx], 0.0)
path = np.cumprod(1 + np.clip(R, -0.95, 3), axis=1) - 1
ok = same[:, 0] & (D.key.values[g0.clip(0, len(D) - 1)] == k0)
E = E[ok].copy(); E["y"] = np.minimum(0, path[ok].min(axis=1)); del D, R, path
d0 = E.day0; month = d0.dt.to_period("M")
sd = (E.idio_vol_pre / np.sqrt(252)).clip(lower=0.003).values
AR = E[[f"ar_d{k}" for k in range(2, 21)]].values / sd[:, None]                      # abnormal returns in units of normal sd
KK = np.arange(2, 21)
s0 = np.log1p(E.ar_day0_sd.abs().fillna(0).values); rs = E.range_shock_01.fillna(0).clip(-2, 3).values
say(f"events with a worst point: {len(E):,}" + ("   [TEST MODE]" if TEST else ""))

def fit_decay(mask):
    """Fit A, B, p on bucket averages of squared standardised abnormal returns (training rows only)."""
    sq = np.clip(AR[mask] ** 2, 0, 100)
    s = s0[mask][:, None] + 0 * KK; r = rs[mask][:, None] + 0 * KK
    sb = pd.qcut(s0[mask] + 0.5 * rs[mask], 10, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(sb):
        mb = sb == b
        for j, k in enumerate(KK):
            v = sq[mb, j]; v = v[np.isfinite(v)]
            if len(v) > 50: rows.append((k, s0[mask][mb].mean(), rs[mask][mb].mean(), v.mean(), len(v)))
    T = np.array(rows)
    def res(th):
        A, B, p, c = th
        return (c * (1 + A * np.clip(T[:, 1] + B * T[:, 2], 0, None) * T[:, 0] ** (-p)) - T[:, 3]) * np.sqrt(T[:, 4] / T[:, 4].mean())
    return least_squares(res, x0=[1.0, 0.5, 0.7, 1.0], bounds=([0, -2, 0, 0.3], [50, 5, 3, 3])).x

def struct_feats(th):
    A, B, p, c = th
    m = c * (1 + A * np.clip(s0 + B * rs, 0, None)[:, None] * KK[None, :] ** (-p))        # events x days 2..20
    sig_m = (E.vix.fillna(E.vix.median()).values / 100) / np.sqrt(252)
    beta = E.beta.fillna(1).clip(0, 3).values
    V = (sd[:, None] ** 2 * m).sum(1) + 19 * (beta * sig_m) ** 2
    return pd.DataFrame({"struct_sd": np.sqrt(V), "struct_q10": -norm.ppf(0.95) * np.sqrt(V),
                         "decay_share5": m[:, :4].sum(1) / m.sum(1)}, index=E.index)  # share of variance in days 2-5

LEVELS = (0.05, 0.10)
X0 = E[FEATS].astype(float); y = E.y
pred = {(b, v): pd.Series(np.nan, index=E.index) for b in LEVELS for v in ("base", "struct")}
P = dict(n_estimators=80 if TEST else 500)
for fi, (tr, te) in enumerate(splits(d0, y)):
    th = fit_decay(tr.values & (d0 < d0[te].min() - PURGE).values)
    S = struct_feats(th)
    say(f"  fold {fi + 1}: decay A={th[0]:.2f} B={th[1]:.2f} p={th[2]:.2f} c={th[3]:.2f}  (share of extra variance gone by day 5: "
        f"{1 - (KK[4:] ** -th[2]).sum() / (KK ** -th[2]).sum():.0%})")
    last = d0[tr].max(); cal = tr & (d0 > last - pd.Timedelta(days=730)); fit = tr & (d0 < last - pd.Timedelta(days=730) - PURGE)
    for v, X in (("base", X0), ("struct", pd.concat([X0, S], axis=1))):
        for b in LEVELS:
            m = _fit(gbm_quantile(b, **P), X[fit], y[fit])
            c = np.quantile(m.predict(X[cal]) - y[cal].values, 1 - b)
            pred[(b, v)][te] = m.predict(X[te]) - c
    if fi == 3: TH_LAST = th

late = (d0.dt.year >= 2016) & pred[(0.10, "base")].notna()
vb = pd.qcut(E.idio_vol_pre[late].rank(method="first"), 5, labels=False) + 1
def pinball(yv, q, b): u = yv - q; return np.where(u >= 0, b * u, (b - 1) * u)
rows, rowsb = [], []
for b in LEVELS:
    pb, ps = pinball(y[late], pred[(b, "base")][late], b), pinball(y[late], pred[(b, "struct")][late], b)
    dm = pd.Series(pb - ps, index=y[late].index).groupby(month[late]).mean()
    rows.append({"beta": b, "pinball_base": pb.mean(), "pinball_struct": ps.mean(), "t_month (struct better)": dm.mean() / (dm.std() / np.sqrt(len(dm))),
                 "breach_base": (y[late] < pred[(b, "base")][late]).mean(), "breach_struct": (y[late] < pred[(b, "struct")][late]).mean(),
                 "median_bound_base_%": 100 * pred[(b, "base")][late].median(), "median_bound_struct_%": 100 * pred[(b, "struct")][late].median()})
    for q in range(1, 6):
        mq = vb == q
        rowsb.append({"beta": b, "vol_fifth": q, "breach_base": (y[late][mq] < pred[(b, "base")][late][mq]).mean(),
                      "breach_struct": (y[late][mq] < pred[(b, "struct")][late][mq]).mean()})
T = pd.DataFrame(rows); TB = pd.DataFrame(rowsb)
say("\n2016-25, worst point over days 2-20 (total return). Lower pinball = better quantile forecast.")
say(T.round(4).to_string(index=False))
say("\nbreach rate by volatility fifth (5 = most volatile):"); say(TB.pivot(index="vol_fifth", columns="beta").round(3).to_string())
top = TB[TB.vol_fifth == 5]
no_worse = all(abs(r.breach_struct - r.beta) <= abs(r.breach_base - r.beta) + 0.005 for r in top.itertuples())
passed = (T["t_month (struct better)"] > 2).all() and no_worse
say(f"\nverdict: {'PASS' if passed else 'FAIL'} (pinball better at both levels with monthly t > 2, and the most volatile fifth "
    "no further from its promise than before, within half a point; final calibration is done per volatility fifth on the board)")
if passed and not TEST:
    joblib.dump({"theta": [float(x) for x in fit_decay(np.ones(len(E), bool))]}, MODELS / "vol_decay.joblib")
    say(f"saved decay parameters -> {MODELS / 'vol_decay.joblib'}")
if not TEST:
    (OUT / ("x04_vol_decay" + ("_live" if LIVE else "") + ".md")).write_text("# Experiment 4: volatility decay and the structural bad case\n\n```\n" + "\n".join(LOG) + "\n```\n")
