"""Step 6: position sizing. "Given my account, how much of it should go into this stock?" with three tiers.

Everything here is in TOTAL returns (what you actually feel in your account), not abnormal returns.
Horizon: buy at the close of day 1 after the event, hold 20 trading days (days 2-20).

Ingredients (all out of sample, same purged walk-forward folds as s03):
  1. Downside: conformalized quantile regression (CQR) of the WORST point of the 20-day path,
     L = min(0, min_k [(1+r_2)...(1+r_k) - 1]),  at levels beta = 5% and 10%.
     Calibrated inside each training window; in the test years the calibration offset is updated
     month by month with adaptive conformal inference (Gibbs & Candes 2021), using only outcomes
     that are fully known (2-month lag):  c_{t+1} = c_t + gamma * (miss_rate_t - beta).
  2. Expected return: CAPM prior + skill-shrunk event alpha (Grinold's fundamental law):
     mu = beta_mkt * ERP_20d + IC_t * sigma_abn * z,
     where z = the s03 drift_1_20 prediction standardised within the month, and IC_t = the
     out-of-sample rank correlation of that prediction measured ONLY on earlier years (0 before 2007,
     clipped to [0, 0.10]). If the signal decays, IC_t falls and alpha shrinks toward zero by itself.
  3. Volatility: sigma_20 = yz_vol_20 * exp(predicted log vol ratio) * sqrt(19/252).
  4. Size (fraction of the account), long only:
        w = min( d / |q_beta| ,  k * mu / sigma^2 ,  cap )
     The first term is the exact single-asset solution of the chance constraint
     P(position loses more than d of the ACCOUNT at any point in 20 days) <= beta
     (loss scales linearly with w); the second is fractional Kelly; the third a hard cap.
     A tier is 0 when mu <= 0.

Tiers (d, beta, Kelly fraction k, cap):
  safest      2% of account, 5% chance,  1/4 Kelly, cap 15%
  recommended 4% of account, 10% chance, 1/2 Kelly, cap 25%
  bold        8% of account, 10% chance, 3/4 Kelly, cap 40%

Backtest (2016-2026, untouched by any choice above): a $200 account takes each event at each tier;
we check whether each tier KEPT ITS PROMISE (realised breach rate <= beta) and what it earned.
Run:  python src/s06_sizing.py        (quick check on 300 firms: SIZING_TEST=1 python src/s06_sizing.py)
Research only. Not investment advice.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from config import PROC, RESULTS, MODELS, OOS_FILE
from modeling import ev, FEATS, FOLDS, PURGE, splits, gbm_quantile, target_relative, _fit

TEST = os.environ.get("SIZING_TEST") == "1"
H = 19                                   # trading days held (days 2..20)
ERP_YEAR = 0.06                          # equity risk premium prior (long-run US average, ~6% a year)
ACCOUNT = 200.0
# d = loss budget as a share of the WHOLE account at the worst point; professional practice risks 1-2% per trade
TIERS = {"safest": dict(d=0.02, beta=0.05, k=0.25, cap=0.15),
         "recommended": dict(d=0.04, beta=0.10, k=0.50, cap=0.25),
         "bold": dict(d=0.08, beta=0.10, k=0.75, cap=0.40)}
LEVELS = sorted({t["beta"] for t in TIERS.values()})
GAMMA = 0.2                              # adaptive conformal step size (return units per unit of miss-rate error)
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(str(s))
def section(n, title):
    say(""); say("=" * 70); say(f"{n}. {title}"); say("=" * 70)

oos = pd.read_parquet(PROC / OOS_FILE)
assert len(oos) == len(ev), "oos_predictions.parquet does not match events; re-run s03_train.py"
E = ev.copy()
E["p_drift20"], E["p_logvr"] = oos["drift_1_20"].values, oos["vol_ratio_2_20"].values
E["p_trough"] = oos["trough_2_20"].values
if TEST:
    E = E[E.permno.isin(np.sort(E.permno.unique())[:300])]

# ---------------------------------------------------------------- total-return path, days 2..20
permnos = [int(p) for p in np.sort(E.permno.unique())]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"], filters=[("permno", "in", permnos)]).to_pandas()
D["key"] = D.permno.astype(np.int64) * 100000 + (D.date.values.astype("datetime64[D]").astype(np.int64) - 7300)
D = D.sort_values("key").reset_index(drop=True)
k0 = E.permno.astype(np.int64).values * 100000 + (E.day0.values.astype("datetime64[D]").astype(np.int64) - 7300)
g0 = np.searchsorted(D.key.values, k0)
idx = (g0[:, None] + np.arange(2, 21)[None, :]).clip(0, len(D) - 1)
same = D.permno.values[idx] == E.permno.values[:, None]
R = np.where(same, D.ret.values[idx], np.nan)
R = np.where(np.isfinite(R), R, 0.0) * same                      # missing day = flat; delisted = no further move
path = np.cumprod(1 + np.clip(R, -0.95, 3.0), axis=1) - 1
E["tot_r20"] = path[:, -1]
E["tot_trough"] = np.minimum(0, path.min(axis=1))
ok_path = same[:, 0] & (D.key.values[g0.clip(0, len(D) - 1)] == k0)
E = E[ok_path].copy()
PATH = path[ok_path].astype(np.float32)                         # kept for the multi-stock backtest
del D, R, path

section(1, "Inputs")
say(f"events with a total-return path: {len(E):,}" + ("   [TEST MODE]" if TEST else ""))
say("realised 20-day total-return path minimum (worst point while holding), percentiles:")
say((E.tot_trough.quantile([0.01, 0.05, 0.10, 0.25, 0.5]) * 100).round(2).to_string())

# ---------------------------------------------------------------- 1. downside quantiles: CQR + ACI
X = E[FEATS].astype(float)
y = E.tot_trough.astype(float)
dates = E.day0
qhat = {b: pd.Series(np.nan, index=E.index) for b in LEVELS}
c0 = {b: {} for b in LEVELS}
for fi, (tr, te) in enumerate(splits(dates, y)):
    last = dates[tr].max()
    cal = tr & (dates > last - pd.Timedelta(days=2 * 365))
    fit = tr & (dates < last - pd.Timedelta(days=2 * 365) - PURGE)
    for b in LEVELS:
        m = _fit(gbm_quantile(b), X[fit], y[fit])                      # unweighted: quantiles describe the real population
        e_cal = m.predict(X[cal]) - y[cal].values                       # >0 means the outcome fell below the bound
        c0[b][fi] = float(np.quantile(e_cal, 1 - b))
        qhat[b][te] = m.predict(X[te])
    say(f"  fold {fi + 1}: fit {int(fit.sum()):,}, calibrate {int(cal.sum()):,}, test {int(te.sum()):,}")

# adaptive conformal offset, month by month, using only outcomes known 2 months later
E["month"] = E.day0.dt.to_period("M")
fold_of = pd.Series(-1, index=E.index)
for fi, (tr, te) in enumerate(splits(dates, y)):
    fold_of[te] = fi
bound = {}
for b in LEVELS:
    lb_static = qhat[b] - fold_of.map(c0[b]).astype(float)
    lb_aci = pd.Series(np.nan, index=E.index)
    months = sorted(E.month[fold_of >= 0].unique())
    c, cur_fold, hist = None, None, {}
    for mth in months:
        mm = (E.month == mth) & (fold_of >= 0)
        f = int(fold_of[mm].iloc[0])
        if f != cur_fold:                                              # new fold: restart from its calibration offset
            c, cur_fold = c0[b][f], f
        lag = mth - 2
        if lag in hist:
            c = c + GAMMA * (hist[lag] - b)
        lb_aci[mm] = qhat[b][mm] - c
        hist[mth] = float((y[mm] < lb_aci[mm]).mean())
    bound[b] = (lb_static.clip(upper=-0.005), lb_aci.clip(upper=-0.005))

section(2, "Downside bound: does 'at most beta chance of falling below it' hold out of sample?")
rows = []
for b in LEVELS:
    for name, lb in (("CQR (fixed)", bound[b][0]), ("CQR + adaptive (ACI)", bound[b][1])):
        for per, msk in (("2006-2015", (dates.dt.year <= 2015)), ("2016-2026", (dates.dt.year >= 2016)), ("2021-2026", (dates.dt.year >= 2021))):
            m = msk & lb.notna()
            rows.append({"beta": b, "method": name, "period": per, "n": int(m.sum()),
                         "breach_rate": float((y[m] < lb[m]).mean()), "median_bound_pct": float(100 * lb[m].median())})
say(pd.DataFrame(rows).round(4).to_string(index=False))
say("breach_rate should be close to beta. The adaptive version should stay closer in recent, shifting years.")
yrs = []
for b in LEVELS:
    lb = bound[b][1]
    t = pd.DataFrame({"yr": dates.dt.year, "miss": (y < lb).astype(float), "ok": lb.notna()})
    yrs.append(t[t.ok].groupby("yr").miss.mean().rename(f"breach_beta_{b}"))
say("breach rate by year (adaptive):"); say(pd.concat(yrs, axis=1).round(3).T.to_string())

# ---------------------------------------------------------------- 2-3. expected return and volatility
z = E.groupby("month").p_drift20.transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
yr_drift = target_relative("drift_1_20").reindex(E.index).astype(float)
ic = {}
for Y in range(2006, 2027):
    m = (dates.dt.year < Y) & (dates.dt.year >= 2006) & E.p_drift20.notna() & yr_drift.notna()
    ic[Y] = float(np.clip(spearmanr(E.p_drift20[m], yr_drift[m])[0], 0, 0.10)) if m.sum() > 2000 else 0.0
IC = dates.dt.year.map(ic).astype(float)
sigma = (E.yz_vol_20.clip(0.05, 3) * np.exp(E.p_logvr.clip(-2, 2)) * np.sqrt(H / 252)).clip(0.01, 1.5)
# calibrate sigma each year with a scale factor estimated ONLY on earlier years: kappa_Y = sqrt(mean(((r - mu0)/sigma)^2))
mu0 = E.beta.clip(0, 3).fillna(1) * ERP_YEAR * H / 252
zz = ((E.tot_r20 - mu0) / sigma) ** 2
kap = {}
for Y in range(2002, 2027):
    m = (dates.dt.year < Y) & zz.notna()
    kap[Y] = float(np.sqrt(zz[m].clip(upper=100).mean())) if m.sum() > 2000 else np.nan
kap = pd.Series(kap).bfill().ffill()
sigma = sigma * dates.dt.year.map(kap).astype(float)
sig_abn = (E.idio_vol_pre.clip(0.05, 3) * np.sqrt(H / 252))
mu = E.beta.clip(0, 3).fillna(1) * ERP_YEAR * H / 252 + IC * sig_abn * z.fillna(0)
section(3, "Expected return and volatility inputs")
say("IC used each year (measured only on earlier years' out-of-sample predictions, clipped to 0..0.10):")
say(pd.Series(ic).round(4).to_frame("IC").T.to_string())
say(f"mu (20-day expected total return): median {100 * mu.median():.2f}%, 10th pct {100 * mu.quantile(.1):.2f}%, 90th {100 * mu.quantile(.9):.2f}%")
say("sigma scale factor by year (estimated on earlier years only): " + ", ".join(f"{y}:{v:.2f}" for y, v in kap.items() if y >= 2006))
say(f"sigma (20-day volatility): median {100 * sigma.median():.1f}%;  realised std of 20-day returns {100 * E.tot_r20.std():.1f}%")
q = pd.qcut(sigma.rank(method="first"), 5, labels=False)
say("calibration of sigma: predicted vs realised 20-day volatility by quintile of prediction (%)")
say(pd.DataFrame({"q": q, "pred": sigma * 100, "real": E.tot_r20 * 100}).groupby("q").agg(pred=("pred", "mean"), real=("real", "std")).round(2).T.to_string())

# ---------------------------------------------------------------- 4. sizes and the $200 backtest
def size(tier, lb):
    t = TIERS[tier]
    w_risk = t["d"] / lb.abs()
    w_kelly = t["k"] * mu / sigma ** 2
    w = np.minimum(np.minimum(w_risk, w_kelly), t["cap"]).clip(lower=0)
    return w.where(mu > 0, 0.0)

section(4, f"Backtest: ${ACCOUNT:.0f} account, one position per event, 2016-2026 (untouched by any choice above)")
test = (dates.dt.year >= 2016) & bound[0.05][1].notna()
res = []
for tier, t in TIERS.items():
    lb = bound[t["beta"]][1]
    w = size(tier, lb)
    for per, msk in (("2016-2026", test), ("2021-2026", test & (dates.dt.year >= 2021))):
        m = msk & w.notna()
        stake = ACCOUNT * w[m]
        worst = stake * E.tot_trough[m]                       # $ at the worst point (<= 0)
        pnl = stake * E.tot_r20[m]
        traded = w[m] > 0
        binding = (t["d"] / lb[m].abs() <= np.minimum(t["k"] * mu[m] / sigma[m] ** 2, t["cap"]))
        res.append({"tier": tier, "period": per, "promise": f"<= {t['beta']:.0%} chance of losing > ${ACCOUNT * t['d']:.0f}",
                    "risk_rule_sets_size": float(binding[traded].mean()),
                    "share_traded": float(traded.mean()), "avg_stake_$": float(stake[traded].mean()),
                    "breach_rate": float((worst[traded] < -ACCOUNT * t["d"]).mean()),
                    "worst_1pct_$": float(np.quantile(worst[traded], 0.01)) if traded.any() else np.nan,
                    "avg_pnl_$": float(pnl[traded].mean()), "avg_pnl_pct_of_stake": float(100 * (pnl[traded] / stake[traded]).mean())})
# benchmark: same events, fixed 10% of account regardless of risk
for per, msk in (("2016-2026", test), ("2021-2026", test & (dates.dt.year >= 2021))):
    stake = pd.Series(ACCOUNT * 0.10, index=E.index)[msk]
    worst = stake * E.tot_trough[msk]; pnl = stake * E.tot_r20[msk]
    res.append({"tier": "fixed 10% (benchmark)", "period": per, "promise": "none", "share_traded": 1.0, "avg_stake_$": 20.0,
                "breach_rate": float((worst < -ACCOUNT * 0.10).mean()), "worst_1pct_$": float(np.quantile(worst, 0.01)),
                "avg_pnl_$": float(pnl.mean()), "avg_pnl_pct_of_stake": float(100 * E.tot_r20[msk].mean())})
say(pd.DataFrame(res).round(3).to_string(index=False))
say("breach_rate = share of traded positions whose worst point cost more than the tier's limit; it should be <= the promised chance.")
say("Sizing does not make returns predictable. Its job is to keep each tier's worst case where it says it will be.")

# out-of-sample trades, saved for the Monte Carlo account simulation in s12_figures.py
if not TEST:
    trades = []
    for tier, t in TIERS.items():
        w = size(tier, bound[t["beta"]][1]); m = test & w.notna() & (w > 0)
        trades.append(pd.DataFrame({"tier": tier, "day0": dates[m].values, "w": w[m].values,
                                    "r20": E.tot_r20[m].values, "trough": E.tot_trough[m].values}))
    for name, wf in (("fixed_10pct", 0.10), ("all_in", 1.0)):                  # benchmarks on every test event
        trades.append(pd.DataFrame({"tier": name, "day0": dates[test].values, "w": wf,
                                    "r20": E.tot_r20[test].values, "trough": E.tot_trough[test].values}))
    pd.concat(trades).to_parquet(RESULTS / "sizing_trades.parquet")

# mean-variance check: risk-adjusted return per $ at risk by tier
section(5, "Risk-adjusted outcome per trade (2016-2026): mean / std of P&L in $")
for tier, t in TIERS.items():
    w = size(tier, bound[t["beta"]][1]); m = test & (w > 0)
    pnl = ACCOUNT * w[m] * E.tot_r20[m]
    say(f"  {tier:12s} mean ${pnl.mean():.2f}  std ${pnl.std():.2f}  ratio {pnl.mean() / pnl.std():.3f}  (n={int(m.sum()):,})")
pnl = ACCOUNT * 0.10 * E.tot_r20[test]
say(f"  {'fixed 10%':12s} mean ${pnl.mean():.2f}  std ${pnl.std():.2f}  ratio {pnl.mean() / pnl.std():.3f}")

# ---------------------------------------------------------------- 6. several stocks at once
section(6, "Splitting one account across 3 stocks on the same day (recommended tier, 2016-2026)")
say("Each stock is first sized alone; then the basket is scaled so its combined bad case stays within the tier's")
say("budget, using a Ledoit-Wolf shrunk correlation of the 3 stocks' returns over the 250 trading days before the event.")
from sizing_lib import shrunk_corr, basket_scale
tier = TIERS["recommended"]; lbR = bound[tier["beta"]][1]; wR = size("recommended", lbR)
pos = pd.Series(np.arange(len(E)), index=E.index)
cand = E[(dates.dt.year >= 2016) & (wR > 0) & lbR.notna()]
days = cand.groupby("day0").permno.nunique()
days = days[days >= 3].index
rng = np.random.default_rng(0)
days = rng.choice(days, min(len(days), 2500 if not TEST else 300), replace=False)
pr = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret"],
                   filters=[("date", ">=", pd.Timestamp("2014-06-01")), ("permno", "in", [int(p) for p in cand.permno.unique()])]).to_pandas()
RP = pr.pivot_table(index="date", columns="permno", values="ret").sort_index(); del pr
rows6 = []
for d0 in days:
    g = cand[cand.day0 == d0].drop_duplicates("permno")
    g = g.loc[rng.choice(g.index, 3, replace=False)]
    w0 = wR[g.index].values; q0 = lbR[g.index].values
    hist = RP.loc[RP.index < d0, [p for p in g.permno.values if p in RP.columns]].tail(250)
    if hist.shape[1] < 3 or len(hist) < 120:
        continue
    C = shrunk_corr(hist.values)
    paths = PATH[pos[g.index].values]                               # 3 x 19 total-return paths
    naive_w = w0 * min(1.0, 0.8 / w0.sum())
    joint_w, est = basket_scale(w0, q0, C, tier["d"])
    for name, ww in (("each sized alone (naive sum)", naive_w), ("basket-scaled (correlation-aware)", joint_w)):
        acct = (ww[:, None] * paths).sum(axis=0)
        rows6.append({"method": name, "invested_share": ww.sum(), "worst": min(0.0, acct.min()), "pnl": acct[-1],
                      "avg_corr": (C.sum() - 3) / 6})
T6 = pd.DataFrame(rows6)
if len(T6):
    say(T6.groupby("method").apply(lambda g: pd.Series({
        "baskets": len(g), "avg_invested_$": ACCOUNT * g.invested_share.mean(),
        f"breach_rate (lose > ${ACCOUNT * tier['d']:.0f}, promise <= {tier['beta']:.0%})": (g.worst < -tier["d"]).mean(),
        "worst_1pct_$": ACCOUNT * g.worst.quantile(0.01), "avg_pnl_$": ACCOUNT * g.pnl.mean()})).round(3).to_string())
    say(f"average pairwise correlation in the baskets: {T6.avg_corr.mean():.2f}")

# ---------------------------------------------------------------- big-drop warning, scored on the events the live board sees
# The live board only shows events that are news days or price/volume shocks, which are riskier than the average event.
# So the warning cutoff is the riskiest 20% of predicted troughs AMONG shock-sized events (|day-0 move| > 3 sd or volume
# > 3x), chosen on years before 2016 and scored on 2016-2026 against the realised total-return worst point.
from sklearn.metrics import roc_auc_score
section(7, "Big-drop warning among shock-sized events (what the live board flags)")
shock = (E.ar_day0_sd.abs() > 3) | (E.abn_volume_0_1 > 3)
early, late = shock & (E.day0.dt.year < 2016) & E.p_trough.notna(), shock & (E.day0.dt.year >= 2016) & E.p_trough.notna()
thr_shock = float(E.p_trough[early].quantile(0.2))
bad = E.tot_trough[late] <= E.tot_trough[late].quantile(0.2)
flag = E.p_trough[late] <= thr_shock
auc = roc_auc_score(bad, -E.p_trough[late])
say(f"shock-sized events: {int(shock.sum()):,} ({shock.mean():.0%} of all); cutoff (pre-2016): predicted relative trough <= {100 * thr_shock:.2f}%")
say(f"2016-2026: flagged {flag.mean():.0%} of shock events; AUC for the worst 20% realised drops {auc:.3f}; "
    f"realised share in the worst 20%: flagged {bad[flag].mean():.0%} vs not flagged {bad[~flag].mean():.0%}")

# ---------------------------------------------------------------- final models for the live tracker
# Split conformal, like each fold: fit on events up to 2 years (+ purge) before the last one, calibrate the offset on the
# last 2 years with the SAME model. (The ACI offset from the fold-4 model does not transfer to a model refit on all data.)
if not TEST:
    allm = y.notna(); last = dates[allm].max()
    calw = allm & (dates > last - pd.Timedelta(days=2 * 365))
    fitw = allm & (dates < last - pd.Timedelta(days=2 * 365) - PURGE)
    finals, last_c = {}, {}
    for b in LEVELS:
        m = _fit(gbm_quantile(b), X[fitw], y[fitw])
        e = m.predict(X[calw]) - y[calw].values                        # >0 means the outcome fell below the bound
        n = len(e); last_c[b] = float(np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - b)) / n)))
        finals[b] = m
        say(f"final {b:.0%} bound: fit {int(fitw.sum()):,} events, calibrated on {n:,} (from {dates[calw].min().date()}); offset {100 * last_c[b]:.2f}%")
    joblib.dump({"q_models": finals, "aci_offset": last_c, "feats": FEATS, "tiers": TIERS, "erp_year": ERP_YEAR,
                 "ic_latest": ic[max(ic)], "horizon_days": H, "gamma": GAMMA, "version": 2,
                 "kappa_latest": float(kap.iloc[-1]), "drift_pred_sd": float(E.p_drift20.std()),
                 "trough_q20": float(oos["trough_2_20"].quantile(0.2)), "trough_q20_shock": thr_shock,
                 "warn_auc_shock_2016_26": float(auc), "cal_start": str(dates[calw].min().date())}, MODELS / "sizing.joblib")
    say(""); say(f"saved {MODELS / 'sizing.joblib'}")
(RESULTS / ("sizing_TEST.md" if TEST else "sizing.md")).write_text("# Position sizing (research only, not investment advice)\n\n```\n" + "\n".join(LOG) + "\n```\n")
say(f"saved {RESULTS / ('sizing_TEST.md' if TEST else 'sizing.md')}")
