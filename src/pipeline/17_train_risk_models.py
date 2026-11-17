"""Step 17 (v2): train and validate.

Targets (all measured AFTER the close of day 1, so they are tradeable):
  drift_1_5 / drift_1_20 / drift_1_60  factor-adjusted abnormal return from day 2 to day h
  trough_2_20                           worst cumulative abnormal return in days 2-20
  vol_ratio_2_20 / abn_volume_2_20      post-event volatility / volume relative to normal (log)
Return targets are trained demeaned-by-month and noise-scaled (see modeling.py) and reported back in % terms,
relative to the average event in the same month.

Validation: purged walk-forward by year blocks, hyperparameters tuned inside each training window only,
shuffled-target placebo, conformalized quantile regression (CQR) intervals, spread-based trading costs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from config import PROC, RESULTS, MODELS, LIVE, OOS_FILE
from modeling import *
from modeling import ev, FEATS, BASE_FEATS, SENT, HAVE_SENT, TARGETS, RET_T, FOLDS, SENT_FOLDS, GBM_NAME, N_ALL, N_UNIVERSE, REGIME

def score(y, p):
    ok = y.notna() & p.notna(); y, p = y[ok], p[ok]
    if len(y) < 50:
        return {}
    base = np.mean((y - y.mean()) ** 2)
    q = pd.qcut(p.rank(method="first"), 10, labels=False)
    return dict(n=len(y), r2_oos=1 - np.mean((y - p) ** 2) / base, spearman=spearmanr(y, p)[0],
                hit=np.mean(np.sign(p - y.mean()) == np.sign(y - y.mean())),
                top_minus_bottom_decile=y[q == 9].mean() - y[q == 0].mean())
spear = lambda y, p: spearmanr(y, p)[0]

X = ev[FEATS].astype(float); Xb = ev[BASE_FEATS].astype(float)
lines = [f"# Results (v2): {len(ev):,} events, {ev.day0.min().date()} to {ev.day0.max().date()}", "",
         "## Data funnel",
         f"- events after linking and event study: {N_ALL:,}",
         f"- tradeable universe (price >= $5, not in the smallest 20% of firms that year): {N_UNIVERSE:,}",
         f"- regime features: {', '.join(REGIME) if REGIME else 'none (run 01_fred_data.py)'}",
         f"- news features: {'yes, ' + str(int(ev.has_news.sum())) + ' events with headlines' if HAVE_SENT else 'no'}", "",
         "## Average reaction by event class (factor-adjusted abnormal returns, tradeable universe)",
         ev.groupby("cls")[["car_0_1", "drift_1_5", "drift_1_20", "drift_1_60"]].agg(["mean", "count"]).round(4).to_string(), "",
         f"## Out-of-sample skill (purged walk-forward, test years 2006-2026; boosting = {GBM_NAME}, Huber loss, tuned per fold)",
         "Return targets are scored relative to the month's average event, in % return units.",
         "Bar for 'real signal': spearman > 0 on the real target and ~0 on PLACEBO.", ""]
# ---- stacked ensemble (risk targets): LightGBM + ExtraTrees + ridge. Blend weights for each test block are learned by
# non-negative least squares on the EARLIER blocks' out-of-sample predictions only (first block: LightGBM alone).
STACK_T = () if LIVE else ("vol_ratio_2_20", "abn_volume_2_20", "trough_2_20")
stack_log = []
def extra_trees():
    from sklearn.ensemble import ExtraTreesRegressor
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(n_estimators=200, min_samples_leaf=100, max_features=0.3, n_jobs=-1, random_state=0))
def huber_lin():
    from sklearn.linear_model import HuberRegressor
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), HuberRegressor(alpha=1.0, max_iter=300))
CNN = None
if (PROC / "cnn_oos.parquet").exists():
    _c = pd.read_parquet(PROC / "cnn_oos.parquet")
    _c["permno"] = _c.permno.astype(ev.permno.dtype)
    CNN = ev[["permno", "day0"]].merge(_c, on=["permno", "day0"], how="left")
    CNN.index = ev.index
def stack(ym, preds):
    from scipy.optimize import nnls
    names = list(preds)
    out, log = pd.Series(np.nan, index=ym.index), []
    tests = [te for tr, te in splits(ev.day0, ym)]
    for f, te in enumerate(tests):
        if f == 0:
            out[te] = preds["lgbm"][te]; log.append("block 1: lgbm only"); continue
        prev = np.zeros(len(ym), bool)
        for g in tests[:f]:
            prev |= g.values
        A = np.column_stack([np.nan_to_num(preds[n].values[prev].astype(float), nan=0.0) for n in names]); yv = ym.values[prev]
        okr = np.isfinite(A).all(axis=1) & np.isfinite(yv)
        w, _ = nnls(A[okr], yv[okr])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0 if n == "lgbm" else 0.0 for n in names])
        out[te] = np.column_stack([np.nan_to_num(preds[n][te].values.astype(float), nan=0.0) for n in names]) @ w
        log.append(f"block {f + 1}: " + ", ".join(f"{n} {wi:.2f}" for n, wi in zip(names, w)))
    return out, log
res, oos, best_params = [], {}, {}
for t in TARGETS:
    print(f"\n=== Target: {t}  (model: {GBM_NAME}) ===", flush=True)
    ym = to_model_target(t).astype(float)                 # what the model learns
    yr = target_relative(t).astype(float)                 # what we score against, original units
    best = tune(X, ym, ev.day0, spear); best_params[t] = best
    pg_m = walk_forward_tuned(X, ym, ev.day0, best); pr_m = walk_forward(ridge, X, ym, ev.day0)
    p_g, p_r = from_model_pred(t, pg_m), from_model_pred(t, pr_m)
    oos[t] = p_g
    res += [{"target": t, "model": "ridge", **score(yr, p_r)}, {"target": t, "model": "lgbm_tuned", **score(yr, p_g)}]
    if t in STACK_T:                                   # stacked ensemble for the risk targets
        pe_m = walk_forward(extra_trees, X, ym, ev.day0)
        ph_m = walk_forward(huber_lin, X, ym, ev.day0)
        members = {"lgbm": pg_m, "extratrees": pe_m, "ridge": pr_m, "huber_linear": ph_m}
        if CNN is not None and f"cnn_{t}" in CNN:
            members["cnn"] = CNN[f"cnn_{t}"]
            res.append({"target": t, "model": "cnn (1-D conv, price/volume window)", **score(yr, from_model_pred(t, CNN[f"cnn_{t}"]))})
        ps_m, wl = stack(ym, members)
        oos[t] = from_model_pred(t, ps_m).where(ps_m.notna(), p_g)
        res += [{"target": t, "model": "huber_linear", **score(yr, from_model_pred(t, ph_m))},
                {"target": t, "model": "extratrees", **score(yr, from_model_pred(t, pe_m))},
                {"target": t, "model": "STACKED", **score(yr, from_model_pred(t, ps_m))}]
        stack_log.append(f"- {t}: " + "; ".join(wl))
    yp = ym.groupby(ev.day0.dt.year).transform(lambda v: v.sample(frac=1, random_state=1).values)
    res.append({"target": t + " PLACEBO", "model": "lgbm", **score(from_model_pred(t, yp), from_model_pred(t, walk_forward(gbm, X, yp, ev.day0)))})
    print(pd.DataFrame(res).tail(3).round(4).to_string(index=False), flush=True)
lines.append(pd.DataFrame(res).round(4).to_string(index=False))
lines += ["", "Stacked-ensemble blend weights (learned on earlier blocks only):"] + stack_log
lines += ["", "Hyperparameters chosen per fold (index into modeling.GRID; {} = default):"]
lines += [f"- {t}: {v}" for t, v in best_params.items()]

# ---- clean events
nxt = ev.sort_values(["permno", "day0"]).groupby("permno").day0.shift(-1)
clean = ((nxt - ev.day0).dt.days > 30) | nxt.isna()
lines += ["", f"## Skill on 'clean' events only (no other event at the firm within 30 days after): {clean.sum():,} events"]
lines.append(pd.DataFrame([{"target": t, **score(target_relative(t)[clean].astype(float), oos[t][clean])} for t in TARGETS]).round(4).to_string(index=False))

if not LIVE:
    # ---- ablation: do this round's additions help? (same folds, default hyperparameters, so only the inputs differ)
    import modeling as M
    NEW = [f for f in FEATS if f in set(M.EXTRA_F) or f.startswith("sestm")]
    OLD = [f for f in FEATS if f not in NEW]
    lines += ["", "## Ablation of this round's additions (default hyperparameters; only inputs / weights differ)",
              f"new inputs: {', '.join(NEW) if NEW else 'none found (run 04_event_features and 14_news_sestm)'}",
              "weights = label uniqueness x class balance (see modeling.py)"]
    ab = []
    for t in ("drift_1_20", "drift_1_60", "trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"):
        ym = to_model_target(t).astype(float); yr = target_relative(t).astype(float)
        for label, cols, wflag in (("old inputs, unweighted", OLD, False), ("old inputs, weighted", OLD, True), ("all inputs, weighted", FEATS, True)):
            M.USE_WEIGHTS = wflag
            p = from_model_pred(t, walk_forward(gbm, ev[cols].astype(float), ym, ev.day0))
            ab.append({"target": t, "setup": label, **score(yr, p)})
        print(pd.DataFrame(ab).tail(3).round(4).to_string(index=False), flush=True)
    M.USE_WEIGHTS = True
    lines.append(pd.DataFrame(ab).round(4).to_string(index=False))

    # ---- round 4 ablation: peer graph + short interest + embedding text features; time-decay weights; ranking objective
    R4 = [f for f in FEATS if f in set(getattr(M, "NEW4", []))]
    lines += ["", "## Round 4 ablation (default hyperparameters)", f"round-4 inputs: {', '.join(R4) if R4 else 'none found (run 05_peer_features and 15_news_embeddings)'}"]
    ab4 = []
    for t in ("drift_1_20", "drift_1_60", "trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"):
        ym = to_model_target(t).astype(float); yr = target_relative(t).astype(float)
        base_cols = [f for f in FEATS if f not in R4]
        for label, cols, hl in (("without round-4 inputs", base_cols, None), ("with round-4 inputs", FEATS, None),
                                ("with round-4 inputs + 5y time decay", FEATS, 5.0)):
            M.DECAY_HL = hl
            p = from_model_pred(t, walk_forward(gbm, ev[cols].astype(float), ym, ev.day0))
            ab4.append({"target": t, "setup": label, **score(yr, p)})
        M.DECAY_HL = None
        print(pd.DataFrame(ab4).tail(3).round(4).to_string(index=False), flush=True)
    # LambdaRank: train directly on the within-month ranking (what spearman scores)
    if M.lgb is not None:
        for t in ("drift_1_20", "drift_1_60"):
            ym = to_model_target(t).astype(float); yr = target_relative(t).astype(float)
            lab = ym.groupby(ev.month).transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) if v.notna().sum() >= 5 else np.nan)
            pr = pd.Series(np.nan, index=ev.index)
            for tr, te in splits(ev.day0, lab):
                g = ev.month[tr].astype(str)
                sizes = g.groupby(g, sort=False).size().values
                r = M.lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=31,
                                     min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
                r.fit(X[tr], lab[tr].astype(int), group=sizes, sample_weight=None)
                pr[te] = r.predict(X[te])
            pr = pr.groupby(ev.month).transform(lambda v: (v - v.mean()) / (v.std() + 1e-12)) * yr.std()
            ab4.append({"target": t, "setup": "LambdaRank (ranking objective)", **score(yr, pr)})
    lines.append(pd.DataFrame(ab4).round(4).to_string(index=False)); print(lines[-1], flush=True)


# ---- downside detection and a naive baseline (what a beginner cares about: did we flag the bad outcomes?)
from sklearn.metrics import roc_auc_score
lines += ["", "## Downside detection: AUC for flagging the worst 20% of outcomes (0.5 = no skill), vs a naive baseline",
          "naive baseline = ridge on only the day 0-1 reaction and pre-event volatility; balanced accuracy uses the model's bottom-20% call"]
dd = []
for t in ("drift_1_20", "drift_1_60", "trough_2_20"):
    yr = target_relative(t).astype(float)
    nb = from_model_pred(t, walk_forward(ridge, ev[["car_0_1", "idio_vol_pre"]].astype(float), to_model_target(t).astype(float), ev.day0))
    for name, p in (("naive baseline", nb), ("final model", oos[t])):
        m = yr.notna() & p.notna()
        bad = (yr[m] <= yr[m].quantile(0.2)).astype(int)
        call = (p[m] <= p[m].quantile(0.2)).astype(int)
        recall = float(call[bad == 1].mean()); spec = float(1 - call[bad == 0].mean())
        dd.append({"target": t, "model": name, "auc_worst20": roc_auc_score(bad, -p[m]), "downside_recall": recall,
                   "balanced_accuracy": (recall + spec) / 2})
lines.append(pd.DataFrame(dd).round(4).to_string(index=False)); print(lines[-1], flush=True)

# ---- skill by size (is it only small caps?)
szq = ev.groupby("year").log_mcap.transform(lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
lines += ["", "## 20-day drift skill by firm-size quartile (within the tradeable universe; 3 = largest)"]
lines.append(pd.DataFrame([{"size_quartile": int(q), **score(target_relative("drift_1_20")[szq == q].astype(float), oos["drift_1_20"][szq == q])}
                           for q in sorted(szq.dropna().unique())]).round(4).to_string(index=False))

# ---- skill by event class, and post-earnings drift by surprise decile (where the literature says drift lives)
lines += ["", "## Out-of-sample skill by event class (drift_1_20 and drift_1_60; hit = share with the right sign)"]
cr = []
for c in sorted(ev.cls.unique()):
    m = (ev.cls == c).values
    for t in ("drift_1_20", "drift_1_60"):
        yv, pv = target_relative(t)[m].astype(float), oos[t][m]
        if pv.notna().sum() > 500:
            cr.append({"cls": c, "target": t, **score(yv, pv)})
lines.append(pd.DataFrame(cr).round(4).to_string(index=False)); print(lines[-1], flush=True)
if "eps_sue" in ev and ev.eps_sue.notna().sum() > 1000:
    e = ev[ev.eps_sue.notna()].copy()
    e["sue_decile"] = pd.qcut(e.eps_sue.rank(method="first"), 10, labels=False)
    tab = e.groupby("sue_decile").agg(n=("eps_sue", "size"), mean_sue=("eps_sue", "mean"), car_0_1_pct=("car_0_1", lambda s: 100 * s.mean()),
                                      drift_1_20_pct=("drift_1_20", lambda s: 100 * s.mean()), drift_1_60_pct=("drift_1_60", lambda s: 100 * s.mean()))
    lines += ["", "## Post-earnings drift by surprise decile (0 = worst miss, 9 = biggest beat), realised averages",
              "PEAD exists in our data if drift_1_60 rises from decile 0 to 9 (Bernard & Thomas 1989).", tab.round(3).to_string()]
    print(lines[-1], flush=True)

# ---- sentiment ablation: only years with news, folds inside 2014-2023, training from 2009
if HAVE_SENT:
    cov = ev.has_news == 1
    lines += ["", f"## Sentiment ablation ({int(cov.sum()):,} events with headlines; test blocks 2014-16, 2017-19, 2020-23; training from 2009)",
              "Same model, same events, same folds; only the news columns differ."]
    ab = []
    for t in ("drift_1_5", "drift_1_20", "drift_1_60", "trough_2_20", "abn_volume_2_20", "vol_ratio_2_20"):
        ym = to_model_target(t).astype(float).where(cov); yr = target_relative(t).astype(float).where(cov)
        for label, XX in (("without_news", Xb), ("with_news", X)):
            p = from_model_pred(t, walk_forward(gbm, XX, ym, ev.day0, SENT_FOLDS, 2009))
            ab.append({"target": t, "features": label, **score(yr, p)})
    lines.append(pd.DataFrame(ab).round(4).to_string(index=False)); print(lines[-1], flush=True)

# ---- CQR intervals: quantile models walk-forward, conformal correction, coverage checked fold by fold
lines += ["", "## Prediction intervals (80%, conformalized quantile regression)",
          "coverage should be close to 0.80; width now varies by stock instead of one band for everyone."]
cqr, iv = {}, []
for t in TARGETS:
    ym = to_model_target(t).astype(float); yr = target_relative(t).astype(float)
    lo = from_model_pred(t, walk_forward(lambda: gbm_quantile(0.1), X, ym, ev.day0))
    hi = from_model_pred(t, walk_forward(lambda: gbm_quantile(0.9), X, ym, ev.day0))
    ok = lo.notna() & yr.notna()
    E = np.maximum(lo - yr, yr - hi)[ok]
    fold_id = pd.cut(ev.day0.dt.year[ok], [2005, 2010, 2015, 2020, 2026], labels=False)
    covered = pd.Series(False, index=E.index)
    for f in fold_id.unique():                                   # correction from the OTHER folds -> honest coverage
        qf = E[fold_id != f].quantile(0.8)
        covered[fold_id == f] = E[fold_id == f] <= qf
    Q = E.quantile(0.8); cqr[t] = Q
    width = (hi - lo)[ok] + 2 * Q
    big = ev.log_mcap[ok] >= ev.log_mcap[ok].median()
    iv.append(dict(target=t, coverage=covered.mean(), mean_width=width.mean(),
                   width_large_caps=width[big].mean(), width_small_caps=width[~big].mean()))
lines.append(pd.DataFrame(iv).round(4).to_string(index=False))

# ---- trading check: relative (market- and month-neutral) and with spread-based costs
lines += ["", "## Trading check: 20-day abnormal return after acting at the close of day 1 (out of sample)",
          "Long when predicted relative drift > +thr, short when < -thr.",
          "Cost per round trip = quoted bid-ask spread (clipped 0.05%-2%) + 0.10%.", ""]
y20 = target_relative("drift_1_20").astype(float); p20 = oos["drift_1_20"]
cost = ev.spread.clip(0.0005, 0.02).fillna(0.005) + 0.001
tb = []
for thr in (0.005, 0.01, 0.02, 0.03):
    for period, mask in (("2006-2026", p20.notna()), ("2021-2026", p20.notna() & (ev.day0.dt.year >= 2021))):
        side = np.where(p20 > thr, 1, np.where(p20 < -thr, -1, 0))
        tr = mask & (side != 0) & y20.notna()
        pnl = side[tr] * y20[tr] - cost[tr]
        tb.append(dict(threshold=thr, period=period, trades=int(tr.sum()), long_share=float((side[tr] > 0).mean()) if tr.any() else np.nan,
                       mean_net=pnl.mean(), hit=(pnl > 0).mean(),
                       t_stat=pnl.mean() / (pnl.std() / np.sqrt(len(pnl))) if len(pnl) > 1 else np.nan,
                       t_month_clustered=(lambda s: s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 else np.nan)(
                           pd.Series(pnl.values).groupby(ev.month[tr].astype(str).values).mean())))
TB = pd.DataFrame(tb)
from scipy.stats import norm
TB["p_one_sided"] = 1 - norm.cdf(TB.t_month_clustered)
order = TB.p_one_sided.sort_values().index; m_ = len(TB)
TB.loc[order, "p_holm"] = np.minimum(1, np.maximum.accumulate(TB.p_one_sided[order].values * (m_ - np.arange(m_))))
lines.append(TB.round(4).to_string(index=False))
lines.append("t_month_clustered: trades in the same month are not independent, so t is computed on monthly averages. "
             "p_holm: Holm adjustment for having tried 8 threshold/period combinations (multiple testing). "
             "Treat a rule as real only if p_holm < 0.05.")

# ---- final models (most recent fold's tuned settings) + explanation of what drives predictions
lines += ["", "## Final models and main drivers (mean |SHAP contribution|, % of total)"]
for t in TARGETS:
    ym = to_model_target(t).astype(float); ok = ym.notna()
    last = list(best_params[t].values())[-1] if best_params[t] else {}
    m = gbm(**last).fit(X[ok], ym[ok])
    qlo = gbm_quantile(0.1, **last).fit(X[ok], ym[ok]); qhi = gbm_quantile(0.9, **last).fit(X[ok], ym[ok])
    joblib.dump({"model": m, "q_lo": qlo, "q_hi": qhi, "cqr": float(cqr[t]), "feats": FEATS, "version": 2,
                 "return_target": t in RET_T}, MODELS / f"{t}.joblib")
    if lgb is not None:
        smp = X[ok].sample(min(20000, int(ok.sum())), random_state=0)
        contrib = np.abs(m.predict(smp, pred_contrib=True)[:, :-1]).mean(0)
        imp = pd.Series(contrib / contrib.sum() * 100, index=FEATS).sort_values(ascending=False).head(8)
        lines.append(f"- {t}: " + ", ".join(f"{k} {v:.0f}%" for k, v in imp.items()))
from final_models import calibrate                      # held-out calibration of the final interval models
lines += ["", "## Final interval models: held-out calibration (last 2 years)", pd.DataFrame([calibrate(t) for t in TARGETS]).to_string(index=False)]
pd.DataFrame(oos).to_parquet(PROC / OOS_FILE)         # used by 18_buy_timing.py
(RESULTS / "report.md").write_text("\n".join(lines))
print("\n".join(lines)); print(f"\nsaved {RESULTS / 'report.md'}")
