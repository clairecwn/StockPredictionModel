"""Experiment 3: earnings preview. Before a scheduled earnings report, how big a move should be expected, and how far
could the stock fall over the following four weeks?

Population: earnings events in the tradeable universe. Prediction time: the close of the trading day BEFORE day 0, so
only features known then are used (reaction-day features, the day-0 market move and the calendar-distance features
are removed; earn_react_hist already uses only earlier reports).
Targets
  move   |CAR(0,1)|, the size of the two-day abnormal reaction (modelled as log(|move| + 0.2%))
  dd     the deepest cumulative abnormal fall from the pre-report close over days 0..20 (0 if it never falls)
Baselines the model must beat
  usual  the stock's own average earnings move over its previous 8 reports (earn_react_hist), the rule of thumb
  vol    pre-event idiosyncratic volatility (a calm stock moves less)
Test: purged walk-forward, Spearman rank correlation month by month on 2016-25, paired t against each baseline;
80% upper bound on |move| from a conformalised 90% quantile model (split conformal on the year before each test block).
Pass: beats both baselines on both targets with monthly t > 2. Then a final model is saved for the board.
Run: LIVE_FEATS=1 python src/experiments/03_earnings_preview.py      (quick check: EXP_TEST=1)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from config import ROOT, MODELS, LIVE
from modeling import ev, FEATS, splits, gbm, gbm_quantile, _fit, UNIQ

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
LOG = []
def say(s=""): print(s, flush=True); LOG.append(str(s))

DROP = {"car_0_1", "ar_day0_sd", "abn_volume_0_1", "range_shock_01", "mkt_day0", "react_per_sue", "sue_react_clash",
        "exp_earn_in_win", "days_to_exp_earn", "days_since_earn", "n_events", "news_30d", "eps_sue", "eps_surp_rel", "eps_beat",
        "peer_car_01", "own_vs_peer", "peer_event", "scandal_word",
        # built from day-0/day-1 headlines or describe the news of day 0 itself: not known the evening before the report
        "news_n_01", "news_abn", "stale_share", "relevant_share", "disagree", "txt_novelty", "txt_dispersion", "has_news",
        "rec_net", "ceo_involved", "exec_departure", "forced_exit", "gov_involved", "log_claim", "activist_stake", "restate_adverse"}
PRE = [f for f in FEATS if f not in DROP and not f.startswith(("has_", "cls_", "sent", "sestm", "emb", "lex", "nov", "ana"))]
EXTRA = [c for c in os.environ.get("EXTRA_PRE", "").split(",") if c and c in ev and c not in PRE]   # e.g. est_dispersion,n_analysts
BASE_PRE = list(PRE); PRE = PRE + EXTRA
E = ev[ev.cls == "earnings_guidance"].copy()
if TEST: E = E[E.permno.isin(np.sort(E.permno.unique())[:600])]
sd = (E.idio_vol_pre / np.sqrt(252)).clip(lower=0.003)
ar0 = E.ar_day0_sd * sd
path = np.column_stack([ar0, E.car_0_1] + [E.car_0_1 + E[[f"ar_d{d}" for d in range(2, k + 1)]].sum(axis=1) for k in range(2, 21)])
E["move"] = E.car_0_1.abs(); E["dd"] = np.minimum(0, np.nanmin(path, axis=1))
E["y_move"] = np.log(E.move + 0.002); E["y_dd"] = E.dd / (sd * np.sqrt(21))          # deepest fall in units of expected noise
E = E[E.move.notna() & E.dd.notna()]
X = E[PRE].astype(float); d0 = E.day0; month = d0.dt.to_period("M")
WU = UNIQ.reindex(E.index).fillna(1.0)          # overlap weights only: one event class here
say(f"earnings events: {len(E):,}; pre-report features: {len(PRE)}" + ("   [TEST MODE]" if TEST else ""))
P = dict(n_estimators=60 if TEST else 500, learning_rate=0.03, num_leaves=31, min_child_samples=300)

def wf(y, fn):
    pr = pd.Series(np.nan, index=y.index)
    for tr, te in splits(d0, y):
        pr[te] = _fit(fn(), X[tr], y[tr], WU[tr].values).predict(X[te])
    return pr
def monthly(y, p, msk):
    g = pd.DataFrame({"y": y[msk], "p": p[msk], "m": month[msk]}).groupby("m")
    return g.apply(lambda x: spearmanr(x.y, x.p).correlation if len(x) >= 30 else np.nan).dropna()

late = (d0.dt.year >= 2016)
rows = []
for tname, y, real in (("move", E.y_move, E.move), ("dd", E.y_dd, E.dd)):
    p = wf(y, lambda: gbm(**P))
    if tname == "dd": p = p * (sd * np.sqrt(21))                                        # back to return units
    ok = late & p.notna() & real.notna()
    bases = {"usual (own past earnings moves)": E.earn_react_hist, "vol (pre-event volatility)": E.idio_vol_pre}
    sign = 1 if tname == "move" else -1                                                  # bigger expected risk = deeper (more negative) fall
    mm = monthly(real, p, ok)
    r = {"target": tname, "model": spearmanr(real[ok], p[ok]).correlation}
    for bn, b in bases.items():
        bo = ok & b.notna()
        mb = monthly(real, sign * b, bo); mm2 = monthly(real, p, bo); dlt = (mm2 - mb).dropna()
        r[f"baseline {bn}"] = spearmanr(real[bo], sign * b[bo]).correlation
        r[f"t vs {bn.split()[0]}"] = dlt.mean() / (dlt.std() / np.sqrt(len(dlt)))
    if tname == "move":
        # log-scale forecasts understate typical sizes; Duan smearing factor, measured on 2006-15 out-of-sample residuals only
        e_ok = (d0.dt.year < 2016) & p.notna()
        smear = float(np.mean(np.exp(y[e_ok] - p[e_ok]))) if e_ok.any() else 1.0
        E["p_move"] = np.exp(p) * smear - 0.002
        usual = E.earn_react_hist[ok]; uo = usual.notna() & (usual > 0)
        big = real[ok][uo] > 2 * usual[uo]                                             # a reaction at least twice the stock's usual
        r["AUC unusually big move (>2x usual)"] = roc_auc_score(big, E.p_move[ok][uo] / usual[uo]) if big.nunique() == 2 else np.nan
        r["smear"] = smear
    else:
        E["p_dd"] = p
    rows.append(r)
T = pd.DataFrame(rows); say("\nOut-of-sample skill on 2016-25 (Spearman; t = monthly paired test against each baseline)"); say(T.round(4).to_string(index=False))
if EXTRA:
    Xb = E[BASE_PRE].astype(float)
    pb = pd.Series(np.nan, index=E.index)
    for tr, te in splits(d0, E.y_move):
        pb[te] = _fit(gbm(**P), Xb[tr], E.y_move[tr], WU[tr].values).predict(Xb[te])
    ok = late & pb.notna() & E.p_move.notna()
    mx, mb_ = monthly(E.move, E.p_move, ok), monthly(E.move, pb, ok); dlt = (mx - mb_).dropna()
    say(f"\nadded {EXTRA}: move skill {spearmanr(E.move[ok], E.p_move[ok]).correlation:.4f} vs {spearmanr(E.move[ok], pb[ok]).correlation:.4f} "
        f"without them; monthly paired t = {dlt.mean() / (dlt.std() / np.sqrt(len(dlt))):.2f}; coverage of the added features "
        f"{E.loc[late, EXTRA].notna().mean().round(3).to_dict()}")

# 90% upper bound on the move. The quantile model is fit per fold; its correction is ROLLING, as on the live board:
# for each report, the 90th percentile of the model's errors on reports in the 365 days ending 7 days before it.
ub = pd.Series(np.nan, index=E.index); y = E.y_move
for tr, te in splits(d0, y):
    last = d0[tr].max(); fit = tr & (d0 < last - pd.Timedelta(days=120))
    if fit.sum() < 2000: continue
    q = _fit(gbm_quantile(0.9, **P), X[fit], y[fit])
    win = (d0 > last - pd.Timedelta(days=365)) | te                     # errors from the year before the block onward
    res = pd.Series(y[win].values - q.predict(X[win]), index=E.index[win]).sort_index()
    dd_ = d0[win].sort_values(); r_ = res.reindex(dd_.index).values; t_ = dd_.values.astype("datetime64[D]").astype(np.int64)
    qte = q.predict(X[te]); tt = d0[te].values.astype("datetime64[D]").astype(np.int64)
    lo_i = np.searchsorted(t_, tt - 372); hi_i = np.searchsorted(t_, tt - 7)
    c = np.array([np.quantile(r_[i:j], 0.9) if j - i >= 200 else np.nan for i, j in zip(lo_i, hi_i)])
    ub[te] = np.exp(qte + c) - 0.002
okb = late & ub.notna()
say(f"\n90% upper bound on the move: realised move above it in {(E.move[okb] > ub[okb]).mean():.1%} of 2016-25 reports (target 10%); "
    f"median bound {100 * ub[okb].median():.1f}%")
dec = pd.qcut(E.p_move[late].rank(method="first"), 5, labels=False) + 1
say("realised average move by predicted quintile (2016-25, %):")
say((100 * pd.DataFrame({"q": dec, "pred": E.p_move[late], "real": E.move[late]}).groupby("q").mean()).round(2).T.to_string())

tcols = [c for c in T.columns if c.startswith("t vs")]
ok_t = {r.target: bool((T.loc[T.target == r.target, tcols] > 2).all(axis=1).iloc[0]) for r in T.itertuples()}
say(f"\nverdict: move {'PASS' if ok_t['move'] else 'FAIL'}, drawdown {'PASS' if ok_t['dd'] else 'FAIL'} "
    "(each must beat both baselines with monthly t > 2; only passing parts are saved for the board)")
if any(ok_t.values()) and not TEST:
    out = {"feats": PRE, "smear": float(T.smear.dropna().iloc[0])}
    if ok_t["move"]:
        out["move"] = gbm(**P).fit(X, E.y_move)
        last = d0.max(); fit = d0 < last - pd.Timedelta(days=120); cal = d0 > last - pd.Timedelta(days=365)
        q = gbm_quantile(0.9, **P).fit(X[fit], E.y_move[fit]); out["q90"] = q
        out["q90_offset"] = float(np.quantile(E.y_move[cal] - q.predict(X[cal]), 0.9))
        out["usual_vs_model"] = {"model": float(T.model[T.target == "move"].iloc[0]),
                                 "usual": float(T.filter(like="baseline usual")[T.target == "move"].iloc[0, 0])}
    if ok_t["dd"]:
        out["dd"] = gbm(**P).fit(X, E.y_dd)
    joblib.dump(out, MODELS / "earnings_preview.joblib"); say(f"saved {MODELS / 'earnings_preview.joblib'} with: {[k for k in ('move', 'dd') if k in out]}")
if not TEST:
    (OUT / ("03_earnings_preview" + ("_live" if LIVE else "") + ".md")).write_text("# Experiment 3: earnings preview\n\n```\n" + "\n".join(LOG) + "\n```\n")
