"""Shared data loading, cleaning filters, features, target transforms, folds and model factories (v2).

Used by s03_train.py, s03c_timing.py and s04_live.py.

v2 changes (each one is reported in results/report.md so its effect can be checked):
  universe  price >= $5 on the day before the event and not in the smallest 20% of firms that year  (tradeable stocks)
  targets   drift and trough are (a) demeaned within each event month, removing "good year / bad year" effects,
            and (b) divided by the stock's expected noise over the horizon (daily idiosyncratic sd * sqrt(h)),
            so a 5% move in a calm mega-cap counts more than a 5% move in a volatile small cap.
            Predictions are converted back to % returns before anyone sees them.
  loss      Huber (robust to the fat tails left after clipping)
  regime    VIX, policy uncertainty, credit spread, consumer sentiment on day 0 (FRED)
  news      sentiment features joined on (permno, day0); interactions with the day 0-1 price move
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try:                                   # LightGBM needs libomp on macOS (brew install libomp); fall back if missing
    import lightgbm as lgb; GBM_NAME = "LightGBM"
except Exception:
    lgb = None; GBM_NAME = "sklearn HistGradientBoosting"
    from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import PROC, PUBLIC
from taxonomy import CLASSES

# ---------------------------------------------------------------- load + clean
ev = pd.read_parquet(PROC / "events.parquet")
N_ALL = len(ev)
ev["year"] = ev.day0.dt.year
size_cut = ev.groupby("year").log_mcap.transform(lambda s: s.quantile(0.2))
price_ok = ev["prc_pre"].abs() >= 5 if "prc_pre" in ev else True
UNIVERSE = price_ok & (ev.log_mcap >= size_cut)
ev = ev[UNIVERSE].copy()
N_UNIVERSE = len(ev)

# ---------------------------------------------------------------- regime features (FRED), value known at the close of day 0
REGIME = []
fred_f = PUBLIC / "fred_daily.csv"
if fred_f.exists():
    fr = pd.read_csv(fred_f, parse_dates=["date"]).sort_values("date")
    if "umcsent" in fr:                                     # monthly survey is published with a lag: use last month's value
        fr["umcsent"] = fr.umcsent.shift(31)
    ev = pd.merge_asof(ev.sort_values("day0"), fr, left_on="day0", right_on="date", direction="backward").drop(columns="date")
    REGIME = [c for c in ("vix", "epu", "hy_oas", "umcsent") if c in ev]

# ---------------------------------------------------------------- news sentiment (s03b), joined on a stable key
SENT = ["news_n_pre", "news_n_01", "news_abn", "sent_pre", "sent_01", "sent_neg_share", "sent_min_01", "sent_change",
        "stale_share", "relevant_share"]
sf = PROC / "sentiment_features.parquet"
HAVE_SENT = False
if sf.exists():
    s = pd.read_parquet(sf)
    if {"permno", "day0"}.issubset(s.columns):
        ev = ev.merge(s, on=["permno", "day0"], how="left")
        ev["has_news"] = ev.news_n_01.notna().astype(int)
        for c in ("news_n_pre", "news_n_01"):
            ev[c] = ev[c].fillna(0)
        ev["sent_x_car"] = ev.sent_01 * ev.car_0_1                         # agreement between tone and price move
        ev["disagree"] = (np.sign(ev.sent_01) * np.sign(ev.car_0_1) < 0).astype(float).where(ev.sent_01.notna())
        SENT = [c for c in SENT if c in ev] + ["sent_x_car", "disagree"]
        HAVE_SENT = True

# extra event-level features (s02c): range-based volatility, attention, reaction-vs-surprise
EXTRA_F = []
xf = PROC / "extra_features.parquet"
if xf.exists():
    x = pd.read_parquet(xf)
    ev = ev.merge(x, on=["permno", "day0"], how="left")
    EXTRA_F = [c for c in x.columns if c not in ("permno", "day0")]
# round 4: peer graph + short interest (s02d) and embedding / lexicon / analogue text features (s03f)
NEW4 = []
for _f in ("peer_features.parquet", "text_features.parquet"):
    if (PROC / _f).exists():
        x = pd.read_parquet(PROC / _f).drop_duplicates(["permno", "day0"])
        x["permno"] = x.permno.astype(ev.permno.dtype)
        ev = ev.merge(x, on=["permno", "day0"], how="left")
        NEW4 += [c for c in x.columns if c not in ("permno", "day0")]
# supervised news sentiment learned from our own returns (s03e, out of sample by year)
xs = PROC / "sestm_features.parquet"
if xs.exists() and HAVE_SENT:
    x = pd.read_parquet(xs)
    ev = ev.merge(x, on=["permno", "day0"], how="left")
    ev["sestm_x_car"] = ev.sestm_react_01 * np.sign(ev.car_0_1)          # text agrees (+) or disagrees (-) with the price move
    _car_z = ev.car_0_1 / ((ev.idio_vol_pre / np.sqrt(252)).clip(lower=0.003) * np.sqrt(2))
    ev["sestm_gap"] = ev.sestm_react_01 - np.tanh(_car_z / 2) / 2                   # text tone minus price tone, both on a -0.5..0.5 scale
    ev["sestm_x_attn"] = ev.sestm_react_01 / np.log1p(ev.get("analyst_cov", pd.Series(0, index=ev.index)).fillna(0) + 1)   # tone where few analysts watch
    SENT = SENT + [c for c in ("txt_novelty", "txt_dispersion", "lex_uncertain", "lex_litig", "lex_fear", "lex_anger", "lex_excite",
                               "ana_trough", "ana_logvr", "ana_drift5", "ana_sim") if c in ev] + ["sestm_react_01", "sestm_drift_01", "sestm_react_pre", "sestm_n", "sestm_x_car", "sestm_gap", "sestm_x_attn"]
ev = ev.sort_values("day0").reset_index(drop=True)
for c in CLASSES:
    ev[f"cls_{c}"] = (ev.cls == c).astype(int)

# ---------------------------------------------------------------- features
FEATS = ["car_0_1", "ar_day0_sd", "abn_volume_0_1", "beta", "idio_vol_pre", "vol_pre", "mom_20", "mom_60", "mom_250",
         "dist_52w_high", "alpha_pre", "amihud", "log_dollar_vol", "log_mcap", "spread", "mkt_day0", "mkt_pre5", "mkt_vol20",
         "ld_smb", "ld_hml", "ld_rmw", "ld_cma", "ld_umd",
         "n_events", "news_30d", "scandal_word", "ceo_involved", "exec_departure", "forced_exit", "gov_involved",
         "log_claim", "activist_stake", "restate_adverse", "rec_net", "ceo_tenure", "ceo_age", "ceo_own_pct", "long_tenure_ceo",
         "yz_vol_5", "yz_vol_20", "yz_vol_60", "yz_term", "range_shock_01", "n_same_day_earn", "friday", "react_per_sue", "sue_react_clash",
         "peer_car_01", "own_vs_peer", "peer_pre20", "peer_event", "peer_corr", "si_pre", "si_chg",
         "eps_sue", "eps_surp_rel", "eps_beat", "est_dispersion", "n_analysts", "analyst_cov"] \
        + [f"has_{c}" for c in CLASSES] + [f"cls_{c}" for c in CLASSES] + REGIME
FEATS = [f for f in FEATS if f in ev]
# Live mode: keep only what can be rebuilt from free live data (daily OHLCV for the stock and SPY, FRED, event class
# from headlines). Factor loadings, Compustat / I/B/E/S / Audit Analytics / ExecuComp fields, peers and news are dropped.
from config import LIVE
LIVE_OK = {"car_0_1", "ar_day0_sd", "abn_volume_0_1", "beta", "idio_vol_pre", "vol_pre", "mom_20", "mom_60", "mom_250",
           "dist_52w_high", "alpha_pre", "amihud", "log_dollar_vol", "log_mcap", "mkt_day0", "mkt_pre5", "mkt_vol20",
           "yz_vol_5", "yz_vol_20", "yz_vol_60", "yz_term", "range_shock_01", "friday", "scandal_word", "n_events",
           "vix", "epu", "hy_oas", "umcsent"} | {f"cls_{c}" for c in CLASSES} | {f"has_{c}" for c in CLASSES}
if LIVE:
    FEATS = [f for f in FEATS if f in LIVE_OK]
    HAVE_SENT = False
BASE_FEATS = list(FEATS)                       # everything except news
if HAVE_SENT:
    FEATS = FEATS + ["has_news"] + SENT

# ---------------------------------------------------------------- targets
RET_T = {"drift_1_5": 4, "drift_1_20": 19, "drift_1_60": 59, "trough_2_20": 19}   # horizon in trading days
LOG_T = ["vol_ratio_2_20", "abn_volume_2_20"]
TARGETS = ["drift_1_5", "drift_1_20", "drift_1_60", "vol_ratio_2_20", "abn_volume_2_20", "trough_2_20"]
for t in RET_T:                                                     # tame extreme outliers (data errors, tiny stocks)
    lo, hi = ev[t].quantile([0.005, 0.995]); ev[t] = ev[t].clip(lo, hi)
for t in LOG_T:
    ev[t] = np.log(ev[t].clip(lower=0.05))
ev["month"] = ev.day0.dt.to_period("M")
SD_DAY = (ev.idio_vol_pre / np.sqrt(252)).clip(lower=0.003)          # daily idiosyncratic sd, floored
SCALE = {t: SD_DAY * np.sqrt(h) for t, h in RET_T.items()}
MONTH_MEAN = {t: ev.groupby("month")[t].transform("mean") for t in RET_T}

def to_model_target(t):
    """Target actually used for training: demeaned within month and noise-scaled (return targets); log ratios as is."""
    if t in RET_T:
        return (ev[t] - MONTH_MEAN[t]) / SCALE[t]
    return ev[t]

def from_model_pred(t, p, idx=None):
    """Convert a model prediction back to the original unit (fraction return), relative to the average event that month."""
    if t in RET_T:
        s = SCALE[t] if idx is None else SCALE[t].loc[idx]
        return p * s
    return p

def target_relative(t):
    """Realised target relative to the month's average event (what the demeaned model is trying to rank)."""
    return ev[t] - MONTH_MEAN[t] if t in RET_T else ev[t]

# ---------------------------------------------------------------- training weights
# (a) label uniqueness (Lopez de Prado 2018): events at the same firm within 30 days share most of their return window,
#     so each one is down-weighted by 1 / (1 + number of such neighbours) instead of being counted as independent news;
# (b) class balance: analyst rating changes and earnings would otherwise dominate what the model learns; each class gets
#     weight proportional to 1/sqrt(its share), a middle ground between ignoring and fully equalising class sizes.
_d = ev[["permno", "day0"]].copy(); _d["i"] = np.arange(len(ev))
_d = _d.sort_values(["permno", "day0"])
_t = _d.day0.values.astype("datetime64[D]").astype(np.int64); _p = _d.permno.values.astype(np.int64)
_key = _p * 100000 + _t
_nb = np.searchsorted(_key, _key + 30, "right") - np.searchsorted(_key, _key - 30, "left") - 1
UNIQ = pd.Series(0.0, index=ev.index); UNIQ.iloc[_d.i.values] = 1.0 / (1.0 + _nb)
_share = ev.cls.map(ev.cls.value_counts(normalize=True))
W = UNIQ * (1.0 / np.sqrt(_share))
W = W / W.mean()
USE_WEIGHTS = True
DECAY_HL = None            # years; if set, older training events are down-weighted: w *= 0.5 ** (age / DECAY_HL)

def wts(tr, dates):
    """Training weights for the rows in mask tr (uniqueness x class balance, optionally x time decay)."""
    w = W[tr].values
    if DECAY_HL:
        age = (dates[tr].max() - dates[tr]).dt.days.values / 365.25
        w = w * 0.5 ** (age / DECAY_HL)
    return w

def _fit(m, X, y, w=None):
    if w is None or not USE_WEIGHTS:
        return m.fit(X, y)
    if hasattr(m, "steps"):                             # sklearn Pipeline: weights go to the final estimator only
        return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m.fit(X, y, sample_weight=w)

# ---------------------------------------------------------------- folds + models
FOLDS = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]
SENT_FOLDS = [(2014, 2016), (2017, 2019), (2020, 2023)]            # only years where news coverage exists
PURGE = pd.Timedelta(days=120)                                      # > 60 trading days so windows never overlap

PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10)
GRID = [dict(), dict(num_leaves=15, min_child_samples=500), dict(num_leaves=63, min_child_samples=100, n_estimators=400),
        dict(learning_rate=0.01, n_estimators=1000)]

def gbm(objective="huber", **over):
    p = {**PARAMS, **over}
    if lgb is None:
        return HistGradientBoostingRegressor(max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                                             max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"],
                                             l2_regularization=p["reg_lambda"], random_state=0)
    extra = {"alpha": 1.0} if objective == "huber" else {}
    return lgb.LGBMRegressor(objective=objective, verbose=-1, **extra, **p)

def gbm_quantile(q, **over):
    p = {**PARAMS, **over}
    if lgb is None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                                             max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"])
    return lgb.LGBMRegressor(objective="quantile", alpha=q, verbose=-1, **p)

def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))

def clf():
    if lgb is None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=10)
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1)

def splits(dates, y, folds=FOLDS, min_year=None):
    """Yield (train_mask, test_mask): train strictly before the test block, minus the purge gap."""
    for a, b in folds:
        te = (dates.dt.year >= a) & (dates.dt.year <= b) & y.notna()
        if not te.any():
            continue
        tr = (dates < dates[te].min() - PURGE) & y.notna()
        if min_year is not None:
            tr &= dates.dt.year >= min_year
        if tr.sum() >= 1000:
            yield tr, te

def walk_forward(fn, X, y, dates, folds=FOLDS, min_year=None):
    pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(dates, y, folds, min_year):
        pred[te] = _fit(fn(), X[tr], y[tr], wts(tr, dates)).predict(X[te])
    return pred

def tune(X, y, dates, score_fn):
    """Pick hyperparameters using only data before each test block: inner validation = last 3 years of the training window."""
    best = {}
    for k, (tr, te) in enumerate(splits(dates, y)):
        last = dates[tr].max()
        inner_va = tr & (dates > last - pd.Timedelta(days=3 * 365))
        inner_tr = tr & (dates < last - pd.Timedelta(days=3 * 365) - PURGE)
        if inner_tr.sum() < 1000 or inner_va.sum() < 500:
            best[k] = {}; continue
        sc = []
        for g in GRID:
            p = pd.Series(_fit(gbm(**g), X[inner_tr], y[inner_tr], wts(inner_tr, dates)).predict(X[inner_va]), index=y[inner_va].index)
            sc.append(score_fn(y[inner_va], p))
        best[k] = GRID[int(np.nanargmax(sc))]
    return best

def walk_forward_tuned(X, y, dates, best):
    pred = pd.Series(np.nan, index=y.index)
    for k, (tr, te) in enumerate(splits(dates, y)):
        pred[te] = _fit(gbm(**best.get(k, {})), X[tr], y[tr], wts(tr, dates)).predict(X[te])
    return pred
