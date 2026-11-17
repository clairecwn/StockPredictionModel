"""
Live engine: rebuilds the live-mode features with the SAME definitions as training, and turns the live models into
signals, risk numbers and position sizes. Used by board.py (hourly) and parity_check.py (checks the rebuild
against the training data). Research tool, not investment advice.

Frame format (one stock): DataFrame indexed by trading date with columns o, h, l, c (prices), vol (shares),
ret (total return), mkt (market return: SPY live, CRSP value-weighted index in the parity check).
"""
import os
import numpy as np, pandas as pd, joblib
from config import EST_WINDOW, MODELS
from taxonomy import CLASSES, classify_headline, is_scandal

KMAX = 15
H_RET = {"drift_1_5": 4, "drift_1_20": 19, "drift_1_60": 59, "trough_2_20": 19}

# data
def alpaca_bars(symbols, years=6):
    """Daily split/dividend-adjusted bars from Alpaca's free market data (consolidated SIP feed, which the free plan
    allows once data is 15+ minutes old). Needs ALPACA_KEY_ID and ALPACA_SECRET. Returns {symbol: DataFrame}."""
    import requests
    kid, sec = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET")
    if not kid or not sec:
        return {}
    end = (pd.Timestamp.utcnow() - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=365 * years + 10)).strftime("%Y-%m-%d")
    rows, tok = {}, None
    for _ in range(50):
        q = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": end, "adjustment": "all",
             "feed": "sip", "limit": 10000}
        if tok: q["page_token"] = tok
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars", params=q, timeout=60,
                         headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
        if r.status_code != 200:
            print(f"  Alpaca: HTTP {r.status_code} {r.text[:120]}"); break
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)
        tok = js.get("next_page_token")
        if not tok: break
    out = {}
    for sym, b in rows.items():
        d = pd.DataFrame(b)
        d.index = pd.to_datetime(d.t).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        out[sym] = d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
    return out

def yahoo_frames(tickers, period="6y"):
    """Daily OHLCV for each ticker plus SPY as the market, from Yahoo Finance (split/dividend adjusted)."""
    import yfinance as yf
    ymap = {t: t.replace(".", "-") for t in list(tickers) + ["SPY"]}             # Yahoo writes BRK.B as BRK-B
    d = yf.download(list(ymap.values()), period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    d.columns = d.columns.set_levels([{v: k for k, v in ymap.items()}.get(x, x) for x in d.columns.levels[0]], level=0)
    missing = [t for t in list(tickers) + ["SPY"] if t not in d.columns.get_level_values(0) or d[t]["Close"].notna().sum() < 300]
    alp = alpaca_bars(missing) if missing else {}
    if missing:
        print(f"  Yahoo returned no usable prices for {missing}; Alpaca filled {sorted(alp)}")
    spy_c = alp["SPY"]["Close"] if "SPY" in alp else d["SPY"]["Close"]
    spy = spy_c.pct_change()
    now_et = pd.Timestamp.now(tz="America/New_York")
    closed = now_et.hour * 60 + now_et.minute >= 16 * 60 + 15        # bars are final ~15 min after the 4pm close
    out = {}
    for t in tickers:
        try:
            x = d[t].dropna(subset=["Close"])
        except KeyError:
            x = pd.DataFrame()
        if len(x) < 300 and t in alp:
            x = alp[t]
        for attempt in range(3):                                    # Yahoo sometimes drops a ticker from a batch: retry it alone
            if len(x) >= 300:
                break
            import time; time.sleep(3 * (attempt + 1))
            try:
                x = yf.download(ymap[t], period=period, auto_adjust=True, progress=False)
                if isinstance(x.columns, pd.MultiIndex):
                    x.columns = x.columns.get_level_values(0)
                x = x.dropna(subset=["Close"])
            except Exception:
                x = pd.DataFrame()
        if len(x) == 0:
            continue
        if len(x) and x.index[-1].date() == now_et.date() and not closed:
            x = x.iloc[:-1]                                         # training features use closing values only
        f = pd.DataFrame({"o": x.Open, "h": x.High, "l": x.Low, "c": x.Close, "vol": x.Volume,
                          "ret": x.Close.pct_change(), "mkt": spy.reindex(x.index)})
        out[t] = f.dropna(subset=["c"])
    return out

def fred_row(day, fred_csv):
    """Regime values known at `day` (same lags as training: consumer sentiment shifted 31 rows)."""
    try:
        fr = pd.read_csv(fred_csv, parse_dates=["date"]).sort_values("date")
        if "umcsent" in fr: fr["umcsent"] = fr.umcsent.shift(31)
        fr = fr.ffill()
        row = fr[fr.date <= pd.Timestamp(day)].iloc[-1]
        return {c: float(row[c]) for c in ("vix", "epu", "hy_oas", "umcsent") if c in row and pd.notna(row[c])}
    except Exception:
        return {}

# range-based volatility (same as 04_event_features)
def _yz_series(f):
    ov = np.log(f.o / f.c.shift(1)); oc = np.log(f.c / f.o)
    rs = np.log(f.h / f.c) * np.log(f.h / f.o) + np.log(f.l / f.c) * np.log(f.l / f.o)
    pk = np.log(f.h / f.l) ** 2 / (4 * np.log(2))
    for s in (ov, oc, rs, pk):
        s[s.abs() > 1] = np.nan
    out = {}
    for n in (5, 20, 60):
        k = 0.34 / (1.34 + (n + 1) / (n - 1)); mp = max(3, int(0.75 * n))
        v = ov.rolling(n, min_periods=mp).var() + k * oc.rolling(n, min_periods=mp).var() + (1 - k) * rs.rolling(n, min_periods=mp).mean()
        out[n] = np.sqrt(v.clip(lower=0) * 252).shift(1)            # window ends on the previous day
    return out, pk

# event features (same as 02_build_events + 04_event_features, market model)
def event_features(f, i0, cls="other", headline="", log_mcap=np.nan, regime=None):
    """Features known at the close of day 1 for an event whose day 0 is row i0 of frame f."""
    n = len(f); r_all, m_all, v_all, p_all = f.ret.values, f.mkt.values, f.vol.values.astype(float), f.c.values
    lo, hi = i0 + EST_WINDOW[0], i0 + EST_WINDOW[1]
    if lo < 0:
        raise ValueError("need ~250 trading days of history before the event")
    r, m = r_all[lo:hi], m_all[lo:hi]; ok = np.isfinite(r) & np.isfinite(m)
    beta, alpha = np.polyfit(m[ok], r[ok], 1)
    sd = np.std(r[ok] - alpha - beta * m[ok])
    ar = np.nan_to_num(r_all - beta * m_all)
    vol_est = np.nanmean(v_all[lo:hi])
    d1 = i0 + 1 < n
    x = {"car_0_1": ar[i0] + (ar[i0 + 1] if d1 else 0.0), "ar_day0_sd": ar[i0] / sd if sd > 0 else np.nan,
         "abn_volume_0_1": np.nanmean(v_all[i0:i0 + 2]) / vol_est, "beta": beta, "alpha_pre": alpha * 252,
         "idio_vol_pre": sd * np.sqrt(252), "vol_pre": np.nanstd(r_all[i0 - 60:i0]) * np.sqrt(252)}
    for k in (20, 60, 250):
        x[f"mom_{k}"] = np.nansum(r_all[max(0, i0 - k):i0])
    x["dist_52w_high"] = p_all[i0 - 1] / np.nanmax(p_all[max(0, i0 - 252):i0]) - 1
    dv = p_all[i0 - 60:i0] * v_all[i0 - 60:i0]
    x["amihud"] = np.nanmean(np.abs(r_all[i0 - 60:i0]) / np.where(dv > 0, dv, np.nan)) * 1e6
    x["log_dollar_vol"] = np.log1p(np.nanmean(dv)); x["log_mcap"] = log_mcap
    # experiment 5: pre-event liquidity and trading activity (same windows as 07_spillover_features; volume ratios equal turnover
    # ratios while shares outstanding are unchanged)
    lg = lambda a: np.log(a) if np.isfinite(a) and a > 0 else np.nan
    v_now, v_base = np.nanmean(v_all[i0 - 20:i0]), np.nanmean(v_all[i0 - 250:i0 - 20])
    x["turn_chg"] = lg(v_now) - lg(v_base)
    x["pre_abvol_5"] = lg(np.nanmean(v_all[i0 - 5:i0])) - lg(v_base)
    dvl = np.abs(p_all[i0 - 250:i0]) * v_all[i0 - 250:i0]
    ill = np.abs(r_all[i0 - 250:i0]) / np.where(dvl > 0, dvl, np.nan) * 1e6
    x["amihud_chg"] = lg(np.nanmean(ill[-20:])) - lg(np.nanmean(ill[:-20]))
    x["zero_ret_60"] = float(np.mean(r_all[i0 - 60:i0] == 0))
    x["mkt_day0"] = m_all[i0]; x["mkt_pre5"] = np.nansum(m_all[i0 - 5:i0]); x["mkt_vol20"] = np.nanstd(m_all[i0 - 20:i0]) * np.sqrt(252)
    yz, pk = _yz_series(f)
    for nn in (5, 20, 60):
        x[f"yz_vol_{nn}"] = float(yz[nn].iloc[i0])
    x["yz_term"] = np.log(max(x["yz_vol_5"], 0.01) / max(x["yz_vol_60"], 0.01)) if np.isfinite(x["yz_vol_5"]) else np.nan
    pk1 = pk.iloc[i0 + 1] if d1 else pk.iloc[i0]
    x["range_shock_01"] = np.log(max(np.sqrt((pk.iloc[i0] + pk1) / 2 * 252), 0.01) / max(x["yz_vol_20"], 0.02))
    # experiment 9: price patterns (same definitions as 10_price_pattern_features)
    pre20, pre60 = r_all[i0 - 20:i0], r_all[i0 - 60:i0]
    x["max_20"], x["min_20"] = float(np.nanmax(pre20)), float(np.nanmin(pre20))
    sd60 = np.nanstd(pre60); x["skew_60"] = float(np.nanmean(((pre60 - np.nanmean(pre60)) / sd60) ** 3)) if sd60 > 0 else np.nan
    x["rev_5"] = float(np.nansum(np.log1p(np.clip(r_all[i0 - 5:i0], -0.99, None))))
    dd = [i0] + ([i0 + 1] if d1 else [])
    o_, c_ = f.o.values, f.c.values
    intra = float(np.sum([np.log(c_[j] / o_[j]) for j in dd if o_[j] > 0 and c_[j] > 0]))
    tot = float(np.sum(np.log1p(np.clip(r_all[dd], -0.99, None))))
    x["intra_01"], x["ovn_01"] = float(np.clip(intra, -1, 1)), float(np.clip(tot - intra, -1, 1))
    den = abs(tot - intra) + abs(intra); x["ovn_share"] = (tot - intra) / den if den > 0 else np.nan
    lr = pd.Series(np.log1p(np.clip(r_all[:i0], -0.99, None)), index=f.index[:i0])
    mon = lr.groupby([lr.index.year, lr.index.month]).sum()
    yr, mo = f.index[i0].year, f.index[i0].month
    past = [mon.get((yr - L, mo)) for L in range(1, 6)]; past = [p_ for p_ in past if p_ is not None]
    x["seas_5y"] = float(np.mean(past)) if len(past) >= 2 else np.nan
    lgv = lambda a, b: float(np.log(max(a, 1e-4) / max(b, 1e-4))) if np.isfinite(a) and np.isfinite(b) else np.nan
    x["lv_long"], x["lv_5"] = lgv(x["idio_vol_pre"], x["vol_pre"]), lgv(x["yz_vol_5"], x["vol_pre"])
    x["lv_20"], x["lv_range"] = lgv(x["yz_vol_20"], x["vol_pre"]), lgv(x["yz_vol_60"], x["vol_pre"])
    x["friday"] = int(f.index[i0].weekday() == 4)
    x["scandal_word"] = int(is_scandal(headline)); x["n_events"] = 1
    for c in CLASSES:
        x[f"has_{c}"] = int(c == cls); x[f"cls_{c}"] = int(c == cls)
    x.update(regime or {})
    x["_ar"], x["_sd_day"] = ar, max(sd, 0.003)
    return x

# calendar features (same rules as 06_calendar_features)
def calendar_features(f, i0, earn_dates):
    """Earnings-calendar and earnings-sensitivity features for an event whose day 0 is row i0 of frame f.
    earn_dates: announcement dates (past and upcoming). Only dates on or before day 0 are used, so the result is what
    an investor could know that day; the expected next report follows the training rule (four quarters back + 364 days,
    else last report + 91 days)."""
    d0 = pd.Timestamp(f.index[i0]).normalize()
    ds = sorted({pd.Timestamp(d).normalize() for d in earn_dates if pd.notna(d)})
    past = [d for d in ds if d <= d0]
    out = {"days_to_exp_earn": np.nan, "days_since_earn": np.nan, "exp_earn_in_win": np.nan, "earn_react_hist": np.nan}
    if not past:
        return out
    prev = past[-1]
    exp = None
    if len(past) >= 4:
        e4 = past[-4] + pd.Timedelta(days=364)
        if e4 > d0 and 45 < (e4 - prev).days < 150:
            exp = e4
    exp = exp or prev + pd.Timedelta(days=91)
    if exp <= d0: exp = d0 + pd.Timedelta(days=1)
    dt = (exp - d0).days
    out.update(days_to_exp_earn=float(np.clip(dt, 0, 120)), days_since_earn=float(np.clip((d0 - prev).days, 0, 120)),
               exp_earn_in_win=float(2 <= dt <= 29))
    # mean |reaction| over the previous 8 earnings days (market-model abnormal return on days 0 and 1)
    r, m = f.ret.values, f.mkt.values; idx = f.index.normalize(); reacts = []
    for d in [d for d in past if d < d0][-8:]:
        j = int(idx.searchsorted(d))
        if j < 250 or j + 1 >= len(f): continue
        lo, hi = j + EST_WINDOW[0], j + EST_WINDOW[1]
        rr, mm = r[lo:hi], m[lo:hi]; ok = np.isfinite(rr) & np.isfinite(mm)
        if ok.sum() < 120: continue
        b = np.polyfit(mm[ok], rr[ok], 1)[0]
        reacts.append(abs((r[j] - b * m[j]) + (r[j + 1] - b * m[j + 1])))
    if len(reacts) >= 2:
        out["earn_react_hist"] = float(np.mean(reacts))
    return out

# daily decision-day features (same as 19_entry_engine put() block)
def daily_features(f, i0, k, ev, sr=None, vix_now=None, vix0=None):
    """Features at the close of decision day k (1..15) after day 0 at row i0. sr = FINRA short-sale ratio series (optional)."""
    ar = ev["_ar"]; sdd = max(ev["idio_vol_pre"] / np.sqrt(252), 0.003)
    path = np.concatenate([[0.0], np.cumsum(ar[i0 + 2:i0 + 1 + k])])          # cum[0] = day 1 close
    ck = path[k - 1]; runlow = path[:k].min(); argl = int(path[:k].argmin()) + 1
    v = f.vol.values.astype(float); pre_vol_med = np.nanmedian(v[max(0, i0 - 60):i0 - 10])
    j = i0 + k
    d = {"k": k, "cum_k": ck, "cum_k_z": ck / (sdd * np.sqrt(k)), "above_low": ck - runlow,
         "above_low_z": (ck - runlow) / (sdd * np.sqrt(k)), "days_since_low": k - argl, "car01_plus_cum": ev["car_0_1"] + ck,
         "ar_k": ar[j] if k >= 2 else np.nan}
    d["ar_k_z"] = d["ar_k"] / sdd if k >= 2 else np.nan
    d["ar_last3_z"] = (ck - (path[k - 4] if k >= 4 else 0.0)) / (sdd * np.sqrt(3))
    d["path_vol_ratio"] = np.std(ar[i0 + 2:i0 + 1 + k]) / sdd if k >= 3 else np.nan
    m = f.mkt.values
    d["mkt_k"] = m[j]; d["mkt_cum_k"] = np.nansum(m[i0 + 2:j + 1]) if k >= 2 else 0.0
    d["vol_rel_k"] = np.log1p(v[j]) - np.log1p(pre_vol_med); d["vol_decay_k"] = np.log1p(v[j]) - np.log1p(v[i0])
    d["vol_rel3"] = np.log1p(np.nanmean(v[max(i0, j - 2):j + 1])) - np.log1p(pre_vol_med)
    h, l, c, o = f.h.values[j], f.l.values[j], f.c.values[j], f.o.values[j]
    d["clv_k"] = (c - l) / (h - l) if h > l else 0.5
    d["range_k_z"] = ((h - l) / c) / sdd
    d["gap_k"] = float(np.clip(o / f.c.values[j - 1] - 1, -0.5, 0.5))
    if sr is not None:
        pre_sr = sr.iloc[max(0, i0 - 30):i0 - 5].mean()
        s_prev = sr.iloc[j - 1]; s3 = sr.iloc[max(0, j - 3):j].mean()
        d["sr_dev_k"] = s_prev - pre_sr; d["sr3_dev"] = s3 - pre_sr; d["sr_day0_dev"] = sr.iloc[i0] - pre_sr
    if vix_now is not None:
        d["vix_k"] = vix_now; d["vix_chg_k"] = vix_now - (vix0 if vix0 is not None else vix_now)
    return d

# event detection
def detect_event(f, headlines=None, earn_days=(), lookback=KMAX):
    """Most recent event in the last `lookback` trading days, with the training convention day0 = announcement day
    (after-close news keeps its own date; the price reaction shows on day 1, which car_0_1 covers).
    Priority: a scheduled earnings date (class earnings_guidance) > a classified headline about the company >
    an unexplained price/volume shock (abnormal return beyond 3 sd or volume above 3x normal). A shock whose news came
    the evening before is dated to that news day. Unexplained shocks get class 'other', which training never saw:
    the page says so.
    headlines: DataFrame with columns day (New York calendar date), headline, relevant (mentions the company).
    Returns (row index of day 0, class, headline) or None."""
    n = len(f)
    if n < 300:
        return None
    idx = f.index.normalize()
    def tday(d):                                                   # calendar date -> trading day on or after it
        j = int(idx.searchsorted(pd.Timestamp(d).normalize()))
        return j if j < n else None
    news = {}
    if headlines is not None and len(headlines):
        for r in headlines[headlines.relevant].itertuples():
            j = tday(r.day)
            if j is None:
                continue
            c = classify_headline(r.headline)
            if j not in news or (news[j][0] == "other" and c != "other"):
                news[j] = (c, r.headline)
    earn = {j for j in (tday(d) for d in earn_days) if j is not None}
    hl = lambda j: news.get(j, ("", ""))[1]
    def move(i):                                                   # abnormal return in sd and volume vs normal on row i
        lo, hi = i + EST_WINDOW[0], i + EST_WINDOW[1]
        r, m = f.ret.values[lo:hi], f.mkt.values[lo:hi]; ok = np.isfinite(r) & np.isfinite(m)
        b, a = np.polyfit(m[ok], r[ok], 1); sd = np.std(r[ok] - a - b * m[ok])
        return (f.ret.values[i] - b * f.mkt.values[i]) / sd, f.vol.values[i] / np.nanmean(f.vol.values[lo:hi])
    def reacted(i):                                                # did the market treat the headline as news? (day 0 or 1)
        for j in (i, i + 1):
            if j < n:
                z, vr = move(j)
                if abs(z) > 2 or vr > 2:
                    return True
        return False
    for i0 in range(n - 1, max(n - 1 - lookback, 260), -1):
        if i0 in earn:
            return i0, "earnings_guidance", hl(i0)
        # Live headline feeds carry commentary and previews ("braces for earnings", "stock will beat the market"),
        # which the training news database did not. A headline counts only if the stock reacted to it.
        # An earnings headline counts only on the report date itself or the day after (the calendar has those dates);
        # anywhere else it is a preview or commentary ("earnings to test AI boom").
        if i0 in news and news[i0][0] == "earnings_guidance" and not (i0 in earn or i0 - 1 in earn):
            pass
        elif i0 in news and news[i0][0] != "other" and reacted(i0):
            return i0, news[i0][0], news[i0][1]
        z, vr = move(i0)
        if abs(z) > 3 or vr > 3:
            if i0 - 1 in earn:
                return i0 - 1, "earnings_guidance", hl(i0 - 1)
            if i0 - 1 in news and news[i0 - 1][0] != "other":
                return i0 - 1, news[i0 - 1][0], news[i0 - 1][1]
            return i0, "other", hl(i0)
    return None

# predictions
from functools import lru_cache
@lru_cache(maxsize=None)                                   # each model file is read once per run, not once per stock
def _load(name):
    p = MODELS / name
    return joblib.load(p) if p.exists() else None

def contributions(model, X, top=6):
    """The inputs that moved this one forecast most (LightGBM's per-prediction contributions, in the model's units).
    Returns [(feature, contribution, value)], largest first."""
    try:
        c = model.predict(X, pred_contrib=True)[0][:-1]                # the last column is the starting value
        order = np.argsort(-np.abs(c))[:top]
        return [(str(X.columns[i]), float(c[i]), float(X.iloc[0, i])) for i in order if np.isfinite(c[i]) and abs(c[i]) > 0]
    except Exception:
        return []


def predict_all(f, i0, ev, sr=None, vix_now=None, vix0=None):
    """All live outputs for one stock. Returns a dict that goes straight into predictions.json."""
    out = {"day0": str(f.index[i0].date()), "car_0_1_pct": 100 * ev["car_0_1"]}
    k = len(f) - 1 - i0                        # decision day = trading days since day 0 (today = close of day k)
    out["decision_day"] = int(k)
    X1 = lambda feats, extra=None: pd.DataFrame([{c: {**ev, **(extra or {})}.get(c, np.nan) for c in feats}]).astype(float)
    sd_day = max(ev["idio_vol_pre"] / np.sqrt(252), 0.003)
    risk = {}
    for t in ("drift_1_20", "trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"):
        mdl = _load(f"{t}.joblib")
        if mdl is None: continue
        X = X1(mdl["feats"]); sc = sd_day * np.sqrt(H_RET[t]) if mdl.get("return_target") else 1.0
        p = float(mdl["model"].predict(X)[0]) * sc
        if "cqr_model" in mdl:                                       # held-out correction, in the model's own units
            lo_ = (float(mdl["q_lo"].predict(X)[0]) - mdl["cqr_model"]) * sc; hi_ = (float(mdl["q_hi"].predict(X)[0]) + mdl["cqr_model"]) * sc
        else:                                                         # older files: correction stored in % units
            lo_ = float(mdl["q_lo"].predict(X)[0]) * sc - mdl["cqr"]; hi_ = float(mdl["q_hi"].predict(X)[0]) * sc + mdl["cqr"]
        if t in ("vol_ratio_2_20", "abn_volume_2_20"):
            p, lo_, hi_ = np.exp(p), np.exp(lo_), np.exp(hi_)
        risk[t] = {"pred": p, "lo80": lo_, "hi80": hi_}
        if t == "trough_2_20":
            risk[t]["drivers"] = contributions(mdl["model"], X)
    out["risk"] = risk
    eng = _load("entry_engine.joblib")
    if eng is not None and 1 <= k <= KMAX:
        dyn = daily_features(f, i0, k, ev, sr, vix_now, vix0)
        X = X1(eng["feats"], dyn)
        fwd = float(eng["fwd_reg"].predict(X)[0])
        dip = float(eng["dip_reg"].predict(X)[0]) * sd_day * np.sqrt(max(20 - k, 1))
        out["entry"] = {"signal": "BUY" if fwd > eng["fwd_threshold"] else "WAIT", "expected_next5_sd": fwd,
                        "expected_further_dip_pct": 100 * max(dip, 0), "at_new_low": bool(dyn["above_low"] <= 0.25 * sd_day),
                        "drivers": contributions(eng["fwd_reg"], X)}
        if "ttl_clf" in eng:                                         # chance the 20-day low comes in each window of days
            m = eng["ttl_clf"]; pr = np.zeros(len(eng["ttl_labels"])); pr[m.classes_.astype(int)] = m.predict_proba(X)[0]
            if 20 - k < 11: pr[4] = 0.0                              # windows that run past day 20 cannot happen
            if 20 - k < 6: pr[3] = 0.0
            pr = pr / pr.sum()
            out["entry"]["days_to_low"] = {"labels": eng["ttl_labels"], "probs": [float(x) for x in pr],
                                           "beats_baseline": bool(eng["ttl_t"] > 2), "days_left": int(20 - k)}
        if out["entry"]["at_new_low"] and "low_clf" in eng:
            out["entry"]["p_low_holds"] = float(eng["low_clf"].predict_proba(X)[0, 1])
    elif k > KMAX:
        out["entry"] = {"signal": "WINDOW OVER", "note": f"day {k} after the event; the timing models cover days 1-{KMAX}"}
    sz = _load("sizing.joblib")
    if sz is not None:
        X = X1(sz["feats"]); H = sz["horizon_days"]
        sigma = max(ev["yz_vol_20"], 0.05) * float(np.exp(np.clip(np.log(risk.get("vol_ratio_2_20", {"pred": 1.0})["pred"]), -2, 2))) \
                * np.sqrt(H / 252) * sz.get("kappa_latest", 1.0)
        z = (risk.get("drift_1_20", {"pred": 0.0})["pred"]) / max(sz.get("drift_pred_sd", 1.0), 1e-6)
        mu = max(min(ev["beta"], 3), 0) * sz["erp_year"] * H / 252 + sz["ic_latest"] * ev["idio_vol_pre"] * np.sqrt(H / 252) * z
        tiers = {}
        for name, t in sz["tiers"].items():
            off = sz["aci_offset"][t["beta"]]
            if "mondrian_offset" in sz and np.isfinite(ev["idio_vol_pre"]):        # per-volatility-bucket calibration (22_sizing_calibration)
                off = sz["mondrian_offset"][t["beta"]][int(np.searchsorted(sz["mondrian_edges"], ev["idio_vol_pre"]))]
            q = float(sz["q_models"][t["beta"]].predict(X)[0]) - off
            q = min(q, -0.005)
            w = min(t["d"] / abs(q), t["k"] * mu / sigma ** 2 if mu > 0 else 0.0, t["cap"])
            tiers[name] = {"weight": max(w, 0.0), "bad_case_pct": 100 * q, "loss_budget_pct": 100 * t["d"], "chance": t["beta"]}
        out["sizing"] = {"tiers": tiers, "mu_pct": 100 * mu, "sigma_pct": 100 * sigma}
        thr = sz.get("trough_q20_shock", sz.get("trough_q20"))     # riskiest 20% among past shock-sized events
        if thr is not None and "trough_2_20" in risk:
            out["warning_big_drop"] = bool(risk["trough_2_20"]["pred"] <= thr)
    return out


# earnings preview (experiment 3, 03_earnings_preview.py)
D_N = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 12: 3.258, 15: 3.472,
       20: 3.735, 25: 3.931, 30: 4.086}
def analyst_dispersion(low, high, avg, n):
    """Standard deviation of analyst estimates from their range: E[range] = d_n * sd for n normal draws (d_n interpolated),
    then scaled as in training: sd / max(|mean|, 0.05), capped at 5."""
    try:
        n = int(n)
        if n < 2 or not np.isfinite(low) or not np.isfinite(high): return np.nan
        ks = sorted(D_N); d = float(np.interp(min(n, 30), ks, [D_N[k] for k in ks]))
        return float(min((high - low) / d / max(abs(avg), 0.05), 5.0))
    except Exception:
        return np.nan

def earnings_preview(f, report_date, earn_dates, regime=None, log_mcap=np.nan, extra=None):
    """Expected size of the two-day reaction to an upcoming report, from features known at the latest close.
    A placeholder row dated on the report day is appended so the event-feature code sees 'day 0' after today's close;
    every feature it uses ends on the previous row, so nothing about the report itself is used."""
    pv = _load("earnings_preview.joblib")
    if pv is None or "move" not in pv:
        return None
    g = f.copy()
    d0 = pd.Timestamp(report_date).normalize()
    if d0 <= g.index[-1]:
        return None
    g.loc[d0] = {"o": np.nan, "h": np.nan, "l": np.nan, "c": g.c.iloc[-1], "vol": np.nan, "ret": 0.0, "mkt": 0.0}
    i0 = len(g) - 1
    x = event_features(g, i0, "earnings_guidance", "", log_mcap, regime)
    x.update(calendar_features(g, i0, earn_dates))
    x.update(extra or {})
    X = pd.DataFrame([{c: x.get(c, np.nan) for c in pv["feats"]}]).astype(float)
    move = float(np.exp(pv["move"].predict(X)[0]) * pv["smear"] - 0.002)
    upper = float(np.exp(pv["q90"].predict(X)[0] + pv["q90_offset"]) - 0.002)
    out = {"report_date": str(d0.date()), "typical_move_pct": 100 * move, "upper_move_pct": 100 * max(upper, move),
           "usual_move_pct": 100 * x["earn_react_hist"] if np.isfinite(x.get("earn_react_hist", np.nan)) else None}
    if "dd" in pv:
        sd = max(x["idio_vol_pre"] / np.sqrt(252), 0.003)
        out["drawdown_pct"] = float(100 * pv["dd"].predict(X)[0] * sd * np.sqrt(21))
    return out
