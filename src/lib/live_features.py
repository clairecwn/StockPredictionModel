"""
Build the same features 02_build_events.py builds, but from live Yahoo Finance prices for one ticker and one event date.
Used by predict_one.py. Market proxy: SPY (CRSP used the value-weighted index; SPY is the closest free equivalent).
"""
import numpy as np, pandas as pd, yfinance as yf
from config import EST_WINDOW
from taxonomy import CLASSES, is_scandal

def prices(ticker, start="2023-01-01"):
    d = yf.download([ticker, "SPY"], start=start, auto_adjust=True, progress=False)
    px = d["Close"]; vol = d["Volume"]
    out = pd.DataFrame({"prc": px[ticker], "vol": vol[ticker], "ret": px[ticker].pct_change(), "mkt": px["SPY"].pct_change()})
    return out.dropna(subset=["prc"])

def market_cap(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info["market_cap"]) / 1e3            # CRSP cap is in thousands of USD
    except Exception:
        return np.nan

def features(ticker, day0, cls, headline="", n_events=1, news_30d=1, extra=None):
    """Return (features dict, DataFrame of daily rows up to today). day0 = event date (str or Timestamp)."""
    p = prices(ticker)
    day0 = pd.Timestamp(day0)
    idx = p.index.searchsorted(day0)
    if idx >= len(p):
        raise ValueError(f"{day0.date()} is after the last available price {p.index[-1].date()}")
    a = {c: p[c].values.astype("float64") for c in p.columns}
    i0, n = idx, len(p)
    lo, hi = i0 + EST_WINDOW[0], i0 + EST_WINDOW[1]
    if lo < 0:
        raise ValueError("need at least 250 trading days of history before the event")
    r, m = a["ret"][lo:hi], a["mkt"][lo:hi]
    ok = np.isfinite(r) & np.isfinite(m)
    beta, alpha = np.polyfit(m[ok], r[ok], 1)
    sd = np.std(r[ok] - alpha - beta * m[ok])
    ar = np.nan_to_num(a["ret"] - beta * a["mkt"])
    vol_est = np.nanmean(a["vol"][lo:hi])
    have_day1 = i0 + 1 < n
    f = {}
    f["car_0_1"] = ar[i0] + (ar[i0 + 1] if have_day1 else 0.0)
    f["ar_day0_sd"] = ar[i0] / sd
    f["abn_volume_0_1"] = np.nanmean(a["vol"][i0:i0 + 2]) / vol_est
    f["beta"], f["alpha_pre"] = beta, alpha * 252
    f["idio_vol_pre"] = sd * np.sqrt(252)
    f["vol_pre"] = np.nanstd(a["ret"][i0 - 60:i0]) * np.sqrt(252)
    for k in (20, 60, 250):
        f[f"mom_{k}"] = np.nansum(a["ret"][i0 - k:i0])
    f["dist_52w_high"] = a["prc"][i0 - 1] / np.nanmax(a["prc"][i0 - 252:i0]) - 1
    dv = a["prc"][i0 - 60:i0] * a["vol"][i0 - 60:i0]
    f["amihud"] = np.nanmean(np.abs(a["ret"][i0 - 60:i0]) / np.where(dv > 0, dv, np.nan)) * 1e6
    f["log_dollar_vol"] = np.log1p(np.nanmean(dv))
    cap = market_cap(ticker)
    f["log_mcap"] = np.log1p(cap) if np.isfinite(cap) else np.nan
    f["spread"] = np.nan                                   # no bid/ask history from Yahoo
    f["mkt_day0"] = a["mkt"][i0]
    f["mkt_pre5"] = np.nansum(a["mkt"][i0 - 5:i0])
    f["mkt_vol20"] = np.nanstd(a["mkt"][i0 - 20:i0]) * np.sqrt(252)
    f["n_events"], f["news_30d"] = n_events, news_30d
    f["scandal_word"] = int(is_scandal(headline))
    for c in CLASSES:
        f[f"has_{c}"] = int(c == cls); f[f"cls_{c}"] = int(c == cls)
    for c in ("ceo_involved", "exec_departure", "forced_exit", "gov_involved", "log_claim", "activist_stake",
              "restate_adverse", "rec_net", "ceo_tenure", "ceo_age", "ceo_own_pct", "long_tenure_ceo"):
        f[c] = np.nan
    if extra:
        f.update(extra)
    f["_day0"], f["_have_day1"], f["_last_price_date"] = p.index[i0], have_day1, p.index[-1]
    f["_realised_ar_since_day1"] = float(ar[i0 + 2:].sum()) if i0 + 2 < n else 0.0
    return f, p
