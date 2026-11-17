"""Live board. For each stock in watchlist.json: find the most recent event (last 15 trading days), rebuild the
features exactly as in training (see parity_check.py), and write:
  docs/data/predictions.json   the board: one summary per stock (signals, risk, sizing) and compact correlations
  docs/data/stocks/<T>.json    everything else the page shows for one stock (price history, quote, headlines), loaded on demand
  docs/data/forward_log.csv    one row per stock per run, so live accuracy can be scored later (out of sample by construction)
  README.md                    the table between the LIVE-BOARD markers is replaced
Models: models/live/ (trained with LIVE_FEATS=1 on features that free live data can rebuild).
Run: python src/live/board.py          Research only. Not investment advice.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import json, os, io, re, warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests
os.environ.setdefault("LIVE_FEATS", "1")
from config import ROOT, PUBLIC, MODELS
from live_engine import yahoo_frames, fred_row, detect_event, event_features, calendar_features, earnings_preview, predict_all
from sizing_lib import shrunk_corr
from drivers import explain

WL = json.loads((ROOT / "watchlist.json").read_text())
# hand-picked stocks (with headline aliases) first, then the S&P 500 and Nasdaq-100 from universe.json
_U = json.loads((ROOT / "universe.json").read_text())["stocks"] if (ROOT / "universe.json").exists() else []
_have = {w["ticker"] for w in WL["stocks"]}
WL["stocks"] += [u for u in _U if u["ticker"] not in _have]
CORE = _have

def _model_json(n):
    p = MODELS / n
    return json.loads(p.read_text()) if p.exists() else None
# Past odds behind each call: entry_odds.json from the entry engine's test years, quiet_rule.json from Experiment 13.
ODDS, QRULE = _model_json("entry_odds.json"), _model_json("quiet_rule.json")
try:                                                            # Experiment 13: how far a stock without news may dip in 20 days
    import joblib
    QDIP = joblib.load(MODELS / "quiet_dip.joblib") if (MODELS / "quiet_dip.joblib").exists() else None
except Exception:
    QDIP = None
def quiet_features(f):
    """The live dip model's inputs, built the same way as in src/experiments/13_quiet_days.py, from the last day of the frame."""
    r, m = f.ret.values.astype(float), f.mkt.values.astype(float); ar = r - m; v = f.vol.values.astype(float)
    lpx = np.cumsum(np.log1p(np.clip(np.nan_to_num(r), -0.95, 10))); px = np.exp(lpx - lpx.max())
    sd = lambda x: float(np.nanstd(x, ddof=1))
    sd5, sd20, sd60, sd250 = sd(r[-5:]), sd(r[-20:]), sd(r[-60:]), sd(r[-250:])
    au, ad = np.nanmean(np.clip(r[-14:], 0, None)), np.nanmean(np.clip(-r[-14:], 0, None))
    x = {"ar_1": ar[-1], "rev_5": np.nansum(ar[-5:]), "ar_20": np.nansum(ar[-20:]), "ar_60": np.nansum(ar[-60:]),
         "mom_250_20": lpx[-21] - lpx[-251], "hi_52": px[-1] / px[-250:].max() - 1, "lo_52": px[-1] / px[-250:].min() - 1,
         "ma50_gap": px[-1] / px[-50:].mean() - 1, "ma200_gap": px[-1] / px[-200:].mean() - 1,
         "rsi_14": 100 - 100 / (1 + au / ad) if ad > 0 else 100.0, "lv_5_60": np.log(sd5 / sd60), "lv_20_250": np.log(sd20 / sd250),
         "vol_60": sd60 * np.sqrt(252), "abvol_5": np.log(np.nanmean(v[-5:]) / np.nanmean(v[-60:])),
         "max_20": np.nanmax(r[-20:]), "min_20": np.nanmin(r[-20:]), "mkt_5": np.nansum(m[-5:]), "mkt_20": np.nansum(m[-20:]),
         "mkt_vol20": sd(m[-20:]) * np.sqrt(252), "vol_chg": np.log(np.nanmean(v[-20:]) / np.nanmean(v[-250:]))}
    return x, max(sd60, 0.003)
def price_call(item, news_day=False):
    """Buy-or-wait lean from price alone (experiment 13): last week's move against the market, with the past odds.
    The rule was tested on days without news, so on a news day it is only shown as a slight lean until the full call."""
    z = item.get("stats", {}).get("rel_5d_z")
    if not QRULE or z is None or not np.isfinite(z):
        return
    B = QRULE["bins"][int(np.searchsorted(QRULE["cut"], z, side="right"))]
    o = B["up_share"] if B["call"] == "BUY" else 1 - B["up_share"]
    item["pcall"] = {"signal": B["call"], "when": B["when"], "odds": round(float(o), 4),
                     "strength": "slight" if news_day else strength(o, B["holds_2021_on"]), "n": B["n"]}
def strength(odds, holds=True):
    """How far the past record sits from a coin flip. odds = share of similar past cases where the call was right."""
    e = odds - 0.5
    return "slight" if (e < 0.015 or not holds) else "mild" if e < 0.03 else "stronger"
TICKERS = [w["ticker"] for w in WL["stocks"]]
import time
_last = [0.0]
def fh(url, params):
    """Finnhub's free plan allows 60 requests a minute; keep below that."""
    wait = 1.1 - (time.time() - _last[0])
    if wait > 0: time.sleep(wait)
    _last[0] = time.time()
    return requests.get(url, params=params, timeout=20)
DOCS = ROOT / "docs" / "data"; DOCS.mkdir(parents=True, exist_ok=True)
now = datetime.now(timezone.utc)
ET = "America/New_York"

def relevant(headline, w):
    """True if the headline is about this company: its ticker or one of its names, case-sensitive and whole-word, within
    the first 7 words (Finnhub company-news also returns articles that only mention the company in passing)."""
    names = ([w["ticker"]] if len(w["ticker"]) > 2 else []) + [w.get("name", "")] + w.get("aliases", [])  # 2-letter tickers are ordinary words
    early = " ".join(headline.split()[:7])                                # the subject usually comes first; later = passing mention
    return any(re.search(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", early) for n in names if n)

_CAL = None
def _calendar(back=40, ahead=30):
    """All earnings dates in the window from Finnhub's calendar, fetched a week at a time (one sweep for every stock)."""
    global _CAL
    if _CAL is not None: return _CAL
    _CAL = {}
    key = os.environ.get("FINNHUB_API_KEY")
    if not key: return _CAL
    d0 = pd.Timestamp(now - pd.Timedelta(days=back)).normalize()
    while d0 < now + pd.Timedelta(days=ahead):
        d1 = d0 + pd.Timedelta(days=6)
        try:
            r = fh("https://finnhub.io/api/v1/calendar/earnings", {"from": str(d0.date()), "to": str(d1.date()), "token": key})
            for c in (r.json().get("earningsCalendar", []) if r.status_code == 200 else []):
                if c.get("date") and c.get("symbol"): _CAL.setdefault(c["symbol"], set()).add(pd.Timestamp(c["date"]))
        except Exception:
            pass
        d0 = d1 + pd.Timedelta(days=1)
    return _CAL

def earnings_days(t, back=40, ahead=30):
    """Earnings announcement dates from Finnhub's calendar. Returns (past dates as Timestamps, next upcoming date or None).
    Training dated earnings events to the announcement day, so after-close reports keep their own date here too."""
    ds = sorted(_calendar(back, ahead).get(t, set()))
    today = pd.Timestamp(now).tz_convert(ET).tz_localize(None).normalize()
    return [d for d in ds if d < today], next((d for d in ds if d >= today), None)

def yahoo_earnings(t, limit=16):
    """Past and upcoming earnings announcement dates from Yahoo (about four years), as calendar dates in New York time.
    Used for the calendar features; Finnhub's free calendar only reaches a few weeks back."""
    try:
        import yfinance as yf
        d = yf.Ticker(t).get_earnings_dates(limit=limit)
        idx = d.index.tz_convert(ET) if getattr(d.index, "tz", None) is not None else d.index
        return [pd.Timestamp(x.date()) for x in idx]
    except Exception:
        return []

def yahoo_estimates(t):
    """Analyst disagreement for the coming quarter from Yahoo's low / high / average EPS estimate (see analyst_dispersion)."""
    try:
        import yfinance as yf
        from live_engine import analyst_dispersion
        q = yf.Ticker(t).earnings_estimate.loc["0q"]
        return {"est_dispersion": analyst_dispersion(float(q["low"]), float(q["high"]), float(q["avg"]), q["numberOfAnalysts"]),
                "n_analysts": float(q["numberOfAnalysts"])}
    except Exception:
        return {}

def finnhub(t, days=25):
    key = os.environ.get("FINNHUB_API_KEY")
    if not key: return pd.DataFrame(columns=["date", "day", "headline", "url"])
    try:
        r = fh("https://finnhub.io/api/v1/company-news", dict(symbol=t, _from=str((now - pd.Timedelta(days=days)).date()), to=str(now.date()), token=key))
        d = pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        d = pd.DataFrame()
    if d.empty: return pd.DataFrame(columns=["date", "day", "headline", "url"])
    d["date"] = pd.to_datetime(d.datetime, unit="s", utc=True).dt.tz_convert(ET).dt.tz_localize(None)   # New York time
    d["day"] = d.date.dt.normalize()
    return d[["date", "day", "headline", "url"]].sort_values("date", ascending=False)

_finra = {}
def finra_ratio(t, dates):
    """FINRA daily short-sale volume / total volume for the given dates (public files, posted each evening)."""
    vals = []
    for d in dates:
        k = d.strftime("%Y%m%d")
        if k not in _finra:
            try:
                r = requests.get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{k}.txt", timeout=15)
                _finra[k] = pd.read_csv(io.StringIO(r.text), sep="|").set_index("Symbol") if r.status_code == 200 else None
            except Exception:
                _finra[k] = None
        x = _finra[k]
        vals.append(float(x.loc[t, "ShortVolume"] / x.loc[t, "TotalVolume"]) if x is not None and t in x.index and x.loc[t, "TotalVolume"] > 0 else np.nan)
    return pd.Series(vals, index=dates)

frames = yahoo_frames(TICKERS)
try:                                                            # last good board, used if a stock's prices fail to load
    PREV = {x["ticker"]: x for x in json.loads((DOCS / "predictions.json").read_text())["stocks"] if x.get("price") is not None}
except Exception:
    PREV = {}
fred_csv = PUBLIC / "fred_daily.csv"
board, log_rows = [], []
for w in WL["stocks"]:
    t = w["ticker"]; f = frames.get(t)
    item = {"ticker": t, "name": w.get("name", t)}
    if w.get("sector"): item["sector"] = w["sector"]
    if (f is None or len(f) < 300 or float(f.c.iloc[-1]) < 5) and t not in CORE:
        continue                                                # new listings and stocks under $5 are outside what the models cover
    if f is None or len(f) < 300:
        if t in PREV:
            item = {**PREV[t], "stale": True}                  # shown as the last update, not logged as a new prediction
        else:
            item["status"] = "no price data"
        board.append(item); continue
    item["price"] = float(f.c.iloc[-1]); item["price_date"] = str(f.index[-1].date())
    item["chg_1d_pct"] = 100 * float(f.ret.iloc[-1])
    # everyday stock facts shown on the page (from the same daily prices; not model outputs)
    try:
        c, r, m = f.c.values, f.ret.values, f.mkt.values
        ok = np.isfinite(r[-250:]) & np.isfinite(m[-250:])
        beta = float(np.polyfit(m[-250:][ok], r[-250:][ok], 1)[0])
        ma50, ma200 = float(np.mean(c[-50:])), float(np.mean(c[-200:]))
        r63, m63 = float(c[-1] / c[-64] - 1), float(np.prod(1 + np.nan_to_num(m[-63:])) - 1)
        trend = "bullish" if (c[-1] > ma50 > ma200 and r63 > m63) else "bearish" if (c[-1] < ma50 < ma200 and r63 < m63) else "mixed"
        item["stats"] = {"beta": beta, "ret_3m_pct": 100 * r63, "mkt_3m_pct": 100 * m63, "from_high_pct": 100 * float(c[-1] / np.max(c[-252:]) - 1),
                         "above_ma200_pct": 100 * float(c[-1] / ma200 - 1), "trend": trend,
                         "dollar_vol_m": float(np.nanmean(c[-20:] * f.vol.values[-20:])) / 1e6,
                         "vol_20_pct": 100 * float(np.nanstd(r[-20:]) * np.sqrt(252)), "vol_1y_pct": 100 * float(np.nanstd(r[-250:]) * np.sqrt(252)),
                         "low_52w": float(np.min(c[-252:])), "high_52w": float(np.max(c[-252:]))}
        dc = np.diff(c[-15:]); up, dn = dc.clip(min=0).mean(), (-dc).clip(min=0).mean()
        item["stats"]["rsi_14"] = float(100 - 100 / (1 + up / dn)) if dn > 0 else 100.0
        v = f.vol.values; item["stats"]["vol_vs_avg"] = float(np.nanmean(v[-5:]) / np.nanmean(v[-65:-5]))
        rel5, sd60 = float(np.nansum(r[-5:] - m[-5:])), max(float(np.nanstd(r[-60:])), 0.003)
        item["stats"]["rel_5d_pct"] = 100 * rel5                   # last 5 days against the market
        item["stats"]["rel_5d_z"] = rel5 / (sd60 * np.sqrt(5))     # the same in units of the stock's usual 5-day swing
    except Exception as e:
        print(f"stats failed for {t}: {e}")
    H = f.iloc[-260:]
    item["hist"] = {"d": [str(x.date()) for x in H.index], "o": [round(float(x), 2) for x in H.o], "h": [round(float(x), 2) for x in H.h],
                    "l": [round(float(x), 2) for x in H.l], "c": [round(float(x), 2) for x in H.c], "v": [int(x) if np.isfinite(x) else 0 for x in H.vol]}
    try:
        tr = np.maximum(f.h.values[-14:], f.c.values[-15:-1]) - np.minimum(f.l.values[-14:], f.c.values[-15:-1])
        item["quote"] = {"open": float(f.o.iloc[-1]), "high": float(f.h.iloc[-1]), "low": float(f.l.iloc[-1]), "prev_close": float(f.c.iloc[-2]),
                         "volume": float(f.vol.iloc[-1]), "avg_volume_3m": float(np.nanmean(f.vol.values[-64:-1])),
                         "atr_pct": 100 * float(np.mean(tr) / f.c.iloc[-1])}
    except Exception as e:
        print(f"quote failed for {t}: {e}")
    item["spark"] = [round(float(x), 2) for x in f.c.iloc[-60::2]]   # small price line for the watchlist (every other day, 3 months)
    tail = f.c.iloc[-45:]                                       # recent closes for the price chart on the page
    item["closes"] = [round(float(x), 2) for x in tail]; item["close_dates"] = [str(d.date()) for d in tail.index]
    news = finnhub(t)
    news["relevant"] = [relevant(h, w) for h in news.headline] if len(news) else []
    past_earn, next_earn = earnings_days(t)
    item["next_earnings"] = str(next_earn.date()) if next_earn is not None else None
    rel = news[news.relevant] if len(news) else news
    item["headlines"] = [{"date": str(r.date)[:16], "headline": r.headline, "url": r.url} for r in rel.head(6).itertuples()]
    found = None
    if w.get("event_date"):                                    # manual override in watchlist.json
        i0 = f.index.searchsorted(pd.Timestamp(w["event_date"]))
        if i0 < len(f): found = (i0, w.get("event_class", "other"), w.get("event_headline", ""))
    if found is None:
        found = detect_event(f, news, past_earn)
    if found is None:
        item["status"] = "quiet: no event in the last 15 trading days"
        # Outlook without a news event: the event models do not apply, so the risk comes from the stock's own history.
        # Bad case = 10th (and 5th) percentile of its lowest point over 20-day holding windows in the last six years,
        # rescaled by how volatile it is now relative to that period.
        try:
            c = f.c.values; lr = np.log(c)
            W = np.lib.stride_tricks.sliding_window_view(lr, 21)
            low20 = np.exp((W[:, 1:].min(axis=1) - W[:, 0])) - 1
            r = f.ret.values; v_now, v_all = np.nanstd(r[-60:]), np.nanstd(r[-1500:])
            k_ = float(np.clip(v_now / v_all, 0.6, 1.8)) if v_all > 0 else 1.0
            q10, q05 = k_ * float(np.quantile(low20, 0.10)), k_ * float(np.quantile(low20, 0.05))
            tiers = {"safest": (0.02, q05, 0.15), "recommended": (0.04, q10, 0.25), "bold": (0.08, q10, 0.40)}
            item["outlook"] = {"vol_20d_pct": 100 * float(np.nanstd(r[-20:]) * np.sqrt(20)), "bad_case_pct": 100 * q10, "bad_case_5_pct": 100 * q05,
                               "vol_now_vs_usual": float(v_now / v_all) if v_all > 0 else None,
                               "weights": {n: float(min(d / max(abs(q), 1e-3), cap)) for n, (d, q, cap) in tiers.items()}}
        except Exception as e:
            print(f"outlook failed for {t}: {e}")
        if QDIP is not None and "outlook" in item:
            try:
                x, sd60 = quiet_features(f)
                pred = float(QDIP["model"].predict(pd.DataFrame([x])[QDIP["features"]])[0])
                item["outlook"]["dip_pct"] = max(0.0, -pred * sd60 * np.sqrt(20) * 100)   # expected deepest fall against the market
                Xq = pd.DataFrame([x])[QDIP["features"]]
                from live_engine import contributions
                item["why"] = {"kind": "dip", "items": explain(contributions(QDIP["model"], Xq), quiet=True)}
            except Exception as e:
                print(f"quiet dip failed for {t}: {e}")
        price_call(item)
        if next_earn is not None and 0 < (next_earn - pd.Timestamp(f.index[-1]).normalize()).days <= 21:
            try:
                cap = float(__import__("yfinance").Ticker(t).fast_info["market_cap"]) / 1e3
            except Exception:
                cap = np.nan
            try:
                item["preview"] = earnings_preview(f, next_earn, past_earn + yahoo_earnings(t), fred_row(f.index[-1], fred_csv),
                                                   np.log1p(cap) if np.isfinite(cap) else np.nan, yahoo_estimates(t))
            except Exception as e:
                print(f"preview failed for {t}: {e}")
        board.append(item); continue
    i0, cls, hl = found
    try:
        import yfinance as yf
        cap = float(yf.Ticker(t).fast_info["market_cap"]) / 1e3
    except Exception:
        cap = np.nan
    ev = event_features(f, i0, cls, hl, np.log1p(cap) if np.isfinite(cap) else np.nan, fred_row(f.index[i0], fred_csv))
    ev.update(calendar_features(f, i0, past_earn + yahoo_earnings(t) + ([next_earn] if next_earn is not None else [])))
    k = len(f) - 1 - i0
    sr = None
    if 1 <= k <= 15:
        need = list(f.index[max(0, i0 - 30):len(f)])
        sr = finra_ratio(t, need).reindex(f.index)
    reg_now = fred_row(f.index[-1], fred_csv)
    out = predict_all(f, i0, ev, sr, reg_now.get("vix"), ev.get("vix"))
    item.update({"status": "event", "event_class": cls, "event_headline": hl, **out})
    if hl and len(news) and "url" in news:
        hit = news[news.headline == hl]
        if len(hit): item["event_url"] = str(hit.url.iloc[0])          # so the page can link the headline to the story
    e_ = item.get("entry") or {}
    if e_.get("drivers"):
        item["why"] = {"kind": "call", "items": explain(e_.pop("drivers"), cls)}
    elif item.get("risk", {}).get("trough_2_20", {}).get("drivers"):
        item["why"] = {"kind": "dip", "items": explain(item["risk"]["trough_2_20"]["drivers"], cls)}
    for r_ in item.get("risk", {}).values():
        r_.pop("drivers", None)
    if e_.get("signal") not in ("BUY", "WAIT"):
        price_call(item, news_day=True)                         # first day after news: the full call needs a second close
    if ODDS and e_.get("signal") in ("BUY", "WAIT") and e_.get("expected_next5_sd") is not None:
        up = ODDS["up_share"][int(np.searchsorted(ODDS["edges"], e_["expected_next5_sd"], side="right"))]
        o = up if e_["signal"] == "BUY" else 1 - up
        e_["odds"], e_["strength"] = round(float(o), 4), strength(o)
    if np.isfinite(cap): item.setdefault("quote", {})["market_cap"] = cap * 1e3
    board.append(item)
    log_rows.append({"run_utc": now.isoformat(timespec="minutes"), "price_date": item["price_date"], "ticker": t,
                     "day0": out["day0"], "cls": cls, "k": out["decision_day"],
                     "price": item["price"], "signal": out.get("entry", {}).get("signal"),
                     "pred_trough": out["risk"].get("trough_2_20", {}).get("pred"), "warning": out.get("warning_big_drop"),
                     "bad_case_10pct": out.get("sizing", {}).get("tiers", {}).get("recommended", {}).get("bad_case_pct"),
                     "pred_drift20": out["risk"].get("drift_1_20", {}).get("pred"),
                     "w_recommended": out.get("sizing", {}).get("tiers", {}).get("recommended", {}).get("weight")})

# correlations for splitting an account across several stocks (Ledoit-Wolf, last 250 trading days)
on_board = [b["ticker"] for b in board]
R = pd.DataFrame({t: frames[t].ret for t in on_board if t in frames}).tail(250).fillna(0.0)
C = shrunk_corr(R.values) if R.shape[1] > 1 else np.eye(1)
# The page needs correlations for any two or three stocks. A full matrix for 500 stocks is too large to download, so it
# is stored as 12 factor loadings per stock: corr(i, j) is about loading_i . loading_j (the diagonal is 1 by definition).
ev_, V_ = np.linalg.eigh(C); top = np.argsort(ev_)[::-1][:12]
L_ = V_[:, top] * np.sqrt(np.clip(ev_[top], 0, None))
skill = json.loads((MODELS / "skill.json").read_text()) if (MODELS / "skill.json").exists() else {}
try:
    import joblib
    _sz = joblib.load(MODELS / "sizing.joblib")
    if "warn_auc_shock_2016_26" in _sz:
        skill["Big-drop warning (shock-sized events, 2016-2026)"] = (f"AUC {_sz['warn_auc_shock_2016_26']:.2f} for spotting the worst 20% of drops "
                                                                     "(0.50 = coin flip)")
except Exception:
    pass
HEAVY = ("hist", "closes", "close_dates", "headlines", "quote")
sd = DOCS / "stocks"; sd.mkdir(exist_ok=True)
for b in board:
    if "headlines" in b:
        hl = b["headlines"] or []
        b["news_n"], b["news_latest"] = len(hl), (hl[0]["date"] if hl else None)
payload = {"generated_utc": now.isoformat(timespec="minutes"), "tickers": list(R.columns), "corr_f": np.round(L_, 3).tolist(),
           "disclaimer": "Research project. Directional guidance, not financial advice. Past patterns can fail.",
           "skill": skill,
           "stocks": board}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
for b in board:                                                   # full detail per stock, loaded when the stock is opened
    if any(k in b for k in HEAVY) and not b.get("stale"):
        (sd / f"{b['ticker']}.json").write_text(json.dumps(clean({k: b[k] for k in HEAVY if k in b}), separators=(",", ":")))
payload["stocks"] = [{k: v for k, v in b.items() if k not in HEAVY} for b in board]
(DOCS / "predictions.json").write_text(json.dumps(clean(payload), separators=(",", ":")))
logf = DOCS / "forward_log.csv"
# one row per stock per trading day (hourly reruns of the same closing data would otherwise count as many predictions);
# rows from before this format are dropped, they repeat the first run
new = pd.DataFrame(log_rows)
old = pd.read_csv(logf) if logf.exists() else pd.DataFrame()
if len(old) and "price_date" not in old: old = pd.DataFrame()
if len(new) or len(old):
    pd.concat([old, new]).drop_duplicates(["ticker", "price_date"], keep="last").to_csv(logf, index=False)

# README table
lines = [f"_Updated {now:%Y-%m-%d %H:%M} UTC. Research only, not financial advice._", "",
         "| Stock | Price | Event (day 0) | Day | Signal | Expected further dip | Most likely timing of the 20-day low | 1-in-10 worst point (20 days) | Risk vs normal | Balanced size |",
         "|---|---|---|---|---|---|---|---|---|---|"]
shown = sorted([b for b in board if b.get("status") == "event" and b.get("entry", {}).get("signal") in ("BUY", "WAIT")], key=lambda b: b["decision_day"])[:40]
lines[1:1] = ["", f"{len(board)} stocks checked; {sum(b.get('status') == 'event' for b in board)} have company news in the last 15 trading days. The 40 most recent calls are listed below; the page shows all of them."]
for b in shown:
    e, r, s = b.get("entry", {}), b["risk"], b.get("sizing", {}).get("tiers", {})
    dip = f"{e['expected_further_dip_pct']:.1f}%" if "expected_further_dip_pct" in e else ""
    vr = f"{r['vol_ratio_2_20']['pred']:.2f}x" if "vol_ratio_2_20" in r else ""
    rec = f"{100 * s['recommended']['weight']:.0f}%" if "recommended" in s else ""
    warn = " (high drop risk)" if b.get("warning_big_drop") else ""
    dl = e.get("days_to_low")
    if dl:
        j = int(np.argmax(dl["probs"])); names = ["already passed", "in 1-2 days", "in 3-5 days", "in 6-10 days", "in 11+ days"]
        low = f"{names[j]} ({100 * dl['probs'][j]:.0f}%)"
    else:
        low = ""
    bad = f"{s['recommended']['bad_case_pct']:.1f}%" if "recommended" in s else ""
    lab = "price or volume shock" if b["event_class"] == "other" else b["event_class"].replace("_", " ")
    lines.append(f"| {b['ticker']} | {b['price']:.2f} | {lab} ({b['day0']}) | {b['decision_day']} | {e.get('signal', '')}{warn} | {dip} | {low} | {bad} | {vr} | {rec} |")
rd = ROOT / "README.md"; txt = rd.read_text()
A, B = "<!-- LIVE-BOARD:START -->", "<!-- LIVE-BOARD:END -->"
block = A + "\n" + "\n".join(lines) + "\n" + B
txt = txt[:txt.index(A)] + block + txt[txt.index(B) + len(B):] if A in txt else txt + "\n\n## Live board\n\n" + block + "\n"
rd.write_text(txt)
print("\n".join(lines))
