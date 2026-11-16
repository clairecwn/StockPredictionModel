"""Step 9: live board. For each stock in watchlist.json: find the most recent event (last 15 trading days), rebuild the
features exactly as in training (see s08_parity.py), and write:
  docs/data/predictions.json   everything the interactive page shows (signals, risk, sizing, headlines, correlations)
  docs/data/forward_log.csv    one row per stock per run, so live accuracy can be scored later (out of sample by construction)
  README.md                    the table between the LIVE-BOARD markers is replaced
Models: models/live/ (trained with LIVE_FEATS=1 on features that free live data can rebuild).
Run: python src/s09_live_board.py          Research only. Not investment advice.
"""
import json, os, io, re, warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests
os.environ.setdefault("LIVE_FEATS", "1")
from config import ROOT, PUBLIC, MODELS
from live_engine import yahoo_frames, fred_row, detect_event, event_features, calendar_features, earnings_preview, predict_all
from sizing_lib import shrunk_corr

WL = json.loads((ROOT / "watchlist.json").read_text())
TICKERS = [w["ticker"] for w in WL["stocks"]]
DOCS = ROOT / "docs" / "data"; DOCS.mkdir(parents=True, exist_ok=True)
now = datetime.now(timezone.utc)
ET = "America/New_York"

def relevant(headline, w):
    """True if the headline is about this company: its ticker or one of its names, case-sensitive and whole-word, within
    the first 7 words (Finnhub company-news also returns articles that only mention the company in passing)."""
    names = ([w["ticker"]] if len(w["ticker"]) > 2 else []) + [w.get("name", "")] + w.get("aliases", [])  # 2-letter tickers are ordinary words
    early = " ".join(headline.split()[:7])                                # the subject usually comes first; later = passing mention
    return any(re.search(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", early) for n in names if n)

def earnings_days(t, back=40, ahead=30):
    """Earnings announcement dates from Finnhub's calendar. Returns (past dates as Timestamps, next upcoming date or None).
    Training dated earnings events to the announcement day, so after-close reports keep their own date here too."""
    key = os.environ.get("FINNHUB_API_KEY")
    if not key: return [], None
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings", timeout=20,
                         params={"from": str((now - pd.Timedelta(days=back)).date()), "to": str((now + pd.Timedelta(days=ahead)).date()),
                                 "symbol": t, "token": key})
        cal = r.json().get("earningsCalendar", []) if r.status_code == 200 else []
    except Exception:
        cal = []
    ds = sorted({pd.Timestamp(c["date"]) for c in cal if c.get("date")})
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
        r = requests.get("https://finnhub.io/api/v1/company-news", timeout=20,
                         params=dict(symbol=t, _from=str((now - pd.Timedelta(days=days)).date()), to=str(now.date()), token=key))
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
    if f is None or len(f) < 300:
        if t in PREV:
            item = {**PREV[t], "stale": True}                  # shown as the last update, not logged as a new prediction
        else:
            item["status"] = "no price data"
        board.append(item); continue
    item["price"] = float(f.c.iloc[-1]); item["price_date"] = str(f.index[-1].date())
    item["chg_1d_pct"] = 100 * float(f.ret.iloc[-1])
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
    board.append(item)
    log_rows.append({"run_utc": now.isoformat(timespec="minutes"), "price_date": item["price_date"], "ticker": t,
                     "day0": out["day0"], "cls": cls, "k": out["decision_day"],
                     "price": item["price"], "signal": out.get("entry", {}).get("signal"),
                     "pred_trough": out["risk"].get("trough_2_20", {}).get("pred"), "warning": out.get("warning_big_drop"),
                     "bad_case_10pct": out.get("sizing", {}).get("tiers", {}).get("recommended", {}).get("bad_case_pct"),
                     "pred_drift20": out["risk"].get("drift_1_20", {}).get("pred"),
                     "w_recommended": out.get("sizing", {}).get("tiers", {}).get("recommended", {}).get("weight")})

# correlations for splitting an account across several stocks (Ledoit-Wolf, last 250 trading days)
R = pd.DataFrame({t: f.ret for t, f in frames.items()}).tail(250)
C = shrunk_corr(R.values) if R.shape[1] > 1 else np.eye(1)
skill = json.loads((MODELS / "skill.json").read_text()) if (MODELS / "skill.json").exists() else {}
try:
    import joblib
    _sz = joblib.load(MODELS / "sizing.joblib")
    if "warn_auc_shock_2016_26" in _sz:
        skill["Big-drop warning (shock-sized events, 2016-2026)"] = (f"AUC {_sz['warn_auc_shock_2016_26']:.2f} for spotting the worst 20% of drops "
                                                                     "(0.50 = coin flip)")
except Exception:
    pass
payload = {"generated_utc": now.isoformat(timespec="minutes"), "tickers": list(R.columns), "corr": np.round(C, 3).tolist(),
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
(DOCS / "predictions.json").write_text(json.dumps(clean(payload), indent=1))
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
         "| Stock | Price | Event (day 0) | Day | Signal | Expected further dip | 1-in-10 worst point (20 days) | Risk vs normal | Recommended share if buying |",
         "|---|---|---|---|---|---|---|---|---|"]
for b in board:
    if b.get("status") != "event":
        lines.append(f"| {b['ticker']} | {b.get('price', float('nan')):.2f} | {b.get('status', '')} | | | | | | |"); continue
    e, r, s = b.get("entry", {}), b["risk"], b.get("sizing", {}).get("tiers", {})
    dip = f"{e['expected_further_dip_pct']:.1f}%" if "expected_further_dip_pct" in e else ""
    vr = f"{r['vol_ratio_2_20']['pred']:.2f}x" if "vol_ratio_2_20" in r else ""
    rec = f"{100 * s['recommended']['weight']:.0f}%" if "recommended" in s else ""
    warn = " ⚠️" if b.get("warning_big_drop") else ""
    bad = f"{s['recommended']['bad_case_pct']:.1f}%" if "recommended" in s else ""
    lab = "price/volume shock, no matched news" if b["event_class"] == "other" else b["event_class"].replace("_", " ")
    lines.append(f"| {b['ticker']} | {b['price']:.2f} | {lab} ({b['day0']}) | {b['decision_day']} | {e.get('signal', '')}{warn} | {dip} | {bad} | {vr} | {rec} |")
rd = ROOT / "README.md"; txt = rd.read_text()
A, B = "<!-- LIVE-BOARD:START -->", "<!-- LIVE-BOARD:END -->"
block = A + "\n" + "\n".join(lines) + "\n" + B
txt = txt[:txt.index(A)] + block + txt[txt.index(B) + len(B):] if A in txt else txt + "\n\n## Live board\n\n" + block + "\n"
rd.write_text(txt)
print("\n".join(lines))
