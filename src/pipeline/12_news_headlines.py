"""
Step 12: shrink the 23 GB FNSPID news file to headlines only, then attach each headline to our events.

Run on your Mac (one pass over the big file, 10-20 minutes):
  python src/pipeline/12_news_headlines.py

Outputs (all git-ignored):
  data/processed/news_headlines.parquet   date, ticker, headline               (all of FNSPID, ~15M rows)
  data/processed/event_headlines.parquet  event_id, rel_day, headline          (headlines on days -5..+1 of our events)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import PUBLIC, PROC, WRDS, find_file

SRC = PUBLIC / "nasdaq_exteral_data.csv"
HL = PROC / "news_headlines.parquet"

# 1. one pass over the big csv, keep three columns ----------
if not HL.exists():
    parts, n = [], 0
    for ch in pd.read_csv(SRC, usecols=["Date", "Article_title", "Stock_symbol"], chunksize=500_000,
                          dtype=str, on_bad_lines="skip", engine="c"):
        ch = ch.dropna(subset=["Date", "Article_title", "Stock_symbol"])
        ch["date"] = pd.to_datetime(ch.Date.str[:19], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        ch = ch.dropna(subset=["date"])
        parts.append(pd.DataFrame({"date": ch.date, "ticker": ch.Stock_symbol.str.upper().str.strip(),
                                   "headline": ch.Article_title.str.strip()}))
        n += len(ch); print(f"  ...{n:,} headlines", flush=True)
    hl = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "ticker", "headline"])
    hl.to_parquet(HL, index=False)
    print(f"saved {len(hl):,} headlines, {hl.ticker.nunique():,} tickers, {hl.date.min().date()} to {hl.date.max().date()} -> {HL.name}")
else:
    hl = pd.read_parquet(HL); print(f"loaded {len(hl):,} headlines from cache")
bad = (hl.date < "1998-01-01") | (hl.date > "2026-12-31")
print(f"dropping {bad.sum():,} headlines with impossible timestamps (e.g. 1969 = empty epoch)")
hl = hl[~bad]
print("note: FNSPID has 15.7M rows but ~10M are general market articles with no ticker; only ticker-tagged rows are usable here")

# 2. map ticker -> permno using CRSP's own Ticker column (by date, since tickers get reused) ----------
tk = pd.read_csv(find_file("crsp_daily"), usecols=["PERMNO", "Ticker", "DlyCalDt"], dtype={"Ticker": str})
tk = tk.dropna(subset=["Ticker"]).rename(columns={"PERMNO": "permno", "Ticker": "ticker", "DlyCalDt": "date"})
tk["date"] = pd.to_datetime(tk.date); tk["ticker"] = tk.ticker.str.upper()
tk = tk.groupby(["permno", "ticker"]).date.agg(["min", "max"]).reset_index()

# 3. attach headlines to events: window = 5 trading days before day 0 through day 1 ----------
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"]).reset_index().rename(columns={"index": "event_id"})
ev = ev.merge(tk, on="permno")
ev = ev[(ev.day0 >= ev["min"] - pd.Timedelta(days=30)) & (ev.day0 <= ev["max"] + pd.Timedelta(days=30))]
hl["day"] = hl.date.dt.normalize()
# news published after 4pm New York counts for the next trading day
hl["eff_day"] = np.where(hl.date.dt.hour >= 16, hl.day + pd.Timedelta(days=1), hl.day)
hl["eff_day"] = pd.to_datetime(hl.eff_day)
m = ev[["event_id", "permno", "ticker", "day0"]].merge(hl[["ticker", "eff_day", "headline", "date"]], on="ticker")
rel = (m.eff_day - m.day0).dt.days
m = m[(rel >= -7) & (rel <= 1)].assign(rel_day=rel)          # calendar days: -7 covers 5 trading days
m = m[["event_id", "permno", "ticker", "day0", "rel_day", "date", "headline"]].drop_duplicates(["event_id", "headline"])

# 4. baseline news flow per event: headlines per day for the ticker in calendar days -97..-8 ----------
cnt = hl.groupby("ticker").eff_day.apply(lambda s: np.sort(s.values.astype("datetime64[D]").astype(np.int64)))
keys = m[["event_id", "ticker", "day0"]].drop_duplicates("event_id")
base = []
for tkr, g in keys.groupby("ticker"):
    arr = cnt.get(tkr)
    d0 = g.day0.values.astype("datetime64[D]").astype(np.int64)
    n = np.searchsorted(arr, d0 - 8, "right") - np.searchsorted(arr, d0 - 97, "left") if arr is not None else np.zeros(len(g))
    base.append(pd.DataFrame({"event_id": g.event_id.values, "news_base_per_day": n / 90.0}))
pd.concat(base).to_parquet(PROC / "event_news_baseline.parquet", index=False)
m.to_parquet(PROC / "event_headlines.parquet", index=False)
print(f"\n{len(m):,} headlines attached to {m.event_id.nunique():,} of {ev.event_id.nunique():,} events "
      f"({m.event_id.nunique() / len(pd.read_parquet(PROC / 'events.parquet')):.1%} of all events have at least one headline)")
print("headlines per covered event: median", int(m.groupby("event_id").size().median()), ", 90th pct", int(m.groupby("event_id").size().quantile(.9)))
