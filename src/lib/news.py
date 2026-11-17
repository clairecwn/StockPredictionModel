"""
Recent headlines for a ticker (Finnhub, free tier) and sentiment scores (FinBERT via the Hugging Face inference API).
Keys are read from .env: FINNHUB_API_KEY, HF_TOKEN. Nothing here is used in model training yet (no news history).
"""
import os, time, requests, pandas as pd
from config import load_env
load_env()

def finnhub_news(ticker, start, end):
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return pd.DataFrame(columns=["date", "headline", "source", "url"])
    r = requests.get("https://finnhub.io/api/v1/company-news",
                     params=dict(symbol=ticker, _from=str(start)[:10], to=str(end)[:10], token=key), timeout=30)
    if r.status_code != 200:                       # never echo the URL: it contains the key
        raise RuntimeError(f"Finnhub returned HTTP {r.status_code}; check FINNHUB_API_KEY in .env")
    d = pd.DataFrame(r.json())
    if d.empty:
        return pd.DataFrame(columns=["date", "headline", "source", "url"])
    d["date"] = pd.to_datetime(d.datetime, unit="s")
    return d[["date", "headline", "source", "url"]].sort_values("date", ascending=False)

def finbert(headlines):
    """Return a list of (label, score) using ProsusAI/finbert through the HF inference API. Empty list if no token."""
    tok = os.environ.get("HF_TOKEN")
    if not tok or not headlines:
        return []
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    out = []
    for i in range(0, len(headlines), 25):
        batch = headlines[i:i + 25]
        for attempt in range(4):
            r = requests.post(url, headers={"Authorization": f"Bearer {tok}"}, json={"inputs": batch}, timeout=60)
            if r.status_code == 503:            # model loading
                time.sleep(10); continue
            break
        if r.status_code != 200:
            return []
        for res in r.json():
            best = max(res, key=lambda x: x["score"])
            out.append((best["label"], best["score"]))
    return out

def sentiment_summary(ticker, start, end):
    n = finnhub_news(ticker, start, end)
    s = finbert(n.headline.tolist())
    if s:
        n["label"] = [x[0] for x in s]; n["conf"] = [x[1] for x in s]
        pos, neg = (n.label == "positive").mean(), (n.label == "negative").mean()
    else:
        n["label"], n["conf"] = "n/a", float("nan"); pos = neg = float("nan")
    return n, {"headlines": len(n), "share_positive": pos, "share_negative": neg, "net_sentiment": pos - neg}
