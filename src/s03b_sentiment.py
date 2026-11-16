"""
Step 3b: score event headlines with FinBERT and build per-event sentiment features.

Run on your Mac after s03a:   python src/s03b_sentiment.py
Needs: pip install torch transformers   (falls back to the Loughran-McDonald word list via pysentiment2 if missing)

Output: data/processed/sentiment_features.parquet, one row per event_id:
  news_n_pre      headlines in the 7 calendar days before day 0
  news_n_01       headlines on day 0 and day 1
  sent_pre        mean FinBERT score (positive prob - negative prob) before day 0
  sent_01         mean FinBERT score on days 0-1
  sent_neg_share  share of day 0-1 headlines labelled negative
  sent_min_01     most negative single headline on days 0-1
  sent_change     sent_01 - sent_pre
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from config import PROC

import re
from config import find_file
MAX_PER_EVENT = 40
hl = pd.read_parquet(PROC / "event_headlines.parquet")

# ---------- relevance: does the headline name the company (or its ticker), or describe an event? ----------
ev_keys = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0", "gvkey"]).drop_duplicates(["permno", "day0"])
names = pd.read_csv(find_file("compustat_ids"), usecols=["gvkey", "conm"], dtype=str).dropna()
names["gvkey"] = pd.to_numeric(names.gvkey, errors="coerce")
names = names.drop_duplicates("gvkey", keep="last")
GENERIC = {"THE", "FIRST", "AMERICAN", "NATIONAL", "UNITED", "GENERAL", "INTERNATIONAL", "GLOBAL", "NEW", "BANK", "GROUP",
           "HOLDINGS", "CORP", "INC", "CO", "COMPANY", "ENERGY", "CAPITAL", "FINANCIAL", "TECHNOLOGIES", "SYSTEMS", "US", "U.S."}
def name_token(conm):
    for w in re.sub(r"[^A-Za-z0-9 ]", " ", conm).upper().split():
        if len(w) >= 3 and w not in GENERIC:
            return w
    return None
names["tok"] = names.conm.map(name_token)
hl = hl.merge(ev_keys, on=["permno", "day0"], how="left").merge(names[["gvkey", "tok"]], on="gvkey", how="left")
EVENT_WORDS = re.compile(r"\b(ceo|cfo|chair|resign|steps down|fired|ousted|lawsuit|sues|sued|probe|investigat|sec\b|doj|"
                         r"settle|fraud|restat|accounting|activist|stake|proxy|upgrade|downgrade|rating|price target|"
                         r"scandal|allegation|subpoena|recall|fine[ds]?\b|penalt)", re.I)
up = hl.headline.str.upper()
hl["relevant"] = [bool((t and t in u) or (k and re.search(rf"\b{re.escape(k)}\b", u)) or EVENT_WORDS.search(h))
                  for t, k, u, h in zip(hl.tok.fillna(""), hl.ticker.fillna("").str.upper(), up, hl.headline)]
hl = hl[~hl.headline.str.contains(r"\.OB\b|\.PK\b|OTCBB", regex=True)]          # OTC tickers are often mis-mapped
print(f"relevance filter: {hl.relevant.mean():.1%} of event headlines name the firm or describe an event")
hl = hl.assign(_d=hl.rel_day.abs()).sort_values(["event_id", "_d"]).groupby("event_id").head(MAX_PER_EVENT).drop(columns="_d")
uniq = hl.headline.drop_duplicates().reset_index(drop=True)
print(f"{len(hl):,} event-headline pairs, {len(uniq):,} unique headlines to score")

cache = PROC / "headline_scores.parquet"
done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["headline", "score", "label"])
todo = uniq[~uniq.isin(done.headline)].tolist()
print(f"{len(done):,} already scored, {len(todo):,} to go")

def score_finbert(texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(dev).eval()
    labels = [mdl.config.id2label[i].lower() for i in range(3)]
    ip, ineg = labels.index("positive"), labels.index("negative")
    out, t0 = [], time.time()
    for i in range(0, len(texts), 128):
        b = texts[i:i + 128]
        enc = tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.no_grad():
            pr = torch.softmax(mdl(**enc).logits, -1).cpu().numpy()
        out.append(pd.DataFrame({"headline": b, "score": pr[:, ip] - pr[:, ineg], "label": np.array(labels)[pr.argmax(1)]}))
        if (i // 128) % 50 == 0:
            print(f"  {i + len(b):,}/{len(texts):,}  ({(i + len(b)) / (time.time() - t0):.0f}/s)", flush=True)
    return pd.concat(out, ignore_index=True)

def score_lexicon(texts):
    import pysentiment2 as ps
    lm = ps.LM(); out = []
    for t in texts:
        s = lm.get_score(lm.tokenize(t)); out.append((t, s["Polarity"], "positive" if s["Polarity"] > 0.1 else "negative" if s["Polarity"] < -0.1 else "neutral"))
    return pd.DataFrame(out, columns=["headline", "score", "label"])

if todo:
    try:
        new = score_finbert(todo); print("scored with FinBERT")
    except Exception as e:
        print(f"FinBERT unavailable ({type(e).__name__}: {e}); using Loughran-McDonald word list")
        new = score_lexicon(todo)
    done = pd.concat([done, new], ignore_index=True).drop_duplicates("headline")
    done.to_parquet(cache, index=False)

hl = hl.merge(done, on="headline", how="left")

# ---------- staleness (Tetlock 2011): share of day 0-1 headlines that mostly repeat words from earlier headlines ----------
def words(h): return set(re.findall(r"[a-z]{3,}", h.lower()))
stale = {}
for eid, g in hl.sort_values("date").groupby("event_id"):
    seen, flags = [], []
    for rd, h in zip(g.rel_day, g.headline):
        w = words(h)
        if rd >= 0:
            sim = max((len(w & p) / max(1, len(w | p)) for p in seen), default=0.0)
            flags.append(sim >= 0.5)
        seen.append(w)
    stale[eid] = np.mean(flags) if flags else np.nan

rel = hl[hl.relevant]
pre, d01 = rel[rel.rel_day < 0], rel[rel.rel_day >= 0]
all01 = hl[hl.rel_day >= 0]
keys = hl.drop_duplicates("event_id").set_index("event_id")[["permno", "day0"]]
f = pd.DataFrame({
    "news_n_pre": hl[hl.rel_day < 0].groupby("event_id").size(),
    "news_n_01": all01.groupby("event_id").size(),
    "relevant_share": all01.groupby("event_id").relevant.mean(),
    "sent_pre": pre.groupby("event_id").score.mean(),
    "sent_01": d01.groupby("event_id").score.mean(),
    "sent_neg_share": d01.groupby("event_id").label.apply(lambda s: (s == "negative").mean()),
    "sent_min_01": d01.groupby("event_id").score.min(),
    "stale_share": pd.Series(stale),
})
f = keys.join(f, how="left")
f[["news_n_pre", "news_n_01"]] = f[["news_n_pre", "news_n_01"]].fillna(0)
f["sent_change"] = f.sent_01 - f.sent_pre
bf = PROC / "event_news_baseline.parquet"
if bf.exists():                                    # abnormal news flow = day 0-1 headlines per day vs the firm's normal rate
    b = pd.read_parquet(bf).set_index("event_id").news_base_per_day
    f["news_abn"] = np.log1p(f.news_n_01 / 2) - np.log1p(b.reindex(f.index).fillna(0))
f = f.reset_index(drop=True)
f.to_parquet(PROC / "sentiment_features.parquet", index=False)
print(f"\nsaved sentiment features for {len(f):,} events -> data/processed/sentiment_features.parquet")
print("\n=== News EDA ===")
print(f.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())
print("\nFinBERT labels on relevant day 0-1 headlines:"); print(d01.label.value_counts(normalize=True).round(3).to_string())
print("\nevents with headlines by year:"); print(f.groupby(f.day0.dt.year).size().to_string())
