"""Step 9 (experiment 11): investor attention from Wikipedia page views (free, daily, from July 2015).

Da, Engelberg and Gao (2011) found that a jump in retail attention (Google searches) predicts higher prices over the
next two weeks and a reversal later. Wikipedia page views are a free, daily, downloadable measure of the same thing.
1. Wikidata links each listed company to its English Wikipedia article, its SEC CIK and its ticker (one SPARQL query).
   Firms are matched by CIK first (Compustat CIK -> gvkey -> CRSP permno), then by Compustat ticker.
2. Daily page views per article come from the Wikimedia REST API (human traffic only), cached in data/raw/wiki/.
3. Event features use views up to day 0 only. Wikimedia counts UTC days, and UTC day 0 ends at 7-8pm New York time,
   before day 1's close, so nothing after the information cutoff is used:
     att_base   log of average daily views, calendar days -120..-31 (how much attention the firm normally gets)
     att_trend  log(views days -30..-6 / views days -120..-31)   attention building before the news
     att_spike  log(views days -1..0 / views days -120..-31)     the attention spike around the news
Output: data/processed/wiki_attention.parquet
Run: python src/pipeline/09_wikipedia_attention.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, json, time, urllib.parse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests
from config import PROC, ROOT, find_file

RAW = ROOT / "data" / "raw" / "wiki"; RAW.mkdir(parents=True, exist_ok=True)
env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.startswith("#"))
UA = {"User-Agent": f"stock-event-research/1.0 ({env.get('SEC_CONTACT_EMAIL', '').strip()})"}

# 1. Wikidata: company -> article, CIK, ticker
links_f = RAW / "wikidata_links.csv"
if not links_f.exists():
    q = """SELECT DISTINCT ?item ?article ?cik ?ticker WHERE {
      ?item p:P414 ?st . ?st ps:P414 ?ex . VALUES ?ex { wd:Q13677 wd:Q82059 wd:Q846626 }
      OPTIONAL { ?st pq:P249 ?ticker . }
      OPTIONAL { ?item wdt:P5531 ?cik . }
      ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }"""
    r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=UA, timeout=300)
    r.raise_for_status()
    b = r.json()["results"]["bindings"]
    L = pd.DataFrame([{k: v["value"] for k, v in x.items()} for x in b])
    L["title"] = L.article.str.split("/wiki/").str[1].map(urllib.parse.unquote)
    L.to_csv(links_f, index=False)
L = pd.read_csv(links_f, dtype=str)
print(f"Wikidata: {L['item'].nunique():,} listed companies with an English article")

ids = pd.read_csv(find_file("compustat_ids"), usecols=["gvkey", "cik", "tic", "datadate"]).sort_values("datadate")
lk = pd.read_csv(find_file("ccm_link")); lk = lk[lk.linkprim.isin(["P", "C"])].dropna(subset=["permno"])
lk["linkenddt"] = pd.to_datetime(lk.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-01-01"))
gv2p = lk.sort_values("linkenddt").drop_duplicates("gvkey", keep="last").set_index("gvkey").permno.astype(np.int64)
cik2gv = ids.dropna(subset=["cik"]).drop_duplicates("cik", keep="last").set_index("cik").gvkey
tic2gv = ids.dropna(subset=["tic"]).drop_duplicates("tic", keep="last").set_index("tic").gvkey
L["cik_n"] = pd.to_numeric(L.get("cik"), errors="coerce")
L["gvkey"] = L.cik_n.map(cik2gv)
L.loc[L.gvkey.isna(), "gvkey"] = L.loc[L.gvkey.isna(), "ticker"].str.upper().map(tic2gv)
L["permno"] = L.gvkey.map(gv2p)
L = L.dropna(subset=["permno"]); L["permno"] = L.permno.astype(np.int64)
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"]).drop_duplicates()
ev["permno"] = ev.permno.astype(np.int64)
L = L[L.permno.isin(set(ev.permno))].drop_duplicates(["permno", "title"])
print(f"matched to event firms: {L.permno.nunique():,} firms, {L.title.nunique():,} articles")

# 2. daily page views (cached per article)
end = (pd.Timestamp.utcnow() - pd.Timedelta(days=2)).strftime("%Y%m%d")
views = []
for i, title in enumerate(sorted(L.title.unique())):
    f = RAW / f"pv_{urllib.parse.quote(title, safe='')[:150]}.csv"
    if not f.exists():
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               f"{urllib.parse.quote(title.replace(' ', '_'), safe='')}/daily/20150701/{end}")
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    it = r.json().get("items", [])
                    pd.DataFrame({"date": [x["timestamp"][:8] for x in it], "views": [x["views"] for x in it]}).to_csv(f, index=False)
                    break
                if r.status_code == 404:
                    pd.DataFrame({"date": [], "views": []}).to_csv(f, index=False); break
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.05)
    if f.exists():
        d = pd.read_csv(f, dtype={"date": str})
        if len(d):
            d["title"] = title; views.append(d)
    if i % 500 == 0:
        print(f"  {i:,} articles", flush=True)
V = pd.concat(views, ignore_index=True)
V["date"] = pd.to_datetime(V.date, format="%Y%m%d")
V = V.merge(L[["title", "permno"]], on="title").groupby(["permno", "date"]).views.sum().reset_index()
print(f"page views: {len(V):,} firm-days, {V.permno.nunique():,} firms, {V.date.min().date()} to {V.date.max().date()}")

# 3. event features (calendar days, views up to day 0)
W = V.pivot(index="date", columns="permno", values="views")
W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="D"))
A = W.values.astype(float)
S = np.vstack([np.zeros((1, A.shape[1])), np.nancumsum(A, 0)]); N = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(np.isfinite(A), 0)])
pos = pd.Series(np.arange(A.shape[1]), index=W.columns)
e = ev[ev.permno.isin(pos.index) & (ev.day0 >= W.index.min() + pd.Timedelta(days=125)) & (ev.day0 <= W.index.max())].copy()
i0 = W.index.get_indexer(e.day0.dt.normalize()); ci = pos[e.permno].values
ok = i0 >= 120; e, i0, ci = e[ok], i0[ok], ci[ok]
def m(a, b):
    s, t = i0 + a, i0 + b + 1
    n = N[t, ci] - N[s, ci]
    return np.where(n > 0, (S[t, ci] - S[s, ci]) / np.maximum(n, 1), np.nan)
base, pre, spike = m(-120, -31), m(-30, -6), m(-1, 0)
F = pd.DataFrame({"permno": e.permno.values, "day0": e.day0.values, "att_base": np.log1p(base),
                  "att_trend": np.log((pre + 1) / (base + 1)), "att_spike": np.log((spike + 1) / (base + 1))})
print(F.describe().T[["count", "mean", "50%", "min", "max"]].round(3).to_string())
F.to_parquet(PROC / "wiki_attention.parquet", index=False)
print(f"saved {len(F):,} events -> data/processed/wiki_attention.parquet")
