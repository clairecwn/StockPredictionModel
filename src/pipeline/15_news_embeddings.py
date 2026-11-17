"""Step 15: modern text features from sentence embeddings, plus analogue retrieval. Run after 12_news_headlines (headlines).

Each headline becomes a 384-number "meaning vector" (all-MiniLM-L6-v2, a sentence-embedding model used in search
engines and retrieval-augmented LLMs; runs locally on the Mac GPU, free). Unlike word counts it sees context:
negation, "beats but cuts guidance", synonyms.

Event features (headlines on days 0-1 unless stated):
  txt_novelty     1 - average best match against the firm's own headlines on days -7..-1 (new news vs rehash; Tetlock 2011)
  txt_dispersion  1 - |mean of unit vectors|: do the day's headlines tell one story or several (disagreement -> volatility)
  lex_uncertain / lex_litig / lex_fear / lex_anger / lex_excite   share of headlines with words from short finance
                  dictionaries (uncertainty and litigation: Loughran-McDonald style; emotions: StockEmotions style)
  ana_trough / ana_logvr / ana_drift5 / ana_sim   ANALOGUE RETRIEVAL: the 50 most similar past events (by headline
                  meaning, from events at least 120 days before 1 Jan of the event's year) and what happened after them:
                  similarity-weighted average 20-day trough, log volatility ratio, 5-day drift; ana_sim = how close they were.

Output: data/processed/text_features.parquet (and a cache of embeddings in data/processed/headline_emb.npy)
Logic check without the embedding model (random vectors): TEXT_TEST=1 python src/pipeline/15_news_embeddings.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from config import PROC

TEST = os.environ.get("TEXT_TEST") == "1"
K_ANA, PURGE_DAYS, MIN_LIB = 50, 120, 5000
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

hl = pd.read_parquet(PROC / "event_headlines.parquet", columns=["permno", "day0", "rel_day", "headline"])
hl = hl.dropna(subset=["headline"]).drop_duplicates(["permno", "day0", "headline"]).reset_index(drop=True)
if TEST:
    keep = hl[["permno", "day0"]].drop_duplicates().sample(20000, random_state=0)
    hl = hl.merge(keep, on=["permno", "day0"]).reset_index(drop=True)
uniq = pd.Series(hl.headline.unique())
hl["hid"] = pd.Index(uniq).get_indexer(hl.headline)
print(f"headlines: {len(hl):,} rows, {len(uniq):,} unique" + ("   [TEST MODE: random vectors]" if TEST else ""))

# embeddings (cached)
cache_v, cache_i = PROC / "headline_emb.npy", PROC / "headline_emb_index.parquet"
if TEST:
    V = np.random.default_rng(0).normal(size=(len(uniq), 384)).astype(np.float32)
else:
    done_txt, V_old = [], None
    if cache_v.exists() and cache_i.exists():
        V_old = np.load(cache_v); done_txt = pd.read_parquet(cache_i).headline.tolist()
    have = pd.Index(done_txt)
    todo = uniq[~uniq.isin(have)].tolist()
    print(f"embedding {len(todo):,} new headlines ({len(have):,} cached)")
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModel
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL); mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(todo), 1024):
                b = tok(todo[i:i + 1024], padding=True, truncation=True, max_length=48, return_tensors="pt").to(dev)
                h = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32))
                if (i // 1024) % 50 == 0:
                    print(f"  {i + len(b['input_ids']):,}/{len(todo):,}", flush=True)
        Vn = np.vstack(out)
        V_all = Vn if V_old is None else np.vstack([V_old, Vn])
        txt_all = list(have) + todo
        np.save(cache_v, V_all); pd.DataFrame({"headline": txt_all}).to_parquet(cache_i, index=False)
        V_old, have = V_all, pd.Index(txt_all)
    V = V_old[have.get_indexer(uniq)]
V = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-9)

# lexicons (short, transparent lists)
LEX = {
    "uncertain": r"\b(may|might|could|uncertain\w*|unclear|possibl\w*|risk\w*|volatil\w*|pending|consider\w*|explor\w*|review\w*|weigh\w*|doubt\w*|question\w*)\b",
    "litig": r"\b(lawsuit\w*|sue[sd]?|suing|litigat\w*|probe\w*|investigat\w*|subpoena\w*|sec\b|doj|settle\w*|class action|fraud\w*|indict\w*|fine[sd]?|penalt\w*|charg\w*)\b",
    "fear": r"\b(fear\w*|panic\w*|crash\w*|plung\w*|tumbl\w*|collaps\w*|warn\w*|worr\w*|concern\w*|threat\w*|crisis|slump\w*|sell-?off|bankrupt\w*|default\w*)\b",
    "anger": r"\b(anger|angry|outrag\w*|slam\w*|blast\w*|attack\w*|accus\w*|disput\w*|feud\w*|fight\w*|oust\w*|fir(e|ed|ing)|resign\w*|scandal\w*|backlash)\b",
    "excite": r"\b(soar\w*|surg\w*|skyrocket\w*|jump\w*|rall\w*|record|breakthrough\w*|beat\w*|blowout|boom\w*|bullish|upgrad\w*|takeover|buyout|acqui\w*)\b",
}
low = hl.headline.str.lower()
for k, rx in LEX.items():
    hl[f"lex_{k}"] = low.str.contains(rx, regex=True).astype(float)

# per-event novelty, dispersion, event vector
hl["key"] = pd.factorize(list(zip(hl.permno, hl.day0)))[0]
d01 = hl[hl.rel_day.between(0, 1)]
pre = hl[hl.rel_day.between(-7, -1)]
pre_groups = {k: g.hid.values for k, g in pre.groupby("key")}
rows = []
for k, g in d01.groupby("key"):
    U = V[g.hid.values]
    mvec = U.mean(axis=0)
    disp = 1 - np.linalg.norm(mvec) if len(U) > 1 else np.nan
    ph = pre_groups.get(k)
    nov = 1 - (U @ V[ph].T).max(axis=1).mean() if ph is not None and len(ph) else np.nan
    rows.append((k, nov, disp, *[g[f"lex_{x}"].mean() for x in LEX]))
F = pd.DataFrame(rows, columns=["key", "txt_novelty", "txt_dispersion"] + [f"lex_{x}" for x in LEX])
keys = hl.drop_duplicates("key").set_index("key")[["permno", "day0"]]
F = F.join(keys, on="key")
EV = d01.groupby("key").hid.apply(lambda h: V[h.values].mean(axis=0)).to_dict()
print(f"events with day 0-1 headlines: {len(F):,}")

# analogue retrieval (past events only)
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0", "trough_2_20", "vol_ratio_2_20", "drift_1_5"]).drop_duplicates(["permno", "day0"])
ev["logvr"] = np.log(ev.vol_ratio_2_20.clip(0.05, 20))
for c in ("trough_2_20", "drift_1_5"):
    lo, hi = ev[c].quantile([0.005, 0.995]); ev[c] = ev[c].clip(lo, hi)
F = F.merge(ev, on=["permno", "day0"], how="left")
M = np.vstack([EV[k] for k in F.key.values]).astype(np.float32)
M = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-9)
for c in ("ana_trough", "ana_logvr", "ana_drift5", "ana_sim"):
    F[c] = np.nan
yr = F.day0.dt.year.values
outc = F[["trough_2_20", "logvr", "drift_1_5"]].values
for Y in sorted(np.unique(yr)):
    q = np.where(yr == Y)[0]
    lib = np.where((F.day0 < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=PURGE_DAYS)).values & np.isfinite(outc).all(axis=1))[0]
    if len(lib) < MIN_LIB:
        continue
    L = M[lib]
    res = np.full((len(q), 4), np.nan)
    for s in range(0, len(q), 2000):
        qq = q[s:s + 2000]
        S = M[qq] @ L.T
        top = np.argpartition(-S, K_ANA, axis=1)[:, :K_ANA]
        st = np.take_along_axis(S, top, axis=1)
        w = np.exp(10 * (st - st.max(axis=1, keepdims=True)))      # softmax-style weights on similarity
        w = w / w.sum(axis=1, keepdims=True)
        o = outc[lib][top]                                          # q x K x 3
        res[s:s + len(qq), :3] = (w[:, :, None] * o).sum(axis=1)
        res[s:s + len(qq), 3] = st.mean(axis=1)
    F.loc[F.index[q], ["ana_trough", "ana_logvr", "ana_drift5", "ana_sim"]] = res
    print(f"  {Y}: {len(q):,} events matched against a library of {len(lib):,} past events", flush=True)

from scipy.stats import spearmanr
print("\n=== Text features: out-of-sample check (analogue forecast vs what actually happened) ===")
for a, t in (("ana_trough", "trough_2_20"), ("ana_logvr", "logvr"), ("ana_drift5", "drift_1_5"),
             ("txt_dispersion", "logvr"), ("lex_uncertain", "logvr"), ("txt_novelty", "logvr")):
    m = F[a].notna() & F[t].notna()
    if m.sum() > 500:
        print(f"spearman({a}, {t}) = {spearmanr(F.loc[m, a], F.loc[m, t])[0]:.3f}   n={m.sum():,}")
print(F[["txt_novelty", "txt_dispersion"] + [f"lex_{x}" for x in LEX] + ["ana_sim"]].describe().T[["count", "mean", "50%"]].round(3).to_string())
cols = ["permno", "day0", "txt_novelty", "txt_dispersion"] + [f"lex_{x}" for x in LEX] + ["ana_trough", "ana_logvr", "ana_drift5", "ana_sim"]
if not TEST:
    F[cols].to_parquet(PROC / "text_features.parquet", index=False)
    print(f"saved {len(F):,} events -> data/processed/text_features.parquet")
