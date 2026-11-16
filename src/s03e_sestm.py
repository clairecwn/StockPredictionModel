"""Step 3e: supervised news sentiment (SESTM; Ke, Kelly & Xiu 2019), learned from OUR headlines and OUR returns.

Why: FinBERT scores tone the way a generic financial reader would. SESTM instead learns which words actually go
with price moves in this data, and how strongly. Every score is out of sample: headlines dated in year Y are scored
by a model trained only on events whose day 0 is at least 120 days before 1 Jan of Y (expanding window, purged).

Two models are learned per year:
  react : words that go with the day 0-1 abnormal reaction  (does the news look good or bad to the market?)
  drift : words that go with the day 2-5 abnormal return     (news the market is slow to absorb?)
Steps per model (paper notation): (1) screen words whose presence is skewed toward positive or negative returns;
(2) estimate positive / negative topic word distributions O+ and O- by regressing word frequencies on the return rank;
(3) score each new headline by penalised maximum likelihood: p_hat = argmax_p sum_j d_j log(p O+_j + (1-p) O-_j) + lam*log(p(1-p)).
Event features (centred so 0 = neutral): sestm_react_01, sestm_drift_01 (headlines on days 0-1), sestm_react_pre (days -7..-1),
sestm_n (headlines that contained at least one screened word).

Output: data/processed/sestm_features.parquet and results/sestm_words.txt (the learned word lists, for inspection).
"""
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from config import PROC, RESULTS

MIN_DOCS, N_SIDE, LAM, PURGE_DAYS, FIRST_TEST_YEAR = 50, 200, 1.0, 120, 2011
GRID = np.linspace(0.01, 0.99, 99)

hl = pd.read_parquet(PROC / "event_headlines.parquet", columns=["permno", "ticker", "day0", "rel_day", "headline"])
ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0", "car_0_1", "drift_1_5"])
hl = hl.merge(ev.drop_duplicates(["permno", "day0"]), on=["permno", "day0"], how="inner")
hl = hl.drop_duplicates(["permno", "day0", "headline"]).reset_index(drop=True)
hl["year"] = hl.day0.dt.year
print(f"headlines: {len(hl):,} for {hl[['permno', 'day0']].drop_duplicates().shape[0]:,} events, {hl.year.min()}-{hl.year.max()}")

# vocabulary (unsupervised, no returns used): lower-case words of 3+ letters, numbers/tickers dropped
vec = CountVectorizer(binary=True, min_df=MIN_DOCS, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b", stop_words="english", lowercase=True)
D = vec.fit_transform(hl.headline.fillna("").astype(str)).tocsr()
vocab = np.array(vec.get_feature_names_out())
# words that are tickers are firm identities, not sentiment: exclude them from screening
TICK = set(hl.ticker.dropna().astype(str).str.lower())
is_ticker = np.array([w in TICK for w in vocab])
firm_code = pd.factorize(hl.permno)[0]
print(f"vocabulary: {len(vocab):,} words appearing in >= {MIN_DOCS} headlines ({is_ticker.sum():,} are tickers and are excluded)")

def fit(Dt, r, firms):
    """Steps 1-2. Dt: training doc-term (binary), r: returns. Returns (screened word idx, O matrix m x 2)."""
    pos = (r > 0).astype(float)
    k = np.asarray(Dt.sum(axis=0)).ravel()
    fpos = np.asarray(Dt.T @ pos).ravel() / np.maximum(k, 1)
    base = pos.mean()
    # a word that mostly appears for one company (its name, its product) is a firm identity: drop it (share > 20%)
    F = sparse.csr_matrix((np.ones(len(firms)), (np.arange(len(firms)), firms)), shape=(len(firms), firms.max() + 1))
    top_firm = np.asarray((Dt.T @ F).max(axis=1).todense()).ravel()
    ok = (k >= MIN_DOCS) & ~is_ticker & (top_firm / np.maximum(k, 1) <= 0.20)
    order = np.argsort(fpos - base)
    neg_idx = [j for j in order if ok[j]][:N_SIDE]
    pos_idx = [j for j in order[::-1] if ok[j]][:N_SIDE]
    S = np.array(sorted(set(neg_idx) | set(pos_idx)))
    Ds = Dt[:, S]
    s = np.asarray(Ds.sum(axis=1)).ravel()
    keep = s > 0
    Dn = sparse.diags(1.0 / s[keep]) @ Ds[keep]
    p = pd.Series(r[keep]).rank(pct=True).values
    W = np.column_stack([p, 1 - p])
    O = np.asarray(Dn.T @ W) @ np.linalg.inv(W.T @ W)       # m x 2
    O = np.clip(O, 1e-6, None); O = O / O.sum(axis=0, keepdims=True)
    return S, O, fpos - base

def score(Dnew, S, O):
    Ds = Dnew[:, S]
    n = np.asarray(Ds.sum(axis=1)).ravel()
    L = np.log(np.outer(O[:, 0], GRID) + np.outer(O[:, 1], 1 - GRID))   # m x G
    ll = np.asarray(Ds @ L) + LAM * np.log(GRID * (1 - GRID))[None, :]
    p = GRID[ll.argmax(axis=1)]
    return np.where(n > 0, p - 0.5, np.nan), n

out = {c: np.full(len(hl), np.nan) for c in ("react", "drift")}
nw = np.zeros(len(hl))
words_log = []
for Y in range(FIRST_TEST_YEAR, int(hl.year.max()) + 1):
    te = (hl.year == Y).values
    if not te.any():
        continue
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=PURGE_DAYS)
    for name, col in (("react", "car_0_1"), ("drift", "drift_1_5")):
        tr = (hl.day0 < cut).values & hl[col].notna().values & hl.rel_day.between(0, 1).values
        if tr.sum() < 5000:
            continue
        S, O, skew = fit(D[tr], hl.loc[tr, col].values, firm_code[tr])
        sc, n = score(D[te], S, O)
        out[name][te] = sc
        if name == "react":
            nw[te] = n
        if Y == int(hl.year.max()) or Y == FIRST_TEST_YEAR:
            top = S[np.argsort(skew[S])]
            words_log.append(f"{name} model used for {Y} (trained on {tr.sum():,} headlines before {cut.date()}):\n"
                             f"  most negative: {', '.join(vocab[top[:25]])}\n  most positive: {', '.join(vocab[top[::-1][:25]])}\n")
    print(f"  {Y}: scored {te.sum():,} headlines", flush=True)

hl["sestm_react"], hl["sestm_drift"], hl["sestm_n"] = out["react"], out["drift"], (nw > 0).astype(int)
win = hl[hl.rel_day.between(0, 1)].groupby(["permno", "day0"]).agg(
    sestm_react_01=("sestm_react", "mean"), sestm_drift_01=("sestm_drift", "mean"), sestm_n=("sestm_n", "sum"))
pre = hl[hl.rel_day.between(-7, -1)].groupby(["permno", "day0"]).agg(sestm_react_pre=("sestm_react", "mean"))
f = win.join(pre, how="outer").reset_index()
f.to_parquet(PROC / "sestm_features.parquet", index=False)
(RESULTS / "sestm_words.txt").write_text("\n".join(words_log))

# ---------------------------------------------------------------- checks
chk = f.merge(ev.drop_duplicates(["permno", "day0"]), on=["permno", "day0"])
from scipy.stats import spearmanr
print("\n=== SESTM out-of-sample checks (events from 2011) ===")
for c, t in (("sestm_react_01", "car_0_1"), ("sestm_drift_01", "drift_1_5"), ("sestm_react_01", "drift_1_5")):
    m = chk[c].notna() & chk[t].notna()
    print(f"spearman({c}, {t}) = {spearmanr(chk.loc[m, c], chk.loc[m, t])[0]:.3f}   n={m.sum():,}")
print("(the first should be clearly positive if the word lists are sensible; the drift rows are the real test)")
print("\n" + "\n".join(words_log))
print(f"saved {len(f):,} events -> data/processed/sestm_features.parquet")
