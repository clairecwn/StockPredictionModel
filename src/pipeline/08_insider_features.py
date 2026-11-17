"""Step 8 (experiment 6): insider trading features from the SEC's free Form 3/4/5 data sets (2006 on).

1. Download every quarterly file from the SEC (cached in data/raw/sec_insider/, so a rerun skips what it has).
   The SEC asks for a contact email in the User-Agent; it is read from SEC_CONTACT_EMAIL in .env.
2. Keep open-market purchases (code P) and sales (code S) of common stock, with the filing date (when the market
   learns about the trade), the insider and their role.
3. Classify each trade the way Cohen, Malloy and Pomorski (2012) do. An insider who traded in the same calendar month in
   each of the previous three years is "routine" (planned, uninformative). One who traded in each of the previous three
   years but not on that pattern is "opportunistic". Others are unclassified.
4. Event features use only trades filed by the day after day 0 (so known at the close of day 1):
     ins_net_90     signed log of net purchase value, trades dated in the 90 days up to day 1
     ins_buyers_90, ins_sellers_90   number of distinct insiders buying / selling
     ins_buy_share  buying insiders / insiders who traded (missing if none traded)
     ins_officer_buy  a CEO, CFO, president or chair bought
     ins_buy_recent   any purchase dated within the 10 days before day 0 through day 1
     opp_buyers_90, opp_sellers_90, opp_net_90   the same for opportunistic insiders only
Output: data/processed/insider_trades.parquet, data/processed/insider_features.parquet
Run: python src/pipeline/08_insider_features.py      (download only: INS_DOWNLOAD_ONLY=1)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, io, re, time, zipfile, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests
from config import PROC, ROOT, find_file

RAW = ROOT / "data" / "raw" / "sec_insider"; RAW.mkdir(parents=True, exist_ok=True)
env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.startswith("#"))
UA = {"User-Agent": f"academic research {env.get('SEC_CONTACT_EMAIL', '').strip()}"}
BASES = ["https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/",
         "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/"]

# 1. download
today = pd.Timestamp.today()
quarters = [f"{y}q{q}" for y in range(2006, today.year + 1) for q in range(1, 5) if pd.Timestamp(y, 3 * q, 1) < today]
for qn in quarters:
    f = RAW / f"{qn}_form345.zip"
    if f.exists() and f.stat().st_size > 1000:
        continue
    for b in BASES:
        r = requests.get(b + f.name, headers=UA, timeout=120)
        time.sleep(0.3)
        if r.status_code == 200 and r.content[:2] == b"PK":
            f.write_bytes(r.content); print(f"  downloaded {f.name} ({len(r.content) / 1e6:.1f} MB)", flush=True); break
    else:
        print(f"  {f.name}: not available yet")
if os.environ.get("INS_DOWNLOAD_ONLY") == "1":
    raise SystemExit("download finished")

# 2. parse
def tsv(z, name, cols):
    n = [m for m in z.namelist() if m.upper().endswith(name)][0]
    d = pd.read_csv(z.open(n), sep="\t", dtype=str, quoting=3, on_bad_lines="skip", low_memory=False)
    d.columns = [c.upper() for c in d.columns]
    return d[[c for c in cols if c in d.columns]]
dt = lambda s: pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
TR = ROOT / "data" / "processed" / "insider_trades.parquet"
parts = []
for f in sorted(RAW.glob("*_form345.zip")):
    with zipfile.ZipFile(f) as z:
        s = tsv(z, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
        o = tsv(z, "REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        t = tsv(z, "NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                          "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    t = t[t.TRANS_CODE.isin(["P", "S"])]
    o = o.drop_duplicates("ACCESSION_NUMBER")
    m = t.merge(s, on="ACCESSION_NUMBER").merge(o, on="ACCESSION_NUMBER", how="left")
    m = m[m.DOCUMENT_TYPE.isin(["4", "4/A", "5", "5/A"])]
    parts.append(pd.DataFrame({"issuer_cik": pd.to_numeric(m.ISSUERCIK, errors="coerce"), "ticker": m.ISSUERTRADINGSYMBOL.str.upper(),
                               "owner": pd.to_numeric(m.RPTOWNERCIK, errors="coerce"), "rel": m.RPTOWNER_RELATIONSHIP.fillna(""),
                               "title": m.RPTOWNER_TITLE.fillna(""), "trans_date": dt(m.TRANS_DATE), "filing_date": dt(m.FILING_DATE),
                               "buy": (m.TRANS_CODE == "P").astype(np.int8),
                               "value": pd.to_numeric(m.TRANS_SHARES, errors="coerce") * pd.to_numeric(m.TRANS_PRICEPERSHARE, errors="coerce")}))
    if f == sorted(RAW.glob("*_form345.zip"))[0]:
        print(f"first file {f.name}: {len(parts[-1]):,} open-market trades")
T = pd.concat(parts, ignore_index=True).dropna(subset=["issuer_cik", "owner", "trans_date", "filing_date"])
T = T[(T.trans_date >= "2000-01-01") & (T.filing_date >= T.trans_date) & (T.filing_date - T.trans_date <= pd.Timedelta(days=365))]
T["value"] = T.value.clip(0, 5e9).fillna(0)
T = T.drop_duplicates(["issuer_cik", "owner", "trans_date", "buy", "value"])
print(f"open-market trades: {len(T):,} ({T.buy.mean():.0%} purchases), {T.trans_date.min().date()} to {T.trans_date.max().date()}")

# 3. routine vs opportunistic
T["y"], T["m"] = T.trans_date.dt.year, T.trans_date.dt.month
ym = T[["issuer_cik", "owner", "y", "m"]].drop_duplicates()
key = lambda d, dy, mm=None: d.issuer_cik.astype(np.int64).astype(str) + "_" + d.owner.astype(np.int64).astype(str) + "_" + (d.y - dy).astype(str) + ("_" + d.m.astype(str) if mm else "")
ym_set, y_set = set(key(ym, 0, True)), set(key(ym, 0))
routine = np.ones(len(T), bool); active3 = np.ones(len(T), bool)
for dy in (1, 2, 3):
    routine &= key(T, dy, True).isin(ym_set).values
    active3 &= key(T, dy).isin(y_set).values
T["kind"] = np.where(active3 & routine, "routine", np.where(active3, "opp", "other"))
T["officer"] = T.title.str.contains(r"\b(?:ceo|chief executive|cfo|chief financial|president|chair)", case=False, regex=True) | \
               T.rel.str.contains("officer", case=False)
T.to_parquet(TR, index=False)
print(T.groupby(["kind", "buy"]).size().unstack().rename(columns={0: "sales", 1: "purchases"}).to_string())

# 4. map issuer CIK -> permno, build event features
ids = pd.read_csv(find_file("compustat_ids"), usecols=["gvkey", "cik"]).dropna().drop_duplicates("cik", keep="last")
lk = pd.read_csv(find_file("ccm_link")); lk = lk[lk.linkprim.isin(["P", "C"])].dropna(subset=["permno"])
lk["linkenddt"] = pd.to_datetime(lk.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-01-01"))
gv2p = lk.sort_values("linkenddt").drop_duplicates("gvkey", keep="last").set_index("gvkey").permno.astype(np.int64)
cik2p = ids.assign(permno=ids.gvkey.map(gv2p)).dropna().drop_duplicates("cik").set_index("cik").permno.astype(np.int64)
T["permno"] = T.issuer_cik.map(cik2p)
print(f"trades matched to a CRSP stock: {T.permno.notna().mean():.0%}")
T = T.dropna(subset=["permno"]); T["permno"] = T.permno.astype(np.int64)
T["sv"] = np.where(T.buy == 1, T.value, -T.value)

ev = pd.read_parquet(PROC / "events.parquet", columns=["permno", "day0"]).drop_duplicates()
ev["permno"] = ev.permno.astype(np.int64)
ev = ev[ev.day0 >= "2006-04-01"].reset_index(drop=True); ev["eid"] = np.arange(len(ev))
covered = set(T.permno)
slog = lambda v: np.sign(v) * np.log1p(np.abs(v))
out = []
for Y in sorted(ev.day0.dt.year.unique()):
    e = ev[ev.day0.dt.year == Y]
    tt = T[(T.trans_date >= pd.Timestamp(f"{Y - 1}-09-01")) & (T.filing_date <= pd.Timestamp(f"{Y + 1}-01-10"))]
    m = e.merge(tt, on="permno")
    lim = m.day0 + pd.Timedelta(days=1)                       # filed by the next calendar day: public by day 1's close
    m = m[(m.filing_date <= lim) & (m.trans_date <= lim) & (m.trans_date >= m.day0 - pd.Timedelta(days=90))]
    g = m.groupby("eid")
    F = pd.DataFrame(index=e.eid)
    F["ins_net_90"] = slog(g.sv.sum())
    F["ins_buyers_90"] = m[m.buy == 1].groupby("eid").owner.nunique()
    F["ins_sellers_90"] = m[m.buy == 0].groupby("eid").owner.nunique()
    F["ins_officer_buy"] = m[(m.buy == 1) & m.officer].groupby("eid").size().gt(0).astype(float)
    F["ins_buy_recent"] = m[(m.buy == 1) & (m.trans_date >= m.day0 - pd.Timedelta(days=10))].groupby("eid").size().gt(0).astype(float)
    mo = m[m.kind == "opp"]
    F["opp_net_90"] = slog(mo.groupby("eid").sv.sum())
    F["opp_buyers_90"] = mo[mo.buy == 1].groupby("eid").owner.nunique()
    F["opp_sellers_90"] = mo[mo.buy == 0].groupby("eid").owner.nunique()
    has = e.permno.isin(covered).values
    F.loc[has] = F.loc[has].fillna(0.0)                       # firms in the insider data with no trades: zero, not missing
    traded = F.ins_buyers_90 + F.ins_sellers_90
    F["ins_buy_share"] = np.where(traded > 0, F.ins_buyers_90 / traded.where(traded > 0), np.nan)
    F["permno"], F["day0"] = e.permno.values, e.day0.values
    out.append(F.reset_index(drop=True))
F = pd.concat(out, ignore_index=True)
print("\n=== Insider features (EDA) ===")
print(F.drop(columns=["permno", "day0"]).describe().T[["count", "mean", "50%", "max"]].round(3).to_string())
print(f"events with any insider trade in the 90 days: {(F.ins_buyers_90 + F.ins_sellers_90).gt(0).mean():.1%}; "
      f"with an opportunistic purchase: {F.opp_buyers_90.gt(0).mean():.1%}")
F.to_parquet(PROC / "insider_features.parquet", index=False)
print(f"saved {len(F):,} rows -> data/processed/insider_features.parquet")
