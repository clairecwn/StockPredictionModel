"""Load the WRDS files saved by hand. Column names differ between CRSP versions and WRDS export
options, so each loader matches columns by a list of possible names (case-insensitive)."""
import numpy as np
import pandas as pd
from config import find_file, PROC

def _pick(cols, options, required=True, what=""):
    low = {c.lower(): c for c in cols}
    for o in options:
        if o in low:
            return low[o]
    if required:
        raise KeyError(f"Could not find a column for {what}. Columns in file: {list(cols)}")
    return None

def _header(path):
    return pd.read_csv(path, nrows=0).columns

def load_crsp_daily():
    """CRSP daily stock file -> parquet cache with columns permno,date,ret,prc,vol,shrout,bid,ask."""
    cache = PROC / "crsp_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    f = find_file("crsp_daily")
    if f is None:
        raise FileNotFoundError("data/raw/wrds/crsp_daily.csv(.gz) not found. See docs/WRDS_GUIDE.md step 3.")
    cols = _header(f)
    m = {"permno": _pick(cols, ["permno"], what="PERMNO"),
         "date": _pick(cols, ["dlycaldt", "date", "caldt"], what="date"),
         "ret": _pick(cols, ["dlyret", "ret"], what="daily return"),
         "prc": _pick(cols, ["dlyprc", "prc"], what="price"),
         "vol": _pick(cols, ["dlyvol", "vol"], what="volume"),
         "shrout": _pick(cols, ["shrout", "dlyshrout"], required=False),
         "cap": _pick(cols, ["dlycap"], required=False),
         "bid": _pick(cols, ["dlybid", "bid"], required=False),
         "ask": _pick(cols, ["dlyask", "ask"], required=False),
         "mkt": _pick(cols, ["vwretd"], required=False)}
    m = {k: v for k, v in m.items() if v}
    print(f"reading {f.name} (first time only; a few minutes)...", flush=True)
    parts = []
    for chunk in pd.read_csv(f, usecols=list(m.values()), chunksize=2_000_000, low_memory=False):
        d = chunk.rename(columns={v: k for k, v in m.items()})
        d["date"] = pd.to_datetime(d.date)
        for c in d.columns.drop(["permno", "date"]):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d["prc"] = d.prc.abs()
        if "cap" not in d and "shrout" in d:
            d["cap"] = d.prc * d.shrout
        parts.append(d.dropna(subset=["permno"]).astype({"permno": "int32"}))
        print(f"  ...{sum(len(x) for x in parts):,} rows", flush=True)
    d = pd.concat(parts, ignore_index=True).sort_values(["permno", "date"])
    d.to_parquet(cache, index=False)
    print(f"  {len(d):,} rows, {d.permno.nunique():,} stocks, {d.date.min().date()} to {d.date.max().date()}")
    return d

def load_market():
    """Daily market return = Fama-French Mkt-RF + RF."""
    f = find_file("ff_daily")
    if f is None:
        raise FileNotFoundError("data/raw/wrds/ff_daily.csv not found. See docs/WRDS_GUIDE.md step 4.")
    d = pd.read_csv(f)
    c = d.columns
    date, mktrf, rf = _pick(c, ["date", "dateff"], what="date"), _pick(c, ["mktrf", "mkt-rf", "mkt_rf"], what="Mkt-RF"), _pick(c, ["rf"], what="RF")
    out = pd.DataFrame({"date": pd.to_datetime(d[date].astype(str)), "mkt": d[mktrf].astype(float) + d[rf].astype(float)})
    if out.mkt.std() > 0.1:          # file in percent rather than decimals
        out["mkt"] /= 100
    return out.sort_values("date")

def load_keydev():
    """Capital IQ Key Developments -> gvkey, ann_date, eventtype, headline (one row per key development)."""
    f = find_file("ciq_keydev")
    if f is None:
        return None
    c = _header(f)
    m = {"keydevid": _pick(c, ["keydevid"], required=False),
         "gvkey": _pick(c, ["gvkey"], what="gvkey"),
         "ann_date": _pick(c, ["announcedate", "announceddate", "mostimportantdateutc"], what="announce date"),
         "eventtype": _pick(c, ["eventtype", "keydeveventtypename"], what="event type"),
         "headline": _pick(c, ["headline"], required=False)}
    m = {k: v for k, v in m.items() if v}
    d = pd.read_csv(f, usecols=list(m.values()), low_memory=False).rename(columns={v: k for k, v in m.items()})
    d["ann_date"] = pd.to_datetime(d.ann_date, errors="coerce")
    d["gvkey"] = pd.to_numeric(d.gvkey, errors="coerce")
    d = d.dropna(subset=["gvkey", "ann_date"])
    if "keydevid" in d:
        d = d.drop_duplicates("keydevid")
    return d

def load_execucomp():
    f = find_file("execucomp_ceo")
    d = pd.read_csv(f, low_memory=False, parse_dates=["becameceo", "leftofc"])
    return d

def load_ccm():
    d = pd.read_csv(find_file("ccm_link"), parse_dates=["linkdt", "linkenddt"])
    d["linkenddt"] = d.linkenddt.fillna(pd.Timestamp("2099-12-31"))
    return d.dropna(subset=["permno"]).astype({"permno": "int32"})
