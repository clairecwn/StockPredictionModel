"""Turns the WRDS event tables NTU subscribes to into one list of dated company events.

Source (file in data/raw/wrds/)            -> class
  aa_officer_changes  (Audit Analytics)     -> leadership       (date = 8-K filing date)
  aa_legal_party + aa_legal_case            -> legal_regulatory (date = case filed; company is defendant)
  aa_activism                               -> activist_short   (date = 13D/proxy filing date)
  aa_restatements                           -> accounting_fraud (date = 8-K item 4.02 / filing date)
  ibes_recs  (I/B/E/S)                      -> analyst          (date = up/downgrade announcement)
  ibes_actuals_q + ibes_summary_q           -> earnings_guidance (date = EPS announcement; surprise vs prior consensus)
Every row is mapped to a Compustat gvkey using compustat_ids (CIK and CUSIP)."""
import re
import numpy as np, pandas as pd
from config import find_file

SCANDAL = re.compile(r"misconduct|violation|investigat|fraud|harass|inappropriate|terminated for cause|"
                     r"personal conduct|relationship|ethic|probe|indict|spy|bribe", re.I)

def col(d, c, default=np.nan):
    """d[c] if the column exists, else a constant Series (so missing optional columns never crash)."""
    return d[c] if c in d else pd.Series(default, index=d.index)

def _read(stem, **kw):
    f = find_file(stem)
    if f is None:
        print(f"  (no {stem} file, skipped)")
        return None
    d = pd.read_csv(f, low_memory=False, **kw)
    d.columns = [c.lower() for c in d.columns]
    return d

def _ids():
    c = _read("compustat_ids", dtype=str).copy()               # keep leading zeros in CUSIP / CIK
    c["gvkey"] = pd.to_numeric(c.gvkey, errors="coerce")
    c["cik"] = pd.to_numeric(c.cik, errors="coerce")
    c["cusip8"] = c.cusip.astype(str).str[:8]
    cik = c.dropna(subset=["cik"]).drop_duplicates("cik", keep="last")[["cik", "gvkey"]]
    cus = c.drop_duplicates("cusip8", keep="last")[["cusip8", "gvkey"]]
    return cik, cus

def _by_cik(d, cik_map, col="company_fkey"):
    d = d.copy()
    d["cik"] = pd.to_numeric(d[col], errors="coerce")
    return d.merge(cik_map, on="cik", how="inner")

def officer_changes(cik_map):
    d = _read("aa_officer_changes")
    if d is None: return None
    top = d[[c for c in ("is_ceo", "is_cfo", "is_chair", "is_president") if c in d]].fillna(0).astype(float).max(axis=1) > 0
    d = _by_cik(d[top].copy(), cik_map)
    text = col(d, "action", "").astype(str) + " " + col(d, "reasons", "").astype(str)
    leaving = text.str.contains(r"resign|retire|terminat|dismiss|death|leave|depart|remov|step", case=False)
    return pd.DataFrame({"gvkey": d.gvkey, "ann_date": pd.to_datetime(d.file_date, errors="coerce"), "cls": "leadership",
                         "scandal_word": text.str.contains(SCANDAL).astype(int),
                         "ceo_involved": pd.to_numeric(col(d, "is_ceo", 0), errors="coerce").fillna(0).astype(int),
                         "exec_departure": leaving.astype(int),
                         "forced_exit": text.str.contains(r"terminat|dismiss|for cause|remov", case=False).astype(int),
                         "source": "aa_officer"})

def legal(cik_map):
    p = _read("aa_legal_party")
    if p is None: return None
    p = p[pd.to_numeric(p.defendant, errors="coerce").fillna(0) > 0]          # company is a defendant
    p = _by_cik(p, cik_map)
    case = _read("aa_legal_case")
    if case is not None:
        cat = [c for c in case.columns if c.startswith("is_category_type_")]
        case["n_categories"] = case[cat].fillna(0).sum(axis=1) if cat else 0
        case = case[["legal_case_key", "claim_dollars", "n_categories"] + [c for c in ("settlement_dollars",) if c in case]]
        p = p.merge(case, on="legal_case_key", how="left")
    date = pd.to_datetime(col(p, "case_start_date"), errors="coerce")
    return pd.DataFrame({"gvkey": p.gvkey, "ann_date": date, "cls": "legal_regulatory",
                         "scandal_word": col(p, "title", "").astype(str).str.contains(SCANDAL).astype(int),
                         "gov_involved": col(p, "gov_name").notna().astype(int),
                         "log_claim": np.log1p(pd.to_numeric(col(p, "claim_dollars"), errors="coerce").fillna(0)),
                         "source": "aa_legal"})

def activism(cik_map):
    d = _read("aa_activism")
    if d is None: return None
    d = _by_cik(d, cik_map)
    return pd.DataFrame({"gvkey": d.gvkey, "ann_date": pd.to_datetime(d.iss_file_date, errors="coerce"), "cls": "activist_short",
                         "scandal_word": 0, "activist_stake": pd.to_numeric(col(d, "agg_percent_class_owned"), errors="coerce"),
                         "source": "aa_activism"})

def restatements(cik_map):
    d = _read("aa_restatements")
    if d is None: return None
    d = _by_cik(d, cik_map)
    date = pd.to_datetime(col(d, "date_of_8k_402"), errors="coerce").fillna(pd.to_datetime(d.file_date, errors="coerce"))
    num = lambda c: pd.to_numeric(col(d, c, 0), errors="coerce").fillna(0)
    return pd.DataFrame({"gvkey": d.gvkey, "ann_date": date, "cls": "accounting_fraud",
                         "scandal_word": num("res_fraud").astype(int), "gov_involved": ((num("res_sec_investigation") + num("res_regulatory_investigation")) > 0).astype(int),
                         "restate_adverse": num("res_adverse").astype(int), "source": "aa_restatement"})

def analyst(cus_map):
    d = _read("ibes_recs", dtype=str, usecols=lambda c: c.lower() in {"cusip", "ticker", "anndats", "amaskcd", "estimid", "ireccd"})
    if d is None: return None
    d["ireccd"] = pd.to_numeric(d.ireccd, errors="coerce")
    d["anndats"] = pd.to_datetime(d.anndats, errors="coerce")
    d = d.dropna(subset=["anndats", "ireccd"]).sort_values(["ticker", "estimid", "amaskcd", "anndats"])
    d["prev"] = d.groupby(["ticker", "estimid", "amaskcd"]).ireccd.shift()
    d = d[d.prev.notna() & (d.ireccd != d.prev)]                               # actual changes only, not reiterations
    d["chg"] = d.ireccd - d.prev                                               # >0 = downgrade (1 strong buy .. 5 sell)
    g = d.groupby(["cusip", "anndats"]).agg(n_downgrades=("chg", lambda s: (s > 0).sum()), n_upgrades=("chg", lambda s: (s < 0).sum())).reset_index()
    g["cusip8"] = g.cusip.astype(str).str[:8]
    g = g.merge(cus_map, on="cusip8", how="inner")
    return pd.DataFrame({"gvkey": g.gvkey, "ann_date": g.anndats, "cls": "analyst", "scandal_word": 0,
                         "rec_net": g.n_downgrades - g.n_upgrades, "source": "ibes"})

def _ibes_q():
    """Quarterly I/B/E/S actuals + the last consensus before each announcement (both on the same split-adjusted basis)."""
    a = _read("ibes_actuals_q", dtype={"cusip": str, "ticker": str})
    s = _read("ibes_summary_q", dtype={"cusip": str, "ticker": str})
    if a is None or s is None:
        return None, None
    a = a[a.get("pdicity", "QTR").eq("QTR") if "pdicity" in a else slice(None)]
    a["anndats"], a["pends"] = pd.to_datetime(a.anndats, errors="coerce"), pd.to_datetime(a.pends, errors="coerce")
    a["value"] = pd.to_numeric(a.value, errors="coerce")
    # an announcement must come after the quarter ends and within ~6 months (drops stale or mis-dated rows)
    a = a.dropna(subset=["anndats", "pends", "value"])
    lag = (a.anndats - a.pends).dt.days
    a = a[(lag >= 0) & (lag <= 180)]
    a = a.sort_values("anndats").drop_duplicates(["ticker", "pends"], keep="first")
    s["statpers"], s["fpedats"] = pd.to_datetime(s.statpers, errors="coerce"), pd.to_datetime(s.fpedats, errors="coerce")
    for c in ("meanest", "medest", "numest", "stdev"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["statpers", "fpedats", "meanest"])
    # consensus = latest monthly snapshot strictly BEFORE the announcement, for the same fiscal quarter, at most 90 days old
    m = a[["ticker", "cusip", "pends", "anndats", "value"]].merge(
        s[["ticker", "fpedats", "statpers", "meanest", "medest", "numest", "stdev"]],
        left_on=["ticker", "pends"], right_on=["ticker", "fpedats"], how="inner")
    m = m[(m.statpers < m.anndats) & ((m.anndats - m.statpers).dt.days <= 90)]
    m = m.sort_values("statpers").drop_duplicates(["ticker", "pends"], keep="last")
    return m, s

def earnings(cus_map):
    """Quarterly earnings announcements with the surprise vs analyst consensus (post-earnings drift, Bernard & Thomas 1989).
    Surprises are scaled by analyst disagreement and by |consensus|, both on I/B/E/S's own split-adjusted basis.
    (Not by CRSP price: I/B/E/S figures are restated to today's split basis, CRSP prices are not, which would distort split stocks.)"""
    m, _ = _ibes_q()
    if m is None: return None
    surp = m.value - m.meanest
    sd = m.stdev.where(m.numest >= 2)
    m["eps_sue"] = (surp / sd.clip(lower=0.01)).clip(-10, 10)                     # surprise in units of analyst disagreement
    m["eps_surp_rel"] = (surp / m.meanest.abs().clip(lower=0.05)).clip(-5, 5)      # surprise relative to the size of the forecast
    m["eps_beat"] = np.sign(surp.round(4))                                         # +1 beat, 0 in line, -1 miss
    m["est_dispersion"] = (sd / m.meanest.abs().clip(lower=0.05)).clip(upper=5)
    m["cusip8"] = m.cusip.astype(str).str[:8]
    n0 = len(m)
    # 1st choice: WRDS's official IBES-CRSP link (ticker -> PERMNO valid on the announcement date, best match score),
    # then PERMNO -> GVKEY through the CCM link valid on that date. Fallback: CUSIP snapshot for anything left.
    lk, ccm = _read("ibes_crsp_link"), _read("ccm_link")
    parts, done = [], pd.Series(False, index=m.index)
    if lk is not None and ccm is not None:
        lk["score"] = pd.to_numeric(lk.score, errors="coerce")
        lk = lk[lk.score <= 2].copy()
        lk["sdate"], lk["edate"] = pd.to_datetime(lk.sdate, errors="coerce"), pd.to_datetime(lk.edate, errors="coerce")
        j = m.reset_index().merge(lk[["ticker", "permno", "sdate", "edate", "score"]], on="ticker", how="inner")
        j = j[(j.anndats >= j.sdate) & (j.anndats <= j.edate)].sort_values("score").drop_duplicates("index")
        ccm["linkdt"] = pd.to_datetime(ccm.linkdt, errors="coerce")
        ccm["linkenddt"] = pd.to_datetime(ccm.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        ccm["gvkey"] = pd.to_numeric(ccm.gvkey, errors="coerce")
        ccm["pri"] = (~col(ccm, "linkprim", "").astype(str).isin(["P", "C"])).astype(int)
        j = j.merge(ccm[["gvkey", "permno", "linkdt", "linkenddt", "pri"]], on="permno", how="inner")
        j = j[(j.anndats >= j.linkdt) & (j.anndats <= j.linkenddt)].sort_values("pri").drop_duplicates("index")
        parts.append(j.set_index("index").drop(columns=["permno", "sdate", "edate", "score", "linkdt", "linkenddt", "pri"]))
        done = m.index.isin(j["index"])
    rest = m[~done].merge(cus_map, on="cusip8", how="inner")
    parts.append(rest)
    m = pd.concat(parts, ignore_index=True)
    print(f"    I/B/E/S quarterly announcements with a prior consensus: {n0:,}; matched: {len(m):,} ({len(m) / max(n0, 1):.0%}; "
          f"{int(done.sum()):,} via the official IBES-CRSP link, {len(rest):,} via CUSIP fallback)")
    return pd.DataFrame({"gvkey": m.gvkey, "ann_date": m.anndats, "cls": "earnings_guidance", "scandal_word": 0,
                         "eps_sue": m.eps_sue, "eps_surp_rel": m.eps_surp_rel, "eps_beat": m.eps_beat,
                         "est_dispersion": m.est_dispersion, "n_analysts": m.numest, "source": "ibes_eps"})

def coverage(cus_map):
    """Number of analysts covering each firm over time (any event): drift is stronger where coverage is thin (Hong, Lim & Stein 2000)."""
    s = _read("ibes_summary_q", dtype={"cusip": str}, usecols=lambda c: c.lower() in {"cusip", "statpers", "numest"})
    if s is None: return None
    s["statpers"] = pd.to_datetime(s.statpers, errors="coerce")
    s["numest"] = pd.to_numeric(s.numest, errors="coerce")
    s["cusip8"] = s.cusip.astype(str).str[:8]
    s = s.dropna(subset=["statpers", "numest"]).merge(cus_map, on="cusip8", how="inner")
    s = s.groupby(["gvkey", "statpers"], as_index=False).numest.max()
    return s.rename(columns={"statpers": "cov_date", "numest": "analyst_cov"})

def build_raw_events(return_maps=False):
    cik_map, cus_map = _ids()
    parts = []
    for name, fn, arg in (("officer changes", officer_changes, cik_map), ("legal cases", legal, cik_map),
                          ("activism", activism, cik_map), ("restatements", restatements, cik_map), ("analyst changes", analyst, cus_map),
                          ("earnings announcements", earnings, cus_map)):
        x = fn(arg)
        if x is not None:
            x = x.dropna(subset=["ann_date"])
            print(f"  {name}: {len(x):,} events")
            parts.append(x)
    ev = pd.concat(parts, ignore_index=True)
    ev["gvkey"] = pd.to_numeric(ev.gvkey, errors="coerce")
    ev = ev.dropna(subset=["gvkey"])
    return (ev, cus_map) if return_maps else ev
