"""Step 11: score the forward log once outcomes are known (truly out of sample: every prediction was logged before
its outcome existed). Writes docs/data/forward_scores.json, shown in the page's reliability panel.

For each logged (stock, trading day) at least 20 trading days old:
  worst point   realised minimum total return over the next 19 trading days vs the logged 1-in-10 bad case
                (promise: breached about 10% of the time or less)
  signal        realised total return over the next 5 trading days for BUY vs WAIT days (BUY should be higher)
  warning       realised worst point for warned vs not-warned rows (warned should be worse)
Counts stay small for months; the page shows n next to every number and says when it is too early to judge.
Run: python src/s11_forward_eval.py      Research only. Not investment advice.
"""
import json, warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np, pandas as pd
from config import ROOT
from live_engine import yahoo_frames

DOCS = ROOT / "docs" / "data"; logf = DOCS / "forward_log.csv"
H, H5, MIN_N = 19, 5, 30
log = pd.read_csv(logf) if logf.exists() else pd.DataFrame()
out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"), "n_logged": int(len(log))}
if len(log) and "price_date" in log:
    log["price_date"] = pd.to_datetime(log.price_date)
    frames = yahoo_frames(sorted(log.ticker.unique()), period="2y")
    rows = []
    for r in log.itertuples():
        f = frames.get(r.ticker)
        if f is None: continue
        i = f.index.searchsorted(r.price_date)
        if i >= len(f) or f.index[i] != r.price_date or i + H >= len(f): continue      # outcome not known yet
        p0 = f.c.iloc[i]; path = f.c.iloc[i + 1:i + 1 + H].values / p0 - 1
        rows.append({"ticker": r.ticker, "signal": r.signal, "warning": r.warning, "bad": r.bad_case_10pct,
                     "worst": min(0.0, path.min()), "ret5": path[H5 - 1]})
    S = pd.DataFrame(rows); out["n_scored"] = int(len(S))
    if len(S):
        b = S.dropna(subset=["bad"])
        out["worst_point"] = {"n": int(len(b)), "breach_rate": float((100 * b.worst < b.bad).mean()) if len(b) else None, "promise": 0.10}
        sg = S[S.signal.isin(["BUY", "WAIT"])]
        out["signal"] = {k: {"n": int((sg.signal == k).sum()), "mean_ret5_pct": float(100 * sg.ret5[sg.signal == k].mean())}
                         for k in ("BUY", "WAIT") if (sg.signal == k).any()}
        w = S.dropna(subset=["warning"]); w["warning"] = w.warning.astype(str).str.lower() == "true"
        out["warning"] = {k: {"n": int((w.warning == v).sum()), "mean_worst_pct": float(100 * w.worst[w.warning == v].mean())}
                          for k, v in (("warned", True), ("not_warned", False)) if (w.warning == v).any()}
    out["too_early"] = out.get("n_scored", 0) < MIN_N
    out["note"] = ("Rows on the same stock overlap in time, so the effective sample is smaller than n. "
                   f"Fewer than {MIN_N} scored rows = too early to judge.")
(DOCS / "forward_scores.json").write_text(json.dumps(out, indent=1, default=lambda o: None if o != o else o))
print(json.dumps(out, indent=1))
