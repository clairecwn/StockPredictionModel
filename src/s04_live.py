"""
Step 4: live prediction for one ticker and one event.

Example (Nvidia, board probe reported Mon 9 Nov 2026):
  python src/s04_live.py --ticker NVDA --date 2026-11-09 --cls leadership \
      --headline "Nvidia board probes CEO over relationship with alleged intelligence asset"

Prints: the features it built, the model prediction with an 80% interval for each target, recent headlines
with FinBERT sentiment, and what has actually happened since day 1 so far. Research tool, not financial advice.
"""
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from config import MODELS, RESULTS
from taxonomy import CLASSES
from live_features import features
from news import sentiment_summary

ap = argparse.ArgumentParser()
ap.add_argument("--ticker", required=True); ap.add_argument("--date", required=True, help="event day 0, YYYY-MM-DD")
ap.add_argument("--cls", required=True, choices=CLASSES); ap.add_argument("--headline", default="")
ap.add_argument("--news-days", type=int, default=30)
a = ap.parse_args()

f, px = features(a.ticker, a.date, a.cls, a.headline)
from config import PUBLIC
fred_f = PUBLIC / "fred_daily.csv"
if fred_f.exists():                                   # market regime on day 0 (re-run s01_fred.py to refresh)
    fr = pd.read_csv(fred_f, parse_dates=["date"]).set_index("date").sort_index()
    if "umcsent" in fr: fr["umcsent"] = fr.umcsent.shift(31)
    row = fr[fr.index <= f["_day0"]].iloc[-1]
    for c in ("vix", "epu", "hy_oas", "umcsent"):
        if c in row: f[c] = float(row[c])
day0 = f["_day0"]
print(f"\n=== {a.ticker}: {a.cls} event, day 0 = {day0.date()}, prices through {f['_last_price_date'].date()} ===")
print("\n--- Features built from prices (same definitions as training) ---")
show = {k: v for k, v in f.items() if not k.startswith("_") and not k.startswith(("has_", "cls_")) and pd.notna(v)}
print(pd.Series(show).round(4).to_string())
if not f["_have_day1"]:
    print("NOTE: day 1 has not closed yet; car_0_1 and abn_volume_0_1 use day 0 only.")

print("\n--- Model predictions (abnormal return relative to the average event that month; 80% interval = conformalized quantile regression) ---")
rows = []
for t in ("drift_1_5", "drift_1_20", "drift_1_60", "trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"):
    mdl = joblib.load(MODELS / f"{t}.joblib")
    X = pd.DataFrame([{k: f.get(k, np.nan) for k in mdl["feats"]}]).astype(float)
    if mdl.get("version", 1) >= 2:
        H = {"drift_1_5": 4, "drift_1_20": 19, "drift_1_60": 59, "trough_2_20": 19}
        sc = max(f["idio_vol_pre"] / np.sqrt(252), 0.003) * np.sqrt(H[t]) if mdl["return_target"] else 1.0
        p = float(mdl["model"].predict(X)[0]) * sc
        lo = (float(mdl["q_lo"].predict(X)[0]) - mdl["cqr"]) * sc
        hi = (float(mdl["q_hi"].predict(X)[0]) + mdl["cqr"]) * sc
    else:
        p = float(mdl["model"].predict(X)[0])
        lo, hi = p + mdl["q10"], p + mdl["q90"]
    if t in ("vol_ratio_2_20", "abn_volume_2_20"):           # trained on log scale
        p, lo, hi = np.exp(p), np.exp(lo), np.exp(hi); unit = "x normal"
        meaning = {"vol_ratio_2_20": "volatility days 2-20 vs pre-event volatility",
                   "abn_volume_2_20": "volume days 2-20 vs normal volume"}[t]
    else:
        p, lo, hi = p * 100, lo * 100, hi * 100; unit = "% abnormal return"
        meaning = {"drift_1_5": "cumulative abnormal return days 2-5", "drift_1_20": "days 2-20",
                   "drift_1_60": "days 2-60", "trough_2_20": "worst cumulative point within days 2-20"}[t]
    rows.append({"target": t, "prediction": round(p, 2), "low_80": round(lo, 2), "high_80": round(hi, 2), "unit": unit, "meaning": meaning})
pred = pd.DataFrame(rows); print(pred.to_string(index=False))

d20 = pred.set_index("target").loc["drift_1_20", "prediction"]
print("\n--- Reading ---")
if d20 > 1.0:    verdict = "model leans POSITIVE: predicted 20-day drift above +1% (the backtest's trading threshold)"
elif d20 < -1.0: verdict = "model leans NEGATIVE: predicted 20-day drift below -1%"
else:            verdict = "model is NEUTRAL: predicted 20-day drift within +/-1%, no edge after costs"
print(verdict)
print("Context: out-of-sample Spearman for 20-day drift is about 0.04, so treat the direction as a weak tilt. "
      "The volume, volatility and drawdown predictions are far more reliable (Spearman 0.3 to 0.6).")
print(f"Realised abnormal return from day 2 to {f['_last_price_date'].date()} so far: {f['_realised_ar_since_day1']*100:+.2f}%")

print(f"\n--- Headlines, last {a.news_days} days (Finnhub) with FinBERT sentiment ---")
try:
    n, summ = sentiment_summary(a.ticker, day0 - pd.Timedelta(days=a.news_days), f["_last_price_date"] + pd.Timedelta(days=1))
    print(pd.Series(summ).round(3).to_string())
    print(n.head(15)[["date", "label", "headline"]].to_string(index=False))
except Exception as e:
    print(f"news unavailable: {e}")

out = RESULTS / "live"; out.mkdir(exist_ok=True)
pred.to_csv(out / f"{a.ticker}_{day0.date()}_prediction.csv", index=False)
print(f"\nsaved -> results/live/{a.ticker}_{day0.date()}_prediction.csv")
