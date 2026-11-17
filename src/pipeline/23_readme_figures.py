"""Step 23: figures for the README (saved to docs/figures/, aggregate statistics only, no row-level licensed data).

  01_events_per_year.png     event counts by year and class (tradeable universe)
  02_car_path.png            average cumulative abnormal return, days 0-1 to 20, by class, with 95% intervals
  03_pead.png                reaction vs 60-day drift by earnings-surprise decile
  04_skill.png               predicted vs realised by prediction decile, out of sample (risk is predictable, direction is not)
  05_bound_by_volatility.png downside-bound breach rate by volatility bucket, one offset vs per-bucket (Mondrian)
  06_monte_carlo.png         Monte Carlo of a $200 account over one year, by sizing tier (block bootstrap by year)
Needs: 17_train_risk_models (live), 20_position_sizing + 22_sizing_calibration (live). Run: LIVE_FEATS=1 python src/pipeline/23_readme_figures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("LIVE_FEATS", "1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import ROOT, PROC, RESULTS, OOS_FILE
from modeling import ev, target_relative

OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999", "#000000"]   # Okabe-Ito
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold", "axes.prop_cycle": plt.cycler(color=C)})
LAB = {"accounting_fraud": "Accounting", "activist_short": "Activist / short seller", "analyst": "Analyst rating",
       "earnings_guidance": "Earnings", "leadership": "Leadership", "legal_regulatory": "Legal / regulatory"}
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("saved", OUT / name)

# 1. events per year
t = ev.groupby([ev.day0.dt.year, "cls"]).size().unstack(fill_value=0)
t = t[[c for c in LAB if c in t]]
fig, ax = plt.subplots(figsize=(10, 4))
bottom = np.zeros(len(t))
for i, c in enumerate(t.columns):
    ax.bar(t.index, t[c], bottom=bottom, label=LAB[c], color=C[i], width=0.85); bottom += t[c].values
ax.set(title=f"Events per year in the tradeable universe ({len(ev):,} events, {ev.permno.nunique():,} firms)",
       xlabel="Year of announcement", ylabel="Events"); ax.legend(ncol=3, fontsize=8, frameon=False); ax.grid(axis="x")
save(fig, "01_events_per_year.png")

# 2. average CAR path
fig, ax = plt.subplots(figsize=(8, 4.5))
days = list(range(1, 21))
for i, c in enumerate(LAB):
    g = ev[ev.cls == c]
    path = pd.concat([g.car_0_1] + [g[f"ar_d{d}"] for d in range(2, 21)], axis=1).fillna(0).cumsum(axis=1) * 100
    m = path.mean(); se = path.std() / np.sqrt(len(path))
    ax.plot(days, m.values, label=f"{LAB[c]} (n={len(g):,})", color=C[i], lw=1.8)
    ax.fill_between(days, (m - 1.96 * se).values, (m + 1.96 * se).values, color=C[i], alpha=0.15)
ax.axhline(0, color="k", lw=0.8)
ax.set(title="Average cumulative abnormal return after the announcement", xlabel="Trading day (day 1 includes the day-0 reaction)",
       ylabel="Cumulative abnormal return (%)", xticks=[1, 5, 10, 15, 20])
ax.legend(fontsize=8, frameon=False); save(fig, "02_car_path.png")

# 3. PEAD
e = ev[(ev.cls == "earnings_guidance") & ev.eps_sue.notna()].copy()
e["dec"] = pd.qcut(e.eps_sue.rank(method="first"), 10, labels=False) + 1
g = e.groupby("dec").agg(r=("car_0_1", "mean"), rs=("car_0_1", "sem"), d=("drift_1_60", "mean"), ds=("drift_1_60", "sem")) * 100
fig, ax = plt.subplots(figsize=(8, 4))
x = g.index.values
ax.bar(x - 0.2, g.r, 0.4, yerr=1.96 * g.rs, label="Reaction, days 0-1", color=C[0], capsize=2)
ax.bar(x + 0.2, g.d, 0.4, yerr=1.96 * g.ds, label="Drift, days 2-60", color=C[1], capsize=2)
ax.axhline(0, color="k", lw=0.8)
ax.set(title="Earnings surprise moves the reaction, not the later drift", xlabel="Earnings-surprise decile (1 = worst miss, 10 = biggest beat)",
       ylabel="Average abnormal return (%)", xticks=x); ax.legend(frameon=False); save(fig, "03_pead.png")

# 4. skill: predicted vs realised by decile
oos = pd.read_parquet(PROC / OOS_FILE); oos.index = ev.index
panels = [("vol_ratio_2_20", "Volatility ratio, days 2-20", True), ("abn_volume_2_20", "Volume ratio, days 2-20", True),
          ("trough_2_20", "Deepest drop vs month average (%)", False), ("drift_1_20", "Drift vs month average (%)", False)]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (tg, title, logt) in zip(axs, panels):
    y = target_relative(tg).astype(float); p = oos[tg].astype(float); ok = y.notna() & p.notna()
    y, p = y[ok], p[ok]; rho = spearmanr(p, y).correlation
    dec = pd.qcut(p.rank(method="first"), 10, labels=False) + 1
    f = (np.exp if logt else (lambda v: 100 * v))
    gp, gy = f(p.groupby(dec).mean()), f(y.groupby(dec).mean())
    ax.plot(gp.index, gy.values, "o-", color=C[0], label="realised")
    ax.plot(gp.index, gp.values, "s--", color=C[1], label="predicted", ms=4)
    ax.set(title=f"{title}\nrank correlation {rho:.2f}", xlabel="Prediction decile", xticks=[1, 5, 10])
axs[0].set_ylabel("Average value"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Out-of-sample predictions vs outcomes (test years 2006-2025)", fontweight="bold"); save(fig, "04_skill.png")

# 5. bound breach by volatility bucket
mp = RESULTS / "mondrian_check.csv"
if mp.exists():
    T = pd.read_csv(mp)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, b in zip(axs, sorted(T.beta.unique())):
        d = T[T.beta == b]; x = d.vol_bucket.values
        ax.bar(x - 0.2, 100 * d.breach_one_offset, 0.4, label="one offset for all stocks", color=C[6])
        ax.bar(x + 0.2, 100 * d.breach_per_bucket, 0.4, label="offset per volatility bucket", color=C[0])
        ax.axhline(100 * b, color=C[3], ls="--", lw=1.5, label=f"promise ({b:.0%})")
        ax.set(title=f"Promise: at most {b:.0%} chance of a worse drop", xlabel="Volatility bucket (1 = calmest fifth, 5 = most volatile)",
               ylabel="Realised breach rate (%)", xticks=x)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Downside bound by volatility: offsets set on 2024, scored on later events", fontweight="bold"); save(fig, "05_bound_by_volatility.png")

# 6. Monte Carlo account simulation
tp = RESULTS / "sizing_trades.parquet"
if tp.exists():
    TR = pd.read_parquet(tp); TR["yr"] = pd.to_datetime(TR.day0).dt.year
    rng = np.random.default_rng(7); N, K, A0 = 10000, 12, 200.0       # 12 sequential 19-day trades = about one year
    tiers = [("safest", "Safest"), ("recommended", "Recommended"), ("bold", "Bold"), ("fixed_10pct", "Fixed 10%"), ("all_in", "All in (100%)")]
    years = sorted(TR.yr.unique()); stats, fans = [], {}
    for key, lab in tiers:
        d = TR[TR.tier == key]; byyr = {y: d[d.yr == y].sort_values("day0") for y in years}
        eq = np.full((N, K + 1), A0); mdd = np.zeros(N)
        for n in range(N):
            g = byyr[years[rng.integers(len(years))]]                  # block bootstrap: one market year per simulated year
            pick = g.iloc[np.sort(rng.choice(len(g), K, replace=len(g) < K))]
            e, peak, dd = A0, A0, 0.0
            for k, r in enumerate(pick.itertuples()):
                low = e * (1 + r.w * r.trough); dd = max(dd, 1 - low / peak)      # worst point inside the trade
                e = e * (1 + r.w * r.r20); peak = max(peak, e); eq[n, k + 1] = e
            mdd[n] = dd
        fans[lab] = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)
        end = eq[:, -1]
        stats.append({"tier": lab, "median_end_$": np.median(end), "p5_end_$": np.percentile(end, 5), "p95_end_$": np.percentile(end, 95),
                      "chance_down_10pct": (end < 0.9 * A0).mean(), "median_max_drawdown": np.median(mdd),
                      "p95_max_drawdown": np.percentile(mdd, 95)})
    S = pd.DataFrame(stats).round(3); print(S.to_string(index=False)); S.to_csv(OUT / "monte_carlo_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    k = np.arange(K + 1)
    for i, (key, lab) in enumerate(tiers[:4]):
        f = fans[lab]
        axs[0].fill_between(k, f[0], f[4], color=C[i], alpha=0.10); axs[0].fill_between(k, f[1], f[3], color=C[i], alpha=0.20)
        axs[0].plot(k, f[2], color=C[i], lw=2, label=lab)
    axs[0].axhline(A0, color="k", lw=0.8)
    axs[0].set(title="Simulated $200 account: median, 50% and 90% bands", xlabel="Trade number (each held 19 trading days)", ylabel="Account value ($)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].barh(S.tier, 100 * S.median_max_drawdown, color=C[0], label="median")
    axs[1].barh(S.tier, 100 * (S.p95_max_drawdown - S.median_max_drawdown), left=100 * S.median_max_drawdown, color=C[3], alpha=0.5, label="up to 95th percentile")
    axs[1].invert_yaxis(); axs[1].set(title="Largest peak-to-trough fall within the year", xlabel="Maximum drawdown (%)"); axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Monte Carlo: {N:,} simulated years drawn from out-of-sample trades, 2016-2025", fontweight="bold")
    save(fig, "06_monte_carlo.png")
