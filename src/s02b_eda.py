"""
Step 2b: exploratory data analysis of the event table produced by s02_events.py.

Prints labelled summary tables to the terminal and saves labelled charts to results/eda/.
Run after s02_events.py:  python src/s02b_eda.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROC, RESULTS

OUT = RESULTS / "eda"; OUT.mkdir(parents=True, exist_ok=True)
ev = pd.read_parquet(PROC / "events.parquet")
TARGETS = [c for c in ["car_0_1", "drift_1_5", "drift_1_20", "drift_1_60", "trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"] if c in ev]

def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig); print(f"  saved chart -> results/eda/{name}")

# ---------------------------------------------------------------- 1. shape
section("1. Dataset shape")
print(f"Events: {len(ev):,}   Firms (permno): {ev.permno.nunique():,}   Columns: {ev.shape[1]}")
print(f"Date range (day0 = event day): {ev.day0.min().date()} to {ev.day0.max().date()}")

# ---------------------------------------------------------------- 2. events per year and class
section("2. Event counts by year and by class")
by_year = ev.groupby([ev.day0.dt.year.rename("year"), "cls"]).size().unstack(fill_value=0)
print(by_year.to_string())
fig, ax = plt.subplots(figsize=(11, 5))
by_year.plot.bar(stacked=True, ax=ax, width=0.85)
ax.set_title("Number of events per year, stacked by event class")
ax.set_xlabel("Year of event (day 0)"); ax.set_ylabel("Number of events")
ax.legend(title="Event class", fontsize=8); save(fig, "01_events_per_year_by_class.png")

# ---------------------------------------------------------------- 3. missing values
section("3. Missing values per column (share of events with NaN)")
miss = ev.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
print((miss * 100).round(2).rename("% missing").to_string() if len(miss) else "No missing values.")
if len(miss):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.3 * len(miss))))
    (miss * 100).sort_values().plot.barh(ax=ax)
    ax.set_title("Share of events with a missing value, by column")
    ax.set_xlabel("% of events missing"); ax.set_ylabel("Column"); save(fig, "02_missing_values.png")

# ---------------------------------------------------------------- 4. target distributions
section("4. Target distributions (raw, before clipping)")
desc = ev[TARGETS].describe(percentiles=[0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995]).T
print(desc.round(4).to_string())
fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
for ax, t in zip(axes, TARGETS):
    x = ev[t].dropna()
    lo, hi = x.quantile([0.005, 0.995])
    ax.hist(x.clip(lo, hi), bins=80, color="steelblue")
    unit = "log ratio" if t.startswith(("vol_ratio", "abn_volume")) else "abnormal return (fraction, 0.01 = 1%)"
    ax.set_title(f"{t}\n(n = {len(x):,}, clipped to 0.5-99.5% for display)", fontsize=9)
    ax.set_xlabel(unit, fontsize=8); ax.set_ylabel("Number of events", fontsize=8)
for ax in axes[len(TARGETS):]: ax.axis("off")
fig.suptitle("Distribution of each prediction target"); save(fig, "03_target_distributions.png")

# ---------------------------------------------------------------- 5. duplicates and overlaps
section("5. Duplicate and overlapping events")
dup_exact = ev.duplicated(["permno", "day0", "cls"]).sum()
dup_day = ev.duplicated(["permno", "day0"]).sum()
print(f"Exact duplicates (same firm, same day, same class): {dup_exact:,}")
print(f"Same firm and same day but different class:          {dup_day - dup_exact:,}")
ev_s = ev.sort_values(["permno", "day0"])
gap = ev_s.groupby("permno").day0.diff().dt.days
for k in (5, 20, 60):
    print(f"Events with another event at the same firm within the previous {k:>2} calendar days: "
          f"{(gap <= k).sum():,} ({(gap <= k).mean():.1%})")
print("Interpretation: when another event falls inside a post-event window, that window's drift mixes both events.")

# ---------------------------------------------------------------- 6. mean abnormal return path by class
section("6. Mean abnormal return by horizon and class")
tab = ev.groupby("cls")[[t for t in TARGETS if t.startswith(("car", "drift"))]].mean()
print((tab * 100).round(3).to_string())
print("Units: percent. car_0_1 = reaction on days 0-1. drift_1_k = cumulative abnormal return over days 2..k.")
fig, ax = plt.subplots(figsize=(9, 5))
horizons = {"car_0_1": 1, "drift_1_5": 5, "drift_1_20": 20, "drift_1_60": 60}
for cls, row in tab.iterrows():
    xs = [0] + [horizons[c] for c in row.index]
    ys = [0] + [row["car_0_1"] * 100] + [(row["car_0_1"] + row[c]) * 100 for c in row.index if c != "car_0_1"]
    ax.plot(xs, ys, marker="o", label=cls)
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("Average cumulative abnormal return after the event, by event class")
ax.set_xlabel("Trading days after event (day 0)"); ax.set_ylabel("Cumulative abnormal return (%)")
ax.legend(title="Event class", fontsize=8); save(fig, "04_mean_car_path_by_class.png")

# ---------------------------------------------------------------- 7. feature correlation
section("7. Correlation between numeric features and the 20-day drift target")
num = ev.select_dtypes("number").drop(columns=[c for c in TARGETS if c in ev] + ["permno", "gvkey", "days_to_trough"], errors="ignore")
corr = num.corrwith(ev["drift_1_20"]).dropna().sort_values()
print(corr.round(4).rename("Pearson corr with drift_1_20").to_string())
fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(corr))))
corr.plot.barh(ax=ax, color=np.where(corr > 0, "seagreen", "indianred"))
ax.set_title("Pearson correlation of each feature with drift_1_20 (20-day post-event abnormal return)")
ax.set_xlabel("Correlation"); ax.set_ylabel("Feature"); save(fig, "05_feature_corr_with_drift_1_20.png")

print(f"\nDone. Charts in {OUT}")
