"""Experiment 5: do spillovers from linked companies, liquidity changes or stock splits add out-of-sample skill?

Part A, same method as 01_feature_ablation: LightGBM on the current features (base) vs base + one group, purged
walk-forward folds, Spearman rank correlation month by month, paired t-test of the monthly differences, Holm-adjusted.
  liquidity   turn_chg, amihud_chg, spread_chg, zero_ret_60, pre_abvol_5
  splits      split_250, rsplit_250, split_now
  contagion   peer_scandal_30, peer_scandal_car, cm_scandal_30
  comention   cm_n, cm_pre20, cm_car_01, cm_event, cm_conflict          (headlines cover 2010-2023 only)
Part B, customer-supplier links exist only from late 2020, so a model cannot learn them from earlier
folds. Instead we ask whether each new feature explains what the base model gets wrong: for events from 2021 on,
the Spearman correlation between the feature and the base model's out-of-sample error is computed month by month and
tested across months (Holm-adjusted). This is run for every new feature, so the two parts can be compared.
Run: python src/experiments/05_spillovers.py          (quick check: EXP_TEST=1)
Output: results/experiments/05_spillovers.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
os.environ["EXP_NO_LIQ"] = "1"          # base = the model before experiment 5
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score
from config import PROC, ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
sp = pd.read_parquet(PROC / "spillover_features.parquet").drop_duplicates(["permno", "day0"])
sp["permno"] = sp.permno.astype(ev.permno.dtype); sp["day0"] = pd.to_datetime(sp.day0)
new = ev[["permno", "day0"]].merge(sp, on=["permno", "day0"], how="left"); new.index = ev.index
GROUPS = {"liquidity": ["turn_chg", "amihud_chg", "spread_chg", "zero_ret_60", "pre_abvol_5"],
          "splits": ["split_250", "rsplit_250", "split_now"],
          "contagion": ["peer_scandal_30", "peer_scandal_car", "cm_scandal_30"],
          "comention": ["cm_n", "cm_pre20", "cm_car_01", "cm_event", "cm_conflict"]}
GROUPS = {g: [c for c in cols if c in new] for g, cols in GROUPS.items()}
CUSTOMER = [c for c in ["cust_n", "cust_pre20", "cust_car_01", "cust_event", "supp_n", "supp_pre20", "supp_car_01"] if c in new]
first = ev.day0[new[CUSTOMER[0]].notna()].min() if CUSTOMER else None
if first is not None and first.year <= 2008:               # historical file: enough years to train on, so test it like the rest
    GROUPS["customer"] = CUSTOMER; CUSTOMER = []
GROUPS["all_new"] = sum(GROUPS.values(), [])
ALLCOLS = list(GROUPS["all_new"])
ONLY = os.environ.get("X05_GROUPS")
if ONLY:
    GROUPS = {g: c for g, c in GROUPS.items() if g in ONLY.split(",")}
TARGETS = ["vol_ratio_2_20", "trough_2_20"] if TEST else ["vol_ratio_2_20", "abn_volume_2_20", "trough_2_20", "drift_1_5", "drift_1_20", "drift_1_60"]
P = dict(n_estimators=40 if TEST else 300, learning_rate=0.04)
Xall = pd.concat([ev[FEATS].astype(float), new[ALLCOLS].astype(float)], axis=1)
month = ev.day0.dt.to_period("M")
print("coverage:", ", ".join(f"{c} {new[c].notna().mean():.0%}" for c in ALLCOLS + CUSTOMER), flush=True)

def oos(t, cols):
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(gbm(**P), Xall.loc[tr, cols], y[tr], wts(tr, ev.day0)).predict(Xall.loc[te, cols])
    return from_model_pred(t, pred)
def monthly_rho(a, b, min_n=30):
    ok = a.notna() & b.notna()
    return pd.DataFrame({"a": a[ok], "b": b[ok], "m": month[ok]}).groupby("m").apply(
        lambda g: spearmanr(g.a, g.b).correlation if len(g) >= min_n and g.a.nunique() > 1 and g.b.nunique() > 1 else np.nan).dropna()
def holm(T, col="t_month"):
    T["p_raw"] = 2 * (1 - norm.cdf(T[col].abs())) if col == "t_resid" else 1 - norm.cdf(T[col])
    order = T.p_raw.sort_values().index; k = len(T)
    T.loc[order, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_raw[order].values * (k - np.arange(k))))
    return T

rows, resid_rows = [], []
for t in TARGETS:
    y = target_relative(t).astype(float)
    base = oos(t, FEATS); mb = monthly_rho(y, base); ok = y.notna() & base.notna()
    rb = spearmanr(y[ok], base[ok]).correlation
    print(f"{t}: base spearman {rb:.4f}", flush=True)
    # Part B: does a new feature line up with the base model's error? (2021 on, events with the feature present)
    err = (y - base).where(ev.day0.dt.year >= 2021)
    for c in ALLCOLS + CUSTOMER:
        m = monthly_rho(new[c], err, min_n=20)
        if len(m) >= 12:
            resid_rows.append({"target": t, "feature": c, "months": len(m), "mean_rho": m.mean(),
                               "t_resid": m.mean() / (m.std() / np.sqrt(len(m)))})
    # Part A: ablation
    for gname, cols in GROUPS.items():
        if not cols:
            continue
        p = oos(t, FEATS + cols); mp = monthly_rho(y, p); d = (mp - mb).dropna()
        late = d[d.index.year >= 2021]
        r = {"target": t, "group": gname, "base": rb, "with_group": spearmanr(y[ok], p[ok]).correlation,
             "monthly_gain": d.mean(), "t_month": d.mean() / (d.std() / np.sqrt(len(d))), "gain_2021_on": late.mean(),
             "share_months_better": (d > 0).mean()}
        if t == "trough_2_20":
            bad = y[ok] <= y[ok].quantile(0.2)
            r["auc_worst20_base"], r["auc_worst20_with"] = roc_auc_score(bad, -base[ok]), roc_auc_score(bad, -p[ok])
        rows.append(r); print(f"   + {gname:12s} {r['with_group']:.4f}  (t = {r['t_month']:.2f})", flush=True)

A = holm(pd.DataFrame(rows))
B = holm(pd.DataFrame(resid_rows), "t_resid").sort_values("p_holm")
txt = ("# Experiment 5: spillovers, liquidity and splits (" + ("live features" if LIVE else "all research features") + " as base)\n\n"
       "## Part A: ablation\n\nA group is kept only if p_holm < 0.05 and the gain also holds from 2021 on.\n\n```\n"
       + A.round(4).to_string(index=False) + "\n```\n\n"
       "## Part B: does the feature explain the base model's error (2021 on)?\n\n"
       "mean_rho is the average monthly rank correlation between the feature and (actual minus predicted). "
       "Two-sided, Holm-adjusted across all rows.\n\n```\n" + B.round(4).to_string(index=False) + "\n```\n")
(OUT / ("05_spillovers" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
