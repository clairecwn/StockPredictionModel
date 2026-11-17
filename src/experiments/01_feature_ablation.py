"""Experiment 1: do the calendar, firm-memory and analyst-revision features (06_calendar_features) add out-of-sample skill?

Same purged walk-forward folds, sample weights and targets as 17_train_risk_models.py. For each target, LightGBM is trained on the
current features (base) and on base + one feature group; a group counts only if it beats base on test years.
Test: Spearman rank correlation is computed month by month for both models; the paired t-statistic of the monthly
differences (months are the independent unit) and a Holm-adjusted p-value across all target x group tests are reported.
Lighter settings than the final models (300 trees) so the whole table takes about half an hour.
Run: python src/experiments/01_feature_ablation.py     (live features only: LIVE_FEATS=1; quick check: EXP_TEST=1)
Output: results/experiments/01_feature_ablation.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, warnings; warnings.filterwarnings("ignore")
os.environ["EXP_ABLATION"] = "1"          # modeling.py must not merge the 06_calendar_features features itself
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score
from config import PROC, ROOT, LIVE
from modeling import ev, FEATS, gbm, splits, wts, _fit, to_model_target, target_relative, from_model_pred

TEST = os.environ.get("EXP_TEST") == "1"
OUT = ROOT / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
cal = pd.read_parquet(PROC / "calendar_features.parquet")
cal["day0"] = pd.to_datetime(cal.day0); cal = cal.drop_duplicates(["permno", "day0"])
new = ev[["permno", "day0"]].merge(cal, on=["permno", "day0"], how="left"); new.index = ev.index
GROUPS = {"calendar": ["exp_earn_in_win", "days_to_exp_earn", "days_since_earn"],
          "earn_sensitivity": ["earn_react_hist"],
          "firm_memory": ["mem_logvr", "mem_logvol", "mem_trough_z", "mem_n"],
          "revisions": ["rev1_rel", "rev3_rel", "rev3_z", "dnum3", "ddisp3"]}
GROUPS["all_new"] = sum(GROUPS.values(), [])
TARGETS = ["vol_ratio_2_20", "trough_2_20"] if TEST else ["vol_ratio_2_20", "abn_volume_2_20", "trough_2_20", "drift_1_20", "drift_1_60"]
P = dict(n_estimators=40 if TEST else 300, learning_rate=0.04)
Xall = pd.concat([ev[FEATS].astype(float), new[GROUPS["all_new"]].astype(float)], axis=1)
month = ev.day0.dt.to_period("M")

def oos(t, cols):
    y = to_model_target(t).astype(float); pred = pd.Series(np.nan, index=y.index)
    for tr, te in splits(ev.day0, y):
        pred[te] = _fit(gbm(**P), Xall.loc[tr, cols], y[tr], wts(tr, ev.day0)).predict(Xall.loc[te, cols])
    return from_model_pred(t, pred)
def monthly_rho(y, p):
    ok = y.notna() & p.notna()
    return pd.DataFrame({"y": y[ok], "p": p[ok], "m": month[ok]}).groupby("m").apply(lambda g: spearmanr(g.y, g.p).correlation if len(g) >= 30 else np.nan).dropna()

rows = []
for t in TARGETS:
    y = target_relative(t).astype(float)
    base = oos(t, FEATS); mb = monthly_rho(y, base); ok = y.notna() & base.notna()
    rb = spearmanr(y[ok], base[ok]).correlation
    print(f"{t}: base spearman {rb:.4f}", flush=True)
    for gname, cols in GROUPS.items():
        p = oos(t, FEATS + cols); mp = monthly_rho(y, p); d = (mp - mb).dropna()
        late = d[d.index.year >= 2021]
        r = {"target": t, "group": gname, "base": rb, "with_group": spearmanr(y[ok], p[ok]).correlation,
             "monthly_gain": d.mean(), "t_month": d.mean() / (d.std() / np.sqrt(len(d))), "gain_2021_on": late.mean(),
             "share_months_better": (d > 0).mean()}
        if t == "trough_2_20":
            bad = y[ok] <= y[ok].quantile(0.2)
            r["auc_worst20_base"], r["auc_worst20_with"] = roc_auc_score(bad, -base[ok]), roc_auc_score(bad, -p[ok])
        rows.append(r); print(f"   + {gname:16s} {r['with_group']:.4f}  (t = {r['t_month']:.2f})", flush=True)
T = pd.DataFrame(rows)
T["p_one_sided"] = 1 - norm.cdf(T.t_month)
order = T.p_one_sided.sort_values().index; k = len(T)
T.loc[order, "p_holm"] = np.minimum(1, np.maximum.accumulate(T.p_one_sided[order].values * (k - np.arange(k))))
txt = ("# Experiment 1: feature ablation (" + ("live features" if LIVE else "all research features") + " as base)\n\n"
       "A group is kept only if p_holm < 0.05 and the gain also holds from 2021 on.\n\n```\n" + T.round(4).to_string(index=False) + "\n```\n")
(OUT / ("01_feature_ablation" + ("_live" if LIVE else "") + ("_TEST" if TEST else "") + ".md")).write_text(txt); print("\n" + txt)
