"""Step 3g: 1-D convolutional network on the pre-event price/volume path (the deep-learning member of the ensemble).

Idea from the sequence-model literature (CNN / LSTM / TCN on price windows): let a network read the SHAPE of the last
60 trading days plus the two reaction days, instead of only hand-made summaries (momentum, volatility).
Input per event: 61 days x 4 channels, all known at the close of day 1:
  1 market-adjusted daily return / stock's own pre-event sd      3 log volume minus its pre-event median
  2 market return / its sd over the window                         4 |channel 1|
Network: three dilated convolutions (receptive field ~29 days) + global average pooling + the last two days' activations,
three outputs (multi-task): trough_2_20, vol_ratio_2_20, abn_volume_2_20 in the same training units as s03.
Validation: the SAME purged walk-forward folds as s03; fixed epochs (no early stopping on test data).
Outputs: data/processed/cnn_oos.parquet (out-of-sample predictions, used as a stacking member in s03_train.py)
         models/cnn.pt (trained on all data, for the live tracker)
Run on the Mac (uses the GPU): python src/s03g_cnn.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from config import PROC, MODELS
from modeling import ev as EV_ALL, splits, to_model_target, target_relative
import os
TEST = os.environ.get("CNN_TEST") == "1"
ev = EV_ALL[EV_ALL.permno.isin(np.sort(EV_ALL.permno.unique())[:150])] if TEST else EV_ALL

T_LIST = ["trough_2_20", "vol_ratio_2_20", "abn_volume_2_20"]
WIN, EPOCHS, BS, LR = 61, 4, 2048, 2e-3
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
print(f"device: {dev}; events: {len(ev):,}")

# ---------------------------------------------------------------- windows days -59..+1
permnos = [int(p) for p in np.sort(ev.permno.unique())]
D = pq.read_table(PROC / "crsp_daily.parquet", columns=["permno", "date", "ret", "mkt", "vol"],
                  filters=[("permno", "in", permnos)]).to_pandas()
D["key"] = D.permno.astype(np.int64) * 100000 + (D.date.values.astype("datetime64[D]").astype(np.int64) - 7300)
D = D.sort_values("key").reset_index(drop=True)
k0 = ev.permno.astype(np.int64).values * 100000 + (ev.day0.values.astype("datetime64[D]").astype(np.int64) - 7300)
g0 = np.searchsorted(D.key.values, k0)
idx = (g0[:, None] + np.arange(-59, 2)[None, :]).clip(0, len(D) - 1)
same = D.permno.values[idx] == ev.permno.values[:, None]
def W(col):
    a = D[col].values[idx].astype(np.float32); a[~same] = np.nan; return a
ret, mkt, vol = W("ret"), W("mkt"), W("vol")
del D
ar = ret - mkt
sd = np.nanstd(ar[:, :59], axis=1, keepdims=True).clip(1e-3)
c1 = ar / sd
c2 = mkt / np.nanstd(mkt, axis=1, keepdims=True).clip(1e-4)
lv = np.log1p(np.nan_to_num(vol, nan=0.0).clip(0))
c3 = lv - np.nanmedian(np.where(same[:, :49], lv[:, :49], np.nan), axis=1, keepdims=True)
X = np.stack([c1, c2, c3, np.abs(c1)], axis=1)                    # N x 4 x 61
X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)
del ret, mkt, vol, ar, c1, c2, c3, lv
Y = np.stack([to_model_target(t).astype(float).loc[ev.index].values for t in T_LIST], axis=1).astype(np.float32)
print(f"windows built: {X.shape}, share of days present: {same.mean():.1%}")

class Net(nn.Module):
    def __init__(self, c=4, h=32, out=3):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(c, h, 5, padding=2), nn.GELU(),
                               nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.GELU(),
                               nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.GELU())
        self.head = nn.Sequential(nn.Linear(h * 3, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, out))
    def forward(self, x):
        z = self.f(x)
        return self.head(torch.cat([z.mean(dim=2), z[:, :, -2], z[:, :, -1]], dim=1))

def train(ix):
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    loss_f = nn.HuberLoss(reduction="none")
    for ep in range(EPOCHS):
        perm = np.random.permutation(ix); tot = 0.0
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            xb = torch.from_numpy(X[b]).to(dev); yb = torch.from_numpy(Y[b]).to(dev)
            m = torch.isfinite(yb); yb = torch.nan_to_num(yb)
            l = (loss_f(net(xb), yb) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l.detach()) * len(b)
        print(f"    epoch {ep + 1}: train loss {tot / len(perm):.4f}", flush=True)
    return net.eval()

def predict(net, ix):
    out = []
    with torch.no_grad():
        for s in range(0, len(ix), 8192):
            out.append(net(torch.from_numpy(X[ix[s:s + 8192]]).to(dev)).cpu().numpy())
    return np.vstack(out)

P = np.full(Y.shape, np.nan, dtype=np.float32)
ref = pd.Series(Y[:, 0], index=ev.index)
for fi, (tr, te) in enumerate(splits(ev.day0, ref)):
    tri, tei = np.where(tr.values)[0], np.where(te.values)[0]
    print(f"  fold {fi + 1}: train {len(tri):,}, test {len(tei):,}", flush=True)
    P[tei] = predict(train(tri), tei)

print("\n=== CNN out-of-sample skill (spearman, same folds as s03) ===")
for j, t in enumerate(T_LIST):
    yr = target_relative(t).astype(float).loc[ev.index].values
    m = np.isfinite(P[:, j]) & np.isfinite(yr)
    print(f"  {t}: spearman {spearmanr(P[m, j], yr[m])[0]:.4f}  n={m.sum():,}")
out = pd.DataFrame({"permno": ev.permno.values, "day0": ev.day0.values, **{f"cnn_{t}": P[:, j] for j, t in enumerate(T_LIST)}})
if TEST:
    raise SystemExit("test mode: nothing saved")
out.to_parquet(PROC / "cnn_oos.parquet", index=False)
allm = np.where(np.isfinite(Y).any(axis=1))[0]
torch.save(train(allm).state_dict(), MODELS / "cnn.pt")
print("saved data/processed/cnn_oos.parquet and models/cnn.pt")
