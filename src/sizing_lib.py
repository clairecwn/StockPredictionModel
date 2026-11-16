"""Shared sizing helpers (used by s06_sizing.py backtests and the live tracker)."""
import numpy as np

STRESS = 0.4   # share of the gap to perfect correlation added in the downside case

def shrunk_corr(R):
    """Ledoit-Wolf shrunk correlation matrix from a days x stocks return array (NaN treated as 0)."""
    from sklearn.covariance import LedoitWolf
    R = np.nan_to_num(np.asarray(R, dtype=float))
    sd = R.std(axis=0); sd[sd == 0] = 1.0
    C = LedoitWolf().fit((R - R.mean(axis=0)) / sd).covariance_
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return np.clip(C / np.outer(d, d), -1, 1)

def basket_scale(w, q, C, d, max_total=0.8):
    """Scale single-stock sizes w (fractions of the account) so the BASKET keeps the tier's promise.
    q = each stock's bad-case path loss (negative, e.g. -0.15). Each stock alone risks l_i = w_i*|q_i|.
    Treating bad cases as moving together with correlation C, the basket's bad case is sqrt(l' C l)
    (= sum of l_i if perfectly correlated, smaller if not). Scale down so it is <= d, and total invested <= max_total.
    Returns the scaled weights and the basket's estimated bad-case loss."""
    w = np.clip(np.asarray(w, float), 0, None); l = w * np.abs(np.asarray(q, float))
    # correlations rise in sell-offs (Longin & Solnik 2001): stress them toward 1 for the downside case
    C = np.asarray(C, float); C = C + STRESS * (1 - C); np.fill_diagonal(C, 1.0)
    joint = float(np.sqrt(max(l @ C @ l, 0.0)))
    s = 1.0
    if joint > d: s = d / joint
    if w.sum() * s > max_total: s = max_total / w.sum()
    return w * s, joint * s
