"""Multi-estimator consensus for the symbol rate, by normalised line prominence.

Each bank member k gives (rate_k, prom_k). Its score is
    z_k = log(prom_k) - mu_k,
mu_k = 95th percentile of log(prom_k) over DEV frames on which member k is wrong
(i.e. the typical height of a false peak for that member). Rules:
    max   take the member with the largest z_k
    vote  cluster the rates (relative tolerance TOL), cluster score = sum of max(z_k, 0);
          take the best cluster, rate = rate of its highest-z member
The member subset is chosen by greedy forward selection on DEV accuracy (within 2 %).
Nothing here sees held-out data: fit() takes DEV arrays only.
"""

from __future__ import annotations

import numpy as np

TOL = 0.01


def correct(rate, rs, tol=0.02):
    return np.abs(rate - rs) / rs < tol


def fit_mu(R, P, rs, q=95):
    logP = np.log(P + 1e-30)
    mu = np.zeros(R.shape[1])
    for k in range(R.shape[1]):
        wrong = ~correct(R[:, k], rs)
        mu[k] = np.percentile(logP[wrong, k], q) if wrong.sum() > 10 else np.median(logP[:, k])
    return mu


def predict(R, P, mu, cols, rule="max"):
    Z = np.log(P[:, cols] + 1e-30) - mu[cols]
    Rc = R[:, cols]
    if rule == "max":
        return Rc[np.arange(len(R)), np.argmax(Z, 1)]
    out = np.empty(len(R))
    W = np.clip(Z, 0, None)
    for i in range(len(R)):
        r, w, z = Rc[i], W[i], Z[i]
        if w.max() <= 0:
            out[i] = r[np.argmax(z)]
            continue
        same = np.abs(r[:, None] / r[None, :] - 1) < TOL          # same[j, k]: j agrees with k
        score = (same * w[:, None]).sum(0)
        best = np.flatnonzero(score == score.max())
        k = best[np.argmax(z[best])]
        out[i] = r[k]
    return out


def greedy(R, P, rs, mu, candidates, rule="max", max_members=12, verbose=False, names=None):
    chosen, best_acc = [], 0.0
    while len(chosen) < max_members:
        best = None
        for c in candidates:
            if c in chosen:
                continue
            acc = correct(predict(R, P, mu, chosen + [c], rule), rs).mean()
            if best is None or acc > best[0]:
                best = (acc, c)
        if best is None or best[0] <= best_acc + 1e-4:
            break
        best_acc = best[0]
        chosen.append(best[1])
        if verbose:
            print(f"   + {names[best[1]] if names else best[1]:>10s} -> {best_acc * 100:.1f} %", flush=True)
    return chosen, best_acc
