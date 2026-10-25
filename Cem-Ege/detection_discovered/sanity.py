"""Theory / known-answer checks, run BEFORE any benchmark.

    python sanity.py tw        Tracy-Widom table + analytic thresholds on iid complex Wishart
                               (known answer) and on the smoothed (Hankel) sub-band covariance
    python sanity.py rise      synthetic correlated signal in noise: every statistic rises
    python sanity.py tkeo      TKEO of a complex tone and of white noise vs closed form
Outputs: results/sanity/*.json (+ png).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disc.eigen import (cond_threshold, eigs, john_asym_threshold, med_tw_threshold, mp_edges,  # noqa: E402
                        stat_from_eigs, tw2_ppf, tw2_table)
from disc.signals import cnoise, qpsk  # noqa: E402
from disc.tkeo import tkeo  # noqa: E402

OUT = os.path.join(HERE, "results", "sanity")
PFAS = [1e-1, 1e-2, 1e-3, 1e-4]
LS = [4, 8, 16, 32]


def _dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def wishart_eigs(rng, L, n, trials, chunk=2000):
    out = []
    for s in range(0, trials, chunk):
        X = cnoise(rng, (min(chunk, trials - s), L, n))
        out.append(np.linalg.eigvalsh(X @ np.conj(np.swapaxes(X, -1, -2)) / n))
    return np.concatenate(out)


def analytic_pfas(ev, L, n):
    """Measured Pfa of each analytic threshold; ev ascending eigenvalues of C (unit noise)."""
    cond = ev[:, -1] / ev[:, 0]
    john = stat_from_eigs(ev, "john")
    r = {}
    for p in PFAS:
        r[str(p)] = {
            "lmax_TW (MED, known s2)": float(np.mean(ev[:, -1] > med_tw_threshold(L, n, p))),
            "lmin_TW (lower edge)": float(np.mean(n * ev[:, 0] < mp_edges(L, n)[2] - mp_edges(L, n)[3] * tw2_ppf(1 - p))),
            "cond_ZL": float(np.mean(cond > cond_threshold(L, n, p, "zl"))),
            "cond_TWC": float(np.mean(cond > cond_threshold(L, n, p, "twc"))),
            "john_asymN": float(np.mean(john > john_asym_threshold(L, n, p))),
        }
    return r


def cmd_tw():
    s, F, f = tw2_table()
    m = float(np.trapezoid(s * f, s))
    v = float(np.trapezoid((s - m) ** 2 * f, s))
    res = {"tw2_table": {"mean": m, "mean_ref": -1.7710868074, "var": v, "var_ref": 0.8131947928,
                         "q0.99": tw2_ppf(0.99)}}
    rng = np.random.default_rng(777)
    res["iid_wishart"], res["hankel_subband"], res["thresholds"] = {}, {}, {}
    for L in LS:
        n = 512 - L + 1
        res["thresholds"][L] = {str(p): {"cond_ZL": cond_threshold(L, n, p, "zl"), "cond_TWC": cond_threshold(L, n, p, "twc"),
                                         "med_TW": med_tw_threshold(L, n, p), "john_asym": john_asym_threshold(L, n, p)}
                                for p in PFAS}
        ev = wishart_eigs(rng, L, n, 100_000 if L <= 16 else 50_000)
        res["iid_wishart"][L] = {"trials": len(ev), "pfa": analytic_pfas(ev, L, n)}
        evh = eigs(cnoise(rng, (200_000, 512)), L)
        # empirical quantiles of the smoothed-covariance condition number vs analytic thresholds
        cond = evh[:, -1] / evh[:, 0]
        res["hankel_subband"][L] = {"trials": len(evh), "pfa": analytic_pfas(evh, L, n),
                                    "empirical_cond_thr": {str(p): float(np.quantile(cond, 1 - p)) for p in PFAS}}
        print(L, "wishart", res["iid_wishart"][L]["pfa"]["0.001"], "\n   hankel", res["hankel_subband"][L]["pfa"]["0.001"], flush=True)
    _dump("tracy_widom.json", res)


def cmd_rise():
    """Blocks of 512 samples (one sub-band block). H1: RRC-QPSK, 8 samples/symbol."""
    rng = np.random.default_rng(31)
    nb = 100_000
    L = 8
    h0 = cnoise(rng, (nb, 512))
    out = {}

    def stats(x, L):
        ev = eigs(x, L)
        return {"agm": stat_from_eigs(ev, "agm"), "john(=EMR-1)": stat_from_eigs(ev, "john"),
                "cond": stat_from_eigs(ev, "cond"), "med(known s2=1)": ev[:, -1],
                "tkeo(mean Psi / s2)": tkeo(x, edges=False).mean(-1), "energy": np.mean(np.abs(x) ** 2, -1)}

    for L in (4, 8, 16, 32):
        s0 = stats(h0, L)
        thr = {k: float(np.quantile(v, 1 - 1e-3)) for k, v in s0.items()}
        out[L] = {}
        for snr in (-10, -5, 0):
            for f0, tag in ((0.1, "offset 0.1 c/s"), (0.0, "centred (0 c/s)")):
                s = qpsk(rng, 4000, 512, f0=f0)
                x = cnoise(rng, s.shape) + s * 10 ** (snr / 20)
                s1 = stats(x, L)
                out[L][f"{snr} dB, {tag}"] = {k: {"mean_h0": float(s0[k].mean()), "mean_h1": float(s1[k].mean()),
                                                  "rises": bool(s1[k].mean() > s0[k].mean()),
                                                  "pd_at_pfa1e-3": float(np.mean(s1[k] > thr[k]))} for k in s0}
    _dump("rise_correlated_signal.json", out)
    for L in out:
        for c, d in out[L].items():
            print(L, c, {k: f"{v['mean_h0']:.3f}->{v['mean_h1']:.3f} Pd {v['pd_at_pfa1e-3']:.2f}" for k, v in d.items()})


def cmd_tkeo():
    n = np.arange(4096)
    rows = []
    for A in (0.5, 1.0, 3.0):
        for w in (0.0, 0.1, np.pi / 4, np.pi / 2, 2.0, 3.0, np.pi):
            x = A * np.exp(1j * (w * n + 0.7))
            psi = tkeo(x, edges=False)
            rows.append({"A": A, "w": float(w), "psi_mean": float(psi.mean()), "psi_spread": float(psi.max() - psi.min()),
                         "theory_A2(1-cos2w)": float(A ** 2 * (1 - np.cos(2 * w))),
                         "brief_formula_A2(1-cos w)": float(A ** 2 * (1 - np.cos(w)))})
    err = max(abs(r["psi_mean"] - r["theory_A2(1-cos2w)"]) + r["psi_spread"] for r in rows)
    err_brief = max(abs(r["psi_mean"] - r["brief_formula_A2(1-cos w)"]) for r in rows)
    rng = np.random.default_rng(5)
    z = cnoise(rng, (2000, 4096))
    p = tkeo(z, edges=False)
    c = [float(np.mean((p[:, :-d] - 1) * (p[:, d:] - 1))) for d in (1, 2, 3)]
    res = {"tone": rows, "max_abs_err_vs_A2(1-cos2w)": err, "max_abs_err_vs_A2(1-cos w)": err_brief,
           "noise": {"mean": float(p.mean()), "mean_theory": 1.0, "var": float(p.var()), "var_theory": 1.5,
                     "autocov_lag1_2_3": c, "var_of_4094_mean": float(p.mean(1).var()),
                     "var_of_mean_theory": 1.5 / p.shape[1]}}
    _dump("tkeo_closed_form.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "tone"}, indent=1))


if __name__ == "__main__":
    {"tw": cmd_tw, "rise": cmd_rise, "tkeo": cmd_tkeo}[sys.argv[1]]()
