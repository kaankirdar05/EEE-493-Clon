"""Sanity checks for the Cramer-von Mises and Anderson-Darling classifiers. Exits non-zero on failure.

    python sanity_symbol.py      -> results/sanity_symbol.json
"""

from __future__ import annotations

import math
import sys
import warnings

import dcommon as dc
import numpy as np
from scipy.stats import anderson_ksamp, cramervonmises_2samp

import dist_classifiers as dcl
import symbol_classifiers as sc

rng = np.random.default_rng(23)
REPORT, FAILS = {}, []


def check(name, ok, detail):
    REPORT[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def ad_standardise(A2, n, m):
    """Scholz & Stephens (1987) standardisation (k = 2), as used by scipy.stats.anderson_ksamp."""
    k, N = 2, n + m
    H = 1 / n + 1 / m
    hs_cs = (1.0 / np.arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / np.arange(2, N)).sum()
    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k ** 2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k ** 2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k ** 2 - 4 * h * k
    s2 = (a * N ** 3 + b * N ** 2 + c * N + d) / ((N - 1.0) * (N - 2.0) * (N - 3.0))
    return (A2 - (k - 1)) / math.sqrt(s2)


def statistic_checks():
    e_cvm, e_ad, e_ad_right = 0.0, 0.0, 0.0
    for _ in range(100):
        a = rng.standard_normal(rng.integers(10, 300))
        b = rng.standard_normal(rng.integers(10, 3000)) * rng.uniform(0.7, 1.4) + rng.uniform(-0.5, 0.5)
        A, B = np.sort(a), np.sort(b)
        e_cvm = max(e_cvm, abs(dcl.cvm2(A, B) - cramervonmises_2samp(a, b).statistic))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_c = anderson_ksamp([a, b], variant="continuous").statistic
            ref_r = anderson_ksamp([a, b], variant="right").statistic
        mine = ad_standardise(dcl.ad2(A, B), len(a), len(b))
        e_ad = max(e_ad, abs(mine - ref_c))
        e_ad_right = max(e_ad_right, abs(mine - ref_r))
    check("cvm_equals_scipy", e_cvm < 1e-10, {"max_abs_diff_vs_cramervonmises_2samp": e_cvm})
    check("ad_equals_scipy", e_ad < 1e-8 and e_ad_right < 1e-8,
          {"max_abs_diff_vs_anderson_ksamp_continuous (standardised)": e_ad,
           "max_abs_diff_vs_anderson_ksamp_right (standardised)": e_ad_right})


def null_checks():
    # frame (n = 120) and reference (m = 25600) iid from the same distribution: |z| of 16QAM at 10 dB
    def raw(nfr, n):
        s = rng.choice(sc.CONS["16QAM"], (nfr, n))
        return np.abs(s + np.sqrt(0.1 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)))
    ref = np.sort(raw(1, 25600)[0])
    fr = [np.sort(f) for f in raw(3000, 120)]
    T = np.array([dcl.cvm2(f, ref) for f in fr])
    A = np.array([dcl.ad2(f, ref) for f in fr])
    res = {"cvm_mean": float(T.mean()), "cvm_theory_mean": (1 + 1 / (120 + 25600)) / 6,
           "cvm_reject_at_0.461": float(np.mean(T > 0.461)),
           "ad_mean": float(A.mean()), "ad_theory_mean": 1.0, "ad_reject_at_2.492": float(np.mean(A > 2.492))}
    check("null_calibration_iid",
          abs(res["cvm_mean"] - 1 / 6) < 0.012 and abs(res["ad_mean"] - 1) < 0.06
          and 0.035 < res["cvm_reject_at_0.461"] < 0.065 and 0.035 < res["ad_reject_at_2.492"] < 0.065, res)
    # with the classifier's per-frame normalisation and phase estimation: correct class vs reference
    ref = sc.reference("16QAM", 10)
    z, _ = sc.synth_symbols("16QAM", 2000, 120, 10, rng)
    T, A = [], []
    for f in z:
        zr = sc.derotate(f, "16QAM")
        T.append(dcl.cvm2(np.sort(np.abs(zr)), ref[2]))
        A.append(dcl.ad2(np.sort(np.abs(zr)), ref[2]))
    T, A = np.array(T), np.array(A)
    REPORT["null_with_per_frame_normalisation (info)"] = {
        "cvm_mean": float(T.mean()), "cvm_reject": float(np.mean(T > 0.461)),
        "ad_mean": float(A.mean()), "ad_reject": float(np.mean(A > 2.492)),
        "note": "parameters estimated from the frame itself -> conservative test (Lilliefors effect), as for KS"}
    print("   ", REPORT["null_with_per_frame_normalisation (info)"])


def synthetic_accuracy():
    """Accuracy on ideal symbol frames (random phase, AWGN) vs the existing KS and the coherent oracle."""
    table = {}
    for nsym, snr, nfr in ((120, 10, 100), (120, 25, 100), (1000, 10, 40), (1000, 25, 40)):
        hits = {f"{t}_{v}": 0 for t in dcl.TESTS for v in ("ap", "iqap")}
        hits["oracle"] = 0
        tot = 0
        for k, c in enumerate(sc.CLASSES):
            z, ph = sc.synth_symbols(c, nfr, nsym, snr, rng)
            S = 1 / (1 + 10 ** (-snr / 10))
            for f, p in zip(z, ph):
                D = dcl.gof_distances(f)
                for t in dcl.TESTS:
                    for v in ("ap", "iqap"):
                        hits[f"{t}_{v}"] += dcl.decide(D, t, v) == k
                hits["oracle"] += sc.oracle_ml(f, np.sqrt(S) * np.exp(1j * p), 1 - S) == k
                tot += 1
        table[f"{nsym} sym, {snr} dB"] = {k: v / tot for k, v in hits.items()}
        print("   ", f"{nsym} sym, {snr} dB", {k: round(v / tot, 3) for k, v in hits.items()}, flush=True)
    ok = all(r["oracle"] >= max(r[f"{t}_ap"] for t in dcl.TESTS) - 0.02 for r in table.values()) \
        and table["1000 sym, 25 dB"]["cvm_ap"] > 0.97 and table["1000 sym, 25 dB"]["ad_ap"] > 0.97
    check("synthetic_accuracy", ok, {"table": table,
                                     "criterion": "oracle not beaten; CvM and AD (ap) > 97 % at 1000 symbols, 25 dB"})


if __name__ == "__main__":
    statistic_checks()
    null_checks()
    synthetic_accuracy()
    REPORT["all_pass"] = not FAILS
    print("saved", dc.save_json(REPORT, "sanity_symbol.json"))
    print("FAILED:" if FAILS else "all checks passed", FAILS)
    sys.exit(1 if FAILS else 0)
