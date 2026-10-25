"""Sanity checks for the KS and EM symbol-domain classifiers on symbol-level synthetic frames
with known answers. Exits non-zero if a check fails.

    python sanity_symbol.py      -> results/sanity_symbol.json
"""

from __future__ import annotations

import sys
import time

import common
import numpy as np
from scipy.stats import ks_2samp

import symbol_classifiers as sc

rng = np.random.default_rng(7)
REPORT, FAILS = {}, []


def check(name, ok, detail):
    REPORT[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def ks_checks():
    # 1. statistic identical to scipy (continuous data and data with ties)
    err = 0.0
    for _ in range(200):
        a = rng.standard_normal(rng.integers(5, 300))
        b = rng.standard_normal(rng.integers(5, 3000)) + rng.uniform(-0.5, 0.5)
        if rng.random() < 0.5:
            a, b = np.round(a, 1), np.round(b, 1)
        err = max(err, abs(sc.ks2(np.sort(a), np.sort(b)) - ks_2samp(a, b).statistic))
    check("ks2_equals_scipy", err < 1e-12, {"max_abs_diff": err})

    # 2. null calibration: frame and reference from the SAME distribution, iid samples ->
    #    sqrt(n m / (n + m)) D follows the Kolmogorov distribution (mean sqrt(pi/2) ln 2 = 0.8687,
    #    5 % point 1.358). Raw (un-normalised) |z| of 16QAM at 10 dB.
    def raw(nfr, n):
        s = rng.choice(sc.CONS["16QAM"], (nfr, n))
        return np.abs(s + np.sqrt(0.1 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)))
    ref = np.sort(raw(1, 25600)[0])
    m = len(ref)
    t = np.array([sc.ks2(np.sort(f), ref) for f in raw(2000, 120)]) * np.sqrt(120 * m / (120 + m))
    mean, rej = float(t.mean()), float(np.mean(t > 1.358))
    check("ks_null_calibration_iid", 0.82 < mean < 0.92 and 0.03 < rej < 0.075,
          {"mean_scaled_D": mean, "theory_mean": 0.8687, "reject_rate_at_5pct": rej})
    # Same with the per-frame unit-power normalisation used by the classifier (and its reference):
    # parameters estimated from the frame itself make D smaller (Lilliefors effect). Informational.
    ref = sc.reference("16QAM", 10)[2]
    m = len(ref)
    z, _ = sc.synth_symbols("16QAM", 2000, 120, 10, rng)
    t = np.array([sc.ks2(np.sort(np.abs(f)), ref) for f in z]) * np.sqrt(120 * m / (120 + m))
    mean, rej = float(t.mean()), float(np.mean(t > 1.358))
    REPORT["ks_null_with_per_frame_normalisation (info)"] = {"mean_scaled_D": mean, "reject_rate_at_5pct": rej}
    print("   normalised-frame null:", REPORT["ks_null_with_per_frame_normalisation (info)"])


def em_checks():
    # 3. parameter recovery + monotone log-likelihood (EM property)
    res, ok = {}, True
    for c in sc.CLASSES:
        errs_a, errs_p, errs_s, mono = [], [], [], True
        for _ in range(20):
            snr = 15
            z, ph = sc.synth_symbols(c, 1, 1000, snr, rng)
            z, ph = z[0], ph[0]
            S = 1 / (1 + 10 ** (-snr / 10))          # after unit-power normalisation (approx.)
            ll, g, s2, tr = sc.em_fit(z, c, return_trace=True)
            mono &= bool(np.all(np.diff(tr) > -1e-6 * abs(tr[-1])))
            sym = 2 * np.pi / sc.POW[c]
            dp = (np.angle(g) - ph + sym / 2) % sym - sym / 2   # phase error modulo symmetry
            errs_a.append(abs(abs(g) - np.sqrt(S)) / np.sqrt(S))
            errs_p.append(abs(dp))
            errs_s.append(abs(s2 - (1 - S)) / (1 - S))
        res[c] = {"amp_rel_err": float(np.median(errs_a)), "phase_err_rad": float(np.median(errs_p)),
                  "noise_var_rel_err": float(np.median(errs_s)), "loglik_monotone": mono}
        ok &= np.median(errs_a) < 0.02 and np.median(errs_p) < 0.03 and np.median(errs_s) < 0.1 and mono
    check("em_parameter_recovery_15dB_1000sym", ok, res)


def accuracy_checks():
    # 4. accuracy on synthetic frames, compared with the coherent oracle ML (true gain, phase,
    #    noise variance). A blind method must not beat the oracle beyond sampling noise.
    out, ok = {}, True
    for nsym, nfr in ((120, 60), (1000, 12)):
        for snr in (0, 5, 10, 15, 20, 25):
            hits = {"oracle": 0, "em": 0, **{f"ks_{v}": 0 for v in sc.KS_VARIANTS}}
            tot = 0
            for k, c in enumerate(sc.CLASSES):
                z, ph = sc.synth_symbols(c, nfr, nsym, snr, rng)
                S = 1 / (1 + 10 ** (-snr / 10))
                for f, p in zip(z, ph):
                    tot += 1
                    hits["oracle"] += sc.oracle_ml(f, np.sqrt(S) * np.exp(1j * p), 1 - S) == k
                    hits["em"] += sc.em_classify(f)[0] == k
                    D = sc.ks_distances(f)
                    for v in sc.KS_VARIANTS:
                        hits[f"ks_{v}"] += sc.ks_decide(D, v) == k
            acc = {m: h / tot for m, h in hits.items()}
            out[f"nsym{nsym}_snr{snr}"] = acc
            print(f"   nsym {nsym:4d} SNR {snr:2d}: " + "  ".join(f"{m} {a:.2f}" for m, a in acc.items()), flush=True)
            se = np.sqrt(0.25 / tot)
            ok &= acc["em"] <= acc["oracle"] + 3 * se and max(acc[f"ks_{v}"] for v in sc.KS_VARIANTS) <= acc["oracle"] + 3 * se
    hi = out["nsym1000_snr25"]
    ok &= hi["em"] >= 0.97 and hi["ks_iqap"] >= 0.9 and out["nsym120_snr25"]["em"] >= 0.9
    check("synthetic_accuracy_vs_oracle", ok, {"accuracy": out})


if __name__ == "__main__":
    t0 = time.time()
    ks_checks()
    em_checks()
    accuracy_checks()
    path = common.save_json({"failed": FAILS, "checks": REPORT, "seconds": time.time() - t0}, "sanity_symbol.json")
    print(f"\n{len(REPORT) - len(FAILS)}/{len(REPORT)} checks passed -> {path}")
    sys.exit(1 if FAILS else 0)
