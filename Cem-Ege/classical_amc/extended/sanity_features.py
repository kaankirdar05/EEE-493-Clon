"""Sanity checks for the extended cumulant, cyclic and Haar-wavelet features.
Every check has a known answer; the script exits non-zero if any check fails.

    python sanity_features.py     -> results/sanity_features.json
"""

from __future__ import annotations

import sys

import common  # noqa: F401  (sys.path)
import numpy as np

from amc.features import _cumulants as base_cumulants
from amc.synthetic import _constellation, _rrc
from cumulants_ext import (_normalize, cumulant_terms, cyclic_features, estimate_cumulants,
                           theoretical_cumulants)
from wavelet import haar, tone_theory, wavelet_features, WAV_NAMES

rng = np.random.default_rng(1)
REPORT, FAILS = {}, []


def check(name, ok, detail):
    REPORT[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def unit(points):
    p = np.asarray(points, complex)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


# ------------------------------------------------------------------ 1. cumulant formula
def cumulant_checks():
    # (a) number of set partitions of an 8-set = Bell(8) = 4140 (formula enumerates all)
    n_part = sum(1 for _ in __import__("cumulants_ext")._partitions(list(range(8))))
    check("bell_number_8", n_part == 4140, {"partitions": n_part})

    # (b) general formula vs the hard-coded C40, C42, C60, C80 of amc/features.py, on frames with
    #     EXACTLY zero odd moments (x concatenated with -x), where those hand formulas are exact.
    z = rng.standard_normal((50, 512)) + 1j * rng.standard_normal((50, 512))
    z = z ** 3                                        # non-Gaussian, non-trivial cumulants
    z = np.concatenate([z, -z], axis=1)
    zn = _normalize(z)
    base = base_cumulants(zn)                         # |C20| |C40| |C41| C42 |C60| |C80|
    est = estimate_cumulants(z, [(2, 0), (4, 0), (4, 1), (4, 2), (6, 0), (8, 0)])
    mine = np.stack([np.abs(est[(2, 0)]), np.abs(est[(4, 0)]), np.abs(est[(4, 1)]), est[(4, 2)].real,
                     np.abs(est[(6, 0)]), np.abs(est[(8, 0)])], 1)
    err = float(np.max(np.abs(mine - base) / np.maximum(np.abs(base), 1)))
    check("general_formula_vs_features.py", err < 1e-9, {"max_rel_err": err})

    # (c) exact theory vs literature (Swami & Sadler 2000, unit-power constellations)
    lit = {"BPSK": {(4, 0): -2, (4, 2): -2, (6, 3): 16, (8, 0): -272},
           "QPSK": {(4, 0): -1, (4, 2): -1, (6, 3): 4, (8, 0): -34},     # sign of C40/C80 depends on rotation
           "8PSK": {(4, 0): 0, (4, 2): -1, (6, 3): 4, (8, 0): 1},
           "16QAM": {(4, 0): -0.68, (4, 2): -0.68, (6, 3): 2.08, (8, 0): -13.98},
           "64QAM": {(4, 0): -0.619, (4, 2): -0.619, (6, 3): 1.797, (8, 0): -11.50}}
    worst = 0.0
    for c, vals in lit.items():
        th = theoretical_cumulants(_constellation(c), list(vals))
        for pq, v in vals.items():
            worst = max(worst, abs(abs(th[pq]) - abs(v)) / max(abs(v), 1))
    check("theory_vs_literature", worst < 2e-3, {"max_rel_err": worst})

    # (d) estimates on 200k noiseless symbols vs theory for the NEW cumulants.
    #     R = 8 independent 200k-symbol realisations per class (no phase rotation, so complex
    #     values are comparable). Pass if every |mean - theory| is within 5 standard errors
    #     (the 8th-order estimators have large variance; a formula bug gives O(1)-O(100) errors)
    #     and, for non-zero theory, the single-run relative error is < 3 %.
    pq_new = [(6, 1), (6, 2), (6, 3), (8, 1), (8, 2), (8, 3), (8, 4)]
    table, worst_z, worst_rel = {}, 0.0, 0.0
    for c in ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4ASK", "16PSK"]:
        pts = _constellation(c)
        th = theoretical_cumulants(pts, pq_new)
        s = rng.choice(unit(pts), (8, 200_000))
        est = estimate_cumulants(s, pq_new)
        row = {}
        for pq in pq_new:
            t, e = th[pq], est[pq]
            se = max(np.sqrt((np.var(e.real) + np.var(e.imag)) / len(e)), 1e-12)
            z = abs(e.mean() - t) / se
            rel = float(np.max(np.abs(e - t)) / abs(t)) if abs(t) > 1e-9 else 0.0
            worst_z, worst_rel = max(worst_z, z), max(worst_rel, rel)
            row[f"C{pq[0]}{pq[1]}"] = {"theory": [round(t.real, 4), round(t.imag, 4)],
                                        "estimate_mean": [round(e.mean().real, 4), round(e.mean().imag, 4)],
                                        "std_single_run": round(float(np.sqrt(np.var(e.real) + np.var(e.imag))), 4)}
        table[c] = row
    check("new_cumulants_200k_noiseless", worst_z < 5 and worst_rel < 0.03,
          {"max_z": worst_z, "max_rel_err_nonzero_theory": worst_rel, "table": table})

    # (e) Gaussian noise: all cumulants of order > 2 are zero
    g = (rng.standard_normal((20, 50_000)) + 1j * rng.standard_normal((20, 50_000))) / np.sqrt(2)
    est = estimate_cumulants(g, pq_new + [(4, 0), (4, 2)])
    mx = {f"C{p}{q}": float(np.abs(np.mean(v))) for (p, q), v in est.items()}
    check("gaussian_cumulants_zero", max(mx.values()) < 0.5, {"mean_abs": mx,
          "note": "8th-order estimator std on 50k samples is O(0.5); means over 20 frames"})


# ------------------------------------------------------------------ 2. cyclic features
def cyclic_checks():
    f0, L, nfr = 0.0123, 1024, 100
    n = np.arange(L)
    res, ok = {}, True
    theory = {"BPSK": 2.0, "QPSK": 1.0, "8PSK": 0.0, "16QAM": 0.68}
    for c, t in theory.items():
        s = rng.choice(unit(_constellation(c)), (nfr, L))          # 1 sample / symbol, noiseless
        s = s * np.exp(1j * (rng.uniform(0, 2 * np.pi, (nfr, 1)) + 2 * np.pi * f0 * n))
        F = cyclic_features(_normalize(s))
        plain = base_cumulants(_normalize(s))[:, 1]              # |C40| without CFO handling
        cc40 = float(F[:, 2].mean())
        res[c] = {"cyc_C40": round(cc40, 3), "theory_C40": t, "plain_abs_C40_under_CFO": round(float(plain.mean()), 3)}
        ok &= abs(cc40 - t) < 0.12
    # x^2 line of BPSK must sit at 2 f0 with |M20| = 1
    s = rng.choice([-1.0, 1.0], (nfr, L)) * np.exp(1j * 2 * np.pi * f0 * n)
    v = _normalize(s) ** 2
    from cumulants_ext import _peak_dtft
    a2, m2 = _peak_dtft(v, 4096)
    res["BPSK_x2_line"] = {"alpha_mean": float(a2.mean()), "expected": 2 * f0, "M20_peak": float(m2.mean())}
    ok &= abs(a2.mean() - 2 * f0) < 2e-5 and abs(m2.mean() - 1) < 5e-3  # fine-grid scalloping + mean removal
    # conjugate cycle frequency of RRC(0.35) QPSK = symbol rate 1/sps
    for sps in (4, 5, 8, 10):
        nsym = L // sps + 40
        up = np.zeros((nfr, nsym * sps), complex)
        up[:, ::sps] = rng.choice(unit(_constellation("QPSK")), (nfr, nsym))
        h = _rrc(0.35, 8, sps)
        y = np.array([np.convolve(r, h, "same") for r in up])[:, 100:100 + L]
        y = y + 0.05 * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
        a = cyclic_features(_normalize(y))[:, 6]
        err = np.abs(a - 1 / sps) * L                          # in frame-DFT bins
        frac = float(np.mean(err < 1))
        res[f"conj_alpha_sps{sps}"] = {"median_alpha": float(np.median(a)), "expected": 1 / sps,
                                       "frac_within_1_over_L": frac, "median_err_bins": float(np.median(err))}
        # Unbiased when it locks (median error << 1 bin); misses are gross picks of the
        # low-frequency |x|^2 continuum (weak line at roll-off 0.35, ~100 symbols per frame).
        ok &= frac > 0.8 and np.median(err) < 0.1
    check("cyclic_features", ok, res)


# ------------------------------------------------------------------ 3. Haar wavelet
def wavelet_checks():
    res, ok = {}, True
    n = np.arange(512)
    errs = []
    for w in (0.05, 0.3, 1.0, 2.0):
        h = np.abs(haar(np.exp(1j * w * n)[None, :], 8))[0]
        errs.append(float(np.max(np.abs(h - tone_theory(w, 8)))))
    res["tone_closed_form_max_err"] = max(errs)
    ok &= max(errs) < 1e-9

    # rect-pulse PSK, sps 8, no CFO: |HWT_a| (a <= 8) is zero except at symbol transitions;
    # peak count == number of symbol changes; median-filtered variance == 0.
    # Balanced symbols (each point 32 times) so the frame mean is exactly 0 and the
    # mean removal inside the features leaves |x| = 1.
    sps, nsym = 8, 128
    sym = rng.permutation(np.repeat(unit(_constellation("QPSK")), nsym // 4))
    x = np.repeat(sym, sps)[None, :]
    h = np.abs(haar(x, 4))[0]
    changes = int(np.sum(sym[1:] != sym[:-1]))
    nz_runs = int(np.sum((h[1:] > 1e-9) & (h[:-1] <= 1e-9)) + (h[0] > 1e-9))
    res["rectQPSK_transitions"] = changes
    res["rectQPSK_nonzero_HWT_runs"] = nz_runs
    ok &= changes == nz_runs
    F = wavelet_features(x)[0]
    vmed = {k: float(F[WAV_NAMES.index(k)]) for k in ("hwt_vmed_2", "hwt_vmed_4", "hwt_amp_8")}
    res["rectQPSK"] = vmed
    ok &= max(vmed.values()) < 1e-12

    # PSK with residual carrier vs 2-FSK (tones +-0.15 rad/sample): flat vs staircase |HWT(x/|x|)|
    w0 = 0.15
    psk = np.repeat(rng.choice(unit(_constellation("BPSK")), nsym), sps) * np.exp(1j * w0 * np.arange(nsym * sps))
    fsk = np.exp(1j * np.cumsum(np.repeat(rng.choice([-w0, 2 * w0], nsym), sps)))  # two tones, different |HWT|
    Fp, Ff = wavelet_features(np.stack([psk, fsk]))
    k = WAV_NAMES.index("hwt_vmedu_4")                 # a = 4 < sps: flat stretches exist inside symbols
    res["vmedu_4_psk_vs_fsk"] = [float(Fp[k]), float(Ff[k])]
    ok &= Fp[k] < 1e-3 and Ff[k] > 10 * max(Fp[k], 1e-6)
    # 16QAM rect: amplitude steps -> hwt_amp > 0; PSK -> 0
    qam = np.repeat(rng.choice(unit(_constellation("16QAM")), nsym), sps)[None, :]
    res["hwt_amp_8_qam"] = float(wavelet_features(qam)[0, WAV_NAMES.index("hwt_amp_8")])
    ok &= res["hwt_amp_8_qam"] > 0.01
    check("haar_wavelet", ok, res)


if __name__ == "__main__":
    print("bell / partition formula count:", len(cumulant_terms(8, 4)), "distinct terms in C84")
    cumulant_checks()
    cyclic_checks()
    wavelet_checks()
    path = common.save_json({"failed": FAILS, "checks": REPORT}, "sanity_features.json")
    print(f"\n{len(REPORT) - len(FAILS)}/{len(REPORT)} checks passed -> {path}")
    sys.exit(1 if FAILS else 0)
