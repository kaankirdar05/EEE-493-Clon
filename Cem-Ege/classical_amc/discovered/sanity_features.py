"""Known-answer / theory checks for the four new feature families. Exits non-zero on failure.

    python sanity_features.py      -> results/sanity_features.json
"""

from __future__ import annotations

import math
import sys
import time

import dcommon as dc
import numpy as np
from scipy.special import erf

import bispectrum as bs
import fractal as fd
import infotheory as it
import stransform as st

rng = np.random.default_rng(11)
REPORT, FAILS = {}, []


def check(name, ok, detail):
    REPORT[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def info(name, detail):
    REPORT[name + " (info)"] = detail
    print(f"[INFO] {name}: {detail}", flush=True)


# =============================================================== bispectrum
def bispectrum_checks():
    K = 256
    L = 64 * K
    # Gaussian: real and complex. E[b^2] ~ 1/K, about 5 % of bins above ln(20)/K
    for kind in ("real", "complex"):
        g = rng.standard_normal((8, L)) + (1j * rng.standard_normal((8, L)) if kind == "complex" else 0)
        B, b2, P, k = bs.bispectrum(g)
        b2m = b2[:, bs.MASK]
        mean_b2 = float(b2m.mean())
        frac = float(np.mean(b2m > np.log(20) / k))
        lvl = float(np.mean(np.abs(B[:, bs.MASK])) / np.mean(P[:, 1:]) ** 1.5)
        check(f"bispectrum_gaussian_{kind}", 0.7 / K < mean_b2 < 1.4 / K and 0.03 < frac < 0.07 and lvl < 0.1,
              {"K": K, "mean_b2": mean_b2, "theory_mean_b2_about_1_over_K": 1 / K, "frac_above_5pct_thr": frac,
               "normalised_|B|_level": lvl})
    # Quadratic phase coupling: f1 = 10, f2 = 17 bins of 64 -> f1+f2 = 27.
    f1, f2 = 10, 17
    n = np.arange(64)

    def qpc(coupled, nseg=K):
        segs = []
        for _ in range(nseg):
            p1, p2 = rng.uniform(0, 2 * np.pi, 2)
            p3 = p1 + p2 if coupled else rng.uniform(0, 2 * np.pi)
            segs.append(np.cos(2 * np.pi * f1 * n / 64 + p1) + np.cos(2 * np.pi * f2 * n / 64 + p2)
                        + np.cos(2 * np.pi * (f1 + f2) * n / 64 + p3) + 0.1 * rng.standard_normal(64))
        return np.concatenate(segs)[None]
    B, b2, P, _ = bs.bispectrum(qpc(True))
    peak = np.unravel_index(np.argmax(b2[0] * bs.MASK), b2[0].shape)
    # equivalent locations of (10, 17) by the symmetries of a real signal's bispectrum
    equiv = {(10, 17), (17, 10), (54, 47), (47, 54), (10, 37), (37, 10), (17, 37), (37, 17),
             (54, 27), (27, 54), (47, 27), (27, 47)}
    Bu, b2u, _, _ = bs.bispectrum(qpc(False))
    check("bispectrum_qpc_peak", b2[0, f1, f2] > 0.9 and tuple(int(v) for v in peak) in equiv and b2u[0, f1, f2] < 5.0 / K,
          {"b2_at_(10,17)_coupled": float(b2[0, f1, f2]), "b2_at_(10,17)_uncoupled": float(b2u[0, f1, f2]),
           "argmax_b2": [int(v) for v in peak], "median_b2_elsewhere": float(np.median(b2[0][bs.MASK]))})
    # Frame-length (K = 16) behaviour on Gaussian frames: what the features see at -20 dB SNR
    g = (rng.standard_normal((400, 1024)) + 1j * rng.standard_normal((400, 1024))) / np.sqrt(2)
    F = bs.bispectrum_features(g)
    info("bispectrum_features_on_gaussian_frames_K16",
         {n_: float(F[:, i].mean()) for i, n_ in enumerate(bs.BIS_NAMES[:10])} | {"theory_mean_b2": 1 / 16})


# =============================================================== fractal
def fbm(H, n, rng):
    """Exact fBm via Davies-Harte circulant embedding of fractional Gaussian noise."""
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(c).real
    lam = np.maximum(lam, 0)
    m = len(c)
    w = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) * np.sqrt(lam / m)
    fgn = np.fft.fft(w).real[:n]
    return np.cumsum(fgn)


def fractal_checks():
    N, R = 1024, 200
    t = np.arange(N, dtype=float)
    line = (3 * t + 1)[None]
    noise = rng.standard_normal((R, N))
    brown = np.cumsum(rng.standard_normal((R, N)), axis=1)
    sine = np.sin(2 * np.pi * t / 256)[None]
    res = {
        "higuchi_line": float(fd.higuchi(line)[0]),
        "higuchi_white": float(fd.higuchi(noise).mean()),
        "higuchi_brown": float(fd.higuchi(brown).mean()),
        "higuchi_fbm_H0.3 (theory 1.7)": float(np.mean(fd.higuchi(np.array([fbm(0.3, N, rng) for _ in range(R)])))),
        "higuchi_fbm_H0.7 (theory 1.3)": float(np.mean(fd.higuchi(np.array([fbm(0.7, N, rng) for _ in range(R)])))),
        "katz_line": float(fd.katz(line)[0]),
        "petrosian_line": float(fd.petrosian(line)[0]),
        "petrosian_white": float(fd.petrosian(noise).mean()),
        "petrosian_white_theory": math.log10(N) / (math.log10(N) + math.log10(N / (N + 0.4 * 2 / 3 * (N - 2)))),
        "box_line": float(fd.boxcount(line)[0]),
        "box_sine": float(fd.boxcount(sine)[0]),
        "box_brown": float(fd.boxcount(brown).mean()),
        "box_white": float(fd.boxcount(noise).mean()),
        "katz_white (no closed form)": float(fd.katz(noise).mean()),
        "katz_brown (no closed form)": float(fd.katz(brown).mean()),
    }
    check("higuchi_known_values",
          abs(res["higuchi_line"] - 1) < 1e-6 and abs(res["higuchi_white"] - 2) < 0.05 and abs(res["higuchi_brown"] - 1.5) < 0.05
          and abs(res["higuchi_fbm_H0.3 (theory 1.7)"] - 1.7) < 0.07 and abs(res["higuchi_fbm_H0.7 (theory 1.3)"] - 1.3) < 0.07,
          {k: v for k, v in res.items() if k.startswith("higuchi")})
    check("katz_petrosian_known_values",
          abs(res["katz_line"] - 1) < 1e-9 and abs(res["petrosian_line"] - 1) < 1e-12
          and abs(res["petrosian_white"] - res["petrosian_white_theory"]) < 0.003,
          {k: v for k, v in res.items() if k.startswith(("katz", "petrosian"))})
    # independent brute-force box count: rasterise the piecewise-linear graph densely and count cells
    def brute(y, js=fd.BOX_J, up=64):
        y = (y - y.min()) / (y.max() - y.min())
        y = np.minimum(y, 1 - 1e-12)
        tt = np.linspace(0, len(y) - 1, (len(y) - 1) * up + 1)
        yy = np.interp(tt, np.arange(len(y)), y)
        xx = np.minimum(tt / len(y), 1 - 1e-12)                      # same time scaling as fd.boxcount
        logs = [np.log(len(set(zip((xx * 2 ** j).astype(int).tolist(), (yy * 2 ** j).astype(int).tolist()))))
                for j in js]
        return np.polyfit(np.log(2.0 ** js), logs, 1)[0]
    bf = [(float(fd.boxcount(b[None])[0]), float(brute(b))) for b in brown[:10]]
    bdiff = float(np.max([abs(a - b) for a, b in bf]))
    order = [float(np.mean(fd.boxcount(np.array([fbm(H, N, rng) for _ in range(50)])))) for H in (0.9, 0.7, 0.5, 0.3, 0.1)]
    res["box_vs_bruteforce_max_abs_diff (brownian)"] = bdiff
    res["box_fbm_H0.9..0.1 (theory 1.1..1.9)"] = order
    check("boxcount_known_values",
          abs(res["box_line"] - 1) < 0.03 and abs(res["box_sine"] - 1) < 0.03 and bdiff < 0.03
          and bool(np.all(np.diff(order) > 0)),
          {k: v for k, v in res.items() if k.startswith("box")} |
          {"note": "finite-size bias: with 1024 samples the box-counting estimate of rough graphs is low "
                   "(Brownian ~1.34 instead of 1.5, white ~1.69 instead of 2); the implementation agrees with "
                   "an independent rasterised count and orders fBm correctly"})


# =============================================================== information theory
def naive_sampen_apen(v, m=2, r_frac=0.2):
    v = np.asarray(v, np.float32)
    N = len(v)
    r = r_frac * float(np.std(v))

    def cnt(mm, ntempl, self_match):
        c = np.zeros(ntempl)
        for i in range(ntempl):
            for j in range(ntempl):
                if i == j and not self_match:
                    continue
                if max(abs(float(v[i + k]) - float(v[j + k])) for k in range(mm)) <= r:
                    c[i] += 1
        return c
    B = cnt(m, N - m, False).sum() / 2
    A = cnt(m + 1, N - m, False).sum() / 2
    sampen = -math.log(A / B)
    phi_m = np.mean(np.log(cnt(m, N - m + 1, True) / (N - m + 1)))
    phi_m1 = np.mean(np.log(cnt(m + 1, N - m, True) / (N - m)))
    return sampen, phi_m - phi_m1


def naive_det_lam(v, m=2, r_frac=0.2):
    v = np.asarray(v, np.float32)
    N = len(v)
    r = r_frac * float(np.std(v))
    n = N - m + 1
    R = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            R[i, j] = i != j and max(abs(float(v[i + k]) - float(v[j + k])) for k in range(m)) <= r
    tot = R.sum()

    def runs(lines):
        s = 0
        for ln in lines:
            run = 0
            for b in list(ln) + [False]:
                if b:
                    run += 1
                else:
                    if run >= 2:
                        s += run
                    run = 0
        return s
    det = runs([np.diagonal(R, k) for k in range(-n + 1, n)]) / tot
    lam = runs([R[:, j] for j in range(n)]) / tot
    return R.sum() / (n * (n - 1)), det, lam


def info_checks():
    N = 1024
    noise = rng.standard_normal((50, N))
    ramp = np.arange(N, dtype=float)[None]
    pe_w = float(it.permutation_entropy(noise, 4, 1).mean())
    pe_w4 = float(it.permutation_entropy(noise, 4, 4).mean())
    pe_r = float(it.permutation_entropy(ramp, 4, 1)[0])
    pe_sine = float(it.permutation_entropy(np.sin(2 * np.pi * np.arange(N) / 64)[None], 4, 1)[0])
    check("permutation_entropy_known_values", pe_w > 0.99 and pe_w4 > 0.99 and pe_r == 0.0,
          {"white_d1": pe_w, "white_d4": pe_w4, "ramp": pe_r, "sine_period64 (info)": pe_sine})

    # SampEn / ApEn / RQA against direct double-loop implementations
    errs = []
    for _ in range(3):
        v = rng.standard_normal(150).cumsum() if rng.random() < 0.5 else rng.standard_normal(150)
        s, a, rr, det, lam = it.entropy_rqa_1d(v)
        s0, a0 = naive_sampen_apen(v)
        rr0, det0, lam0 = naive_det_lam(v)
        errs.append(max(abs(s - s0), abs(a - a0), abs(rr - rr0), abs(det - det0), abs(lam - lam0)))
    check("sampen_apen_rqa_equal_naive_loops", max(errs) < 1e-9, {"max_abs_diff": float(max(errs))})

    se = np.array([it.entropy_rqa_1d(f) for f in noise[:20]])
    theory = -math.log(erf(0.1))
    tt = np.arange(N)
    # exactly periodic, period 16: with r = 0.05 std only exact repeats match -> SampEn = 0, DET = 1
    s16 = it.entropy_rqa_1d(np.sin(2 * np.pi * tt / 16), r_frac=0.05)
    # default r = 0.2 std: near-matches around the extrema make SampEn > 0 for coarse sampling; it -> 0
    # as the sampling gets denser (informational)
    dense = {P: [float(v) for v in it.entropy_rqa_1d(np.sin(2 * np.pi * tt / P))] for P in (16, 64, 256)}
    check("sampen_known_values",
          abs(se[:, 0].mean() - theory) < 0.1 and abs(s16[0]) < 1e-12 and abs(s16[1]) < 0.01 and dense[256][0] < 0.08,
          {"white_sampen": float(se[:, 0].mean()), "theory_-ln_erf(0.1)": theory, "white_apen": float(se[:, 1].mean()),
           "sine_p16_r0.05_sampen": float(s16[0]), "sine_p16_r0.05_apen": float(s16[1]),
           "sine_r0.2_[sampen,apen,rr,det,lam]_by_period": dense})
    p = erf(0.1)
    check("rqa_known_values", abs(se[:, 2].mean() - p ** 2) < 0.002 and s16[3] == 1.0 and dense[256][3] > 0.95,
          {"white_RR": float(se[:, 2].mean()), "theory_RR_white_p^2": p ** 2, "white_DET": float(se[:, 3].mean()),
           "sine_p16_r0.05_DET": float(s16[3]), "sine_p256_DET": dense[256][3]})

    u = rng.uniform(0, 3, (20, 100000))
    h1, h2 = it.hist_entropy(u, 0, 3)
    c1, c2 = it.hist_entropy(np.full((1, 1000), 1.234), 0, 3)
    check("histogram_entropy_known_values", np.all(h1 > 0.999) and np.all(h2 > 0.998) and c1[0] == 0 and abs(c2[0]) < 1e-12,
          {"uniform_shannon": float(h1.mean()), "uniform_renyi2": float(h2.mean()), "constant_shannon": float(c1[0]),
           "constant_renyi2": float(c2[0])})


# =============================================================== S-transform
def st_checks():
    N = 1024
    k = np.arange(N)
    all_v = np.concatenate([np.arange(1, 512), -np.arange(1, 513)])
    f0, A = 37, 1.7
    x = A * np.exp(2j * np.pi * f0 * k / N)
    S = st.stransform(x, all_v)[0]
    mag = np.abs(S)
    peak_v = all_v[np.argmax(mag, axis=0)]
    dm = (f0 - all_v + N // 2) % N - N // 2                          # the DFT is N-periodic in frequency
    theory = A * np.exp(-2 * np.pi ** 2 * dm ** 2 / all_v.astype(float) ** 2)
    err_theory = float(np.max(np.abs(mag - theory[:, None])))
    check("stransform_tone_peak", np.all(peak_v == f0) and abs(mag[all_v == f0].mean() - A) < 1e-12 and err_theory < 1e-12,
          {"peak_voice_all_times": int(np.unique(peak_v)[0]) if len(np.unique(peak_v)) == 1 else str(np.unique(peak_v)),
           "|S| at f0": float(mag[all_v == f0].mean()), "max_err_vs_closed_form": err_theory})
    # real tone: peaks at +-f0
    xr = np.cos(2 * np.pi * 100 * k / N)
    Sr = np.abs(st.stransform(xr, all_v)[0])
    pv = all_v[np.argmax(Sr[:, 200:800], axis=0)]
    check("stransform_real_tone_peak", set(np.unique(np.abs(pv)).tolist()) == {100},
          {"peak_voices": sorted(set(pv.tolist()))})
    # time integration returns the spectrum
    z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    Sz = st.stransform(z, all_v)[0]
    X = np.fft.fft(z)
    err = float(np.max(np.abs(Sz.sum(1) - X[all_v % N])))
    check("stransform_time_integral_equals_dft", err < 1e-9, {"max_abs_err": err})
    # chirp: peak voice follows the instantaneous frequency (info + loose check)
    f_inst = 20 + 200 * k / N                                       # bins
    ch = np.exp(2j * np.pi * np.cumsum(f_inst) / N)
    Sc = np.abs(st.stransform(ch, all_v)[0])
    pv = all_v[np.argmax(Sc, axis=0)]
    mid = slice(100, 924)
    med = float(np.median(np.abs(pv[mid] - f_inst[mid])))
    check("stransform_chirp_tracking", med < 3, {"median_abs_err_bins (100<k<924)": med})


# =============================================================== extraction smoke test
def smoke():
    from amc.synthetic import _constellation
    x = []
    for c in ("BPSK", "QPSK", "16QAM"):
        s = rng.choice(_constellation(c).astype(complex), (20, 128))
        up = np.repeat(s, 8, axis=1)
        x.append(up + 0.1 * (rng.standard_normal(up.shape) + 1j * rng.standard_normal(up.shape)))
    x = np.concatenate(x).astype(np.complex64)
    out = {}
    for name, fn, names in (("bis", bs.bispectrum_features, bs.BIS_NAMES), ("fd", fd.fractal_features, fd.FD_NAMES),
                            ("info", it.info_features, it.INFO_NAMES), ("st", st.st_features, st.ST_NAMES)):
        t0 = time.perf_counter()
        F = fn(x)
        ms = (time.perf_counter() - t0) / len(x) * 1e3
        ok = F.shape == (len(x), len(names)) and np.all(np.isfinite(F)) and np.all(F.std(0) > 0)
        out[name] = {"n_features": len(names), "shape_ok": F.shape == (len(x), len(names)),
                     "finite": bool(np.all(np.isfinite(F))), "constant_columns": [names[i] for i in np.flatnonzero(F.std(0) == 0)],
                     "ms_per_frame_single_process": ms}
        check(f"extraction_smoke_{name}", ok, out[name])


if __name__ == "__main__":
    for f in (bispectrum_checks, fractal_checks, info_checks, st_checks, smoke):
        f()
    REPORT["all_pass"] = not FAILS
    print("saved", dc.save_json(REPORT, "sanity_features.json"))
    print("FAILED:" if FAILS else "all checks passed", FAILS)
    sys.exit(1 if FAILS else 0)
