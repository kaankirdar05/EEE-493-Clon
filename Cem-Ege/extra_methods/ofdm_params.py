"""Blind OFDM parameter estimation: FFT size and cyclic-prefix length (van de Beek / Sandell style).

An OFDM symbol repeats its last cp samples as a prefix, so x[n] and x[n + N] are correlated for the
cp samples of every prefix. For each candidate (N, cp) from the grid below, fold the lag-N products
p[n] = x[n] x*[n+N] into one period (N + cp), sum a window of cp samples at every timing offset theta and take
the maximum (the timing is unknown):

    rho(theta) = |sum_window p| / (0.5 * sum_window (|x[n]|^2 + |x[n+N]|^2))        correlation coefficient
    z = rho * sqrt(K),  K = (symbols) * cp                                           standardised under noise

Standardising by sqrt(K) makes candidates with different cp comparable (under noise rho ~ 1/sqrt(K));
the estimate is the candidate with the largest z. Frames hold 16 OFDM symbols, QPSK on 75 % of the
subcarriers (guard bands), random timing offset, carrier phase and a frequency offset of up to +-0.3
subcarrier spacings (the correlation magnitude ignores a common phase rotation; the lag-N correlation of
a frequency offset only rotates it).

Grid: N in {64, 128, 256, 512, 1024}, cp = N/{4, 8, 16, 32}  (20 candidates; covers LTE/DAB/DVB-T-type ratios).
Checks (--check): clean 25 dB frames are identified 100 %; the standardised z under noise has the same mean
and spread for every candidate (so the argmax is not biased towards one of them).
Benchmark: exact (N, cp) match and N-only match vs per-sample SNR.

    python ofdm_params.py --check
    python ofdm_params.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (64, 128, 256, 512, 1024)
CPD = (4, 8, 16, 32)
GRID = [(n, n // d) for n in NS for d in CPD]
NSYM = 16


def ofdm_frame(n, cp, snr_db, rng, cfo=True):
    used = np.zeros(n, bool)
    k = int(0.75 * n) // 2
    used[1:k + 1] = True
    used[-k:] = True
    sym = np.zeros((NSYM + 2, n), complex)
    sym[:, used] = (rng.choice([-1, 1], (NSYM + 2, used.sum())) + 1j * rng.choice([-1, 1], (NSYM + 2, used.sum()))) / np.sqrt(2)
    t = np.fft.ifft(sym, axis=1) * np.sqrt(n / used.sum())
    s = np.concatenate([np.concatenate([row[-cp:], row]) for row in t])
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    off = int(rng.integers(0, n + cp))
    s = s[off:off + NSYM * (n + cp)]
    if cfo:
        s = s * np.exp(2j * np.pi * rng.uniform(-0.3, 0.3) / n * np.arange(len(s)))
    s = s * np.exp(1j * rng.uniform(0, 2 * np.pi))
    w = np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
    return s + w


def z_stat(x, n, cp):
    period = n + cp
    m = (len(x) - n) // period
    if m < 2:
        return 0.0
    L = m * period
    a, b = x[:L], x[n:n + L]
    p = (a * np.conj(b)).reshape(m, period).sum(0)
    e = (0.5 * (np.abs(a) ** 2 + np.abs(b) ** 2)).reshape(m, period).sum(0)
    ext = lambda v: np.concatenate([v, v[:cp - 1]])
    win = lambda v: np.convolve(ext(v), np.ones(cp), "valid")
    rho = np.abs(win(p)) / np.maximum(win(e), 1e-30)
    return float(rho.max() * np.sqrt(m * cp))


def estimate(x):
    z = [z_stat(x, n, cp) for n, cp in GRID]
    return GRID[int(np.argmax(z))], np.array(z)


def check():
    rng = np.random.default_rng(0)
    ok = tot = 0
    for n, cp in GRID:
        for _ in range(5):
            ok += estimate(ofdm_frame(n, cp, 25, rng))[0] == (n, cp); tot += 1
    zs = np.array([[z_stat(rng.standard_normal(16 * 1280) + 1j * rng.standard_normal(16 * 1280), n, cp) for n, cp in GRID] for _ in range(60)])
    print(f"clean 25 dB frames, all {len(GRID)} candidates: {ok}/{tot} identified exactly")
    print(f"standardised z under noise, per candidate: mean {zs.mean(0).min():.2f}..{zs.mean(0).max():.2f}, std {zs.std(0).min():.2f}..{zs.std(0).max():.2f} "
          "(equal across candidates = unbiased argmax)")
    return ok == tot and (zs.mean(0).max() - zs.mean(0).min()) < 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--per-cell", type=int, default=30)
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    rng = np.random.default_rng(7)
    snrs = list(range(-14, 7, 2))
    res = {}
    print("SNR (dB):      " + "  ".join(f"{s:4d}" for s in snrs))
    exact = {s: [] for s in snrs}; nonly = {s: [] for s in snrs}; bycp = {d: {s: [] for s in snrs} for d in CPD}
    for n, cp in GRID:
        for s in snrs:
            for _ in range(args.per_cell):
                est = estimate(ofdm_frame(n, cp, s, rng))[0]
                exact[s].append(est == (n, cp)); nonly[s].append(est[0] == n); bycp[n // cp][s].append(est == (n, cp))
    for name, d in (("exact (N, cp)", exact), ("N only", nonly)):
        print(f"{name:>14s} " + "  ".join(f"{np.mean(d[s]) * 100:4.0f}" for s in snrs))
    for dd in CPD:
        print(f"  cp = N/{dd:<2d}     " + "  ".join(f"{np.mean(bycp[dd][s]) * 100:4.0f}" for s in snrs))
    res = {"snr_db": snrs, "exact": [float(np.mean(exact[s])) for s in snrs], "n_only": [float(np.mean(nonly[s])) for s in snrs],
           "exact_by_cp_ratio": {f"N/{d}": [float(np.mean(bycp[d][s])) for s in snrs] for d in CPD},
           "snr_for_90pct_exact": next((s for s in snrs if np.mean(exact[s]) >= 0.9), None), "grid": GRID, "frames_per_cell": args.per_cell}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "ofdm_params.json"), "w"), indent=2)
    print("90 % exact from", res["snr_for_90pct_exact"], "dB per-sample SNR")


if __name__ == "__main__":
    main()
