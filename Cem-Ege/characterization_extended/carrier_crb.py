"""Task 1 - carrier-frequency estimators vs the Cramer-Rao bound (synthetic).

Signals (complex baseband, AWGN, SNR = signal power / noise power per sample):
  tone : A exp(j(2 pi f0 n + phi)),  f0 ~ U(-0.4, 0.4) cycles/sample
  qpsk : RRC (beta 0.35, 4 samples/symbol) QPSK, unit power, f0 ~ U(-0.1, 0.1)
N = 1024 and 4096.

Bound (cycles/sample^2):  CRB(f) = 6 / ((2 pi)^2 SNR N (N^2 - 1))   (Rife & Boorstyn 1974)
For QPSK this is the data-aided / modified CRB (known symbols); NDA estimators cannot
beat it and are expected to sit above it (self-noise, higher threshold).

Estimators
  pdg_parab_N    N-point periodogram peak + parabola on |X|        (biased: error floor)
  pdg_parab_4N   4N zero-padded periodogram peak + parabola on |X|
  jacobsen       N-point DFT, Jacobsen 3-bin estimator with Candan's (2011) bias correction
  quinn2         Quinn's second estimator (1997)
  ml             periodogram maximiser: 4N peak then bounded scalar search (= ML for one tone)
  kay            Kay (1989) weighted phase-difference estimator (on the raw signal)
  kay_coarse     Kay after removing the N-point FFT peak frequency (avoids +-pi phase wrapping)
  music_M        single-signal MUSIC: peak of |a(f)^H u1|^2 (u1 = principal eigenvector of the
                 forward-backward M x M sample covariance), refined by scalar search
  esprit_M       single-signal LS-ESPRIT on u1
QPSK only
  p4_jacobsen    4th power x^4, Jacobsen/Candan on its N-point DFT, / 4
  p4_ml          4th power x^4, periodogram maximiser, / 4
  da_ml          data-aided reference: x * conj(known waveform), then ml

    python carrier_crb.py [--trials 300]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar

import common
from amc.synthetic import _rrc

TWO_PI = 2 * np.pi


def crb_tone(snr_lin, n):
    return 6.0 / (TWO_PI**2 * snr_lin * n * (n**2 - 1))


# ---------------------------------------------------------------- estimators
def _peak(X):
    return int(np.argmax(np.abs(X)))


def pdg_parab(x, pad=1):
    n = len(x) * pad
    X = np.abs(np.fft.fft(x, n))
    k = _peak(X)
    a, b, c = X[k - 1], X[k], X[(k + 1) % n]
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return common.wrap((k + d) / n)


def jacobsen(x):
    n = len(x)
    X = np.fft.fft(x)
    k = _peak(X)
    a, b, c = X[k - 1], X[k], X[(k + 1) % n]
    d = np.real((a - c) / (2 * b - a - c))
    d *= np.tan(np.pi / n) / (np.pi / n)            # Candan 2011 bias correction
    return common.wrap((k + d) / n)


def _tau(x):
    return 0.25 * np.log(3 * x**2 + 6 * x + 1) - np.sqrt(6) / 24 * np.log(
        (x + 1 - np.sqrt(2 / 3)) / (x + 1 + np.sqrt(2 / 3)))


def quinn2(x):
    n = len(x)
    X = np.fft.fft(x)
    k = _peak(X)
    ap = np.real(X[(k + 1) % n] / X[k])
    am = np.real(X[k - 1] / X[k])
    dp = -ap / (1 - ap)
    dm = am / (1 - am)
    d = (dp + dm) / 2 + _tau(dp**2) - _tau(dm**2)
    return common.wrap((k + d) / n)


def _refine_peak(x, f0, half_width):
    """Maximise |sum x[n] exp(-j 2 pi f n)|^2 over f in [f0 - hw, f0 + hw]."""
    n = np.arange(len(x))
    obj = lambda f: -np.abs(np.dot(x, np.exp(-1j * TWO_PI * f * n)))
    r = minimize_scalar(obj, bounds=(f0 - half_width, f0 + half_width), method="bounded",
                        options={"xatol": 1e-12 / len(x)})
    return r.x


def ml(x):
    n4 = 4 * len(x)
    X = np.fft.fft(x, n4)
    k = _peak(X)
    return common.wrap(_refine_peak(x, k / n4, 1.0 / n4))


KAY_W = {}


def kay(x):
    n = len(x)
    if n not in KAY_W:
        t = np.arange(n - 1)
        w = 1.5 * n / (n**2 - 1) * (1 - ((t - (n / 2 - 1)) / (n / 2)) ** 2)
        KAY_W[n] = w
    return float(np.sum(KAY_W[n] * np.angle(x[1:] * np.conj(x[:-1])))) / TWO_PI


def kay_coarse(x):
    """Kay after removing a coarse (N-point FFT peak) frequency, so that the phase
    increments stay near 0 and do not wrap at +-pi for |f0| near 0.5."""
    n = len(x)
    fc = _peak(np.fft.fft(x)) / n
    return common.wrap(fc + kay(x * np.exp(-1j * TWO_PI * fc * np.arange(n))))


def _principal_vec(x, m):
    H = sliding_window_view(x, m)                      # (K, M) snapshots
    R = H.T @ H.conj() / H.shape[0]                    # R[i,j] = E x[n+i] x*[n+j]
    J = np.eye(m)[::-1]
    R = 0.5 * (R + J @ R.conj() @ J)                   # forward-backward
    w, V = np.linalg.eigh(R)
    return V[:, -1]


def music(x, m):
    u = _principal_vec(x, m)
    nf = 64 * m
    U = np.abs(np.fft.fft(u, nf))                      # |a(f)^H u| on a grid; a(f) = exp(j2pi f k)
    k = _peak(U)
    # u is proportional to a(f0), so |sum_k u_k exp(-j 2 pi f k)| peaks at f0
    return common.wrap(_refine_peak(u, k / nf, 1.0 / nf))


def esprit(x, m):
    u = _principal_vec(x, m)
    phi = np.vdot(u[:-1], u[1:]) / np.vdot(u[:-1], u[:-1])
    return float(np.angle(phi)) / TWO_PI


def p4_jacobsen(x):
    return common.wrap(jacobsen(x**4) / 4, 0.25)


def p4_ml(x):
    return common.wrap(ml(x**4) / 4, 0.25)


# ---------------------------------------------------------------- signals
RRC = _rrc(0.35, 8, 4)
QPSK = np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))


def qpsk_wave(n, rng):
    nsym = n // 4 + 40
    up = np.zeros(nsym * 4, complex)
    up[::4] = rng.choice(QPSK, nsym)
    s = np.convolve(up, RRC, "same")[80:80 + n]
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


def run(kind, n, snrs, trials, rng):
    if kind == "tone":
        ests = {"pdg_parab_N": pdg_parab, "pdg_parab_4N": lambda x: pdg_parab(x, 4), "jacobsen": jacobsen,
                "quinn2": quinn2, "ml": ml, "kay": kay, "kay_coarse": kay_coarse,
                "music_16": lambda x: music(x, 16), "esprit_16": lambda x: esprit(x, 16),
                "music_64": lambda x: music(x, 64), "esprit_64": lambda x: esprit(x, 64)}
    else:
        ests = {"p4_jacobsen": p4_jacobsen, "p4_ml": p4_ml, "da_ml": None}
    mse = {k: [] for k in ests}
    bias = {k: [] for k in ests}
    nn = np.arange(n)
    for snr in snrs:
        err = {k: [] for k in ests}
        sig2 = 10 ** (-snr / 10)
        for _ in range(trials):
            if kind == "tone":
                f0 = rng.uniform(-0.4, 0.4)
                s = np.ones(n, complex)
            else:
                f0 = rng.uniform(-0.1, 0.1)
                s = qpsk_wave(n, rng)
            ph = rng.uniform(0, TWO_PI)
            x = s * np.exp(1j * (TWO_PI * f0 * nn + ph))
            x = x + np.sqrt(sig2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            for k, f in ests.items():
                if k == "da_ml":
                    fh = ml(x * np.conj(s))
                    e = common.wrap(fh - f0)
                elif kind == "qpsk":
                    e = common.wrap(f(x) - f0, 0.25)
                else:
                    e = common.wrap(f(x) - f0)
                err[k].append(e)
        for k in ests:
            e = np.array(err[k])
            mse[k].append(float(np.mean(e**2)))
            bias[k].append(float(np.mean(e)))
    return mse, bias


def threshold_and_efficiency(snrs, mse, crb):
    """Threshold = lowest SNR from which MSE/CRB stays < 3 at all higher SNRs;
    efficiency = median MSE/CRB over SNRs >= threshold + 5 dB."""
    r = np.array(mse) / np.array(crb)
    thr = None
    for i in range(len(snrs)):
        if np.all(r[i:] < 3.0):
            thr = snrs[i]
            break
    eff = None
    if thr is not None:
        sel = np.array(snrs) >= thr + 5
        if sel.any():
            eff = float(np.median(r[sel]))
    return thr, eff, r.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    out = {}
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    t0 = time.perf_counter()
    for row, kind in enumerate(["tone", "qpsk"]):
        snrs = list(np.arange(-25, 41, 2.5)) if kind == "tone" else list(np.arange(-10, 41, 2.5))
        for col, n in enumerate([1024, 4096]):
            mse, bias = run(kind, n, snrs, args.trials, rng)
            crb = [crb_tone(10 ** (s / 10), n) for s in snrs]
            res = {}
            ax = axs[row, col]
            ax.semilogy(snrs, crb, "k-", lw=2.5, label="CRB" if kind == "tone" else "MCRB (data-aided)")
            for k in mse:
                thr, eff, ratio = threshold_and_efficiency(snrs, mse[k], crb)
                res[k] = {"mse": mse[k], "bias": bias[k], "mse_over_crb": ratio, "threshold_snr_db": thr,
                          "median_mse_over_crb_above_threshold_plus5dB": eff}
                ax.semilogy(snrs, mse[k], marker=".", label=k)
                print(f"{kind:4s} N={n:4d} {k:>13s}: threshold {thr} dB, MSE/CRB above it {eff}  "
                      f"(at max SNR {ratio[-1]:.2f})")
            ax.set_title(f"{kind}, N = {n}")
            ax.set_xlabel("SNR per sample (dB)"); ax.set_ylabel("MSE (cycles/sample)$^2$")
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
            ax.set_ylim(max(min(crb) / 10, 1e-22), 1e-1)
            out[f"{kind}_N{n}"] = {"snr_db": [float(s) for s in snrs], "crb": crb, "estimators": res}
    fig.tight_layout()
    fig.savefig(os.path.join(common.RESULTS, "carrier_crb.png"), dpi=120)
    with open(os.path.join(common.RESULTS, "carrier_crb.json"), "w") as f:
        json.dump({"trials_per_snr": args.trials, "crb_formula": "6/((2pi)^2 SNR N (N^2-1)) cycles^2/sample^2",
                   "results": out, "runtime_s": time.perf_counter() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
