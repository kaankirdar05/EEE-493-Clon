"""Symbol-domain classifiers for BPSK / QPSK / 8PSK / 16QAM / 64QAM.

Input: unit-power, symbol-spaced, CFO-corrected symbols z (practical_alrt.to_symbols).
Unknown per frame: carrier phase, signal amplitude, noise variance.

KS  - Kolmogorov-Smirnov goodness-of-fit (after Wang & Wang 2010, "Fast and robust
      modulation classification via Kolmogorov-Smirnov test"). For each candidate k:
        * class-conditional SNR estimate (M2M4 with that class's E|c|^4),
        * class-conditional phase estimate theta = arg(<z^M> / <c^M>) / M, derotate,
        * KS distances between the empirical CDFs of I, Q, |z| and arg(z) and REFERENCE CDFs
          obtained by simulating class k at the nearest grid SNR and passing the
          simulated frames through the same phase estimator,
      decide argmin_k of the summed KS distance (components chosen on the training split).
      NOTE: I and Q marginals alone cannot separate BPSK from QPSK: the QPSK phase estimator
      rotates BPSK to +-45 deg, where its I and Q marginals equal QPSK's (only the joint
      differs). The phase component arg(z) fixes this (found by the synthetic sanity check).
EM  - Gaussian-mixture constellation fit. For each candidate k, the symbols are modelled as
          z ~ (1/M) sum_m CN(g c_m, s2)
      with unknown complex gain g (amplitude + phase) and noise variance s2, estimated by EM
      (closed-form M-step; 4 phase restarts). Every candidate has the same 3 free parameters,
      so the BIC decision equals the maximum-likelihood decision.
"""

from __future__ import annotations

from functools import lru_cache

import common  # noqa: F401
import numpy as np
from scipy.special import logsumexp

from amc.synthetic import _constellation

CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
POW = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4, "64QAM": 4}   # phase-estimator power = symmetry order


def _unit(p):
    p = np.asarray(p, complex)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


CONS = {c: _unit(_constellation(c)) for c in CLASSES}
KURT = {c: float(np.mean(np.abs(p) ** 4)) for c, p in CONS.items()}
MC = {c: complex(np.mean(p ** POW[c])) for c, p in CONS.items()}


def synth_symbols(cls, n_frames, nsym, snr_db, rng, random_phase=True):
    """Ideal symbol-level frames: unit-power constellation, random phase, AWGN, then
    per-frame unit-power normalisation (as to_symbols does). Returns (frames, phases)."""
    s = rng.choice(CONS[cls], (n_frames, nsym))
    ph = rng.uniform(0, 2 * np.pi, (n_frames, 1)) if random_phase else np.zeros((n_frames, 1))
    s2 = 10 ** (-snr_db / 10)
    z = s * np.exp(1j * ph) + np.sqrt(s2 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    return z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)), ph[:, 0]


# ======================================================================= KS
def ks2(a_sorted, b_sorted):
    """Exact two-sample KS statistic (same as scipy.stats.ks_2samp(...).statistic)."""
    allv = np.concatenate([a_sorted, b_sorted])
    fa = np.searchsorted(a_sorted, allv, side="right") / len(a_sorted)
    fb = np.searchsorted(b_sorted, allv, side="right") / len(b_sorted)
    return float(np.max(np.abs(fa - fb)))


def m2m4_snr_db(z, cls):
    """Class-conditional M2M4 SNR estimate for unit-power z (M2 = 1)."""
    m4 = np.mean(np.abs(z) ** 4)
    S = np.sqrt(max(2.0 - m4, 0.0) / (2.0 - KURT[cls]))
    S = min(max(S, 1e-3), 1 - 1e-4)
    return 10 * np.log10(S / (1 - S))


def derotate(z, cls):
    M = POW[cls]
    th = np.angle(np.mean(z ** M) / MC[cls]) / M
    return z * np.exp(-1j * th)


SNR_GRID = np.arange(-10, 41, 1)


@lru_cache(maxsize=None)
def reference(cls, snr_db, nsym=128, n_frames=200, seed=12345):
    """Sorted reference samples (I, Q, |z|, arg z) of class cls at snr_db, pooled over frames,
    with the same per-frame phase estimation as applied to the received frames."""
    rng = np.random.default_rng(seed + 1000 * CLASSES.index(cls) + int(snr_db) + 50)
    z, _ = synth_symbols(cls, n_frames, nsym, snr_db, rng)
    zr = np.concatenate([derotate(f, cls) for f in z])
    return np.sort(zr.real), np.sort(zr.imag), np.sort(np.abs(zr)), np.sort(np.angle(zr))


def ks_distances(z):
    """(5 classes, 4 components [I, Q, |z|, arg z]) KS distances of frame z."""
    D = np.empty((len(CLASSES), 4))
    for k, c in enumerate(CLASSES):
        snr = int(SNR_GRID[np.argmin(np.abs(SNR_GRID - m2m4_snr_db(z, c)))])
        ref = reference(c, snr)
        zr = derotate(z, c)
        D[k] = [ks2(np.sort(v), r) for v, r in zip((zr.real, zr.imag, np.abs(zr), np.angle(zr)), ref)]
    return D


KS_VARIANTS = {"amp": [2], "iq": [0, 1], "iqa": [0, 1, 2], "ap": [2, 3], "iqap": [0, 1, 2, 3]}


def ks_decide(D, variant):
    return int(np.argmin(D[:, KS_VARIANTS[variant]].sum(1)))


# ======================================================================= EM
def em_fit(z, cls, n_restarts=4, max_iter=100, tol=1e-7, return_trace=False):
    """Fit z ~ (1/M) sum CN(g c_m, s2). Returns (loglik, g, s2[, trace of best run])."""
    c = CONS[cls]
    M = len(c)
    n = len(z)
    A0 = np.sqrt(10 ** (m2m4_snr_db(z, cls) / 10) / (1 + 10 ** (m2m4_snr_db(z, cls) / 10)))
    th0 = np.angle(np.mean(z ** POW[cls]) / MC[cls]) / POW[cls]
    sym = 2 * np.pi / POW[cls]
    best = (-np.inf, None, None, None)
    for r in range(n_restarts):
        g = A0 * np.exp(1j * (th0 + r * sym / n_restarts))
        s2 = max(1 - A0**2, 0.02)
        trace, ll_old = [], -np.inf
        for _ in range(max_iter):
            d = np.abs(z[:, None] - g * c[None, :]) ** 2                   # (n, M)
            logp = -d / s2 - np.log(np.pi * s2) - np.log(M)
            lse = logsumexp(logp, axis=1)
            ll = float(lse.sum())
            trace.append(ll)
            R = np.exp(logp - lse[:, None])                                 # responsibilities
            g = np.sum(R * z[:, None] * np.conj(c)[None, :]) / np.sum(R * np.abs(c)[None, :] ** 2)
            s2 = max(float(np.sum(R * np.abs(z[:, None] - g * c[None, :]) ** 2) / n), 1e-4)
            if ll - ll_old < tol * n:
                break
            ll_old = ll
        # final log-likelihood at the final parameters
        d = np.abs(z[:, None] - g * c[None, :]) ** 2
        ll = float(logsumexp(-d / s2 - np.log(np.pi * s2) - np.log(M), axis=1).sum())
        trace.append(ll)
        if ll > best[0]:
            best = (ll, g, s2, trace)
    return best if return_trace else best[:3]


def em_classify(z):
    lls = np.array([em_fit(z, c)[0] for c in CLASSES])
    return int(np.argmax(lls)), lls


def oracle_ml(z, g, s2):
    """Coherent ML with the TRUE gain and noise variance (upper reference for sanity checks)."""
    ll = [float(logsumexp(-np.abs(z[:, None] - g * CONS[c][None, :]) ** 2 / s2, axis=1).sum()
                - len(z) * np.log(len(CONS[c]))) for c in CLASSES]
    return int(np.argmax(ll))
