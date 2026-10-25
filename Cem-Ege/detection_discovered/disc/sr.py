"""Stochastic-resonance (SR) pre-treatment before energy detection.

Zheng, Li, Djouadi & Wang, "Spectrum Sensing in Low SNR Regime via Stochastic
Resonance", arXiv:0906.0739 (2009):
  * overdamped bistable system  dx/dt = a x - b x^3 + s(t) + eta(t), a = b = 1,
    driven by the received signal plus ADDED noise of power D (tuned; the paper
    finds D = 0.43 for a 10 Hz tone of amplitude 0.3);
  * block metric  M = (1/K) sum_n max_t |Y(n, t)|^2, Y(n, .) = FFT of the n-th
    output window (FFT 256, sensing window 512 -> K = 2);
  * the paper reports gains at very low SNR and says SR is worse "when SNR is
    not quite low, e.g. -10 dB".

Here: Euler integration with step h per input sample,
    x[k+1] = x[k] + h (a x[k] - b x[k]^3 + g y[k] + sqrt(D) xi[k]),
xi ~ N(0, 1) i.i.d. (the added SR noise), g an input gain. g, D and h are the SR
design parameters and are tuned on dev seeds. Complex input: I and Q are passed
through two independent bistable systems.
"""

from __future__ import annotations

import numpy as np


def bistable(u: np.ndarray, h: float, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """u (..., N) real drive -> state x (..., N). Starts at x = 0."""
    x = np.zeros(u.shape[:-1])
    out = np.empty(u.shape)
    for k in range(u.shape[-1]):
        x = x + h * (a * x - b * x * x * x + u[..., k])
        x = np.clip(x, -50, 50)
        out[..., k] = x
    return out


def sr_filter(y: np.ndarray, g: float, D: float, h: float, rng, a=1.0, b=1.0) -> np.ndarray:
    """SR pre-treatment of a real or complex array (..., N)."""
    def one(r):
        return bistable(g * r + np.sqrt(D) * rng.standard_normal(r.shape), h, a, b)
    if np.iscomplexobj(y):
        return one(y.real) + 1j * one(y.imag)
    return one(y)


def fft_max_metric(z: np.ndarray, nfft: int = 256, exclude_dc: int = 1) -> np.ndarray:
    """Paper metric M = mean over K windows of max_bin |FFT|^2 (window mean removed,
    bins |k| < exclude_dc excluded so a slowly drifting well state is not 'signal')."""
    K = z.shape[-1] // nfft
    w = z[..., : K * nfft].reshape(*z.shape[:-1], K, nfft)
    w = w - w.mean(-1, keepdims=True)
    P = np.abs(np.fft.fft(w, axis=-1)) ** 2 / nfft
    k = np.abs(np.fft.fftfreq(nfft) * nfft)
    P = P[..., k >= exclude_dc]
    return P.max(-1).mean(-1)


def energy_metric(z: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(z - z.mean(-1, keepdims=True)) ** 2, axis=-1)
