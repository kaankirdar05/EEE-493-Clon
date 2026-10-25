"""Synthetic test signals (unit average power)."""

from __future__ import annotations

import numpy as np


def rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, v in enumerate(t):
        if abs(v) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(v) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * v * (1 - beta)) + 4 * beta * v * np.cos(np.pi * v * (1 + beta))) / (np.pi * v * (1 - (4 * beta * v) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def qpsk(rng, n_trials: int, N: int, sps: int = 8, beta: float = 0.35, f0=None) -> np.ndarray:
    """RRC-shaped QPSK, `sps` samples/symbol, carrier offset f0 (cycles/sample;
    array per trial, scalar, or None = uniform in +-0.4). Unit power."""
    h = rrc(beta, sps)
    nsym = N // sps + 2 * len(h) // sps + 2
    sym = (rng.choice([-1, 1], (n_trials, nsym)) + 1j * rng.choice([-1, 1], (n_trials, nsym))) / np.sqrt(2)
    up = np.zeros((n_trials, nsym * sps), complex)
    up[:, ::sps] = sym
    from scipy.signal import fftconvolve
    s = fftconvolve(up, h[None, :], axes=1)
    st = len(h) + int(rng.integers(0, sps))
    s = s[:, st: st + N]
    s /= np.sqrt(np.mean(np.abs(s) ** 2, axis=1, keepdims=True))
    if f0 is None:
        f0 = rng.uniform(-0.4, 0.4, n_trials)
    f0 = np.broadcast_to(np.asarray(f0, float), (n_trials,))
    return s * np.exp(2j * np.pi * f0[:, None] * np.arange(N)[None, :])


def tone(rng, n_trials: int, N: int, f0=None) -> np.ndarray:
    if f0 is None:
        f0 = rng.uniform(-0.5, 0.5, n_trials)
    f0 = np.broadcast_to(np.asarray(f0, float), (n_trials,))
    ph = rng.uniform(0, 2 * np.pi, n_trials)
    return np.exp(1j * (2 * np.pi * f0[:, None] * np.arange(N)[None, :] + ph[:, None]))


def cnoise(rng, shape) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
