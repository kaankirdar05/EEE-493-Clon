"""Noiseless known-rate test signals at ARBITRARY (non-integer samples/symbol) symbol rates, evaluated from the continuous-time definitions."""
import sys
sys.dont_write_bytecode = True
import numpy as np


def rrc_t(t, beta):
    """Root-raised-cosine pulse at times t (in symbol periods), unit-symbol-period normalisation (peak ~ 1 - beta + 4 beta/pi)."""
    t = np.asarray(t, float)
    out = np.empty_like(t)
    z = np.abs(t) < 1e-9
    s = np.abs(np.abs(4 * beta * t) - 1) < 1e-9
    g = ~(z | s)
    out[z] = 1 - beta + 4 * beta / np.pi
    out[s] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    tg = t[g]
    out[g] = (np.sin(np.pi * tg * (1 - beta)) + 4 * beta * tg * np.cos(np.pi * tg * (1 + beta))) / (np.pi * tg * (1 - (4 * beta * tg) ** 2))
    return out


def gen_qpsk(rs, n, rng, beta=0.35, span=10):
    nsym = int(np.ceil(n * rs)) + 2 * span + 4
    s = np.exp(2j * np.pi * (rng.integers(0, 4, nsym) + 0.5) / 4)
    t0 = span + 2 + rng.uniform(0, 1)                    # random symbol-timing phase
    tt = t0 + np.arange(n) * rs                          # time in symbol periods
    k0 = np.floor(tt).astype(int)
    x = np.zeros(n, complex)
    for j in range(-span, span + 1):
        k = k0 + j
        x += s[k] * rrc_t(tt - k, beta)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def gen_msk(rs, n, rng, span=4):
    nsym = int(np.ceil(n * rs)) + 2 * span + 4
    b = rng.choice([-1.0, 1.0], nsym)
    t0 = span + 2 + rng.uniform(0, 1)
    tt = t0 + np.arange(n) * rs
    k0 = np.floor(tt).astype(int)
    cum = np.concatenate([[0.0], np.cumsum(b)])
    phase = (np.pi / 2) * (cum[k0] + b[k0] * (tt - k0))      # frequency +-Rs/4 : phase advance +-pi/2 per symbol
    return np.exp(1j * phase)
