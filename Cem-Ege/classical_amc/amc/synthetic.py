"""Tiny synthetic signal generator, used ONLY to smoke-test the pipeline without
the 20 GB RadioML file. It is not dataset v1 (job 2.2): impairments are minimal
(random phase, small frequency offset, AWGN) and the SNR definition is simply
signal power / noise power per complex sample.
"""

from __future__ import annotations

import numpy as np

SPS = 8  # samples per symbol


def _rrc(beta: float = 0.35, span: int = 8, sps: int = SPS) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


_RRC = _rrc()


def _constellation(name: str) -> np.ndarray | None:
    if name == "BPSK":
        return np.array([1, -1], dtype=complex)
    if name.endswith("PSK") and name[:-3].isdigit():
        m = int(name[:-3])
        return np.exp(2j * np.pi * np.arange(m) / m)
    if name == "QPSK":
        return np.exp(2j * np.pi * (np.arange(4) + 0.5) / 4)
    if name.endswith("QAM") and name[:-3].isdigit():
        m = int(name[:-3])
        k = int(round(np.sqrt(m)))
        if k * k != m:
            return None
        lv = np.arange(-(k - 1), k, 2)
        return (lv[:, None] + 1j * lv[None, :]).ravel()
    if name == "OOK":
        return np.array([0, 1], dtype=complex)
    if name.endswith("ASK") and name[:-3].isdigit():
        return np.arange(int(name[:-3])).astype(complex)
    return None


def _linear(points: np.ndarray, n: int, length: int, rng) -> np.ndarray:
    nsym = length // SPS + 2 * len(_RRC) // SPS + 2
    sym = rng.choice(points, size=(n, nsym))
    up = np.zeros((n, nsym * SPS), dtype=complex)
    up[:, ::SPS] = sym
    y = np.array([np.convolve(r, _RRC, mode="same") for r in up])
    start = len(_RRC)
    return y[:, start:start + length]


def _smooth_message(n: int, length: int, rng, cutoff: float = 0.02) -> np.ndarray:
    """Low-pass random message in [-1, 1] (speech-like stand-in)."""
    taps = np.sinc(2 * cutoff * np.arange(-64, 65)) * np.hamming(129)
    m = np.array([np.convolve(r, taps, mode="same") for r in rng.standard_normal((n, length))])
    return m / np.max(np.abs(m), axis=1, keepdims=True)


def _gmsk(n: int, length: int, rng, bt: float = 0.3) -> np.ndarray:
    nsym = length // SPS + 4
    bits = rng.choice([-1.0, 1.0], size=(n, nsym))
    nrz = np.repeat(bits, SPS, axis=1)
    t = np.arange(-2 * SPS, 2 * SPS + 1) / SPS
    g = np.exp(-2 * np.pi**2 * bt**2 * t**2 / np.log(2))
    g /= g.sum()
    freq = np.array([np.convolve(r, g, mode="same") for r in nrz])
    phase = np.cumsum(freq, axis=1) * (np.pi / 2) / SPS  # modulation index 0.5
    return np.exp(1j * phase[:, :length])


def generate(name: str, n: int, snr_db: float, length: int, rng) -> np.ndarray:
    """(n, length) complex64 frames of modulation `name` at `snr_db`.
    Unsupported names return pure noise (they only fill the fake file)."""
    pts = _constellation(name)
    if pts is not None:
        s = _linear(pts, n, length, rng)
    elif name == "FM":
        s = np.exp(1j * 2 * np.pi * 0.05 * np.cumsum(_smooth_message(n, length, rng), axis=1))
    elif name in ("AM-DSB-WC", "AM-DSB-SC"):
        m = _smooth_message(n, length, rng)
        s = (1 + 0.7 * m) if name == "AM-DSB-WC" else m
        s = s.astype(complex)
    elif name == "GMSK":
        s = _gmsk(n, length, rng)
    else:
        s = np.zeros((n, length), dtype=complex)

    p = np.mean(np.abs(s) ** 2, axis=1, keepdims=True)
    s = s / np.sqrt(np.where(p > 0, p, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    cfo = rng.uniform(-0.002, 0.002, size=(n, 1))  # cycles / sample
    s = s * np.exp(1j * (phase + 2 * np.pi * cfo * np.arange(length)))
    noise_p = 10 ** (-snr_db / 10) if pts is not None or np.any(p > 0) else 1.0
    w = np.sqrt(noise_p / 2) * (rng.standard_normal((n, length)) + 1j * rng.standard_normal((n, length)))
    return (s + w).astype(np.complex64)
