"""Haar wavelet transform (HWT) features, after Ho, Prakasam & Chan (2000),
"Modulation identification of digital signals by the wavelet transform".

Continuous Haar transform at (even) scale a, discrete time:

    HWT_a[t] = ( sum_{k=t}^{t+a/2-1} x[k]  -  sum_{k=t+a/2}^{t+a-1} x[k] ) / sqrt(a)

Theory used by Ho et al. (and checked in sanity_checks.py): for a tone e^{j w n}
(the baseband equivalent of a PSK symbol with residual carrier w, or one FSK tone),

    |HWT_a| = (2 / sqrt(a)) |sin(w a / 4)| * |sin(w a / 4) / sin(w / 2)|

i.e. constant inside a symbol. Hence |HWT| of PSK is flat apart from spikes at
symbol transitions (phase jumps); a median filter longer than the spikes removes
them, leaving ~zero variance. QAM adds amplitude steps (non-zero variance of the
median-filtered |HWT| of |x|, and of |HWT(x)| when w != 0); FSK gives a staircase
(different w per symbol) in |HWT| of the amplitude-normalised signal x / |x|.

Features per scale a in SCALES (frames normalised to zero mean, unit power):
    m_a        mean |HWT_a(x)|
    vraw_a     var |HWT_a(x)| / m_a^2
    vmed_a     var(median_filter(|HWT_a(x)|, 2a+1)) / m_a^2       (Ho et al.'s statistic; a transition
               spike is a-1 samples wide, so the window must exceed 2(a-1) to remove it)
    peaks_a    rate of local maxima of |HWT_a(x)| above mean + 2 std (transitions / sample)
    vmedu_a    as vmed_a but for u = x / |x|                        (FSK vs PSK)
    amp_a      mean |HWT_a(|x|)|                                    (amplitude steps: QAM/ASK vs PSK)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

SCALES = (2, 4, 8, 16)
WAV_NAMES = [f"{f}_{a}" for a in SCALES for f in ("hwt_m", "hwt_vraw", "hwt_vmed", "hwt_peaks", "hwt_vmedu", "hwt_amp")]


def haar(x: np.ndarray, a: int) -> np.ndarray:
    """(N, L) -> (N, L - a + 1) Haar CWT at scale a (even)."""
    S = np.concatenate([np.zeros((x.shape[0], 1), x.dtype), np.cumsum(x, axis=1)], axis=1)
    h = a // 2
    first = S[:, h:-h] - S[:, :-2 * h]
    second = S[:, 2 * h:] - S[:, h:-h]
    return (first - second) / np.sqrt(a)


def tone_theory(w: float, a: int) -> float:
    return 2 / np.sqrt(a) * abs(np.sin(w * a / 4)) * abs(np.sin(w * a / 4) / np.sin(w / 2))


def _medvar(h, a):
    mf = median_filter(h, size=(1, 2 * a + 1), mode="nearest")
    return mf.var(axis=1)


def _peak_rate(h):
    thr = h.mean(1, keepdims=True) + 2 * h.std(1, keepdims=True)
    c = h[:, 1:-1]
    pk = (c > h[:, :-2]) & (c >= h[:, 2:]) & (c > thr)
    return pk.sum(1) / h.shape[1]


def wavelet_features(x: np.ndarray, batch: int = 1024) -> np.ndarray:
    out = []
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].astype(np.complex128)
        xb = xb - xb.mean(1, keepdims=True)
        xb = xb / np.sqrt(np.maximum(np.mean(np.abs(xb) ** 2, 1, keepdims=True), 1e-20))
        amp = np.abs(xb)
        u = xb / np.maximum(amp, 1e-9)
        cols = []
        for a in SCALES:
            h = np.abs(haar(xb, a))
            m = np.maximum(h.mean(1), 1e-12)
            hu = np.abs(haar(u, a))
            mu = np.maximum(hu.mean(1), 1e-12)
            cols += [m, h.var(1) / m**2, _medvar(h, a) / m**2, _peak_rate(h),
                     _medvar(hu, a) / mu**2, np.abs(haar(amp, a)).mean(1)]
        out.append(np.stack(cols, axis=1))
    F = np.concatenate(out).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
