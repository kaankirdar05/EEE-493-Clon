"""Box refinement after segmentation (4.3): sharper frequency and time edges.

wavelet_freq_edges  Tian & Giannakis (2006) wavelet edge detection: the box's
                    log-PSD is convolved with derivative-of-Gaussian wavelets at
                    several scales; the multiscale product peaks at band edges
                    (positive = rising edge, negative = falling edge). The
                    strongest rising edge left of the box centre and the strongest
                    falling edge right of it become the new frequency edges.
cusum_time_edges    single change-point detection (CUSUM of the log power over
                    the box's bins): the start edge is searched around the box's
                    first frame, the stop edge around its last frame.

Both take a box (t0, t1, f0, f1) in grid cells (half-open) and return a refined
box. They only move edges within a margin around the original box, so they
correct segmentation errors but cannot create or delete detections.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d


def _dog(scale: float) -> np.ndarray:
    """Derivative-of-Gaussian kernel for correlation: rising input -> positive output."""
    t = np.arange(-4 * scale, 4 * scale + 1)
    g = t * np.exp(-t**2 / (2 * scale**2))
    return g / np.sum(np.abs(g))


def wavelet_freq_edges(spec, box, scales=(1, 2, 4), margin_frac=0.5, min_margin=6):
    t0, t1, f0, f1 = box
    T, F = spec.shape
    w = f1 - f0
    m = max(min_margin, int(round(margin_frac * w)))
    lo, hi = max(0, f0 - m), min(F, f1 + m)
    if hi - lo < 8:
        return box
    psd = 10 * np.log10(spec[t0:t1, lo:hi].mean(0) + 1e-12)
    prod = np.ones(hi - lo)
    for s in scales:
        # derivative estimate at scale s; edge-replicate padding avoids false edges at the region ends
        prod *= correlate1d(psd, _dog(s), mode="nearest")
    centre = (f0 + f1) // 2 - lo
    left, right = prod[:max(centre, 1)], prod[centre:]
    if left.max() <= 0 or right.min() >= 0:
        return box  # no clear edge pair: keep the segmentation's box
    nf0 = lo + int(np.argmax(left))
    nf1 = lo + centre + int(np.argmin(right)) + 1
    return (t0, t1, nf0, nf1) if nf1 - nf0 >= 2 else box


def _change_point(v: np.ndarray) -> int:
    """Index k maximising |S_k - k/n S_n| (CUSUM for a single mean change)."""
    n = len(v)
    s = np.cumsum(v - v.mean())
    return int(np.argmax(np.abs(s[:-1]))) + 1 if n > 2 else n // 2


def cusum_time_edges(spec, box, margin_frac=0.25, min_margin=8):
    t0, t1, f0, f1 = box
    T = spec.shape[0]
    h = max(min_margin, int(round(margin_frac * (t1 - t0))))
    p = np.log(spec[:, f0:f1].mean(1) + 1e-12)
    a0, a1 = max(0, t0 - h), min(T, t0 + h)          # start edge window
    b0, b1 = max(0, t1 - h), min(T, t1 + h)          # stop edge window
    nt0 = a0 + _change_point(p[a0:a1]) if a1 - a0 > 4 and t0 > 0 else t0
    nt1 = b0 + _change_point(p[b0:b1]) if b1 - b0 > 4 and t1 < T else t1
    return (nt0, nt1, f0, f1) if nt1 - nt0 >= 2 else box
