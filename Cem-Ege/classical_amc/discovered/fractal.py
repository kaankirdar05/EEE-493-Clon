"""Fractal-dimension features (vectorised over frames).

    Higuchi (1988)   L(k) = < sum_i |x[m+ik] - x[m+(i-1)k]| (N-1) / (floor((N-m)/k) k) / k >_m ,
                     FD = slope of log L(k) against log(1/k), k = 1..KMAX.
                     Known values: straight line 1, Brownian motion 1.5, white noise 2,
                     fractional Brownian motion 2 - H.
    Katz (1988)      FD = log10(n) / (log10(n) + log10(d / L)), n = N-1 steps, L = sum|dx|,
                     d = max |x_i - x_0| (scale-free variant used by e.g. antropy). Straight line: 1.
    Petrosian (1995) FD = log10(N) / (log10(N) + log10(N / (N + 0.4 N_delta))), N_delta = number of
                     sign changes of the first difference. Straight line: 1.
    Box counting     graph of the series scaled to the unit square, columns of width eps = 2^-j,
                     boxes per column = floor(max/eps) - floor(min/eps) + 1 (column extended by the first
                     sample of the next column so the graph stays connected); FD = slope of
                     log N(eps) against log(1/eps), j = 4..9. Line: 1; graph of white noise -> 2 only as
                     resolution grows (finite N gives < 2).

Signals (unit-power frame x): amplitude |x|, I and Q (FD averaged over the two), unwrapped and
linearly detrended phase, and wrapped instantaneous frequency. 4 signals x 4 dimensions = 16.
Multi-scale variant (after the multiscale-entropy coarse graining of Costa et al. 2002): Higuchi FD
of the amplitude and of I/Q after averaging non-overlapping blocks of 2, 4 and 8 samples: 6 more.
Total 22 features.
"""

from __future__ import annotations

import numpy as np

import dcommon as dc

KMAX = 8
BOX_J = np.arange(4, 10)   # eps = 1/16 .. 1/512 (chosen on synthetic known-answer signals, see sanity_features.py)
MS_SCALES = (2, 4, 8)
SIGNALS = ["amp", "iq", "phase", "freq"]
DIMS = ["higuchi", "katz", "petrosian", "box"]
FD_NAMES = [f"fd_{s}_{d}" for s in SIGNALS for d in DIMS] + \
           [f"fd_ms{a}_{s}_higuchi" for a in MS_SCALES for s in ("amp", "iq")]


def _slope(xv, Y):
    """Least-squares slope of each row of Y against xv."""
    xc = xv - xv.mean()
    return (Y - Y.mean(1, keepdims=True)) @ xc / np.sum(xc ** 2)


def higuchi(x: np.ndarray, kmax: int = KMAX) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, float))
    N = x.shape[1]
    L = np.empty((x.shape[0], kmax))
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            sub = x[:, m::k]
            nm = sub.shape[1] - 1                                   # = floor((N - 1 - m) / k)
            if nm < 1:
                continue
            Lm.append(np.abs(np.diff(sub, axis=1)).sum(1) * (N - 1) / (nm * k) / k)
        L[:, k - 1] = np.mean(Lm, axis=0)
    return _slope(np.log(1.0 / np.arange(1, kmax + 1)), np.log(np.maximum(L, 1e-300)))


def katz(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, float))
    Ltot = np.abs(np.diff(x, axis=1)).sum(1)
    d = np.max(np.abs(x - x[:, :1]), axis=1)
    n = x.shape[1] - 1
    return np.log10(n) / (np.log10(n) + np.log10(np.maximum(d, 1e-300) / np.maximum(Ltot, 1e-300)))


def petrosian(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, float))
    N = x.shape[1]
    dx = np.diff(x, axis=1)
    nd = np.sum(dx[:, 1:] * dx[:, :-1] < 0, axis=1)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nd)))


def boxcount(x: np.ndarray, js=BOX_J) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, float))
    n, N = x.shape
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    y = (x - lo) / np.maximum(hi - lo, 1e-300)
    y = np.minimum(y, 1 - 1e-12)
    logs = []
    for j in js:
        c = 2 ** j
        w = N // c
        cols = y[:, :c * w].reshape(n, c, w)
        nxt = np.concatenate([y[:, w:c * w:w], y[:, -1:]], axis=1)[:, :c, None]   # first sample of next column
        cols = np.concatenate([cols, nxt], axis=2)
        eps = 1.0 / c
        boxes = np.floor(cols.max(2) / eps) - np.floor(cols.min(2) / eps) + 1
        logs.append(np.log(boxes.sum(1)))
    return _slope(np.log(2.0 ** np.asarray(js, float)), np.stack(logs, axis=1))


def all_dims(s):
    return [higuchi(s), katz(s), petrosian(s), boxcount(s)]


def coarse(s, a):
    n, N = s.shape
    return s[:, :N // a * a].reshape(n, N // a, a).mean(2)


def fractal_features(x: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    for s0 in range(0, len(x), batch):
        xb = dc.normalize(x[s0:s0 + batch])
        amp = np.abs(xb)
        ph = np.unwrap(np.angle(xb), axis=1)
        t = np.arange(ph.shape[1])
        ph = ph - (np.outer(_slope(t.astype(float), ph), t - t.mean()) + ph.mean(1, keepdims=True))
        fr = dc.inst_freq(xb)
        cols = all_dims(amp)
        cols += [(a + b) / 2 for a, b in zip(all_dims(xb.real), all_dims(xb.imag))]
        cols += all_dims(ph) + all_dims(fr)
        for a in MS_SCALES:
            cols += [higuchi(coarse(amp, a)),
                     (higuchi(coarse(xb.real, a)) + higuchi(coarse(xb.imag, a))) / 2]
        out.append(np.stack(cols, axis=1))
    return dc.clean(np.concatenate(out))
