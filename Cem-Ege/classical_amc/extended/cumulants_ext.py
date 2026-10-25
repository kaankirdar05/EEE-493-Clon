"""Extended higher-order cumulants and cyclic moments / cumulants.

Cumulants
---------
All cumulants are computed with the general moment-to-cumulant (Leonov-Shiryaev) formula

    cum(x_1..x_n) = sum over set partitions P of {1..n}:
                    (-1)^{|P|-1} (|P|-1)!  prod_{B in P} E[prod_{i in B} x_i]

with p - q copies of x and q copies of x* (C_pq). Every block moment only depends on
(#x, #x*) in the block, so the 4140 partitions of an 8-set collapse to a handful of
terms per C_pq. The same code computes (a) exact theoretical values from a
constellation's moments and (b) per-frame estimates from sample moments, so a bug in
the formula would show up as a mismatch with the hard-coded C40/C42/C60/C80 of
amc/features.py and with literature values (checked in sanity_checks.py).

Frames are normalised to zero mean and unit power first (C21 = 1), as in features.py,
so C_pq is already "normalised by C21^(p/2)". Three ratios that do not depend on
signal power or on (Gaussian) noise level are added: |C40|/|C42|, |C63|/|C42|^1.5,
|C84|/|C42|^2 (Gaussian noise has zero cumulants above order 2).

Cyclic features (CFO-robust; zero-padded FFT + fine local search of the peak)
-----------------------------------------------------------------------------
    cyc_M20      max_a |<x^2 e^{-j2pi a n}>|           2nd-order cyclic moment peak
    cyc_M40      max_a |<x^4 e^{-j2pi a n}>|           4th-order cyclic moment peak (alpha_4)
    cyc_C40      |M40(alpha_4) - 3 M20(alpha_4/2)^2|   4th-order cyclic CUMULANT at alpha_4
                 (= |C40| of the CFO-corrected signal; |C40| itself is destroyed by a CFO)
    cyc_M80      max_a |<x^8 e^{-j2pi a n}>|
    cyc_conj0    |<|x|^2 e^{-j2pi a_c n}>| / <|x|^2>   conjugate cyclic autocorrelation, lag 0,
                 at a_c = strongest non-zero conjugate cycle frequency in [1/32, 1/2]
                 (the symbol rate for linearly modulated signals)
    cyc_conjT2   same at lag tau = round(1 / (2 a_c)) (half a symbol)
    cyc_conj_a   a_c itself (cycles/sample)
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from math import factorial

import numpy as np

CUM_NAMES = ["abs_C61", "abs_C62", "C63", "abs_C81", "abs_C82", "abs_C83", "C84",
             "r_C40_C42", "r_C63_C42", "r_C84_C42"]
CYC_NAMES = ["cyc_M20", "cyc_M40", "cyc_C40", "cyc_M80", "cyc_conj0", "cyc_conjT2", "cyc_conj_a"]
EXT_NAMES = CUM_NAMES + CYC_NAMES


# ----------------------------------------------------------------- partitions / formula

def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        yield [[first]] + p                          # first alone
        for i in range(len(p)):                      # first joins an existing block
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


@lru_cache(maxsize=None)
def cumulant_terms(p: int, q: int):
    """C_pq as a list of (coef, ((a1,b1),(a2,b2),...)) where each (a,b) is the moment
    E[x^a x*^b] and the term is coef * prod of those moments."""
    kinds = [0] * (p - q) + [1] * q                  # 0 -> x, 1 -> conj(x)
    acc = defaultdict(int)
    for part in _partitions(list(range(p))):
        k = len(part)
        coef = (-1) ** (k - 1) * factorial(k - 1)
        blocks = tuple(sorted((sum(1 for i in B if kinds[i] == 0), sum(1 for i in B if kinds[i] == 1))
                              for B in part))
        acc[blocks] += coef
    return [(c, b) for b, c in acc.items() if c != 0]


def moment_table(x: np.ndarray, order: int = 8) -> dict:
    """(N, L) samples -> {(a, b): (N,) E[x^a conj(x)^b]} for 1 <= a + b <= order."""
    xp = [np.ones_like(x)]
    for _ in range(order):
        xp.append(xp[-1] * x)
    xc = [np.conj(v) for v in xp]
    return {(a, b): (xp[a] * xc[b]).mean(axis=-1)
            for a in range(order + 1) for b in range(order + 1 - a) if a + b >= 1}


def cumulant_from_moments(m: dict, p: int, q: int):
    out = 0
    for coef, blocks in cumulant_terms(p, q):
        t = coef
        for ab in blocks:
            t = t * m[ab]
        out = out + t
    return out


def theoretical_cumulants(points: np.ndarray, pq_list) -> dict:
    """Exact C_pq of an equiprobable constellation, normalised to zero mean, unit power."""
    c = np.asarray(points, complex)
    c = c - c.mean()
    c = c / np.sqrt(np.mean(np.abs(c) ** 2))
    m = moment_table(c[None, :])
    return {pq: complex(cumulant_from_moments(m, *pq)[0]) for pq in pq_list}


def _normalize(x):
    x = x.astype(np.complex128)
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.sqrt(np.maximum(np.mean(np.abs(x) ** 2, axis=1, keepdims=True), 1e-20))


def estimate_cumulants(x: np.ndarray, pq_list) -> dict:
    """Per-frame estimates of C_pq for (N, L) frames (normalised inside)."""
    m = moment_table(_normalize(x))
    return {pq: cumulant_from_moments(m, *pq) for pq in pq_list}


def cumulant_features(xn: np.ndarray) -> np.ndarray:
    m = moment_table(xn)
    C = {pq: cumulant_from_moments(m, *pq) for pq in
         [(4, 0), (4, 2), (6, 1), (6, 2), (6, 3), (8, 1), (8, 2), (8, 3), (8, 4)]}
    c42 = np.maximum(np.abs(C[(4, 2)]), 1e-3)
    return np.stack([np.abs(C[(6, 1)]), np.abs(C[(6, 2)]), C[(6, 3)].real,
                     np.abs(C[(8, 1)]), np.abs(C[(8, 2)]), np.abs(C[(8, 3)]), C[(8, 4)].real,
                     np.abs(C[(4, 0)]) / c42, np.abs(C[(6, 3)]) / c42 ** 1.5, np.abs(C[(8, 4)]) / c42 ** 2],
                    axis=1)


# ----------------------------------------------------------------- cyclic

def _peak_dtft(v: np.ndarray, nfft: int, lo: float = None, hi: float = None, n_fine: int = 17):
    """Per row of v: (alpha, |<v e^{-j2pi alpha n}>|) of the strongest DTFT peak.
    Coarse zero-padded FFT, then a fine search over +-1 coarse bin. Optional band
    [lo, hi] (cycles/sample, positive frequencies) for real-valued v."""
    N, L = v.shape
    P = np.abs(np.fft.fft(v, nfft, axis=1))
    f = np.fft.fftfreq(nfft)
    if lo is not None:
        mask = (f >= lo) & (f <= hi)
        P = np.where(mask[None, :], P, -1.0)
    a0 = f[np.argmax(P, axis=1)]
    d = np.linspace(-1.0 / nfft, 1.0 / nfft, n_fine)
    alphas = a0[:, None] + d[None, :]                                   # (N, F)
    n = np.arange(L)
    # <v e^{-j2pi(a0+d)n}> = (v e^{-j2pi a0 n}) @ e^{-j2pi d n}
    vals = np.abs((v * np.exp(-2j * np.pi * a0[:, None] * n[None, :])) @ np.exp(-2j * np.pi * np.outer(n, d))) / L
    k = np.argmax(vals, axis=1)
    return alphas[np.arange(N), k], vals[np.arange(N), k]


def _dtft_at(v: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    n = np.arange(v.shape[1])
    return (v * np.exp(-2j * np.pi * alpha[:, None] * n[None, :])).mean(axis=1)


def cyclic_features(xn: np.ndarray, nfft: int = 4096) -> np.ndarray:
    x2 = xn * xn
    x4 = x2 * x2
    x8 = x4 * x4
    _, m20 = _peak_dtft(x2, nfft)
    a4, m40 = _peak_dtft(x4, nfft)
    c40 = np.abs(_dtft_at(x4, a4) - 3 * _dtft_at(x2, a4 / 2) ** 2)
    _, m80 = _peak_dtft(x8, nfft)
    e = np.abs(xn) ** 2
    p0 = e.mean(axis=1)
    ac, r0 = _peak_dtft(e - p0[:, None], nfft, lo=1 / 32, hi=0.5)
    tau = np.clip(np.round(1 / (2 * ac)).astype(int), 1, 16)
    rT = np.empty(len(xn))
    for t in np.unique(tau):
        rows = tau == t
        lag = xn[rows, t:] * np.conj(xn[rows, :-t])
        rT[rows] = np.abs(_dtft_at(lag, ac[rows]))
    return np.stack([m20, m40, c40, m80, r0 / p0, rT / p0, ac], axis=1)


def extract_extended(x: np.ndarray, batch: int = 1024) -> np.ndarray:
    """(N, L) complex frames -> (N, len(EXT_NAMES)) float32."""
    out = []
    for s in range(0, len(x), batch):
        xn = _normalize(x[s:s + batch])
        out.append(np.concatenate([cumulant_features(xn), cyclic_features(xn)], axis=1))
    F = np.concatenate(out).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
