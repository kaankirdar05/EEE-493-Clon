"""Eigenvalue-family detectors per sub-band block (smoothing factor L) and the
random-matrix (Tracy-Widom) thresholds used for the analytic comparison.

Sample covariance: the L x L "smoothed" covariance of one complex sub-band block
x[0..N-1] (Zeng & Liang 2009), C[i, j] = 1/n sum_{k=0}^{n-1} x[k+i] x*[k+j],
n = N - L + 1 (the same matrix as wp4.detectors.EigenMME builds from a Hankel
matrix, here computed from cumulative lag products so memory is O(L*N)).

Statistics (all scale-invariant except MED):
    agm    arithmetic / geometric mean of the eigenvalues. The GLRT of
           H0: C = s^2 I (s unknown) against an unstructured C (Mauchly's
           sphericity LRT; Zhang, Mallik & Letaief 2010 for spectrum sensing).
    med    lambda_max / s_hat^2, s_hat^2 = noise power estimated from the scene
           (max-eigenvalue detection, Zeng & Liang 2009, with estimated noise).
    john   John's sphericity statistic U = (1/L) tr[(C / (tr C / L) - I)^2]
           = L sum(l^2) / (sum l)^2 - 1 (John 1971; Ledoit & Wolf 2002).
    emr    eigenvalue moment ratio (Huang, Fang, Liu, So & Li, IEEE TVT 2015):
           T = (1/L) sum(l^2) / ((1/L) sum l)^2 = U + 1. Affine in John's U, so
           with an empirically calibrated threshold the two make IDENTICAL
           decisions; they differ only in how an analytic threshold is derived.
    cond   lambda_max / lambda_min (MME, Zeng & Liang 2009), with
           (i) empirical threshold, (ii) Tracy-Widom thresholds:
           'zl'  Zeng-Liang form: TW2 for lambda_max, lambda_min at its MP edge;
           'twc' Penna, Garello & Spirito (IEEE Commun. Lett. 2009): ratio of two
                 asymptotically independent TW2 variables, Curtiss (1941) ratio
                 integral.
Complex data -> beta = 2 Tracy-Widom (TW2) and complex-Wishart centring/scaling
(Johnstone 2001; El Karoui 2006; Feldheim & Sodin 2010 for the smallest one).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import airy, ndtri

from . import WP4  # noqa: F401
from wp4.detectors import N_IQ, BlockDetector, Detector, SB_LEN, subbands  # noqa: E402


# ------------------------------------------------------------------ covariance and eigenvalues

def smoothed_cov(x: np.ndarray, L: int) -> np.ndarray:
    """(..., N) complex -> (..., L, L) Hermitian sample covariance with smoothing factor L."""
    N = x.shape[-1]
    n = N - L + 1
    C = np.empty(x.shape[:-1] + (L, L), dtype=np.complex128)
    for d in range(L):
        p = x[..., d:] * np.conj(x[..., : N - d])                     # x[k+d] x*[k], k = 0..N-d-1
        cs = np.concatenate([np.zeros(x.shape[:-1] + (1,), np.complex128), np.cumsum(p, axis=-1)], axis=-1)
        j = np.arange(L - d)
        v = (cs[..., j + n] - cs[..., j]) / n                          # C[j+d, j]
        C[..., j + d, j] = v
        C[..., j, j + d] = np.conj(v)
    return C


def eigs(x: np.ndarray, L: int, chunk: int = 4096) -> np.ndarray:
    """Ascending eigenvalues (..., L) of the smoothed covariance, chunked over blocks."""
    shp = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    out = np.empty((len(xf), L))
    for s in range(0, len(xf), chunk):
        out[s: s + chunk] = np.linalg.eigvalsh(smoothed_cov(xf[s: s + chunk], L))
    return np.maximum(out, 1e-300).reshape(*shp, L)


def stat_from_eigs(ev: np.ndarray, kind: str) -> np.ndarray:
    L = ev.shape[-1]
    if kind == "agm":
        return np.exp(np.log(ev.mean(-1)) - np.log(ev).mean(-1))
    if kind == "john":
        return L * np.sum(ev ** 2, -1) / np.sum(ev, -1) ** 2 - 1
    if kind == "emr":
        return L * np.sum(ev ** 2, -1) / np.sum(ev, -1) ** 2
    if kind == "cond":
        return ev[..., -1] / ev[..., 0]
    raise ValueError(kind)


# ------------------------------------------------------------------ scene noise-power estimate

def separable_noise(P: np.ndarray, q: float = 0.25, iterations: int = 3) -> np.ndarray:
    """Noise level gain(t) x shape(sub-band) from a (T, B) map of per-block levels.

    shape = q-quantile over time per sub-band, gain = q-quantile over sub-bands of
    P / shape per time block (handles receiver gain steps), alternated. A constant
    bias factor of the quantile is absorbed by the empirical threshold calibration
    (done on whole white-noise scenes through this same estimator)."""
    shape = np.quantile(P, q, axis=0)
    gain = np.ones(P.shape[0])
    for _ in range(iterations):
        gain = np.quantile(P / shape[None, :], q, axis=1)
        gain = gain / np.median(gain)
        shape = np.quantile(P / gain[:, None], q, axis=0)
    return np.maximum(gain[:, None] * shape[None, :], 1e-30)


# ------------------------------------------------------------------ detectors

class EigenBlock(BlockDetector):
    """Blind (scale-invariant) eigenvalue statistic per sub-band block."""

    def __init__(self, kind: str = "agm", L: int = 8):
        self.kind, self.L = kind, L
        self.name = f"{kind}_L{L}"

    def block_stat(self, x):
        return stat_from_eigs(eigs(x, self.L), self.kind)


class MED(BlockDetector):
    """lambda_max / s_hat^2 with s_hat^2 = separable scene noise estimate of the
    block power tr(C)/L. Needs the whole (time blocks, sub-bands) layout, so the
    threshold is calibrated on whole white-noise scenes (400 scenes = 204800 blocks)."""

    def __init__(self, L: int = 8, n_cal_scenes: int = 400):
        self.L, self.n_cal_scenes = L, n_cal_scenes
        self.name = f"med_L{L}"

    def block_stat(self, x):
        ev = eigs(x, self.L)
        P = ev.mean(-1)                                  # = tr(C)/L = block power
        return ev[..., -1] / separable_noise(P)

    def h0_samples(self, rng, n_scenes=None):
        return scene_h0(self, rng, n_scenes or self.n_cal_scenes)


def scene_h0(det, rng, n_scenes):
    out = []
    for _ in range(n_scenes):
        iq = ((rng.standard_normal(N_IQ) + 1j * rng.standard_normal(N_IQ)) / np.sqrt(2)).astype(np.complex64)
        out.append(det.block_stat(subbands(iq)).ravel())
    return np.concatenate(out)


class CondTW(EigenBlock):
    """Condition-number detector with an ANALYTIC Tracy-Widom threshold (no
    calibration on noise). method 'zl' (Zeng-Liang form) or 'twc' (TW-Curtiss)."""

    def __init__(self, L: int = 8, method: str = "twc"):
        super().__init__("cond", L)
        self.method = method
        self.name = f"cond_{method}_L{L}"

    def calibrate(self, pfas, cache_dir=None, seed=None):
        n = SB_LEN - self.L + 1
        self.thr = {float(p): cond_threshold(self.L, n, p, self.method) for p in pfas}


# ------------------------------------------------------------------ Tracy-Widom (beta = 2)

def _tw2_cdf_point(s: float, m: int = 80) -> float:
    """F2(s) = det(I - K_Airy) on L^2(s, inf) by Gauss-Legendre (Bornemann 2010)."""
    hi = max(s, 0.0) + 14.0
    t, w = np.polynomial.legendre.leggauss(m)
    x = s + (t + 1) * (hi - s) / 2
    w = w * (hi - s) / 2
    ai, aip, _, _ = airy(x)
    X, Y = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (ai[:, None] * aip[None, :] - aip[:, None] * ai[None, :]) / (X - Y)
    K[np.diag_indices(m)] = aip ** 2 - x * ai ** 2
    sw = np.sqrt(w)
    return float(np.linalg.det(np.eye(m) - sw[:, None] * K * sw[None, :]))


_GRID = None


def tw2_table():
    """(s, F2(s), f2(s)) tabulated on [-9, 7], step 0.005 (cached on disk)."""
    global _GRID
    if _GRID is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tw2_table.npz")
        if os.path.exists(path):
            d = np.load(path)
            _GRID = (d["s"], d["F"], d["f"])
        else:
            s = np.arange(-9.0, 7.0 + 1e-9, 0.005)
            F = np.clip(np.array([_tw2_cdf_point(v) for v in s]), 0, 1)
            f = np.gradient(F, s)
            np.savez(path, s=s, F=F, f=f)
            _GRID = (s, F, f)
    return _GRID


def tw2_cdf(s):
    g, F, _ = tw2_table()
    return np.interp(s, g, F, left=0.0, right=1.0)


def tw2_pdf(s):
    g, _, f = tw2_table()
    return np.interp(s, g, f, left=0.0, right=0.0)


def tw2_ppf(p: float) -> float:
    g, F, _ = tw2_table()
    return float(brentq(lambda v: tw2_cdf(v) - p, g[0], g[-1]))


def mp_edges(L: int, n: int):
    """Complex-Wishart edge centring/scaling for n*lambda of C = W/n, W ~ CW_L(n, I).
    (n l_max - mu_p)/s_p -> TW2 ; (mu_m - n l_min)/s_m -> TW2 (L < n)."""
    a, b = np.sqrt(n), np.sqrt(L)
    mu_p, s_p = (a + b) ** 2, (a + b) * (1 / a + 1 / b) ** (1 / 3)
    mu_m, s_m = (a - b) ** 2, (a - b) * (1 / b - 1 / a) ** (1 / 3)
    return mu_p, s_p, mu_m, s_m


def cond_tail(gamma: float, L: int, n: int) -> float:
    """P(l_max / l_min > gamma) with independent TW2 edges (Curtiss ratio integral)."""
    mu_p, s_p, mu_m, s_m = mp_edges(L, n)
    g, _, f = tw2_table()
    y = g                                   # Y = (mu_m - n l_min)/s_m ~ TW2
    den = mu_m - s_m * y
    ok = den > 0
    xthr = (gamma * den - mu_p) / s_p
    tail = np.where(ok, 1 - tw2_cdf(xthr), 1.0)
    return float(np.trapezoid(f * tail, y))


def cond_threshold(L: int, n: int, pfa: float, method: str = "twc") -> float:
    mu_p, s_p, mu_m, s_m = mp_edges(L, n)
    if method == "zl":
        return float((mu_p + s_p * tw2_ppf(1 - pfa)) / mu_m)
    if method == "twc":
        lo, hi = 1.0 + 1e-9, 1e3 * mu_p / mu_m
        return float(brentq(lambda gm: cond_tail(gm, L, n) - pfa, lo, hi))
    raise ValueError(method)


def med_tw_threshold(L: int, n: int, pfa: float) -> float:
    """Threshold on l_max / s^2 (known s^2): (mu_p + s_p F2^-1(1-pfa)) / n."""
    mu_p, s_p, _, _ = mp_edges(L, n)
    return float((mu_p + s_p * tw2_ppf(1 - pfa)) / n)


def john_asym_threshold(L: int, n: int, pfa: float) -> float:
    """Complex Gaussian, iid columns: n U - L -> N(-1/L, 2) (complex analogue of
    Ledoit & Wolf 2002, whose real-valued limit is N(1, 4); the linear-spectral-
    statistic variance halves for beta = 2). The mean -1/L follows from
    E tr W^2 = L n (L + n) for complex Wishart. A first version used N(0, 1): the
    iid-Wishart known-answer check gave Pfa 0.016-0.019 at target 1e-3 and the
    measured moments (var 1.88-1.98, mean -0.25/-0.13/-0.08 for L = 4/8/16)
    exposed the error."""
    return float((L - 1 / L + np.sqrt(2) * ndtri(1 - pfa)) / n)
