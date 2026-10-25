"""Cramer-von Mises and Anderson-Darling goodness-of-fit classifiers in the framework of the KS
classifier of extended/symbol_classifiers.py (5 digital classes, symbol-spaced unit-power z).

For every candidate class k the frame is processed exactly as for KS (class-conditional M2M4 SNR
estimate -> nearest 1-dB reference set, class-conditional M-th power phase estimate and de-rotation),
and the empirical CDFs of I, Q, |z| and arg z are compared with the class's REFERENCE samples
(200 simulated frames x 128 symbols, same phase estimator; sc.reference) by a two-sample statistic:

    KS   D   = max |F_n - G_m|                                          (as in symbol_classifiers.ks2)
    CvM  T   = n m / (n+m)^2  sum_{z in pooled sample} (F_n(z) - G_m(z))^2           (Anderson 1962;
               identical to scipy.stats.cramervonmises_2samp's statistic)
    AD   A^2 = 1/(n m) sum_{j=1}^{N-1} (N M_j - j n)^2 / (j (N - j))   (Pettitt 1976; = the k = 2 case of
               Scholz & Stephens' A^2_kN, scipy.stats.anderson_ksamp before its standardisation)
               M_j = number of frame samples among the j smallest pooled values, N = n + m.

The decision is argmin_k of the statistic summed over a component set chosen on training frames
(the KS variants of symbol_classifiers). CvM weights the whole CDF difference instead of its maximum;
AD additionally weights the tails by 1/(F(1-F)) - the standard argument why both are more powerful
than KS (Stephens 1974); see also the distribution-test classifiers in Zhu & Nandi (2015),
"Automatic Modulation Classification: Principles, Algorithms and Applications".

Under H0 with iid samples: T -> mean 1/6, 5 % point 0.461; A^2 -> mean 1, 5 % point 2.492.
"""

from __future__ import annotations

import dcommon  # noqa: F401
import numpy as np

import symbol_classifiers as sc

TESTS = ["ks", "cvm", "ad"]
COMPONENTS = ["I", "Q", "amp", "phase"]


def cvm2(a_sorted, b_sorted):
    n, m = len(a_sorted), len(b_sorted)
    allv = np.concatenate([a_sorted, b_sorted])
    fa = np.searchsorted(a_sorted, allv, side="right") / n
    fb = np.searchsorted(b_sorted, allv, side="right") / m
    return float(n * m / (n + m) ** 2 * np.sum((fa - fb) ** 2))


def ad2(a_sorted, b_sorted):
    """Two-sample Anderson-Darling A^2 (continuous data, no ties)."""
    n, m = len(a_sorted), len(b_sorted)
    N = n + m
    pos = np.searchsorted(b_sorted, a_sorted, side="right") + np.arange(n)     # 0-based pooled positions of a
    ind = np.zeros(N)
    ind[pos] = 1.0
    M = np.cumsum(ind)[:-1]                                                     # M_j, j = 1..N-1
    j = np.arange(1, N, dtype=float)
    return float(np.sum((N * M - j * n) ** 2 / (j * (N - j))) / (n * m))


FUNCS = {"ks": sc.ks2, "cvm": cvm2, "ad": ad2}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cut_offset(cls, grid=3600):
    """Offset c such that the phase branch cut (at angle c + pi) lies in the widest angular gap of the
    class's constellation (after the same de-rotation), so no cluster is split between +pi and -pi."""
    ang = np.angle(sc.CONS[cls])
    cand = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    gap = np.min(np.abs(_wrap(cand[:, None] - ang[None, :])), axis=1)
    return float(_wrap(cand[np.argmax(gap)] - np.pi))


CUT = {c: cut_offset(c) for c in sc.CLASSES}
_REF_CUT = {}


def reference_cut(cls, snr):
    key = (cls, snr)
    if key not in _REF_CUT:
        r = sc.reference(cls, snr)
        _REF_CUT[key] = (r[0], r[1], r[2], np.sort(_wrap(r[3] - CUT[cls])))
    return _REF_CUT[key]


def gof_distances(z, tests=TESTS, cut=False):
    """(len(tests), 5 classes, 4 components) statistics of symbol frame z.
    cut=False: phase = arg z in (-pi, pi] exactly as in the KS classifier of symbol_classifiers.py.
    cut=True : phase measured from a class-specific branch cut placed in the constellation's widest
               angular gap (post-hoc diagnostic variant, see README)."""
    D = np.empty((len(tests), len(sc.CLASSES), 4))
    for k, c in enumerate(sc.CLASSES):
        snr = int(sc.SNR_GRID[np.argmin(np.abs(sc.SNR_GRID - sc.m2m4_snr_db(z, c)))])
        ref = reference_cut(c, snr) if cut else sc.reference(c, snr)
        zr = sc.derotate(z, c)
        ph = _wrap(np.angle(zr) - CUT[c]) if cut else np.angle(zr)
        comps = [np.sort(v) for v in (zr.real, zr.imag, np.abs(zr), ph)]
        for t, name in enumerate(tests):
            D[t, k] = [FUNCS[name](v, r) for v, r in zip(comps, ref)]
    return D


def decide(D, test, variant):
    return int(np.argmin(D[TESTS.index(test)][:, sc.KS_VARIANTS[variant]].sum(1)))
