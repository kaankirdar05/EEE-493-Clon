"""Analytic (genie-aligned, phase-0) references for the ADS-B detectors and the PPM bit decision."""
import numpy as np
from scipy import integrate
from scipy.stats import ncx2, gamma, beta


def pb_theory(snr_db, N):
    """Bit error probability of the half-bit energy comparison with N independent samples per half bit:
    P(Y0 > Y1), Y0 ~ Gamma(N) (noise), Y1 = ncx2(2N, 2N*snr)/2 (pulse + noise)."""
    s = 10 ** (snr_db / 10)
    f = lambda y: 2 * ncx2.pdf(2 * y, 2 * N, 2 * N * s) * gamma.sf(y, N)
    hi = N * (1 + s) + 40 * np.sqrt(N * (1 + 2 * s)) + 40
    return integrate.quad(f, 0, hi, limit=400, epsabs=1e-14, epsrel=1e-10)[0]


def pd_energy(snr_db, Np, Nw, thr):
    s = 10 ** (snr_db / 10)
    f = lambda y: 2 * ncx2.pdf(2 * y, 2 * Np, 2 * Np * s) * gamma.sf(thr - y, Nw - Np) if y < thr else 2 * ncx2.pdf(2 * y, 2 * Np, 2 * Np * s)
    hi = Np * (1 + s) + 40 * np.sqrt(Np * (1 + 2 * s)) + 40
    return integrate.quad(f, 0, hi, limit=400, points=[thr], epsabs=1e-12)[0]


def pd_peak(snr_db, Np, Nw, thr):
    s = 10 ** (snr_db / 10)
    off = (1 - np.exp(-thr)) ** (Nw - Np)
    on = ncx2.cdf(2 * thr, 2, 2 * s) ** Np
    return 1 - off * on


def pd_ratio(snr_db, Np, Nw, thr):
    """P(Yp / (Yp + Yo) > thr), Yp = ncx2(2Np, 2Np s)/2, Yo ~ Gamma(Nw-Np)."""
    s = 10 ** (snr_db / 10)
    c = thr / (1 - thr)
    f = lambda y: gamma.pdf(y, Nw - Np) * ncx2.sf(2 * c * y, 2 * Np, 2 * Np * s)
    hi = (Nw - Np) + 40 * np.sqrt(Nw - Np) + 40
    return integrate.quad(f, 0, hi, limit=400, epsabs=1e-12)[0]


def snr_at(pd_fun, target=0.9, lo=-30.0, hi=30.0):
    from scipy.optimize import brentq
    return brentq(lambda x: pd_fun(x) - target, lo, hi, xtol=1e-3)


def pd_sum(snr_db, Np, thr):
    """P(sum of Np pulse samples / sigma^2 > thr)."""
    s = 10 ** (snr_db / 10)
    return float(ncx2.sf(2 * thr, 2 * Np, 2 * Np * s))
