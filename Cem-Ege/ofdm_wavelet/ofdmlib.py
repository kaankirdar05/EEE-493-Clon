"""Shared OFDM / single-carrier signal generation, features and the subcarrier receiver (Part A).

Imports (never edits) extra_methods/ofdm_params.py (blind (N, cp) estimator), classical_amc/requirements_methods.py
(Swami-Sadler thresholds) and classical_amc/extended/symbol_classifiers.py (EM constellation classifier).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "extra_methods"), os.path.join(ROOT, "classical_amc"), os.path.join(ROOT, "classical_amc", "extended")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy.signal import welch

import ofdm_params as op                        # noqa: E402  existing: ofdm_frame, z_stat, estimate, GRID
from amc.synthetic import _rrc                  # noqa: E402  existing RRC taps

L_FRAME = 8192
PILOT_SPACING = 4      # assumption: comb pilots on every 4th FFT bin (k % 4 == 0, k != 0) inside the used band, same known QPSK sequence in every OFDM symbol
MODS = {"QPSK": np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2),
        "16QAM": (np.arange(-3, 4, 2)[:, None] + 1j * np.arange(-3, 4, 2)[None, :]).ravel() / np.sqrt(10),
        "64QAM": (np.arange(-7, 8, 2)[:, None] + 1j * np.arange(-7, 8, 2)[None, :]).ravel() / np.sqrt(42),
        "BPSK": np.array([1.0, -1.0], complex)}


def layout(n):
    """Same layout as ofdm_params.ofdm_frame: used bins 1..k and -k..-1 (k = int(0.75 n)//2), DC and guard bands null."""
    k = int(0.75 * n) // 2
    used = np.zeros(n, bool)
    used[1:k + 1] = True
    used[-k:] = True
    kk = np.fft.fftfreq(n, 1 / n).astype(int)
    pil = used & (kk % PILOT_SPACING == 0)
    return used, pil, kk


def pilot_values(n):
    rng = np.random.RandomState(1000 + n)
    return (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)


def make_channel(cp, rng):
    """Static 2-3 tap channel, all delays <= D = min(cp//2, 4) samples (< cp). Exponentially decaying Rayleigh taps."""
    D = max(1, min(cp // 2, 4))
    nt = 3 if D >= 2 else 2
    d = np.concatenate([[0], np.sort(rng.choice(np.arange(1, D + 1), nt - 1, replace=False))])
    g = (rng.standard_normal(nt) + 1j * rng.standard_normal(nt)) / np.sqrt(2) * np.exp(-0.5 * np.arange(nt))
    h = np.zeros(D + 1, complex)
    h[d] = g
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def make_ofdm(n, cp, mod, snr_db, rng, L=L_FRAME, channel=None, cfo=True, pilots=True, noise=True):
    """OFDM frame of L samples (random timing offset / carrier phase / CFO up to +-0.3 subcarrier spacings).
    Returns x and a dict with the truth (tx subcarrier symbols, CP start theta in frame coordinates, ...)."""
    used, pil, kk = layout(n)
    data = used & ~pil if pilots else used
    period = n + cp
    nsym = (L + 2 * period) // period + 2
    sym = np.zeros((nsym, n), complex)
    c = MODS[mod]
    sym[:, data] = rng.choice(c, (nsym, int(data.sum())))
    if pilots:
        sym[:, pil] = pilot_values(n)[pil]
    t = np.fft.ifft(sym, axis=1) * np.sqrt(n / used.sum())
    s = np.concatenate([np.concatenate([row[-cp:], row]) for row in t])
    if channel is not None:
        s = np.convolve(s, channel)[:len(s)]
    s = s[period:]; sym = sym[1:]                       # drop symbol 0 (no channel history)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    off = int(rng.integers(0, period))
    s = s[off:off + L]
    eps = rng.uniform(-0.3, 0.3) if cfo else 0.0
    ph = rng.uniform(0, 2 * np.pi)
    s = s * np.exp(1j * (ph + 2 * np.pi * eps / n * np.arange(len(s))))
    x = s
    if noise:
        x = s + np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
    # CP start of OFDM symbol j (numbering after the drop) is at j*period - off in frame coordinates
    theta = (-off) % period
    first = int(np.ceil(off / period))                  # first symbol whose CP starts at >= 0
    return x, {"n": n, "cp": cp, "off": off, "theta": theta, "sym": sym, "eps": eps, "phase": ph, "used": used, "pil": pil,
               "first_sym": first, "period": period, "clean": s}


def make_sc(mod, snr_db, rng, sps=None, L=L_FRAME, beta=0.35):
    """Single-carrier RRC(0.35) frame of L samples; sps random in {2,3,4,6,8}; random phase, CFO up to +-0.01 cycles/sample."""
    if sps is None:
        sps = int(rng.choice([2, 3, 4, 6, 8]))
    nsym = L // sps + 64
    up = np.zeros(nsym * sps, complex)
    up[::sps] = rng.choice(MODS[mod], nsym)
    s = np.convolve(up, _rrc(beta, 8, sps), "same")
    off = int(rng.integers(8 * sps, 16 * sps))
    s = s[off:off + L]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    s = s * np.exp(1j * (rng.uniform(0, 2 * np.pi) + 2 * np.pi * rng.uniform(-0.01, 0.01) * np.arange(L)))
    return s + np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L)), {"sps": sps}


# --------------------------------------------------------------------------- cyclic prefix statistic with timing / CFO
def cp_fold(x, n, cp):
    """Same statistic as ofdm_params.z_stat, but also returns the CP start theta (fold index of the maximising window)
    and the complex correlation (for the fractional CFO)."""
    period = n + cp
    m = (len(x) - n) // period
    if m < 2:
        return 0.0, 0, 0j
    Lm = m * period
    a, b = x[:Lm], x[n:n + Lm]
    p = (a * np.conj(b)).reshape(m, period).sum(0)
    e = (0.5 * (np.abs(a) ** 2 + np.abs(b) ** 2)).reshape(m, period).sum(0)
    ext = lambda v: np.concatenate([v, v[:cp - 1]])
    win = lambda v: np.convolve(ext(v), np.ones(cp), "valid")
    wp = win(p)
    rho = np.abs(wp) / np.maximum(win(e), 1e-30)
    th = int(np.argmax(rho))
    return float(rho[th] * np.sqrt(m * cp)), th, complex(wp[th])


# --------------------------------------------------------------------------- cumulant features (sample domain)
_NOISE_BIAS = None


def _welch(x):
    return welch(x, nperseg=256, noverlap=128, return_onesided=False, detrend=False, window="hann")[1]


def noise_floor(x):
    """Blind noise-power estimate from the spectrum: mean of the lowest 20 % of a 256-bin Welch PSD, bias-corrected on pure noise."""
    global _NOISE_BIAS
    v = np.sort(_welch(x))[:51].mean()
    if _NOISE_BIAS is None:
        r = np.random.default_rng(5)
        vals = []
        for _ in range(40):
            w = (r.standard_normal(L_FRAME) + 1j * r.standard_normal(L_FRAME)) / np.sqrt(2)
            vals.append(np.sort(_welch(w))[:51].mean())
        _NOISE_BIAS = float(np.mean(vals))
    return v / _NOISE_BIAS


def cumulants(x):
    """dict: c21 (power), |C40| at zero frequency, peak of |m40(omega)| over omega (4th-power line search, robust to CFO),
    C42, normalised by the blind SIGNAL power ps = c21 - N_hat (Gaussian noise leaves C40/C42 unchanged), plus raw-normalised versions."""
    x = x - x.mean()
    c21 = float(np.mean(np.abs(x) ** 2))
    m20 = np.mean(x ** 2)
    x4 = x ** 4
    m40 = np.mean(x4)
    m42 = float(np.mean(np.abs(x) ** 4))
    nh = noise_floor(x)
    ps = max(c21 - nh, 0.05 * c21)
    c40_0 = abs(m40 - 3 * m20 ** 2)
    line = np.abs(np.fft.fft(x4, 2 * len(x))).max() / len(x)
    c42 = m42 - abs(m20) ** 2 - 2 * c21 ** 2
    return {"c21": c21, "snr_hat": 10 * np.log10(ps / max(nh, 1e-9)), "c40_0": c40_0 / ps ** 2, "c40_peak": line / ps ** 2,
            "c42": c42 / ps ** 2, "c40_0_raw": c40_0 / c21 ** 2, "c40_peak_raw": line / c21 ** 2, "c42_raw": c42 / c21 ** 2}


# --------------------------------------------------------------------------- subcarrier receiver
def used_mask_from_power(P, n):
    """Used-bin mask from the average power profile P (length n): circular moving-average over frequency, threshold half-way between
    the noise floor (5 % quantile) and the used level (60 % quantile) of the smoothed profile, then drop the transition bins."""
    w = max(3, n // 64)
    Ps = np.convolve(np.concatenate([P[-w:], P, P[:w]]), np.ones(w) / w, "same")[w:-w]
    lo = np.quantile(Ps, 0.05)
    hi = np.quantile(Ps, 0.6)
    m = Ps > 0.5 * (lo + hi)
    for _ in range(2):                                   # erosion by 2 bins: removes the (blurred) transition bins
        m = m & np.roll(m, 1) & np.roll(m, -1)
    m[0] = False                                         # assumption: DC bin unused (as in LTE / DVB-T / DAB)
    return m


def subcarrier_symbols(x, n, cp, theta, equalize=True, backoff=None, drop_fade=0.3, return_info=False):
    """Subcarrier extraction. Steps: fractional CFO from the CP correlation, derotate; FFT window starts `backoff` samples
    before the end of the CP (timing theta = CP start, e.g. from cp_fold); used bins from the power profile; if equalize:
    pilot-based per-symbol common-phase tracking + channel fit (delay-slope removal, ridge LS on a short FIR) + one-tap ZF,
    deep-faded bins (|H|^2 < drop_fade * mean) dropped. Only data bins are returned (pilot comb removed), normalised to unit power."""
    period = n + cp
    z, th, corr = cp_fold(x, n, cp)
    if theta is None:
        theta = th
    eps = -np.angle(corr) / (2 * np.pi)
    y = x * np.exp(-2j * np.pi * eps / n * np.arange(len(x)))
    b = max(1, cp // 4) if backoff is None else backoff
    s0 = theta + cp - b
    starts = np.arange(s0, len(y) - n + 1, period)
    starts = starts[starts >= 0]
    Y = np.fft.fft(np.stack([y[s:s + n] for s in starts]), axis=1)          # (M, n)
    M = Y.shape[0]
    used_true, pil, kk = layout(n)
    used = used_mask_from_power(np.mean(np.abs(Y) ** 2, 0), n)
    pil_u = pil & used
    data = used & ~pil
    # blind noise variance per FFT bin: mean power of the guard-band bins beyond the outermost used bin (+ margin)
    # (the guard band beyond the outermost used bin only: faded USED bins inside the band are not noise)
    far = np.abs(layout(n)[2]) > (np.abs(layout(n)[2][used]).max() + max(4, n // 32)) if used.any() else np.zeros(n, bool)
    sig2 = float(np.mean(np.abs(Y[:, far]) ** 2)) if far.sum() >= 3 else float("nan")
    info = {"used": used, "M": M, "eps": eps, "sig2_bin": sig2}
    if equalize and pil_u.sum() >= 4:
        X = pilot_values(n)
        r = Y[:, pil_u] / X[pil_u][None, :]                                   # per-symbol pilot LS channel (+ noise)
        ref = r[0]
        for _ in range(2):                                                    # common-phase tracking vs. a running reference
            cpe = np.angle(np.sum(r * np.conj(ref)[None, :], axis=1))
            ref = (r * np.exp(-1j * cpe)[:, None]).mean(0)
        hp = ref                                                              # channel at the pilot bins (up to a global phase)
        r_al = r * np.exp(-1j * cpe)[:, None]
        sig_h = float(np.mean(np.abs(r_al - hp[None, :]) ** 2)) / max(M - 1, 1)   # noise variance of the symbol-averaged pilot estimate
        kp = kk[pil_u].astype(float)
        dd = hp[1:] * np.conj(hp[:-1])
        ok = np.abs(kp[1:] - kp[:-1] - PILOT_SPACING) < 0.5                   # true neighbours only (not across DC / gaps)
        tau = 0.0
        if ok.sum() >= 2:
            tau = -np.angle(np.sum(dd[ok])) / (2 * np.pi * PILOT_SPACING) * n   # bulk delay in samples
        hs = hp * np.exp(1j * 2 * np.pi * kp * tau / n)
        taps = np.arange(-3, max(1, min(cp // 2, 4)) + 4)                    # residual delays after slope removal (< N/PILOT_SPACING taps: no aliasing)
        A = np.exp(-2j * np.pi * np.outer(kp, taps) / n)
        lam = sig_h * len(taps) + 1e-6                                       # MMSE-style ridge: noise variance / prior tap variance (1/ntaps)
        coef = np.linalg.solve(A.conj().T @ A + lam * np.eye(len(taps)), A.conj().T @ hs)
        kd_all = kk[data].astype(float)
        Hd = np.exp(-1j * 2 * np.pi * kd_all * tau / n) * (np.exp(-2j * np.pi * np.outer(kd_all, taps) / n) @ coef)
        Yd = Y[:, data] * np.exp(-1j * cpe)[:, None]
        pw = np.abs(Hd) ** 2
        keep = pw > drop_fade * pw.mean()
        Z = Yd[:, keep] / Hd[keep][None, :]
        nz = sig2 * float(np.mean(1.0 / pw[keep]))                            # noise variance after one-tap ZF
        info.update(tau=tau, cpe=cpe, keep=float(keep.mean()))
    else:
        Z = Y[:, data]
        nz = sig2
    Z = Z.ravel()
    pz = float(np.mean(np.abs(Z) ** 2))
    info["nvar"] = min(nz / pz, 0.99) if np.isfinite(nz) else None             # noise variance / total power of the returned symbols
    Z = Z / np.sqrt(pz)
    return (Z, info) if return_info else Z
