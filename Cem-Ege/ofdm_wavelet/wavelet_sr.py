"""Part B - Morlet continuous-wavelet-transform (CWT) symbol-rate estimator.

Method (transition detection by a scalogram, then a spectral line at the symbol rate)
  1. Occupied band: carrier centre fc and 99 % occupied bandwidth OBW from a Welch PSD above its noise floor (fsk_rate.psd_band, existing).
  2. CWT with the Morlet wavelet  psi(t) = exp(j 2 pi fc' t) exp(-t^2 / (2 s_t^2)),  s_t = 1 / (2 pi s_f)   (analytic Morlet; its Fourier transform is a Gaussian
     of centre fc' and width s_f).  The scale is matched to the signal's occupied bandwidth:  s_f = kappa * OBW  (a short wavelet resolves the symbol transitions:
     the time resolution 1/s_f is a fraction of a symbol interval 1/Rs ~ (1+rolloff)/OBW).  The CWT is evaluated at the band centre fc' = fc and optionally at
     further centres fc + o*OBW (o in a set): tones of FSK sit away from the centre, PSK/QAM transitions are visible at the centre.
     |CWT_{fc'}(t)| is evaluated by FFT: W = IFFT( X(f) * exp(-(f - fc')^2 / (2 s_f^2)) ), edge samples (4 s_t) discarded.
     (This is the standard Morlet CWT of the frequency-shifted signal; |W| is invariant to the shift, so the real-passband definition with omega0 = 6
     is the special case s_f = fc/6.  Complex baseband has no fixed carrier, hence the explicit centre / width.)
  3. Transition feature: v_c(t) = |W_c(t)|^p (p = 1 or 2). PSK/QAM symbol transitions make |W| dip / jump once per symbol, FSK tone changes switch the energy in
     and out of a tone-centred wavelet -> v_c is cyclostationary with period 1/Rs.
  4. Spectral line: periodogram of v_c - mean (Hann, 8x zero padding), each channel divided by its median in the search band, summed over the centres, then the
     SAME peak picking as fsk_rate.line (search 1/25 .. 1/2.5 cycles/sample, local-median-background prominence (".l") or band-median prominence (".g"),
     parabolic interpolation), so that (rate, prominence) can join the consensus vote of characterization_extended.

Parameters (kappa, centre set, p) are chosen on the DEV set only.
References: Ho, Prokopiw & Chan, "Modulation identification of digital signals by the wavelet transform", IEE Proc.-Radar Sonar Navig. 147(4), 2000 (Haar transform,
transition detection); Morlet-wavelet symbol-rate estimators: "A symbol rate estimation algorithm based on Morlet wavelet transform and autocorrelation" (MORAL),
and the wavelet-vs-cyclic-spectrum comparison in Int. J. Performability Eng. 15(8), 2019 (see README).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CE = os.path.join(ROOT, "characterization_extended")
for p in (CE, os.path.join(ROOT, "classical_amc"), os.path.join(ROOT, "extra_methods")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import common as cm          # noqa: E402  characterization_extended/common.py (paths only)
import fsk_rate as fr        # noqa: E402  existing: psd_band, _bg, _pick_band, BG_WIDTH
import consensus as cs       # noqa: E402  existing vote machinery

A_MIN, A_MAX, PAD = fr.A_MIN, fr.A_MAX, 8

KAPPAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
CENTRE_SETS = {"c1": (0.0,), "c3": (-0.3, 0.0, 0.3), "c3w": (-0.45, 0.0, 0.45), "c5": (-0.4, -0.2, 0.0, 0.2, 0.4), "c7": (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45)}
POWERS = (1, 2)
CONFIGS = [(k, c, p) for k in KAPPAS for c in CENTRE_SETS for p in POWERS]


def cfg_name(cfg):
    return f"morl_k{cfg[0]}_{cfg[1]}_p{cfg[2]}"


def morlet_cwt(x, fc, sigma_f):
    """Complex Morlet CWT coefficients of x at centre frequency fc (cycles/sample), Fourier-domain Gaussian width sigma_f (circular FFT convolution)."""
    n = len(x)
    f = np.fft.fftfreq(n)
    d = (f - fc + 0.5) % 1.0 - 0.5
    return np.fft.ifft(np.fft.fft(x) * np.exp(-d ** 2 / (2 * sigma_f ** 2)))


def morlet_cwt_direct(x, fc, sigma_f):
    """Reference implementation by direct time-domain convolution with the truncated Morlet wavelet (for the sanity check)."""
    st = 1 / (2 * np.pi * sigma_f)
    m = int(np.ceil(6 * st))
    t = np.arange(-m, m + 1)
    psi = np.exp(2j * np.pi * fc * t) * np.exp(-t ** 2 / (2 * st ** 2))
    psi = psi / np.sum(np.exp(-t ** 2 / (2 * st ** 2)))
    # the FFT version has unit gain at fc (Gaussian peak 1) and correlates x with the wavelet: W(t) = sum_u x(u) conj(psi)(u - t) ... for the analytic Gaussian filter
    # H(f) = exp(-(f-fc)^2 / 2 sf^2) <-> impulse response h(t) = sum-normalised exp(j 2 pi fc t) exp(-t^2 / 2 st^2); y = x * h
    return np.convolve(x, psi, "same")


def _band(n):
    nn = PAD * n
    lo = int(np.ceil(A_MIN * nn)) - int(fr.BG_WIDTH * nn)
    hi = int(np.floor(A_MAX * nn)) + int(fr.BG_WIDTH * nn)
    lo, hi = max(lo, 1), min(hi, nn // 2 - 1)
    kgrid = np.arange(lo, hi + 1)
    inband = (kgrid >= A_MIN * nn) & (kgrid <= A_MAX * nn)
    return nn, kgrid, inband


def channel_spectrum(v, nn, kgrid, inband):
    v = v - v.mean()
    P = np.abs(np.fft.rfft(v * np.hanning(len(v)), nn)) ** 2
    F = P[kgrid]
    return F / (np.median(F[inband]) + 1e-30)


def pick(F, kgrid, inband, nn, nbase):
    """(rate, prom_local, prom_global) from a summed channel spectrum F (same peak picking as fsk_rate.line)."""
    old = fr.LOCAL_BG
    out = []
    for mode in (True, False):
        fr.LOCAL_BG = mode
        r, p = fr._pick_band(F, kgrid, inband, nn, PAD, nbase)
        out.append((r, p))
    fr.LOCAL_BG = old
    return out[0][0], out[0][1], out[1][0], out[1][1]


def frame_features(x, configs=CONFIGS):
    """For one frame: dict cfg -> (rate_l, prom_l, rate_g, prom_g) for every (kappa, centre set, p) config."""
    x = np.asarray(x, complex)
    fc, obw = fr.psd_band(x)
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n)
    out = {}
    cache = {}
    for kap, cname, p in configs:
        sf = kap * obw
        st = 1 / (2 * np.pi * sf)
        trim = int(min(np.ceil(4 * st), n // 4))
        spec = []
        nn = None
        for o in CENTRE_SETS[cname]:
            key = (kap, o)
            if key not in cache:
                d = (f - (fc + o * obw) + 0.5) % 1.0 - 0.5
                cache[key] = np.abs(np.fft.ifft(X * np.exp(-d ** 2 / (2 * sf ** 2))))[trim:n - trim]
            v = cache[key] ** p
            nn, kgrid, inband = _band(len(v))
            if (key, p) not in cache:
                cache[(key, p)] = channel_spectrum(v, nn, kgrid, inband)
            spec.append(cache[(key, p)])
        F = np.sum(spec, axis=0)
        out[(kap, cname, p)] = pick(F, kgrid, inband, nn, len(v))
    return out


def estimate(x, cfg, mode="l"):
    """Rate estimate (cycles/sample) of one frame for one configuration (cfg = (kappa, centre-set name, power)); mode l/g = local / global prominence peak picking."""
    return frame_features(x, [cfg])[cfg][0 if mode == "l" else 2]
