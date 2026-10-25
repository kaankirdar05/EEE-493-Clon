"""Task 3 - FSK-oriented classical symbol-rate estimators + multi-estimator consensus.

All estimators return (rate_hat, prominence): rate in cycles/sample (search range 1/25 .. 1/2.5,
as in classical_amc/symbol_rate.py), prominence = spectral line height / median spectrum
height in the search band (amplitude ratio). Prominence is what the consensus uses to weigh
estimators against each other.

Building blocks
  bandlimit(x)       Welch PSD -> noise floor (low quantile) -> centroid and 99 % occupied band
                     of the excess power -> shift to 0 Hz, low-pass FIR at 0.6 x OBW. Raises the
                     FM-discriminator SNR for narrow signals (TorchSig FSK: 8-45 % of fs).
  discriminator(y)   f[n] = arg(y[n+1] y*[n])  (instantaneous frequency, rad/sample)
Transition features of the discriminator output (each -> spectral line at Rs)
  if_dev_L      | f_s - median(f_s) |, f_s = L-tap moving average            (as inst_freq_sL)
  haar_L        | mean(f[n:n+L]) - mean(f[n-L:n]) |  (Haar wavelet of the frequency at scale L:
                peaks at frequency transitions; Ho/Prokopiw/Chan-style transition detector)
  dsq_L         (f_s[n+L] - f_s[n])^2   (squared discriminator increment)
  *_auto        the scale L in a fixed set chosen per frame by the highest prominence
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, welch

A_MIN, A_MAX = 1 / 25, 1 / 2.5


BG_WIDTH = 0.012     # local-background window (cycles/sample) for line detection
LOCAL_BG = True


def _bg(F, pad, nbase):
    """Local background of spectrum F (padded grid, factor pad): running median over
    BG_WIDTH on the un-padded grid, interpolated back. A line is ~1/N wide, the continuous
    spectrum of the feature is much broader, so F / bg isolates lines."""
    from scipy.ndimage import median_filter
    base = F[::pad]
    w = max(11, int(round(BG_WIDTH * nbase)) | 1)
    b = median_filter(base, size=w, mode="nearest")
    return np.interp(np.arange(len(F)), np.arange(len(base)) * pad, b) + 1e-30


def line(v, a_min=A_MIN, a_max=A_MAX, pad=8):
    """Strongest spectral line of real sequence v in [a_min, a_max] cycles/sample.
    Returns (rate, prominence = power / local background)."""
    v = np.asarray(v, float)
    v = v - v.mean()
    n = pad * len(v)
    P = np.abs(np.fft.rfft(v * np.hanning(len(v)), n)) ** 2
    lo = int(np.ceil(a_min * n)) - int(BG_WIDTH * n)
    hi = int(np.floor(a_max * n)) + int(BG_WIDTH * n)
    lo, hi = max(lo, 1), min(hi, len(P) - 1)
    kgrid = np.arange(lo, hi + 1)
    F = P[kgrid]
    inband = (kgrid >= a_min * n) & (kgrid <= a_max * n)
    r, p = _pick_band(F, kgrid, inband, n, pad, len(v))
    return r, p


def _pick_band(F, kgrid, inband, n, pad, nbase):
    R = F / _bg(F, pad, nbase) if LOCAL_BG else F / (np.median(F[inband]) + 1e-30)
    Rb = np.where(inband, R, -np.inf)
    k = int(np.argmax(Rb))
    kk = float(kgrid[k])
    if 0 < k < len(F) - 1:
        a, b, c = np.log(F[k - 1] + 1e-30), np.log(F[k] + 1e-30), np.log(F[k + 1] + 1e-30)
        den = a - 2 * b + c
        if den != 0:
            kk += 0.5 * (a - c) / den
    return kk / n, float(R[k])


def psd_band(x, nfft=256):
    f, P = welch(x, nperseg=nfft, noverlap=nfft // 2, return_onesided=False, detrend=False, window="hann")
    f, P = np.fft.fftshift(f), np.fft.fftshift(P)
    floor = np.quantile(P, 0.2)
    E = np.clip(P - floor, 0, None)
    if E.sum() <= 0:
        return 0.0, 1.0
    fc = float(np.sum(f * E) / E.sum())
    c = np.cumsum(E) / E.sum()
    lo, hi = f[np.searchsorted(c, 0.005)], f[min(np.searchsorted(c, 0.995), len(f) - 1)]
    return fc, float(max(hi - lo, 2.0 / nfft))


_FIR = {}


def bandlimit(x, factor=0.6, ntaps=129):
    fc, obw = psd_band(x)
    y = x * np.exp(-2j * np.pi * fc * np.arange(len(x)))
    cut = min(max(factor * obw, 0.02), 0.49)
    key = (round(cut, 3), ntaps)
    if key not in _FIR:
        _FIR[key] = firwin(ntaps, key[0] * 2)   # firwin cutoff relative to Nyquist (= 0.5 cyc/sample)
    return np.convolve(y, _FIR[key], "same")


def discriminator(y):
    return np.angle(y[1:] * np.conj(y[:-1]))


def smooth(f, L):
    return f if L <= 1 else np.convolve(f, np.ones(L) / L, "same")


def feat_if_dev(f, L):
    fs = smooth(f, L)
    return np.abs(fs - np.median(fs))


def feat_haar(f, L):
    c = np.concatenate([[0.0], np.cumsum(f)])
    n = len(f)
    i = np.arange(L, n - L + 1)
    right = (c[i + L] - c[i]) / L
    left = (c[i] - c[i - L]) / L
    return np.abs(right - left)


def feat_dsq(f, L):
    fs = smooth(f, max(L // 2, 1))
    return (fs[L:] - fs[:-L]) ** 2


FEATS = {"if_dev": feat_if_dev, "haar": feat_haar, "dsq": feat_dsq}
SCALES = (1, 2, 3, 5, 8)


def est_feature(x, feat, L, bl=True):
    y = bandlimit(x) if bl else x
    return line(FEATS[feat](discriminator(y), L))


def est_auto(x, feat, scales=SCALES, bl=True):
    y = bandlimit(x) if bl else x
    f = discriminator(y)
    best = (0.0, -1.0)
    for L in scales:
        r = line(FEATS[feat](f, L))
        if r[1] > best[1]:
            best = r
    return best


def line_folded(P, n, a_min=A_MIN, a_max=A_MAX, pad=8):
    """P = power spectrum of a complex sequence on an n-point FFT grid; fold +/- alpha,
    pick the strongest line in [a_min, a_max]."""
    lo = max(int(np.ceil(a_min * n)) - int(BG_WIDTH * n), 1)
    hi = min(int(np.floor(a_max * n)) + int(BG_WIDTH * n), n // 2 - 1)
    m = np.arange(lo, hi + 1)
    F = P[m] + P[(-m) % n]
    inband = (m >= a_min * n) & (m <= a_max * n)
    return _pick_band(F, m, inband, n, pad, n // pad)


def caf_spectrum(y, lags, pad=8):
    """sum_d |FFT(y[n+d] y*[n] - mean)|^2: cyclic autocorrelation power summed over lags."""
    N = len(y)
    n = pad * N
    w = np.hanning(N - max(lags))
    P = np.zeros(n)
    for d in lags:
        z = y[d:d + len(w)] * np.conj(y[:len(w)])
        z = z - z.mean()
        P += np.abs(np.fft.fft(z * w, n)) ** 2
    return P, n


def est_caf(x, lags=(1, 2, 3, 4), bl=True):
    y = bandlimit(x) if bl else x
    P, n = caf_spectrum(y, lags)
    return line_folded(P, n)


def est_caf_auto(x, lag_sets=((1,), (2,), (3,), (4,), (6,), (8,)), bl=True):
    y = bandlimit(x) if bl else x
    best = (0.0, -1.0)
    for ls in lag_sets:
        P, n = caf_spectrum(y, ls)
        r = line_folded(P, n)
        if r[1] > best[1]:
            best = r
    return best


CANDIDATES = {}
for _feat in FEATS:
    for _L in (1, 3):
        CANDIDATES[f"{_feat}{_L}_raw"] = (lambda a, b: (lambda x: est_feature(x, a, b, bl=False)))(_feat, _L)
        CANDIDATES[f"{_feat}{_L}_bl"] = (lambda a, b: (lambda x: est_feature(x, a, b)))(_feat, _L)
    CANDIDATES[f"{_feat}_auto_bl"] = (lambda a: (lambda x: est_auto(x, a)))(_feat)
CANDIDATES["caf1234_raw"] = lambda x: est_caf(x, bl=False)
CANDIDATES["caf1234_bl"] = lambda x: est_caf(x)
CANDIDATES["caf12_bl"] = lambda x: est_caf(x, (1, 2))
CANDIDATES["caf_auto_bl"] = lambda x: est_caf_auto(x)
