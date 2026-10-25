"""Part B - time-frequency tools for linear-chirp detection.

All frames are N complex samples, time index centred: n' = n - N/2.
A linear chirp is s(n) = exp(j(2 pi f0 n' + pi k n'^2 + phi)): instantaneous
frequency IF(n) = f0 + k n' (cycles/sample), sweep bandwidth B = |k| N.

    pwvd(x)            pseudo Wigner-Ville distribution (lag window h), optional time
                       smoothing g -> smoothed pseudo-WVD (SPWVD). x is upsampled by 2
                       first so the discrete WVD of a full-band complex signal does not alias.
    stft_power(x)      spectrogram on the same (time, frequency) grid as pwvd
    line_integrals(W)  Radon transform restricted to straight IF lines f = f0 + k n'
                       on a (f0, k) grid -> Radon-Wigner (on a WVD) or spectrogram Radon
    dechirp_search(x)  |FFT(x * exp(-j pi k n'^2))|^2 over a grid of k. This is the
                       magnitude of the fractional Fourier transform at angle
                       alpha = arccot(-k N) (normalised coordinates t = n'/sqrt(N)),
                       i.e. a discrete FrFT / Radon-Wigner peak search.
"""

from __future__ import annotations

import numpy as np


def chirp(N, f0, k, amp=1.0, phi=0.0):
    n = np.arange(N) - N / 2
    return amp * np.exp(1j * (2 * np.pi * f0 * n + np.pi * k * n**2 + phi))


def upsample2(x):
    N = len(x)
    X = np.fft.fft(x)
    Y = np.zeros(2 * N, complex)
    Y[: N // 2] = X[: N // 2]
    Y[-N // 2:] = X[N // 2:]
    return np.fft.ifft(Y) * 2


def pwvd(x, M=512, hop=16, L=255, g_len=0, sub=1, lag_win="hann", agg=True):
    """(Smoothed) pseudo-WVD on times n_i = hop/2 + hop*i (original samples) and
    M frequency bins f = (b - M/2)/M (fftshifted). Lag window: Hann, |m| <= L
    lags of the 2x-upsampled signal. g_len > 0: time smoothing with a Hann of
    g_len original samples (SPWVD), computed on a finer grid (every `sub`
    samples) and then decimated to `hop`.
    agg=True (and g_len = 0): the WVD is computed at EVERY sample and summed
    over each hop (boxcar g of length hop) before decimation. Without this, a
    line integral over the decimated grid only uses the sample pairs whose
    midpoint lies on the grid (1/hop of them) and loses ~sqrt(hop) in
    deflection - found with diag_wvd_deflection.py (full-frame WVD on a hop-16
    grid: deflection 8.3 vs 46.8 for the equivalent dechirp)."""
    N = len(x)
    xu = np.concatenate([np.zeros(L, complex), upsample2(x), np.zeros(L + 1, complex)])
    if agg and not g_len:
        g_len, g_box = hop, True
    else:
        g_box = False
    step = sub if g_len else hop
    n_t = np.arange((hop // 2) % step, N, step)
    m = np.arange(-L, L + 1)
    h = np.hanning(2 * L + 3)[1:-1] if lag_win == "hann" else np.ones(2 * L + 1)
    c = 2 * n_t[:, None] + L
    K = xu[c + m] * np.conj(xu[c - m]) * h
    Kf = np.zeros((len(n_t), M), complex)
    Kf[:, m % M] = K
    W = np.fft.fftshift(np.fft.fft(Kf, axis=1).real, axes=1)
    if g_len:
        g = np.ones(g_len // sub) if g_box else np.hanning(g_len // sub + 2)[1:-1]
        g /= g.sum()
        from scipy.ndimage import convolve1d
        W = convolve1d(W, g, axis=0, mode="constant")
        W = W[(hop // 2) // step:: hop // step]
    return W  # (N/hop, M)


def stft_power(x, M=512, hop=16, win=128):
    N = len(x)
    xp = np.concatenate([np.zeros(win // 2, complex), x, np.zeros(win // 2, complex)])
    n_t = np.arange(hop // 2, N, hop)
    idx = n_t[:, None] + np.arange(win)[None, :]
    fr = xp[idx] * np.hanning(win)
    return np.fft.fftshift(np.abs(np.fft.fft(fr, n=M, axis=1)) ** 2, axes=1) / np.sum(np.hanning(win) ** 2)


def k_grid_tf(N, M=512, n_k=257, kmax=None):
    kmax = 0.5 / N if kmax is None else kmax
    return np.linspace(-kmax, kmax, n_k)


def line_integrals(W, N, ks, hop=16):
    """Sum W along f(n') = f0 + k n' for all f0 bins and every k in ks.
    W: (T, M), times n_i = hop/2 + hop*i. Returns (len(ks), M) array of line
    sums (index 1 = f0 bin at the centre time, fftshifted)."""
    T, M = W.shape
    n_c = np.arange(hop // 2, N, hop) - N / 2
    out = np.empty((len(ks), M))
    rows = np.arange(T)[:, None]
    base = np.arange(M)[None, :]
    for j, k in enumerate(ks):
        off = np.rint(k * n_c * M).astype(int)[:, None]
        out[j] = W[rows, (base + off) % M].sum(0)
    return out


def dechirp_search(x, ks, nfft=None, chunk=256):
    """max_f |FFT(x e^{-j pi k n'^2})|^2 / N for each k. Returns (peak per k, argmax-f per k)."""
    N = len(x)
    nfft = nfft or 2 * N
    n = np.arange(N) - N / 2
    peaks = np.empty(len(ks))
    fidx = np.empty(len(ks), int)
    for s in range(0, len(ks), chunk):
        kk = ks[s: s + chunk]
        Y = np.abs(np.fft.fft(x[None, :] * np.exp(-1j * np.pi * kk[:, None] * n[None, :] ** 2), n=nfft, axis=1)) ** 2 / N
        peaks[s: s + chunk] = Y.max(1)
        fidx[s: s + chunk] = Y.argmax(1)
    return peaks, fidx


def frft_angle(k, N):
    """FrFT rotation angle (rad, in (0, pi)) that compresses a chirp of rate k
    (cycles/sample^2) into an impulse: cot(alpha) = -k N."""
    return np.arctan2(1.0, -k * N) % np.pi


def frft_mag(x, alpha, nfft=None):
    """|X_alpha(u)| of the continuous-kernel FrFT sampled with t = n'/sqrt(N):
    |X_alpha(u)| = |sum_t x(t) exp(j pi cot(alpha) t^2) exp(-j 2 pi u t csc(alpha))| / sqrt(N|sin alpha|)
    evaluated on the FFT grid (u csc alpha = bin/nfft * sqrt(N))."""
    N = len(x)
    t = (np.arange(N) - N / 2) / np.sqrt(N)
    nfft = nfft or 2 * N
    return np.abs(np.fft.fft(x * np.exp(1j * np.pi * t**2 / np.tan(alpha)), n=nfft)) / np.sqrt(N * abs(np.sin(alpha)))
