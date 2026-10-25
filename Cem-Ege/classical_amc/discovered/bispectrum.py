"""Bispectrum (third-order spectrum) features.

Direct (FFT, segment-averaged) estimator, as in the HOSA toolbox `bispecd` and Nikias & Petropulu
(1993), "Higher-Order Spectra Analysis":

    B(f1, f2) = < X_k(f1) X_k(f2) X_k*(f1 + f2) >_k            (k = segment, Hann window, mean removed)
    b^2(f1, f2) = |B|^2 / ( <|X_k(f1) X_k(f2)|^2>_k <|X_k(f1 + f2)|^2>_k )      (bicoherence, Kim & Powers 1979)

b^2 is in [0, 1] (Cauchy-Schwarz). For a Gaussian process B = 0 and b^2 only shows the estimator's
floor, E[b^2] ~ 1/K with K segments; under quadratic phase coupling (phase(f1)+phase(f2) =
phase(f1+f2) in every segment) b^2(f1, f2) -> 1.

Bins with f1 = 0, f2 = 0 or f1 + f2 = 0 (mod nfft) are excluded (the segment mean is removed).
Statistics follow the bispectrum-feature literature for AMC / fault diagnosis:
    bic_mean, bic_max          mean and max bicoherence
    bic_frac                   fraction of bins whose b^2 exceeds the 5 % Gaussian-null threshold ln(20)/K
                               (2K b^2 ~ chi^2_2 under the null, Elgar & Guza 1988)
    log_skew                   log10( mean|B| / mean(P)^1.5 ): scale-free bispectral level
    ent1, ent2                 normalised bispectral entropies of |B| and |B|^2 (Chua et al. 2010)
    diag_frac                  share of sum|B| on the diagonal slice f1 = f2
    peak_r                     radial frequency sqrt(f1^2 + f2^2) of the |B| maximum (signed frequencies)
    wcob                       weighted centre of bispectrum  sum|f1||B| / sum|B|  (Zhang et al. 2001)
    aib_pmr                    peak-to-mean ratio of the axially integrated bispectrum |sum_f2 B(f1, f2)|
                               (Tugnait 1994; Chandran & Elgar 1993)

Signals: the unit-power complex frame x (third-order moments x x x*), its envelope |x| and its
wrapped instantaneous frequency. 3 signals x 10 statistics = 30 features.
"""

from __future__ import annotations

import numpy as np

import dcommon as dc

NFFT = 64
STATS = ["bic_mean", "bic_max", "bic_frac", "log_skew", "ent1", "ent2", "diag_frac", "peak_r", "wcob", "aib_pmr"]
SIGNALS = ["x", "amp", "freq"]
BIS_NAMES = [f"bis_{s}_{t}" for s in SIGNALS for t in STATS]

_I, _J = np.meshgrid(np.arange(NFFT), np.arange(NFFT), indexing="ij")
_S = (_I + _J) % NFFT
MASK = (_I != 0) & (_J != 0) & (_S != 0)
FREQ = np.fft.fftfreq(NFFT)


def segment_fft(s: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    """(N, L) -> (N, K, nfft) Hann-windowed FFTs of non-overlapping, mean-removed segments."""
    K = s.shape[1] // nfft
    seg = s[:, :K * nfft].reshape(s.shape[0], K, nfft)
    seg = seg - seg.mean(axis=2, keepdims=True)
    return np.fft.fft(seg * np.hanning(nfft), axis=2)


def bispectrum(s: np.ndarray, nfft: int = NFFT):
    """Return B (N, nfft, nfft) complex, bicoherence b2 (N, nfft, nfft), power P (N, nfft), K."""
    F = segment_fft(s, nfft)
    K = F.shape[1]
    I, J = np.meshgrid(np.arange(nfft), np.arange(nfft), indexing="ij")
    S = (I + J) % nfft
    X12 = F[:, :, :, None] * F[:, :, None, :]                        # (N, K, nfft, nfft)
    X3 = F[:, :, S]
    B = np.mean(X12 * np.conj(X3), axis=1)
    den = np.mean(np.abs(X12) ** 2, axis=1) * np.mean(np.abs(X3) ** 2, axis=1)
    b2 = np.abs(B) ** 2 / np.maximum(den, 1e-30)
    P = np.mean(np.abs(F) ** 2, axis=1)
    return B, b2, P, K


def _stats(B, b2, P, K):
    n = B.shape[0]
    m = MASK
    aB = np.abs(B)
    aBm = aB[:, m]                                                    # (N, nbins)
    b2m = b2[:, m]
    nb = aBm.shape[1]
    sB = np.maximum(aBm.sum(1), 1e-30)
    p1 = aBm / sB[:, None]
    q = aBm ** 2
    p2 = q / np.maximum(q.sum(1), 1e-30)[:, None]
    ent1 = -np.sum(p1 * np.log(np.maximum(p1, 1e-300)), 1) / np.log(nb)
    ent2 = -np.sum(p2 * np.log(np.maximum(p2, 1e-300)), 1) / np.log(nb)
    Pm = np.maximum(P[:, 1:].mean(1), 1e-30)
    log_skew = np.log10(np.maximum(aBm.mean(1), 1e-30) / Pm ** 1.5)
    diag = np.zeros((NFFT, NFFT), bool)
    np.fill_diagonal(diag, True)
    diag_frac = (aB * (diag & m)).reshape(n, -1).sum(1) / sB
    flat = (aB * m).reshape(n, -1)
    k = np.argmax(flat, axis=1)
    i, j = np.unravel_index(k, (NFFT, NFFT))
    peak_r = np.sqrt(FREQ[i] ** 2 + FREQ[j] ** 2)
    wcob = (aB * m * np.abs(FREQ)[None, :, None]).reshape(n, -1).sum(1) / sB
    aib = np.abs(np.sum(B * m, axis=2))[:, 1:]                        # (N, nfft-1), f1 != 0
    aib_pmr = aib.max(1) / np.maximum(aib.mean(1), 1e-30)
    thr = np.log(20.0) / K
    return [b2m.mean(1), b2m.max(1), np.mean(b2m > thr, 1), log_skew, ent1, ent2, diag_frac, peak_r, wcob, aib_pmr]


def bispectrum_features(x: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for s0 in range(0, len(x), batch):
        xb = dc.normalize(x[s0:s0 + batch])
        cols = []
        for sig in (xb, np.abs(xb), dc.inst_freq(xb)):
            cols += _stats(*bispectrum(sig))
        out.append(np.stack(cols, axis=1))
    return dc.clean(np.concatenate(out))
