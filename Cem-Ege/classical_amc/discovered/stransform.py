"""Stockwell (S-) transform features.

Discrete S-transform (Stockwell, Mansinha & Lowe 1996), voice n != 0 of an N-sample frame with
DFT X[m] (numpy convention, unnormalised):

    S[k, n] = (1/N) sum_m X[m + n] exp(-2 pi^2 m^2 / n^2) exp(+j 2 pi m k / N),   k = 0..N-1

computed as an inverse FFT per voice; S[k, 0] = mean(x). Two exact properties are used in the
sanity checks: sum_k S[k, n] = X[n] (time-integration returns the spectrum), and for a tone
A exp(j 2 pi f0 k / N) on an integer bin, |S[k, n]| = A exp(-2 pi^2 (f0 - n)^2 / n^2), maximal (= A) at
n = f0 for every k. Complex baseband: positive and negative voices are both needed.

Voices (log-spaced, constant-Q, dense near DC where RadioML's 8-samples-per-symbol signals live):
    n in {0, +-1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384}  (bins of 1024)
grouped into 8 bands by |n|: {1,2} {4,6} {8,12} {16,24} {32,48} {64,96} {128,192} {256,384}.
Features (unit-power frame), in the spirit of S-transform features for power-quality and
modulation recognition (e.g. Dash et al. 2003; Lee & Kim 2008):
    st_eband_b      band energy share (8)
    st_cv_b         coefficient of variation over time of the band energy envelope (8)
    st_asym         (E_pos - E_neg) / (E_pos + E_neg) over all voices (SSB / spectral asymmetry)
    st_tfent        normalised Shannon entropy of the |S|^2 time-frequency distribution
    st_pk_std       std over time of the peak voice's log2|n| (voice-hopping, FSK/GMSK)
    st_pk_jump      rate at which the peak voice changes between consecutive samples
    st_max_mean, st_max_cv, st_max_kurt   statistics of the max-over-voices amplitude contour
Total 23 features.
"""

from __future__ import annotations

import numpy as np

import dcommon as dc

N_FFT = 1024
POS = np.array([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384])
VOICES = np.concatenate([[0], POS, -POS])
BANDS = [(1, 2), (4, 6), (8, 12), (16, 24), (32, 48), (64, 96), (128, 192), (256, 384)]
ST_NAMES = ([f"st_eband{i}" for i in range(len(BANDS))] + [f"st_cv{i}" for i in range(len(BANDS))] +
            ["st_asym", "st_tfent", "st_pk_std", "st_pk_jump", "st_max_mean", "st_max_cv", "st_max_kurt"])


def windows(N: int, voices=VOICES):
    m = np.fft.fftfreq(N, 1.0 / N)                                   # signed integer frequency
    G = np.zeros((len(voices), N))
    for i, n in enumerate(voices):
        if n == 0:
            G[i, 0] = 1.0                                            # -> S[k, 0] = mean(x)
        else:
            G[i] = np.exp(-2 * np.pi ** 2 * m ** 2 / float(n) ** 2)
    idx = (np.arange(N)[None, :] + np.asarray(voices)[:, None]) % N  # X[m + n]
    return G, idx


def stransform(x: np.ndarray, voices=VOICES) -> np.ndarray:
    """(B, N) -> (B, n_voices, N) complex S-transform at the given voices."""
    x = np.atleast_2d(x)
    N = x.shape[1]
    G, idx = windows(N, voices)
    X = np.fft.fft(x, axis=1)
    return np.fft.ifft(X[:, idx] * G[None], axis=2)


def st_features(x: np.ndarray, batch: int = 128) -> np.ndarray:
    out = []
    lv = np.log2(np.abs(VOICES[1:]))
    for s0 in range(0, len(x), batch):
        xb = dc.normalize(x[s0:s0 + batch])
        S = np.abs(stransform(xb))[:, 1:, :]                         # drop the DC voice (mean = 0)
        P = S ** 2                                                   # (B, 32, N)
        av = np.abs(VOICES[1:])
        Etot = np.maximum(P.sum((1, 2)), 1e-30)
        cols = []
        env = []
        for lo, hi in BANDS:
            sel = (av >= lo) & (av <= hi)
            e = P[:, sel, :].sum(1)                                  # (B, N) band energy envelope
            env.append(e)
            cols.append(e.sum(1) / Etot)
        for e in env:
            cols.append(e.std(1) / np.maximum(e.mean(1), 1e-30))
        pos = P[:, VOICES[1:] > 0, :].sum((1, 2))
        neg = P[:, VOICES[1:] < 0, :].sum((1, 2))
        cols.append((pos - neg) / np.maximum(pos + neg, 1e-30))
        p = P.reshape(len(xb), -1) / Etot[:, None]
        cols.append(-np.sum(p * np.log(np.maximum(p, 1e-300)), 1) / np.log(p.shape[1]))
        pk = np.argmax(S, axis=1)                                    # (B, N) peak voice index
        cols.append(lv[pk].std(1))
        cols.append(np.mean(pk[:, 1:] != pk[:, :-1], 1))
        mx = S.max(1)
        mu = mx.mean(1)
        sd = mx.std(1)
        cols += [mu, sd / np.maximum(mu, 1e-30),
                 np.mean((mx - mu[:, None]) ** 4, 1) / np.maximum(sd ** 4, 1e-30)]
        out.append(np.stack(cols, axis=1))
    return dc.clean(np.concatenate(out))
