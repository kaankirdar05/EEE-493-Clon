"""Part A - additional per-block detectors in the WP4 calibrated framework.

All four work on the same coarse grid as the existing `eigen_mme` / `cyclo`:
32 time blocks x 16 sub-bands, each block a 512-sample complex sub-band signal
(ideal FFT channelisation, `wp4.detectors.subbands`). Every statistic is
scale-invariant (a ratio of second-order quantities), so no noise-power estimate
is needed; thresholds are the (1 - Pfa) quantiles of the statistic on white
complex Gaussian blocks (`Detector.calibrate`), exactly as for the existing ones.

    cav          covariance absolute value, Zeng & Liang (2009)
    entropy_spec spectral (Shannon) entropy deficit of the Welch PSD of the block
    entropy_hist Zhang et al. (2010) entropy of the FFT-magnitude histogram
    acorr        sum of squared normalised autocorrelations, lags 1..L
    scd_fam      max spectral coherence at nonzero cyclic frequency, FFT Accumulation
                 Method (Roberts, Brown & Loomis 1991), conjugate + non-conjugate
"""

from __future__ import annotations

import numpy as np

from . import WP4  # noqa: F401  (puts wp4_detection on sys.path)
from wp4.detectors import SB_LEN, BlockDetector


def _lags(x: np.ndarray, L: int) -> np.ndarray:
    """Biased sample autocorrelation lambda(l) = 1/N sum_n x[n] x*[n-l], l = 0..L-1. x (..., N)."""
    N = x.shape[-1]
    return np.stack([np.sum(x[..., l:] * np.conj(x[..., : N - l]), axis=-1) / N for l in range(L)], axis=-1)


class CAV(BlockDetector):
    """Covariance absolute value detector (Zeng & Liang, IEEE TVT 2009).

    R = L x L sample covariance (Toeplitz, smoothing factor L).
    T1 = 1/L sum_nm |r_nm|, T2 = 1/L sum_n |r_nn|; statistic T1/T2.
    White noise: R ~ sigma^2 I, so T1/T2 -> 1. Any correlation between nearby
    samples (narrowband / oversampled signal) raises T1. With the Toeplitz form
    T1/T2 = 1 + 2/L sum_{l=1}^{L-1} (L - l) |lambda(l)| / lambda(0).
    """
    name = "cav"

    def __init__(self, L=8):
        self.L = L
        self.name = "cav" if L == 8 else f"cav_L{L}"

    def block_stat(self, x):
        lam = _lags(x, self.L)
        l = np.arange(1, self.L)
        return 1 + 2 / self.L * np.sum((self.L - l) * np.abs(lam[..., 1:]), axis=-1) / np.maximum(lam[..., 0].real, 1e-30)


class AutoCorr(BlockDetector):
    """Normalised autocorrelation detector: Q = N * sum_{l=1}^{L} |rho(l)|^2,
    rho(l) = lambda(l) / lambda(0). Under white noise each N|rho(l)|^2 ~ Exp(1)
    (Ljung-Box type portmanteau statistic), so Q ~ Gamma(L, 1)."""
    name = "acorr"

    def __init__(self, L=8):
        self.L = L
        self.name = "acorr" if L == 8 else f"acorr_L{L}"

    def block_stat(self, x):
        lam = _lags(x, self.L + 1)
        rho = lam[..., 1:] / np.maximum(lam[..., :1].real, 1e-30)
        return x.shape[-1] * np.sum(np.abs(rho) ** 2, axis=-1)


class SpectralEntropy(BlockDetector):
    """Spectral entropy deficit: Welch PSD (Hann, non-overlapping segments of
    length m) normalised to a pmf p_k; statistic = log(m) - H(p).
    White noise has a flat expected PSD (maximum entropy up to estimation noise);
    a signal that shapes the spectrum inside the sub-band lowers H.
    A signal that is flat across the whole sub-band (wider than fs/16) is
    invisible - by construction, not a bug."""
    name = "entropy_spec"

    def __init__(self, m=64):
        self.m = m
        self.name = "entropy_spec" if m == 64 else f"entropy_spec_m{m}"
        self.w = np.hanning(m)

    def block_stat(self, x):
        n = (x.shape[-1] // self.m) * self.m
        seg = x[..., :n].reshape(*x.shape[:-1], -1, self.m)
        P = np.mean(np.abs(np.fft.fft(seg * self.w, axis=-1)) ** 2, axis=-2)
        p = P / np.maximum(P.sum(-1, keepdims=True), 1e-30)
        H = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=-1)
        return np.log(self.m) - H


class HistEntropy(BlockDetector):
    """Zhang et al. (IET Commun. 2010) entropy detector: FFT of the block,
    histogram of |X_k| with `bins` equal-width bins between min and max, Shannon
    entropy of the histogram. Statistic = log(bins) - H (larger = signal).
    Under noise the magnitude law is Rayleigh regardless of sigma, so the entropy
    distribution is noise-power invariant."""
    name = "entropy_hist"

    def __init__(self, bins=16):
        self.bins = bins
        self.name = "entropy_hist" if bins == 16 else f"entropy_hist_b{bins}"

    def block_stat(self, x):
        a = np.abs(np.fft.fft(x, axis=-1))
        lo, hi = a.min(-1, keepdims=True), a.max(-1, keepdims=True)
        idx = np.minimum(((a - lo) / np.maximum(hi - lo, 1e-30) * self.bins).astype(np.int64), self.bins - 1)
        flat = idx.reshape(-1, a.shape[-1]) + self.bins * np.arange(idx.size // a.shape[-1])[:, None]
        cnt = np.bincount(flat.ravel(), minlength=self.bins * (idx.size // a.shape[-1])).reshape(-1, self.bins)
        p = cnt / a.shape[-1]
        H = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=-1)
        return (np.log(self.bins) - H).reshape(a.shape[:-1])


# ------------------------------------------------------------------ FAM spectral correlation

def fam(x: np.ndarray, Np: int = 32, hop: int | None = None, Pp: int | None = None, conj: bool = False,
        q_all: bool = False):
    """FFT Accumulation Method estimate of the spectral coherence.

    x: (..., N) complex. Channeliser: Hamming window of Np samples, hop `hop`
    (default Np/4), P frames. Complex demodulates X(p, k) are phase-corrected to
    absolute time, products X_k X_l^* (or X_k X_l for the conjugate SCD) are
    Fourier-transformed over p (length Pp >= P, rectangular window) and only the
    central |q| <= Pp*hop/(2*Np) cyclic bins of each channel pair are kept
    (the FAM 'valid' region; Roberts et al. 1991).

    Returns (C, alpha, f): C (..., Np, Np, Q) = |S|^2 / (P_k P_l), the squared
    spectral coherence magnitude; alpha and f (Np, Np, Q) in cycles/sample.
    Non-conjugate: alpha = f_k - f_l + q/(Pp hop), f = (f_k + f_l)/2.
    Conjugate:     alpha = f_k + f_l + q/(Pp hop), f = (f_k - f_l)/2.
    """
    hop = hop or Np // 4
    N = x.shape[-1]
    P = (N - Np) // hop + 1
    Pp = Pp or int(2 ** np.ceil(np.log2(P)))
    idx = np.arange(P)[:, None] * hop + np.arange(Np)[None, :]
    fr = x[..., idx] * np.hamming(Np)                                   # (..., P, Np)
    X = np.fft.fftshift(np.fft.fft(fr, axis=-1), axes=-1)
    fk = (np.arange(Np) - Np // 2) / Np
    X = X * np.exp(-2j * np.pi * np.outer(np.arange(P) * hop, fk))      # absolute-time phase
    pw = np.mean(np.abs(X) ** 2, axis=-2)                               # (..., Np)
    qmax = Pp // 2 if q_all else int(Pp * hop // (2 * Np))
    q = np.arange(-qmax, qmax + 1)
    D = np.exp(-2j * np.pi * np.outer(np.arange(P), q) / Pp) / P        # (P, Q)
    Y = np.conj(X) if not conj else X
    Z = X[..., :, :, None] * Y[..., :, None, :]                         # (..., P, Np, Np)
    sh = Z.shape
    S = (np.swapaxes(Z.reshape(*sh[:-2], Np * Np), -1, -2) @ D).reshape(*sh[:-3], Np, Np, len(q))  # = einsum("...pkl,pq->...klq")
    C = np.abs(S) ** 2 / np.maximum(pw[..., :, None, None] * pw[..., None, :, None], 1e-300)
    sgn = -1 if not conj else 1
    alpha = fk[:, None, None] + sgn * fk[None, :, None] + q[None, None, :] / (Pp * hop)
    f = (fk[:, None, None] - sgn * fk[None, :, None]) / 2 + 0 * q[None, None, :]
    return C, alpha, f


class SCDFam(BlockDetector):
    """Cyclostationary detector on the full spectral correlation density (FAM).

    Statistic per block = max over (f, alpha) of the squared spectral coherence,
    over both the non-conjugate SCD (alpha != 0: symbol rate, chip rate, ...) and
    the conjugate SCD (alpha = 2 f_c, 2 f_c +- R_s: BPSK, ASK, AM-DSB, MSK-type).
    Cells with |alpha| < min_alpha / N are excluded from the non-conjugate part
    (they are the PSD itself, plus slow gain drift). Compared with the existing
    `cyclo` detector (two lines, no frequency resolution), FAM resolves the
    feature in spectral frequency f, so a feature confined to a few frequency
    channels is not diluted by noise from the rest of the band.
    """
    name = "scd_fam"

    def __init__(self, Np=32, min_alpha=4, chunk=64):
        self.Np, self.min_alpha, self.chunk = Np, min_alpha, chunk
        self.name = "scd_fam" if Np == 32 else f"scd_fam_Np{Np}"

    def block_stat(self, x):
        shp = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        out = np.empty(len(xf))
        for s in range(0, len(xf), self.chunk):
            xb = xf[s: s + self.chunk]
            C, a, _ = fam(xb, self.Np)
            ok = np.abs(a) >= self.min_alpha / x.shape[-1]
            m1 = np.where(ok, C, 0).reshape(len(xb), -1).max(-1)
            Cc, _, _ = fam(xb, self.Np, conj=True)
            m2 = Cc.reshape(len(xb), -1).max(-1)
            out[s: s + len(xb)] = np.maximum(m1, m2)
        return out.reshape(shp)

    def h0_samples(self, rng, n_blocks: int = 200_000):
        out = []
        for _ in range(n_blocks // 10_000):
            x = (rng.standard_normal((10_000, SB_LEN)) + 1j * rng.standard_normal((10_000, SB_LEN))) / np.sqrt(2)
            out.append(self.block_stat(x))
        return np.concatenate(out)


def new_detectors(**over):
    dets = [CAV(**over.get("cav", {})), SpectralEntropy(**over.get("entropy_spec", {})),
            HistEntropy(**over.get("entropy_hist", {})), AutoCorr(**over.get("acorr", {})),
            SCDFam(**over.get("scd_fam", {}))]
    return {d.name: d for d in dets}


# ------------------------------------------------------------------ prewhitened variants

def whiten_subbands(x: np.ndarray, q: float = 0.25, smooth_bins: int = 9) -> np.ndarray:
    """Prewhiten (time blocks, sub-bands, n) blocks with a per-sub-band noise PSD
    shape estimated from the scene itself: q-quantile over time blocks of each
    block's periodogram (normalised to unit block power, so gain steps do not
    matter), smoothed over `smooth_bins` frequency bins. Blocks are divided by
    sqrt(shape) in the frequency domain. Rationale: the impaired TorchSig
    receiver filter colours the noise in the edge sub-bands (|rho1| ~ 0.4-0.5 vs
    0.04 for white noise), which every whiteness test flags as 'signal'.
    Cost: a signal present in > (1-q) of the time blocks of a sub-band is
    whitened away (persistent carriers), as is the DC artefact."""
    from scipy.ndimage import uniform_filter1d
    X = np.fft.fft(x, axis=-1)
    P = np.abs(X) ** 2
    # Quantile over time of the RAW periodograms (no per-block normalisation): a signal can only
    # raise a block's bins, so the low quantile stays a noise estimate; a gain step scales all
    # bins of a block alike and leaves the shape intact. (Two earlier versions normalised each
    # block by its mean / median bin first; a signal filling > half the sub-band then pushed the
    # quantile DOWN in its spectral nulls and the 'whitened' noise blocks became coloured -
    # free-block Pfa 0.08-0.10 instead of 1e-3 on clean dev scenes.)
    Pn = P
    shape = np.quantile(Pn, q, axis=0, keepdims=True)
    shape = uniform_filter1d(shape, smooth_bins, axis=-1, mode="wrap")
    return np.fft.ifft(X / np.sqrt(np.maximum(shape, 1e-12)), axis=-1)


class Whitened(BlockDetector):
    """Any block detector applied after scene-adaptive prewhitening.
    Calibrated on whole white-noise scenes through the same whitening step."""

    def __init__(self, base: BlockDetector, n_cal_scenes: int = 200):
        self.base, self.n_cal_scenes = base, n_cal_scenes
        self.name = base.name + "_pw"

    def block_stat(self, x):
        return self.base.block_stat(whiten_subbands(x))

    def h0_samples(self, rng, n_scenes=None):
        from wp4.detectors import N_IQ, subbands
        out = []
        for _ in range(n_scenes or self.n_cal_scenes):
            iq = ((rng.standard_normal(N_IQ) + 1j * rng.standard_normal(N_IQ)) / np.sqrt(2)).astype(np.complex64)
            out.append(self.block_stat(subbands(iq)).ravel())   # block level = cell level (each block -> 512 cells)
        return np.concatenate(out)
