"""Classical WP4 detectors behind one interface, with thresholds calibrated on noise.

Every detector computes a test-statistic map R (T, F) on the 512 x 512 grid.
The threshold for a target Pfa is the (1 - Pfa) quantile of R under pure white
noise, measured once by simulation (`calibrate`). This gives every detector the
same false-alarm meaning without hand-derived formulas, which do not exist in
closed form for several of them (correlated Hann bins, ordered statistics,
eigenvalue ratios, maxima over cyclic frequencies).

Detectors
    energy      time-smoothed spectrogram / gain(t) x shape(f) noise floor (4.1)
    energy_mt   same on a multitaper (DPSS) spectrogram: lower variance per frame
    fcme        forward consecutive mean excision noise floor, per frequency bin
    lad         localisation algorithm based on double thresholding (hysteresis)
    cfar_ca     cell-averaging CFAR along frequency (reference for 4.2)
    cfar_go     greatest-of CFAR
    cfar_so     smallest-of CFAR
    cfar_os     ordered-statistic CFAR (75th percentile of the training cells)
    eigen_mme   max/min eigenvalue ratio per sub-band block (coarse grid)
    cyclo       cyclostationary feature detector per sub-band block (coarse grid)

Coarse detectors decide per block of 16 frames x 32 bins (32 x 16 blocks), so
their boxes are blocky by construction; compare them on emitter Pd vs Pfa.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage
from scipy.signal.windows import dpss

from .energy import effective_looks, noise_floor, smooth
from .spectrum import spectrogram

N_IQ = 512 * 512
FFT = 512


class Detector:
    name = "base"
    coarse = False

    def stat(self, spec: np.ndarray, iq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def h0_samples(self, rng, n_scenes: int = 8) -> np.ndarray:
        out = []
        for _ in range(n_scenes):
            iq = ((rng.standard_normal(N_IQ) + 1j * rng.standard_normal(N_IQ)) / np.sqrt(2)).astype(np.complex64)
            out.append(self.stat(spectrogram(iq, FFT), iq).ravel())
        return np.concatenate(out)

    def mask(self, R: np.ndarray, pfa: float) -> np.ndarray:
        return R > self.thr[pfa]

    def calibrate(self, pfas, cache_dir: str, seed: int = 12345):
        path = os.path.join(cache_dir, f"{self.name}.npz")
        pfas = sorted(pfas)
        if os.path.exists(path):
            d = np.load(path)
            if all(p in d["pfas"] for p in pfas):
                self.thr = {float(p): float(t) for p, t in zip(d["pfas"], d["thr"])}
                return
        h0 = self.h0_samples(np.random.default_rng(seed))
        self.thr = {p: float(np.quantile(h0, 1 - p)) for p in pfas}
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, pfas=np.array(pfas), thr=np.array([self.thr[p] for p in pfas]), n_h0=len(h0))

    def masks(self, spec, iq, pfas):
        R = self.stat(spec, iq)
        return {p: self.mask(R, p) for p in pfas}


# ------------------------------------------------------------------ energy family

class Energy(Detector):
    name = "energy"

    def __init__(self, kt=8, kf=1, noise_method="excision"):
        self.kt, self.kf, self.noise_method = kt, kf, noise_method

    def _spec(self, spec, iq):
        return spec

    def stat(self, spec, iq):
        s = smooth(self._spec(spec, iq), self.kt, self.kf)
        looks = effective_looks(self.kt, self.kf, s.shape[1])
        return s / noise_floor(s, looks, self.noise_method)


class EnergyMultitaper(Energy):
    """Thomson multitaper: K orthogonal DPSS tapers per frame, averaged."""
    name = "energy_mt"

    def __init__(self, nw=2.5, k=4, **kw):
        super().__init__(**kw)
        self.tapers = dpss(FFT, nw, k)  # (k, FFT), unit energy each

    def _spec(self, spec, iq):
        frames = iq[: (len(iq) // FFT) * FFT].reshape(-1, FFT).astype(np.complex128)
        p = np.zeros(frames.shape)
        for w in self.tapers:
            p += np.abs(np.fft.fft(frames * w, axis=1)) ** 2
        return np.fft.fftshift(p / len(self.tapers), axes=1)


class FCME(Detector):
    """Forward consecutive mean excision per frequency bin (Saarnisaari et al.).

    Start from the smallest 10 % of a bin's time samples, then repeatedly keep
    every sample below T_cme x (mean of the kept set) until the set stops
    changing. The kept-set mean is the noise estimate for that bin.
    """
    name = "fcme"

    def __init__(self, kt=8, t_cme=2.0, iterations=10):
        self.kt, self.t_cme, self.iterations = kt, t_cme, iterations

    def stat(self, spec, iq):
        s = smooth(spec, self.kt, 1)
        start = np.sort(s, axis=0)[: max(1, s.shape[0] // 10)].mean(0)
        est = start
        for _ in range(self.iterations):
            keep = s < self.t_cme * est
            new = (s * keep).sum(0) / np.maximum(keep.sum(0), 1)
            if np.allclose(new, est):
                break
            est = np.where(keep.sum(0) > 0, new, est)
        return s / np.maximum(est, 1e-30)


class LAD(Energy):
    """Double threshold: keep connected regions above a LOW threshold that
    contain at least one cell above the HIGH (target-Pfa) threshold."""
    name = "lad"

    def __init__(self, pfa_low=2e-2, **kw):
        super().__init__(**kw)
        self.pfa_low = pfa_low

    def calibrate(self, pfas, cache_dir, seed=12345):
        super().calibrate(sorted(set(pfas) | {self.pfa_low}), cache_dir, seed)

    def mask(self, R, pfa):
        high = R > self.thr[pfa]
        low = R > min(self.thr[self.pfa_low], self.thr[pfa])
        lab, n = ndimage.label(low)
        if n == 0:
            return high
        keep = np.unique(lab[high])
        return np.isin(lab, keep[keep > 0])


# ------------------------------------------------------------------ CFAR family

class CFAR(Detector):
    """1-D CFAR along frequency on the time-smoothed spectrogram.

    Training cells: `train` bins on each side, separated from the cell under
    test by `guard` bins. Statistic = cell / noise estimate from the training cells.

    Guard/train chosen on dev scenes 0-99 (impaired). Emitters here are 5-64 bins
    wide, so a narrow guard puts the training cells inside the signal and CFAR
    masks itself: guard 4 / train 16 found 5 % of emitters (CA) vs 21-37 % with
    guard 16-40. guard 16 / train 32 gave the best box F1 (CA 29 %, OS 42 %).
    """

    def __init__(self, kind="ca", kt=8, guard=16, train=32, os_rank=0.75):
        self.kind, self.kt, self.guard, self.train, self.os_rank = kind, kt, guard, train, os_rank
        self.name = f"cfar_{kind}"

    def stat(self, spec, iq):
        s = smooth(spec, self.kt, 1)
        g, n = self.guard, self.train
        pad = g + n
        sp = np.pad(s, ((0, 0), (pad, pad)), mode="reflect")
        F = s.shape[1]
        if self.kind == "os":
            cells = [sp[:, pad - g - k: pad - g - k + F] for k in range(1, n + 1)] + \
                    [sp[:, pad + g + k: pad + g + k + F] for k in range(1, n + 1)]
            stack = np.stack(cells)
            kth = int(self.os_rank * (2 * n)) - 1
            est = np.partition(stack, kth, axis=0)[kth]
        else:
            c = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(sp, axis=1)], axis=1)
            idx = np.arange(F) + pad
            left = (c[:, idx - g] - c[:, idx - g - n]) / n
            right = (c[:, idx + g + 1 + n] - c[:, idx + g + 1]) / n
            est = {"ca": (left + right) / 2, "go": np.maximum(left, right), "so": np.minimum(left, right)}[self.kind]
        return s / np.maximum(est, 1e-30)


# ------------------------------------------------------------------ block (coarse) detectors

N_TB, N_SB = 32, 16              # time blocks, sub-bands
BLK = N_IQ // N_TB               # 8192 samples per time block
SB_LEN = BLK // N_SB             # 512 samples per sub-band after decimation


def subbands(iq: np.ndarray) -> np.ndarray:
    """(32 time blocks, 16 sub-bands, 512) complex: ideal channelisation by FFT masking."""
    X = np.fft.fftshift(np.fft.fft(iq[: N_TB * BLK].reshape(N_TB, BLK).astype(np.complex128), axis=1), axes=1)
    X = X.reshape(N_TB, N_SB, SB_LEN)
    return np.fft.ifft(np.fft.ifftshift(X, axes=2), axis=2)


def to_grid(block_stat: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(block_stat, 512 // N_TB, axis=0), 512 // N_SB, axis=1)


class BlockDetector(Detector):
    coarse = True

    def block_stat(self, x: np.ndarray) -> np.ndarray:
        """x: (..., 512) complex sub-band blocks -> (...) statistic."""
        raise NotImplementedError

    def stat(self, spec, iq):
        return to_grid(self.block_stat(subbands(iq)))

    def h0_samples(self, rng, n_blocks: int = 200_000):
        out = []
        for _ in range(n_blocks // 20_000):
            x = (rng.standard_normal((20_000, SB_LEN)) + 1j * rng.standard_normal((20_000, SB_LEN))) / np.sqrt(2)
            out.append(self.block_stat(x))
        return np.concatenate(out)


class EigenMME(BlockDetector):
    """Maximum-minimum eigenvalue detector (Zeng & Liang 2009) with smoothing
    factor L: eigenvalues of the L x L sample covariance of the sub-band block.
    White noise -> all eigenvalues ~equal; any correlated signal -> ratio grows."""
    name = "eigen_mme"

    def __init__(self, L=8):
        self.L = L

    def block_stat(self, x):
        L = self.L
        n = x.shape[-1] - L + 1
        H = np.stack([x[..., i: i + n] for i in range(L)], axis=-2)        # (..., L, n)
        C = H @ np.conj(np.swapaxes(H, -1, -2)) / n                        # (..., L, L)
        ev = np.linalg.eigvalsh(C)
        return ev[..., -1] / np.maximum(ev[..., 0], 1e-30)


class Cyclo(BlockDetector):
    """Cyclostationary feature detector: second-order cyclic features.

    Man-made signals have periodic statistics (symbol rate, carrier), noise does
    not. Two cheap cyclic tests per sub-band block, normalised so each bin is
    Exp(1) under white noise; the statistic is the strongest feature:
      * |FFT(|x|^2 - mean)|^2   envelope cyclic frequencies (symbol rate of
        pulse-shaped linear modulations); |alpha| < 4 bins excluded (gain drift, AM)
      * |FFT(x^2)|^2            conjugate cyclic frequency at 2 x carrier offset
        (BPSK, ASK, MSK/GMSK, AM-DSB)
    Constant-envelope signals without x^2 lines (FM, M-FSK, M-PSK with M>2 at
    raw rate) are invisible to these two features: a known limitation.
    """
    name = "cyclo"

    def __init__(self, min_alpha=4):
        self.min_alpha = min_alpha

    def block_stat(self, x):
        n = x.shape[-1]
        p = np.abs(x) ** 2
        pc = p - p.mean(-1, keepdims=True)
        env = np.abs(np.fft.fft(pc, axis=-1)) ** 2 / (n * np.maximum(pc.var(-1, keepdims=True), 1e-30))
        k = np.fft.fftfreq(n) * n
        env = env[..., np.abs(k) >= self.min_alpha].max(-1)
        x2 = x * x
        conj = np.abs(np.fft.fft(x2, axis=-1)) ** 2 / (n * np.maximum((p * p).mean(-1, keepdims=True), 1e-30))
        return np.maximum(env, conj.max(-1))


class CFAR2D(Detector):
    """2-D cell-averaging CFAR: rectangular training ring in time AND frequency
    around a guard rectangle. Uses time context too, so a signal that is wide in
    frequency but short in time still has noise cells above/below it."""
    name = "cfar_2d"

    def __init__(self, kt=8, guard=(24, 16), train=(16, 32)):
        self.kt, self.guard, self.train = kt, guard, train

    def stat(self, spec, iq):
        s = smooth(spec, self.kt, 1)
        gt, gf = self.guard
        tt, tf = self.train
        outer = (2 * (gt + tt) + 1, 2 * (gf + tf) + 1)
        inner = (2 * gt + 1, 2 * gf + 1)
        so = ndimage.uniform_filter(s, outer, mode="reflect") * (outer[0] * outer[1])
        si = ndimage.uniform_filter(s, inner, mode="reflect") * (inner[0] * inner[1])
        est = (so - si) / (outer[0] * outer[1] - inner[0] * inner[1])
        return s / np.maximum(est, 1e-30)


class PSDOccupancy(Energy):
    """Frequency-only spectrum occupancy: Welch PSD over the whole capture vs the
    noise shape, broadcast over time. The classic 'is this channel used' test;
    it has no time resolution, so every detection is a full-length stripe."""
    name = "psd_occupancy"

    def stat(self, spec, iq):
        R = super().stat(spec, iq)  # time-smoothed, noise-normalised
        return np.broadcast_to(R.mean(0, keepdims=True), R.shape).copy()


class SpectralKurtosis(Detector):
    """Spectral kurtosis (Antoni 2006) of complex STFT coefficients per bin over a
    sliding window of `win` frames. Complex Gaussian noise has SK = 0; constant-
    envelope or bursty components move it away from 0. Statistic = |SK|.
    Gaussian-like signals (OFDM, dense QAM at the STFT output) stay near 0: a
    detector of non-Gaussianity, not of power."""
    name = "spectral_kurtosis"

    def __init__(self, win=16):
        self.win = win

    def stat(self, spec, iq):
        frames = iq[: (len(iq) // FFT) * FFT].reshape(-1, FFT).astype(np.complex128)
        X = np.fft.fftshift(np.fft.fft(frames * np.hanning(FFT), axis=1), axes=1)
        p2 = np.abs(X) ** 2
        m2 = ndimage.uniform_filter1d(p2, self.win, axis=0, mode="nearest")
        m4 = ndimage.uniform_filter1d(p2**2, self.win, axis=0, mode="nearest")
        return np.abs(m4 / np.maximum(m2**2, 1e-30) - 2)


class GoFAnderson(BlockDetector):
    """Goodness-of-fit detector (Wang et al. 2009): Anderson-Darling test of the
    sub-band block's |x|^2 / s^2 against Exp(1), the noise-only distribution.
    s^2 is the sub-band's noise power, estimated as a low quantile of block
    powers across the scene (most blocks are noise)."""
    name = "gof_ad"

    @staticmethod
    def _ad(u):
        """Anderson-Darling A^2 of samples u (..., n) against Exp(1)."""
        n = u.shape[-1]
        z = np.sort(u, axis=-1)
        F = np.clip(1 - np.exp(-z), 1e-12, 1 - 1e-12)
        i = np.arange(1, n + 1)
        return -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[..., ::-1])), axis=-1)

    def block_stat(self, x):
        """x: (time blocks, sub-bands, n). Noise power per sub-band = 25th
        percentile of its block powers, de-biased by the Gamma(n)/n quantile."""
        from scipy.special import gammaincinv
        p = np.abs(x) ** 2
        n = p.shape[-1]
        s2 = np.quantile(p.mean(-1), 0.25, axis=0, keepdims=True)[..., None] / (gammaincinv(n, 0.25) / n)
        return self._ad(p / s2)

    def h0_samples(self, rng, n_scenes: int = 40):
        # Calibrate through the same noise-power estimate used on scenes.
        return Detector.h0_samples(self, rng, n_scenes)


def all_detectors() -> dict[str, Detector]:
    dets = [Energy(), EnergyMultitaper(), FCME(), LAD(), PSDOccupancy(),
            CFAR("ca"), CFAR("go"), CFAR("so"), CFAR("os"), CFAR2D(),
            SpectralKurtosis(), EigenMME(), Cyclo(), GoFAnderson()]
    return {d.name: d for d in dets}
