"""Teager-Kaiser energy operator (TKEO) detectors.

Complex TKEO (Hamila, Astola, Alaya Cheikh, Gabbouj & Renfors, EURASIP JASP 2003):
    Psi[x](n) = |x(n)|^2 - Re( x(n-1) x*(n+1) )
For a complex tone A exp(j w n): Psi = A^2 (1 - cos 2w) = 2 A^2 sin^2 w.
For complex white noise of power s^2: E Psi = s^2, Var Psi = 1.5 s^4, and Psi(n),
Psi(m) are uncorrelated for n != m. So TKEO is a quadratic detector with a fixed
frequency weighting (1 - cos 2w): it responds twice as strongly as energy at
w = +-pi/2 and not at all at w = 0 and w = pi.

    tkeo_stft  Psi along TIME of each STFT bin (non-overlapping Hann frames, as
               wp4.spectrum.spectrogram), then the same kt-frame smoothing and
               gain(t) x shape(f) excision noise floor as wp4 `energy`.
    tkeo_sb    Psi of each 512-sample sub-band block (time domain), block mean,
               divided by a separable scene estimate of the noise level made from
               the block means themselves.
"""

from __future__ import annotations

import numpy as np

from . import WP4  # noqa: F401
from wp4.detectors import FFT, BlockDetector, Detector  # noqa: E402
from wp4.energy import noise_floor, smooth  # noqa: E402

from .eigen import scene_h0, separable_noise  # noqa: E402


def tkeo(x: np.ndarray, axis: int = -1, edges: bool = True) -> np.ndarray:
    """Complex TKEO along `axis`. edges=True: same length, first/last sample = |x|^2;
    edges=False: length - 2 (only fully defined samples)."""
    x = np.moveaxis(x, axis, -1)
    core = np.abs(x[..., 1:-1]) ** 2 - np.real(x[..., :-2] * np.conj(x[..., 2:]))
    if edges:
        core = np.concatenate([np.abs(x[..., :1]) ** 2, core, np.abs(x[..., -1:]) ** 2], axis=-1)
    return np.moveaxis(core, -1, axis)


class TKEOSTFT(Detector):
    name = "tkeo_stft"

    def __init__(self, kt: int = 8):
        self.kt = kt
        self.name = "tkeo_stft" if kt == 8 else f"tkeo_stft_kt{kt}"

    def stat(self, spec, iq):
        n = (len(iq) // FFT) * FFT
        frames = iq[:n].reshape(-1, FFT).astype(np.complex128)
        w = np.hanning(FFT)
        X = np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1) / np.sqrt(np.sum(w ** 2))
        psi = tkeo(X, axis=0)                       # along time, per bin
        s = smooth(psi, self.kt, 1)
        looks = self.kt / 1.5                       # Var Psi = 1.5 s^4 per frame
        return s / noise_floor(np.maximum(s, 1e-12 * np.abs(s).mean()), looks, "excision")

    def h0_samples(self, rng, n_scenes: int = 16):
        return Detector.h0_samples(self, rng, n_scenes)


class TKEOSubband(BlockDetector):
    name = "tkeo_sb"

    def __init__(self, n_cal_scenes: int = 400):
        self.n_cal_scenes = n_cal_scenes

    def block_stat(self, x):
        m = tkeo(x, edges=False).mean(-1)
        return m / separable_noise(np.maximum(m, 1e-30))

    def h0_samples(self, rng, n_scenes=None):
        return scene_h0(self, rng, n_scenes or self.n_cal_scenes)
