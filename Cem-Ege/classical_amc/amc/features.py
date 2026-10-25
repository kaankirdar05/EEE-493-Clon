"""Hand-crafted features for classical automatic modulation classification (job 5.1).

Three families, all computed per frame and vectorised over a batch:

1. Higher-order cumulants (Swami & Sadler 2000). After normalising each frame
   to zero mean and unit power (C21 = 1), |C40|, |C42|, |C60|, |C80| take
   known theoretical values per constellation, e.g. |C40| = 2 for BPSK,
   1 for QPSK, 0 for 8PSK, 0.68 for 16QAM.
2. Power-law spectral lines. Raising an M-PSK signal to the M-th power strips
   the modulation and leaves a tone at M times the carrier offset. The
   peak-to-mean ratio of |FFT(x^k)| for k = 1, 2, 4, 8 separates
   BPSK / QPSK / 8PSK even when a frequency offset smears C40.
3. Amplitude / instantaneous-frequency statistics (Azzouz & Nandi 1996),
   which cumulants handle poorly: they separate AM, FM and GMSK from the
   digital constellations.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    # cumulants (magnitudes, C21 normalised to 1)
    "abs_C20", "abs_C40", "abs_C41", "C42", "abs_C60", "abs_C80",
    # power-law spectral lines, log10(peak / mean)
    "line_x1", "line_x2", "line_x4", "line_x8",
    # amplitude
    "sigma_aa", "gamma_max", "kurt_amp",
    # instantaneous frequency (strong-sample gated)
    "sigma_f", "sigma_af", "kurt_f",
    # spectrum shape
    "spec_flatness", "bw99_frac", "spec_asym",
]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex128)
    x = x - x.mean(axis=1, keepdims=True)
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    return x / np.sqrt(np.maximum(p, 1e-20))


def _cumulants(x: np.ndarray) -> np.ndarray:
    """|C20|, |C40|, |C41|, C42, |C60|, |C80| for zero-mean, unit-power frames."""
    xc = np.conj(x)
    x2 = x * x
    m20 = x2.mean(1)
    m21 = (x * xc).real.mean(1)
    m40 = (x2 * x2).mean(1)
    m41 = (x2 * x * xc).mean(1)
    m42 = (x2 * xc * xc).real.mean(1)
    m60 = (x2 * x2 * x2).mean(1)
    m80 = (x2 * x2 * x2 * x2).mean(1)

    c40 = m40 - 3 * m20**2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21**2
    c60 = m60 - 15 * m20 * m40 + 30 * m20**3
    c80 = m80 - 28 * m60 * m20 - 35 * m40**2 + 420 * m40 * m20**2 - 630 * m20**4
    return np.stack(
        [np.abs(m20), np.abs(c40), np.abs(c41), c42, np.abs(c60), np.abs(c80)], axis=1
    )


def _spectral_lines(x: np.ndarray) -> np.ndarray:
    out = []
    win = np.hanning(x.shape[1])
    for k in (1, 2, 4, 8):
        p = np.abs(np.fft.fft((x**k) * win, axis=1)) ** 2
        out.append(np.log10(p.max(1) / np.maximum(p.mean(1), 1e-20)))
    return np.stack(out, axis=1)


def _kurtosis(v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Kurtosis along axis 1, optionally weighted by a 0/1 mask `w`."""
    if w is None:
        w = np.ones_like(v)
    n = np.maximum(w.sum(1), 1)
    mu = (v * w).sum(1) / n
    d = (v - mu[:, None]) * w
    var = (d**2).sum(1) / n
    return (d**4).sum(1) / n / np.maximum(var**2, 1e-20)


def _amplitude_and_frequency(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    acn = a / np.maximum(a.mean(1, keepdims=True), 1e-20) - 1  # normalised-centred amplitude
    sigma_aa = acn.std(1)
    gamma_max = (np.abs(np.fft.fft(acn, axis=1)) ** 2).max(1) / x.shape[1]
    kurt_amp = _kurtosis(a)

    # Instantaneous frequency from consecutive-sample phase difference, only on
    # samples above the mean amplitude: phase of weak samples is mostly noise.
    f = np.angle(x[:, 1:] * np.conj(x[:, :-1]))
    strong = ((a[:, 1:] > a.mean(1, keepdims=True)) & (a[:, :-1] > a.mean(1, keepdims=True)))
    strong = strong.astype(np.float64)
    n = np.maximum(strong.sum(1), 1)
    fc = f - ((f * strong).sum(1) / n)[:, None]
    sigma_f = np.sqrt(((fc**2) * strong).sum(1) / n)
    af = np.abs(fc)
    sigma_af = np.sqrt((((af - (af * strong).sum(1, keepdims=True) / n[:, None]) ** 2) * strong).sum(1) / n)
    kurt_f = _kurtosis(f, strong)
    return np.stack([sigma_aa, gamma_max, kurt_amp, sigma_f, sigma_af, kurt_f], axis=1)


def _spectrum_shape(x: np.ndarray) -> np.ndarray:
    p = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(x.shape[1]), axis=1), axes=1)) ** 2
    p = p + 1e-20
    flatness = np.exp(np.log(p).mean(1)) / p.mean(1)

    # Fraction of the band between the 0.5 % and 99.5 % points of the power CDF.
    cdf = np.cumsum(p, axis=1) / p.sum(1, keepdims=True)
    lo = np.argmax(cdf >= 0.005, axis=1)
    hi = np.argmax(cdf >= 0.995, axis=1)
    bw99 = (hi - lo + 1) / x.shape[1]

    half = x.shape[1] // 2
    lower, upper = p[:, :half].sum(1), p[:, half:].sum(1)
    asym = np.abs(lower - upper) / (lower + upper)  # ~1 for SSB, ~0 for DSB / PSK
    return np.stack([flatness, bw99, asym], axis=1)


def extract_features(x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """(N, L) complex frames -> (N, len(FEATURE_NAMES)) float32 features."""
    out = []
    for start in range(0, len(x), batch):
        xb = _normalize(x[start:start + batch])
        out.append(np.concatenate([
            _cumulants(xb),
            _spectral_lines(xb),
            _amplitude_and_frequency(xb),
            _spectrum_shape(xb),
        ], axis=1))
    feats = np.concatenate(out, axis=0).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
