"""Job 4.1 - noise-floor estimate + energy detector on the time-frequency grid.

Model: under noise only, each spectrogram cell is s^2 * Exp(1). Averaging L
cells (Kt frames x Kf bins) gives s^2/L * Gamma(L, 1). So for a target
false-alarm rate Pfa the threshold is

    threshold = s^2 * gammaincinv(L, 1 - Pfa) / L

and the only unknown is the noise power s^2, estimated robustly from the
spectrogram itself (most cells are noise in a typical scene).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from scipy.special import gammaincinv


def effective_looks(kt: int, kf: int, fft_size: int = 512) -> float:
    """Independent looks in a kt x kf average.

    Time frames do not overlap, so they are independent. Adjacent Hann-windowed
    frequency bins are correlated (power correlation r_d at bin distance d), so
    kf bins are worth only kf^2 / (kf + 2 * sum_d (kf - d) r_d) looks.
    The smoothed cell is then approximated as Gamma(L_eff).
    """
    w2 = np.hanning(fft_size) ** 2
    n = np.arange(fft_size)
    r = [abs(np.sum(w2 * np.exp(-2j * np.pi * n * d / fft_size)) / np.sum(w2)) ** 2 for d in range(1, kf)]
    denom = kf + 2 * sum((kf - d) * r[d - 1] for d in range(1, kf))
    return kt * kf**2 / denom


def smooth(spec: np.ndarray, kt: int = 8, kf: int = 1) -> np.ndarray:
    """Moving average over kt time frames x kf frequency bins."""
    if kt == 1 and kf == 1:
        return spec
    return uniform_filter(spec, size=(kt, kf), mode="nearest")


def noise_floor(spec_smoothed: np.ndarray, looks: float, method: str = "separable", q: float = 0.25,
                freq_window: int = 65, time_window: int = 5, iterations: int = 2):
    """Estimate per-cell noise power s^2 from an L-look smoothed spectrogram.

    separable:  noise = gain(t) x shape(f), shape (T, F). shape(f) as in per_bin;
        gain(t) is the q-quantile across frequency of spec / shape(f) in each
        frame, lightly median-filtered over `time_window` frames. Alternating the
        two estimates `iterations` times handles receiver gain steps / AGC
        (TorchSig CoarseGainChange, DigitalAGC) on top of a non-flat passband.
    per_bin:    noise power per frequency bin, shape (F,). For each bin take the
        q-quantile over time, then a running median across `freq_window` bins so
        a signal that is on for the whole capture in a few bins is not taken as
        noise. Needed whenever the receiver passband is not flat (real SDRs,
        TorchSig impairment levels 1-2).
    percentile: one scalar for the whole grid, the q-quantile of all cells.
        Correct only for white noise.
    median:     scalar, q = 0.5 (simplest, biased upward in busy scenes).

    Quantiles are divided by the q-quantile of Gamma(L, 1)/L to turn them into
    the mean noise power.
    """
    if method == "median":
        q = 0.5
    scale = gammaincinv(looks, q) / looks
    if method in ("percentile", "median"):
        return float(np.quantile(spec_smoothed, q) / scale)

    def shape_f(s):
        return median_filter(np.quantile(s, q, axis=0) / scale, size=freq_window, mode="nearest")

    if method == "per_bin":
        return shape_f(spec_smoothed)
    if method == "separable":
        tiny = 1e-12 * max(float(spec_smoothed.mean()), 1e-30)
        nf = np.maximum(shape_f(spec_smoothed), tiny)
        gain = np.ones(spec_smoothed.shape[0])
        for _ in range(iterations):
            gain = np.quantile(spec_smoothed / nf, q, axis=1) / scale
            gain = np.maximum(median_filter(gain, size=time_window, mode="nearest"), 1e-12)
            nf = np.maximum(shape_f(spec_smoothed / gain[:, None]), tiny)
        return gain[:, None] * nf[None, :]
    if method == "excision":
        return _excision_floor(spec_smoothed, looks, noise_method_init=noise_floor(
            spec_smoothed, looks, "separable", q, freq_window, time_window, iterations), time_window=time_window)
    raise ValueError(method)


def _excision_floor(s: np.ndarray, looks: float, noise_method_init: np.ndarray, pfa_excise: float = 1e-2,
                    guard: tuple[int, int] = (9, 5), iterations: int = 3, time_window: int = 5,
                    min_blob: int = 40) -> np.ndarray:
    """Separable noise floor re-estimated from noise-only cells.

    Start from the quantile-based separable estimate, flag cells above a loose
    threshold (Pfa 1e-2), grow the flags by `guard` cells to drop signal skirts,
    then re-fit shape(f) and gain(t) as means over the remaining cells. Bins or
    frames with too few noise cells are interpolated from their neighbours. This
    stops wide, long-lasting signals from raising the floor under themselves.
    """
    from scipy.ndimage import binary_dilation, label, sum as region_sum

    noise = noise_method_init
    k = gammaincinv(looks, 1 - pfa_excise) / looks
    for _ in range(iterations):
        # Excise only blobs big enough to be signals: removing isolated noise
        # peaks too would bias the floor low and break the Pfa calibration.
        flags = s > noise * k
        lab, n = label(flags)
        if n:
            big = np.flatnonzero(region_sum(flags, lab, np.arange(1, n + 1)) >= min_blob) + 1
            flags = np.isin(lab, big)
        keep = ~binary_dilation(flags, structure=np.ones(guard, bool))
        gain = noise.mean(axis=1) / noise.mean(axis=1).mean()
        r = s / gain[:, None]
        cnt_f = keep.sum(0)
        nf = np.where(cnt_f >= 16, (r * keep).sum(0) / np.maximum(cnt_f, 1), np.nan)
        nf = _fill_nan(nf)
        nf = median_filter(nf, size=9, mode="nearest")
        r2 = s / np.maximum(nf, 1e-30)[None, :]
        cnt_t = keep.sum(1)
        g = np.where(cnt_t >= 16, (r2 * keep).sum(1) / np.maximum(cnt_t, 1), np.nan)
        g = median_filter(_fill_nan(g), size=time_window, mode="nearest")
        noise = np.maximum(g[:, None] * nf[None, :], 1e-30)
    return noise


def _fill_nan(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.all() or not ok.any():
        return np.nan_to_num(v, nan=1.0)
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def energy_detector(spec: np.ndarray, pfa: float = 1e-3, kt: int = 8, kf: int = 1,
                    noise_method: str = "excision") -> tuple[np.ndarray, dict]:
    """Return a boolean (T, F) detection mask and diagnostics."""
    looks = effective_looks(kt, kf, spec.shape[1])
    s = smooth(spec, kt, kf)
    sigma2 = noise_floor(s, looks, noise_method)
    thr = sigma2 * gammaincinv(looks, 1 - pfa) / looks
    return s > thr, {"noise_power": sigma2, "threshold": thr, "looks": looks}
