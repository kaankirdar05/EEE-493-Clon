"""Spectrogram and box <-> grid conversions shared by all WP4 detectors."""

from __future__ import annotations

import numpy as np


def spectrogram(iq: np.ndarray, fft_size: int = 512) -> np.ndarray:
    """Power spectrogram, shape (time frames, frequency bins), frequency fftshifted.

    Non-overlapping Hann-windowed frames, scaled so that complex white noise of
    power s^2 per sample gives cells with mean s^2. Under noise only each cell is
    then s^2 * Exp(1), which is what the detector thresholds assume.
    """
    n = (len(iq) // fft_size) * fft_size
    frames = iq[:n].reshape(-1, fft_size).astype(np.complex128)
    w = np.hanning(fft_size)
    spec = np.abs(np.fft.fft(frames * w, axis=1)) ** 2 / np.sum(w**2)
    return np.fft.fftshift(spec, axes=1)


def freq_to_bin(f_hz, sample_rate: float, n_bins: int):
    """Baseband frequency (-fs/2..fs/2) -> fractional bin index (0..n_bins)."""
    return (np.asarray(f_hz) + sample_rate / 2) / sample_rate * n_bins


def bin_to_freq(b, sample_rate: float, n_bins: int):
    return np.asarray(b) / n_bins * sample_rate - sample_rate / 2


def boxes_to_mask(boxes, shape, sample_rate: float) -> np.ndarray:
    """Rasterise truth boxes (t_start, t_stop in 0..1; f_low, f_high in Hz) onto the grid."""
    T, F = shape
    mask = np.zeros(shape, dtype=bool)
    for b in boxes:
        t0, t1 = int(np.floor(b["t_start"] * T)), int(np.ceil(b["t_stop"] * T))
        f0 = int(np.floor(freq_to_bin(b["f_low"], sample_rate, F)))
        f1 = int(np.ceil(freq_to_bin(b["f_high"], sample_rate, F)))
        mask[max(t0, 0):min(t1, T), max(f0, 0):min(f1, F)] = True
    return mask


def box_to_cells(b, shape, sample_rate: float) -> tuple[int, int, int, int]:
    """Truth box -> (t0, t1, f0, f1) half-open cell ranges, clipped to the grid."""
    T, F = shape
    t0, t1 = int(np.floor(b["t_start"] * T)), int(np.ceil(b["t_stop"] * T))
    f0 = int(np.floor(freq_to_bin(b["f_low"], sample_rate, F)))
    f1 = int(np.ceil(freq_to_bin(b["f_high"], sample_rate, F)))
    return max(t0, 0), min(t1, T), max(f0, 0), min(f1, F)
