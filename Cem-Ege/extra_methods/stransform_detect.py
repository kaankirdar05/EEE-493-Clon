"""S-transform (Stockwell) vs STFT for detecting short bursts.

The S-transform uses a Gaussian window whose width scales with 1/frequency, so
it resolves short events at high frequencies better than a fixed-window STFT.
Test: 4,096-sample frames of white noise; half contain a short tone burst
(64 samples, Hann envelope) at a random time and a random normalised frequency
in [0.1, 0.45]. Detector statistic for each representation: maximum of the map
normalised by the map's median (scale-free); threshold set for Pfa = 1e-2 per
frame on noise-only frames (simulated). Pd vs per-sample burst SNR.

Sanity check (--check): the S-transform of a pure tone peaks at the tone's
frequency row, and its time marginal is flat.

    python stransform_detect.py --check
    python stransform_detect.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

N = 4096
RNG = np.random.default_rng(7)
FREQS = np.arange(64, N // 2, 16)            # analysed frequency bins (k), 0.0156..0.5 cycles/sample


def stransform(x, freqs=FREQS):
    """Discrete S-transform rows for frequency bins k: S[k, t] = IFFT( X[m + k] * exp(-2 pi^2 m^2 / k^2) )."""
    X = np.fft.fft(x)
    m = np.fft.fftfreq(len(x)) * len(x)
    rows = []
    for k in freqs:
        g = np.exp(-2 * np.pi**2 * m**2 / k**2)
        rows.append(np.fft.ifft(np.roll(X, -k) * g))
    return np.abs(np.array(rows)) ** 2


def stft(x, nfft=256, hop=64):
    w = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop] * w
    return np.abs(np.fft.fft(frames, axis=1)) ** 2


def frame(snr_db, present, rng=RNG):
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    if present:
        L = 64
        t0 = int(rng.integers(0, N - L))
        f = rng.uniform(0.1, 0.45)
        b = np.hanning(L) * np.exp(2j * np.pi * f * np.arange(L) + 1j * rng.uniform(0, 6.28))
        b /= np.sqrt(np.mean(np.abs(b) ** 2))
        x[t0:t0 + L] += np.sqrt(10 ** (snr_db / 10)) * b
    return x


def stat(M):
    return M.max() / np.median(M)


DET = {"stft_256": lambda x: stat(stft(x, 256, 64)), "stft_64": lambda x: stat(stft(x, 64, 16)),
       "s_transform": lambda x: stat(stransform(x))}


def check():
    n = np.arange(N)
    S = stransform(np.exp(2j * np.pi * (512 / N) * n))
    row = FREQS[np.argmax(S.mean(1))]
    flat = S[np.argmax(S.mean(1))].std() / S[np.argmax(S.mean(1))].mean()
    print(f"S-transform check: tone at bin 512 -> peak row bin {row}; time-marginal variation {flat:.3f} (expect ~512 and ~0)")
    return row == 512 and flat < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    thr = {k: np.quantile([f(frame(0, False)) for _ in range(1500)], 0.99) for k, f in DET.items()}
    snrs = list(range(-6, 13, 2))
    pd = {k: {s: float(np.mean([f(frame(s, True)) > thr[k] for _ in range(200)])) for s in snrs} for k, f in DET.items()}
    pfa = {k: float(np.mean([f(frame(0, False)) > thr[k] for _ in range(1500)])) for k, f in DET.items()}
    res = {k: {"pd_vs_snr": pd[k], "measured_pfa": pfa[k], "snr_for_pd90": next((s for s in snrs if pd[k][s] >= 0.9), None)} for k in DET}
    for k, v in res.items():
        print(f"{k:>12s}: Pd 90 % at {v['snr_for_pd90']} dB, measured Pfa {v['measured_pfa']:.3f}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "stransform.json"), "w") as f:
        json.dump({"burst_len": 64, "target_pfa": 1e-2, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
