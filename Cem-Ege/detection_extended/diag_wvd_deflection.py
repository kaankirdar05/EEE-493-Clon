"""Diagnostic for Part B: deflection (mean shift / H0 std) of the TRUE-line statistic (no grid
search) for the spectrogram, the PWVD with growing lag window, the SPWVD, and the dechirp
(= |FrFT|^2 at the right angle), at a fixed SNR.

Purpose: check that a WVD line integral approaches the dechirp/FrFT optimum as the lag window
grows to the full frame (Radon-Wigner identity) when every time sample is used, so a weak
short-lag PWVD result is a property of the estimator, not a bug; and quantify the loss from
summing the WVD only on a decimated (hop-16) time grid.
Line sums here are taken along the exact IF line at every sample (hop 1) unless noted.
Writes results/partB/diag_wvd_deflection.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext.chirp import chirp, pwvd, stft_power  # noqa: E402

N, SNR_DB, R = 4096, -20, 100
F0, K = 0.0625, 0.25 / N


def line_sum(W, hop, M):
    n_c = np.arange(hop // 2, N, hop) - N / 2
    b = np.rint((F0 + K * n_c) * M + M / 2).astype(int) % M
    return W[np.arange(len(n_c)), b].sum()


def stats(x):
    out = {"spectrogram_w128_hop16": line_sum(stft_power(x, 512, 16, 128), 16, 512)}
    for L, M in [(255, 512), (1023, 2048), (4095, 8192)]:
        lw = "rect" if L == 4095 else "hann"
        out[f"pwvd_L{L}_every_sample"] = line_sum(pwvd(x, M, 1, L, lag_win=lw, agg=False), 1, M)
        out[f"pwvd_L{L}_hop16_grid_only"] = line_sum(pwvd(x, M, 16, L, lag_win=lw, agg=False), 16, M)
    out["pwvd_L255_hop16_boxcar_agg (bench)"] = line_sum(pwvd(x, 512, 16, 255), 16, 512)
    out["spwvd_L255_g64 (bench)"] = line_sum(pwvd(x, 512, 16, 255, g_len=64), 16, 512)
    n = np.arange(N) - N / 2
    out["dechirp_true_k_f"] = abs(np.sum(x * np.exp(-1j * (2 * np.pi * F0 * n + np.pi * K * n**2)))) ** 2 / N
    return out


rng = np.random.default_rng(11)
h0, h1 = [], []
for _ in range(R):
    w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    h0.append(stats(w))
    h1.append(stats(w + chirp(N, F0, K, 10 ** (SNR_DB / 20), rng.uniform(0, 2 * np.pi))))
res = {}
for m in h0[0]:
    a, b = np.array([d[m] for d in h0]), np.array([d[m] for d in h1])
    res[m] = float((b.mean() - a.mean()) / a.std())
print(json.dumps(res, indent=1))
json.dump({"snr_db": SNR_DB, "trials": R, "theory_dechirp_deflection_N_snr": N * 10 ** (SNR_DB / 10),
           "deflection_true_line": res},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "partB", "diag_wvd_deflection.json"), "w"),
          indent=2)
