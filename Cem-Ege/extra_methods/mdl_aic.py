"""MDL / AIC estimation of the number of signals (Wax & Kailath 1985), single channel.

With an antenna array, the number of sources is the number of covariance
eigenvalues above the noise floor. With one channel we build snapshots by time-
delay embedding (M consecutive samples) with forward-backward averaging; each
complex tone then occupies one dimension of the signal subspace, so MDL/AIC
estimate the number of tones (narrowband emitters) in the band.

Test: K = 0..4 complex tones with random, well-separated frequencies (min spacing
2/M), random phases and per-tone SNR, N = 1024 samples, M = 16, AWGN.
Score: fraction of trials where the estimate equals K.

    for k in 0..M-1:
        L(k)   = -(M - k) * n * log( geometric_mean(lambda_{k+1..M}) / arithmetic_mean(lambda_{k+1..M}) )
        AIC(k) = 2 L(k) + 2 k (2M - k)
        MDL(k) = L(k) + 0.5 k (2M - k) log n
    K_hat = argmin

Expected behaviour (theory): MDL is consistent (-> 100 % at high SNR), AIC tends
to overestimate; with overlapping (non-independent) embedded snapshots both
become optimistic about n, so a residual over-estimate at high SNR is expected.

Outputs: results/mdl_aic.json and .png
"""

from __future__ import annotations

import json
import os

import numpy as np

N, M = 1024, 16
RNG = np.random.default_rng(3)


def tones(k, snr_db, rng):
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    freqs = []
    while len(freqs) < k:
        f = rng.uniform(-0.5, 0.5)
        if all(min(abs(f - g), 1 - abs(f - g)) > 2 / M for g in freqs):
            freqs.append(f)
    for f in freqs:
        x += np.sqrt(10 ** (snr_db / 10)) * np.exp(1j * (2 * np.pi * f * np.arange(N) + rng.uniform(0, 2 * np.pi)))
    return x


def eigenvalues(x):
    n = N - M + 1
    X = np.stack([x[i:i + n] for i in range(M)])                 # (M, n) Hankel snapshots
    R = X @ X.conj().T / n
    J = np.eye(M)[::-1]
    R = 0.5 * (R + J @ R.conj() @ J)                              # forward-backward averaging
    return np.sort(np.linalg.eigvalsh(R))[::-1], n


def estimate(x):
    lam, n = eigenvalues(x)
    lam = np.maximum(lam, 1e-12)
    aic, mdl = [], []
    for k in range(M):
        tail = lam[k:]
        L = -(M - k) * n * (np.mean(np.log(tail)) - np.log(np.mean(tail)))
        aic.append(2 * L + 2 * k * (2 * M - k))
        mdl.append(L + 0.5 * k * (2 * M - k) * np.log(n))
    return int(np.argmin(aic)), int(np.argmin(mdl))


def main(trials=200):
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)
    snrs = list(range(-20, 11, 2))
    res = {"aic": {}, "mdl": {}}
    over = {"aic": {}, "mdl": {}}
    for s in snrs:
        hits = {"aic": 0, "mdl": 0}
        overs = {"aic": 0, "mdl": 0}
        tot = 0
        for k in range(5):
            for _ in range(trials // 5 if k else trials // 5):
                a, m = estimate(tones(k, s, RNG))
                hits["aic"] += a == k
                hits["mdl"] += m == k
                overs["aic"] += a > k
                overs["mdl"] += m > k
                tot += 1
        for c in res:
            res[c][s] = hits[c] / tot
            over[c][s] = overs[c] / tot
    noise_only = {c: float(np.mean([estimate(tones(0, 0, RNG))[i] == 0 for _ in range(300)])) for i, c in enumerate(["aic", "mdl"])}
    summary = {c: {"accuracy_vs_snr": res[c], "overestimate_rate_vs_snr": over[c],
                   "snr_for_90pct_correct": next((s for s in snrs if res[c][s] >= 0.9), None),
                   "correct_on_noise_only": noise_only[c]} for c in res}
    with open(os.path.join(out, "mdl_aic.json"), "w") as f:
        json.dump({"N": N, "M": M, "K_range": [0, 4], "summary": summary}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for c in res:
        ax.plot(snrs, [res[c][s] * 100 for s in snrs], marker="o", label=f"{c.upper()} correct")
        ax.plot(snrs, [over[c][s] * 100 for s in snrs], ls="--", label=f"{c.upper()} over-estimate")
    ax.set_xlabel("Per-tone SNR (dB)")
    ax.set_ylabel("% of trials")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Number of signals (0-4 tones), single channel, M = 16")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mdl_aic.png"), dpi=150)
    for c, v in summary.items():
        print(f"{c.upper()}: 90 % correct from {v['snr_for_90pct_correct']} dB; correct on noise only {v['correct_on_noise_only'] * 100:.0f} %; "
              f"over-estimate at +10 dB {v['overestimate_rate_vs_snr'][10] * 100:.0f} %")


if __name__ == "__main__":
    main()
