"""OFDM cyclic-prefix (CP) autocorrelation detector vs energy detector.

OFDM symbols repeat their last L samples as a prefix, so x[n] and x[n + Nfft]
are correlated inside each CP and uncorrelated in noise. A detector built on
that correlation needs no knowledge of the noise power, unlike the energy
detector, which suffers an "SNR wall" when the noise power is only known to
within +-u dB (Tandra & Sahai 2008). This script shows both effects.

Frame: 4096 complex samples. Signal present in half the frames: OFDM with
Nfft = 64, CP = 16, QPSK subcarriers, filling the frame, random phase.
Noise power is either known exactly, or drawn uniformly in +-1 dB around the
nominal value (noise uncertainty) while the detectors assume the nominal value.

Detectors (threshold for Pfa = 1e-2 per frame from noise-only simulation):
    energy        total frame energy / nominal noise power
    cp_known      CP correlation at lag 64, combined over all symbol timings:
                  max over offset theta of |sum_m sum_{k<L} x[t] x*[t+N]| / sum |x|^2,
                  t = theta + m (N+L) + k (van de Beek / Sandell style metric)
    cp_blind      same, but Nfft and CP unknown: maximum over Nfft in {32, 64, 128, 256}
                  with CP = Nfft / 4 (thresholds calibrated for the max)

Outputs: results/ofdm_cp.json and .png
"""

from __future__ import annotations

import json
import os

import numpy as np

N = 4096
RNG = np.random.default_rng(1)
CONFIGS = [(32, 8), (64, 16), (128, 32), (256, 64)]


def ofdm(nfft=64, cp=16, rng=RNG):
    nsym = N // (nfft + cp) + 1
    qpsk = (rng.choice([-1, 1], (nsym, nfft)) + 1j * rng.choice([-1, 1], (nsym, nfft))) / np.sqrt(2)
    t = np.fft.ifft(qpsk, axis=1) * np.sqrt(nfft)
    s = np.concatenate([np.concatenate([row[-cp:], row]) for row in t])[:N]
    return s / np.sqrt(np.mean(np.abs(s) ** 2)) * np.exp(1j * rng.uniform(0, 2 * np.pi))


def frame(snr_db, present, uncertainty_db=0.0, rng=RNG):
    npow = 10 ** (rng.uniform(-uncertainty_db, uncertainty_db) / 10)
    x = np.sqrt(npow / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    if present:
        x = x + np.sqrt(10 ** (snr_db / 10)) * ofdm(rng=rng)
    return x


def stat_energy(x):
    return np.mean(np.abs(x) ** 2)  # nominal noise power = 1


def cp_metric(x, nfft, cp):
    p = x[:-nfft] * np.conj(x[nfft:])                 # lag-Nfft products
    period = nfft + cp
    n = (len(p) // period) * period
    folded = p[:n].reshape(-1, period).sum(0)          # sum over symbols, per offset within a period
    win = np.concatenate([folded, folded[:cp]])
    c = np.abs(np.convolve(win, np.ones(cp), "valid"))[:period]   # CP-length window at every timing
    return c.max() / np.sum(np.abs(x) ** 2)


def stat_cp_known(x):
    return cp_metric(x, 64, 16)


def stat_cp_blind(x):
    return max(cp_metric(x, n, c) for n, c in CONFIGS)


DET = {"energy": stat_energy, "cp_known": stat_cp_known, "cp_blind": stat_cp_blind}


def main(n_noise=3000, n_sig=300):
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)
    h0 = {k: np.array([f(frame(0, False)) for _ in range(n_noise)]) for k, f in DET.items()}
    thr = {k: np.quantile(v, 1 - 1e-2) for k, v in h0.items()}
    snrs = list(range(-20, 3, 2))
    res = {}
    for unc in (0.0, 1.0):
        pfa = {k: float(np.mean([f(frame(0, False, unc)) > thr[k] for _ in range(n_noise)])) for k, f in DET.items()}
        pd = {k: {s: float(np.mean([f(frame(s, True, unc)) > thr[k] for _ in range(n_sig)])) for s in snrs} for k, f in DET.items()}
        res[f"uncertainty_{unc:g}dB"] = {"measured_pfa": pfa, "pd_vs_snr": pd,
                                         "snr_for_pd90": {k: next((s for s in snrs if pd[k][s] >= 0.9), None) for k in DET}}
    # Theory check for the energy detector with known noise (Gaussian approximation, N samples):
    # threshold 1 + Q^-1(Pfa)/sqrt(N); under signal the mean is 1 + snr and the std (1 + snr)/sqrt(N).
    from scipy.optimize import brentq
    from scipy.stats import norm
    thr_t = 1 + norm.isf(1e-2) / np.sqrt(N)
    snr_t = brentq(lambda s: norm.sf((thr_t - (1 + s)) / ((1 + s) / np.sqrt(N))) - 0.9, 1e-4, 10)
    print(f"theory check: energy detector, known noise, Pd 90 % expected near {10 * np.log10(snr_t):.1f} dB")
    with open(os.path.join(out, "ofdm_cp.json"), "w") as f:
        json.dump({"target_pfa": 1e-2, "nfft": 64, "cp": 16, "results": res,
                   "theory_energy_known_noise_snr_for_pd90_db": round(float(10 * np.log10(snr_t)), 2)}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (key, r) in zip(axs, res.items()):
        for k in DET:
            ax.plot(snrs, [r["pd_vs_snr"][k][s] * 100 for s in snrs], marker="o",
                    label=f"{k} (Pfa {r['measured_pfa'][k]:.3f})")
        ax.set_title(f"Noise power {key.replace('_', ' ').replace('uncertainty', 'uncertainty +-')}", fontsize=9)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Pd (%)")
    fig.suptitle("OFDM (Nfft 64, CP 16): cyclic-prefix vs energy detection, target Pfa 1e-2", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ofdm_cp.png"), dpi=150)
    for key, r in res.items():
        print(key, "measured Pfa", {k: round(v, 3) for k, v in r["measured_pfa"].items()}, "Pd90 at", r["snr_for_pd90"])


if __name__ == "__main__":
    main()
