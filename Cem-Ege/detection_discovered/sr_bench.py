"""Stochastic-resonance pre-treated energy detection, in the setting of
Zheng, Li, Djouadi & Wang, arXiv:0906.0739:
real tone at 10 Hz sampled at fs = 100 Hz (0.1 cycles/sample), random phase, real
AWGN of unit variance, sensing window N = 512, FFT 256 (K = 2), bistable SR
dx/dt = a x - b x^3 + drive, a = b = 1, Euler step h per sample.

    python sr_bench.py tune     dev seeds: grid over input gain g, added noise D, step h
    python sr_bench.py run      held-out seeds: Pd vs SNR at Pfa 1e-2 (and 1e-1, the paper's Fig. 8)

SNR = (A^2 / 2) / sigma^2 (real tone). Detectors (thresholds = empirical (1 - Pfa)
quantile on 20000 noise-only windows, separate seed):
  radiometer          mean y^2                                  (plain energy detector)
  fftmax              paper metric M on the raw input           ("ordinary" detector of Fig. 7a)
  known_f             |DTFT of y at 10 Hz|^2 / N                 (quadrature matched filter, unknown phase:
                                                                  the optimal detector when f is known)
  sr_fftmax           paper metric M on the SR output            (the paper's SR detector)
  sr_energy           mean-removed output power of the SR system
  sr_known_f          |DTFT of SR output at 10 Hz|^2
SR parameters are tuned (generously, per detector family) on dev seeds at -15 dB.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disc.sr import bistable, energy_metric, fft_max_metric  # noqa: E402

OUT = os.path.join(HERE, "results", "sr_paper_setting")
N, F0, NFFT = 512, 0.1, 256
GRID = [(g, D, h) for g in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0) for D in (0.0, 0.01, 0.1, 0.43, 1.0, 3.0) for h in (0.01, 0.03, 0.1, 0.3)]
SNRS = np.arange(-30, 1, 1)
PAPER = (1.0, 0.43, 0.01)


def gen(rng, n, snr):
    A = np.sqrt(2 * 10 ** (snr / 10)) if snr is not None else 0.0
    ph = rng.uniform(0, 2 * np.pi, (n, 1))
    return A * np.cos(2 * np.pi * F0 * np.arange(N)[None, :] + ph) + rng.standard_normal((n, N))


def known_f(z):
    e = np.exp(-2j * np.pi * F0 * np.arange(N))
    zc = z - z.mean(-1, keepdims=True)
    return np.abs(zc @ e) ** 2 / N


def raw_stats(y):
    return {"radiometer": np.mean(y ** 2, -1), "fftmax": fft_max_metric(y, NFFT), "known_f": known_f(y)}


def sr_stats(y, p, rng):
    g, D, h = p
    z = bistable(g * y + np.sqrt(D) * rng.standard_normal(y.shape), h)
    return {"sr_fftmax": fft_max_metric(z, NFFT), "sr_energy": energy_metric(z), "sr_known_f": known_f(z)}


def cmd_tune():
    res = []
    for p in GRID:
        rng = np.random.default_rng(11)
        h0 = sr_stats(gen(rng, 4000, None), p, rng)
        rng = np.random.default_rng(12)
        h1 = sr_stats(gen(rng, 1000, -15), p, rng)
        r = {"g": p[0], "D": p[1], "h": p[2]}
        for k in h0:
            r[k] = float(np.mean(h1[k] > np.quantile(h0[k], 0.99)))
        res.append(r)
    best = {k: max(res, key=lambda r: r[k]) for k in ("sr_fftmax", "sr_energy", "sr_known_f")}
    # effect of the ADDED noise D at the best (g, h) of the paper's detector
    b = best["sr_fftmax"]
    d_curve = [r for r in res if r["g"] == b["g"] and r["h"] == b["h"]]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"grid": res, "best": best, "D_curve_at_best_g_h": d_curve}, open(os.path.join(OUT, "tuning_dev.json"), "w"), indent=2)
    print(json.dumps(best, indent=1))
    print("D sweep at best g,h:", [(r["D"], r["sr_fftmax"]) for r in d_curve])


def cmd_run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = json.load(open(os.path.join(OUT, "tuning_dev.json")))["best"]
    cfg = {k: (best[k]["g"], best[k]["D"], best[k]["h"]) for k in best}
    cfg["paper (g=1, D=0.43, h=0.01)"] = PAPER

    def all_stats(y, rng):
        out = raw_stats(y)
        for k, p in cfg.items():
            s = sr_stats(y, p, rng)
            if k.startswith("paper"):
                out["sr_fftmax, paper params"] = s["sr_fftmax"]
            else:
                out[k] = s[k]
        return out

    rng = np.random.default_rng(2027)
    h0 = [all_stats(gen(rng, 5000, None), rng) for _ in range(4)]
    h0 = {k: np.concatenate([d[k] for d in h0]) for k in h0[0]}
    res = {"cfg": cfg, "snr": SNRS.tolist(), "pd": {}}
    for pfa in (1e-2, 1e-1):
        thr = {k: np.quantile(v, 1 - pfa) for k, v in h0.items()}
        pd = {k: [] for k in thr}
        for snr in SNRS:
            r1 = np.random.default_rng(int(9000 + snr))
            h1 = all_stats(gen(r1, 2000, snr), r1)
            for k in thr:
                pd[k].append(float(np.mean(h1[k] > thr[k])))
        res["pd"][str(pfa)] = pd
        print(pfa, {k: [round(v, 2) for v in vv[10:26:3]] for k, vv in pd.items()}, flush=True)
    from synth_bench import snr_at
    res["snr_at_pd50"] = {p: {k: snr_at(v, SNRS) for k, v in d.items()} for p, d in res["pd"].items()}
    json.dump(res, open(os.path.join(OUT, "sr_bench.json"), "w"), indent=2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, pfa in zip(axs, ("0.01", "0.1")):
        for k, v in res["pd"][pfa].items():
            ax.plot(SNRS, v, "--" if not k.startswith("sr") else "-", marker="o", ms=2.5, label=k)
        ax.set_title(f"10 Hz tone, fs 100 Hz, N = 512, Pfa = {pfa}")
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Pd")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sr_pd_vs_snr.png"), dpi=140)
    print(json.dumps(res["snr_at_pd50"], indent=1))


if __name__ == "__main__":
    {"tune": cmd_tune, "run": cmd_run}[sys.argv[1]]()
