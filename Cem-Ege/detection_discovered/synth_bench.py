"""Synthetic narrowband benchmark: one signal in complex AWGN, N = 4096 samples,
Pd vs SNR at Pfa 1e-2, every detector next to the plain energy detector.

    python synth_bench.py tune-sr     dev seeds: choose SR parameters (g, D, h) for this test
    python synth_bench.py run         held-out seeds: Pd vs SNR, json + png

SNR = signal power / noise power per complex sample (whole band). Signals:
  tone  complex exponential, frequency U(-0.5, 0.5) cycles/sample, random phase
  qpsk  RRC (beta 0.35) QPSK, 8 samples/symbol (bandwidth ~0.17 of fs), carrier U(-0.4, 0.4)
Noise power is known (= 1) to `energy`, `tkeo`, `med_known` (oracle noise level);
agm / john_emr / cond are blind. Thresholds: empirical (1 - Pfa) quantile of each
statistic on 20000 noise-only frames (seed differs from the H1 seeds); the
condition-number detector is ALSO run with its analytic TW-Curtiss threshold,
whose measured Pfa is reported. Energy has a closed form for the tone:
2N T ~ noncentral chi2(2N, 2N SNR) -> analytic Pd curve (known-answer check of the harness).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import chi2, ncx2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disc.eigen import cond_threshold, eigs, stat_from_eigs  # noqa: E402
from disc.signals import cnoise, qpsk, tone  # noqa: E402
from disc.sr import energy_metric, fft_max_metric, sr_filter  # noqa: E402
from disc.tkeo import tkeo  # noqa: E402

OUT = os.path.join(HERE, "results", "synthetic")
N = 4096
PFA = 1e-2
SNRS = np.arange(-26, 3, 1)
N_H0, N_H1 = 20000, 1000
SR_GRID = [(g, D, h) for g in (0.3, 1.0, 3.0) for D in (0.0, 0.1, 0.43, 1.0) for h in (0.01, 0.1)]


def load_L():
    p = os.path.join(HERE, "results", "dev_tuning", "chosen.json")
    return json.load(open(p))["L"] if os.path.exists(p) else {"agm": 8, "john": 8, "cond": 8, "med": 8}


def stats(x, Ls, sr, rng):
    out = {"energy": np.mean(np.abs(x) ** 2, -1),
           "fft_max (ref, tone GLRT)": np.max(np.abs(np.fft.fft(x, axis=-1)) ** 2, -1) / N,
           "tkeo": tkeo(x, edges=False).mean(-1)}
    cache = {}
    for k, L in Ls.items():
        if L not in cache:
            cache[L] = eigs(x, L)
        ev = cache[L]
        if k == "med":
            out[f"med_known_L{L}"] = ev[..., -1]
        elif k == "john":
            out[f"john_emr_L{L}"] = stat_from_eigs(ev, "john")
        else:
            out[f"{k}_L{L}"] = stat_from_eigs(ev, k)
    if sr is not None:
        g, D, h = sr
        z = sr_filter(x, g, D, h, rng)
        out["sr_energy"] = energy_metric(z)
        out["sr_fftmax"] = fft_max_metric(z, 256)
    return out


def batched(fn, n, bs=1000):
    parts = [fn(min(bs, n - s)) for s in range(0, n, bs)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def gen(rng, kind, n, snr):
    s = tone(rng, n, N) if kind == "tone" else qpsk(rng, n, N)
    return cnoise(rng, (n, N)) + s * 10 ** (snr / 20)


def cmd_tune_sr():
    """Dev seeds (101, 102): pick (g, D, h) maximising mean Pd of sr_energy over both
    signals at -15 and -10 dB, Pfa 1e-2."""
    res = []
    for p in SR_GRID:
        rng = np.random.default_rng(101)
        h0 = batched(lambda m: {"s": energy_metric(sr_filter(cnoise(rng, (m, N)), *p, rng))}, 4000)["s"]
        thr = np.quantile(h0, 1 - PFA)
        pd = []
        for kind in ("tone", "qpsk"):
            for snr in (-15, -10):
                rng1 = np.random.default_rng(102)
                h1 = energy_metric(sr_filter(gen(rng1, kind, 500, snr), *p, rng1))
                pd.append(float(np.mean(h1 > thr)))
        res.append({"g": p[0], "D": p[1], "h": p[2], "pd": pd, "mean_pd": float(np.mean(pd))})
        print(res[-1], flush=True)
    best = max(res, key=lambda r: r["mean_pd"])
    os.makedirs(OUT, exist_ok=True)
    json.dump({"grid": res, "best": best}, open(os.path.join(OUT, "sr_tuning_dev.json"), "w"), indent=2)
    print("best", best)


def snr_at(pd, snrs, target=0.5):
    pd = np.asarray(pd)
    idx = np.flatnonzero(pd >= target)
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(snrs[0])
    return float(snrs[i - 1] + (target - pd[i - 1]) / (pd[i] - pd[i - 1]) * (snrs[i] - snrs[i - 1]))


def cmd_run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = load_L()
    tp = os.path.join(OUT, "sr_tuning_dev.json")
    sr = tuple(json.load(open(tp))["best"][k] for k in ("g", "D", "h"))
    rng = np.random.default_rng(2026)
    h0 = batched(lambda m: stats(cnoise(rng, (m, N)), Ls, sr, rng), N_H0)
    thr = {k: float(np.quantile(v, 1 - PFA)) for k, v in h0.items()}
    Lc = Ls["cond"]
    thr_tw = cond_threshold(Lc, N - Lc + 1, PFA, "twc")
    ck = f"cond_L{Lc}"
    res = {"N": N, "pfa": PFA, "L": Ls, "sr_params": sr, "thr": thr, "cond_twc_threshold": thr_tw,
           "cond_twc_measured_pfa": float(np.mean(h0[ck] > thr_tw)), "snr": SNRS.tolist(), "pd": {}}
    for kind in ("tone", "qpsk"):
        pd = {k: [] for k in thr}
        pd["cond_twc (analytic thr)"] = []
        for snr in SNRS:
            rng1 = np.random.default_rng(int(5000 + snr + (0 if kind == "tone" else 1000)))
            h1 = batched(lambda m: stats(gen(rng1, kind, m, snr), Ls, sr, rng1), N_H1)
            for k in thr:
                pd[k].append(float(np.mean(h1[k] > thr[k])))
            pd["cond_twc (analytic thr)"].append(float(np.mean(h1[ck] > thr_tw)))
            print(kind, snr, {k: round(v[-1], 3) for k, v in pd.items()}, flush=True)
        res["pd"][kind] = pd
    # analytic energy Pd (tone, known noise): 2N*T ~ ncx2(2N, 2N*snr)
    g = chi2.ppf(1 - PFA, 2 * N)
    res["energy_analytic_tone"] = [float(ncx2.sf(g, 2 * N, 2 * N * 10 ** (s / 10))) for s in SNRS]
    res["energy_threshold_analytic"] = float(g / (2 * N))
    res["snr_at_pd50"] = {kind: {k: snr_at(v, SNRS) for k, v in res["pd"][kind].items()} for kind in res["pd"]}
    res["snr_at_pd50"]["tone"]["energy (analytic)"] = snr_at(res["energy_analytic_tone"], SNRS)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "synthetic_bench.json"), "w"), indent=2)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kind in zip(axs, ("tone", "qpsk")):
        for k, v in res["pd"][kind].items():
            ls = "-" if not k.startswith(("energy", "fft")) else "--"
            ax.plot(SNRS, v, ls, marker="o", ms=2.5, lw=2.2 if k == "energy" else 1.2, label=k)
        if kind == "tone":
            ax.plot(SNRS, res["energy_analytic_tone"], "k:", lw=1.5, label="energy (analytic)")
        ax.set_title(f"{kind}, N = {N}, Pfa = {PFA:g}")
        ax.set_xlabel("SNR (dB, per complex sample)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Pd")
    axs[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "synthetic_pd_vs_snr.png"), dpi=140)
    print(json.dumps(res["snr_at_pd50"], indent=1), "\ncond_twc measured Pfa", res["cond_twc_measured_pfa"])


if __name__ == "__main__":
    {"tune-sr": cmd_tune_sr, "run": cmd_run}[sys.argv[1]]()
