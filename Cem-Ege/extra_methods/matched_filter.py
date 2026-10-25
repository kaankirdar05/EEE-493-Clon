"""Matched filter (preamble correlator) vs energy detector for a burst with a KNOWN preamble.

The matched filter is the optimal detector when the waveform is known, but it is
waveform-specific (ADS-B preamble, GSM midamble, WiFi STF, DMR sync...), so it
is benchmarked here on synthetic bursts rather than on the mixed TorchSig scenes.

Frame: 4096 complex samples of unit-power AWGN. With probability 1/2 a burst is
inserted at a random position: a 64-symbol known BPSK preamble (4 samples/symbol,
RRC roll-off 0.35, 256 samples) followed by 512 samples of random QPSK data.
Per-sample SNR (signal power / noise power) is swept.

Detectors (each decides per frame; threshold set for Pfa = 1e-3 per frame on
noise-only frames, measured by simulation):
    energy        max over positions of the energy in a 256-sample sliding window
    mf            max over positions of |<x, preamble>|^2 / (window energy x preamble energy)
                  (normalised correlation, so it does not need the noise power)
    mf_cfo        mf on bursts with an unknown carrier offset of up to +-0.002 cycles/sample
    mf_seg_cfo    segmented matched filter: 4 sub-correlations of 64 samples combined
                  non-coherently, robust to that offset

Outputs: results/matched_filter.json and .png (Pd vs SNR), timing-error statistics.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical_amc"))
from amc.synthetic import _rrc  # noqa: E402

N, SPS, NPRE, NDATA = 4096, 4, 64, 512
RNG = np.random.default_rng(0)
H = _rrc(0.35, 8, SPS)


def shape(sym):
    up = np.zeros(len(sym) * SPS, complex)
    up[::SPS] = sym
    return np.convolve(up, H, "same")


PRE = shape(np.random.default_rng(42).choice([-1.0, 1.0], NPRE))
PRE /= np.sqrt(np.mean(np.abs(PRE) ** 2))
LP = len(PRE)


def frame(snr_db, present, cfo=0.0, rng=RNG):
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    pos = -1
    if present:
        data = shape(rng.choice(np.array([1, 1j, -1, -1j]) * np.exp(1j * np.pi / 4), NDATA // SPS))
        data /= np.sqrt(np.mean(np.abs(data) ** 2))
        b = np.concatenate([PRE, data]) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        b = b * np.exp(2j * np.pi * cfo * np.arange(len(b)))
        pos = int(rng.integers(0, N - len(b)))
        x[pos:pos + len(b)] += np.sqrt(10 ** (snr_db / 10)) * b
    return x, pos


def win_energy(x, L):
    c = np.concatenate([[0], np.cumsum(np.abs(x) ** 2)])
    return c[L:] - c[:-L]


def stat_energy(x):
    e = win_energy(x, LP)
    return e.max() / np.median(e), int(np.argmax(e))


def stat_mf(x):
    corr = np.abs(np.correlate(x, PRE, "valid")) ** 2
    s = corr / (win_energy(x, LP) * np.sum(np.abs(PRE) ** 2))
    return s.max(), int(np.argmax(s))


def stat_mf_seg(x, nseg=4):
    L = LP // nseg
    tot = 0
    for k in range(nseg):
        seg = PRE[k * L:(k + 1) * L]
        c = np.abs(np.correlate(x, seg, "valid")) ** 2 / (win_energy(x, L) * np.sum(np.abs(seg) ** 2))
        tot = tot + c[k * L: k * L + N - LP + 1]
    return tot.max(), int(np.argmax(tot))


DET = {"energy": stat_energy, "mf": stat_mf, "mf_seg": stat_mf_seg}


def main(n_noise=4000, n_sig=400):
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)
    h0 = {k: np.array([f(frame(0, False)[0])[0] for _ in range(n_noise)]) for k, f in DET.items()}
    thr = {k: np.quantile(v, 1 - 1e-3) for k, v in h0.items()}
    snrs = list(range(-24, 1, 2))
    cases = {"energy": ("energy", 0.0), "mf": ("mf", 0.0), "mf_cfo": ("mf", 0.002), "mf_seg_cfo": ("mf_seg", 0.002),
             "energy_cfo": ("energy", 0.002)}
    res = {k: {} for k in cases}
    toa = {k: [] for k in cases}
    for s in snrs:
        for name, (det, cfo) in cases.items():
            hits = 0
            for _ in range(n_sig):
                x, pos = frame(s, True, cfo=RNG.uniform(-cfo, cfo))
                v, at = DET[det](x)
                if v > thr[det]:
                    hits += 1
                    toa[name].append((s, abs(at - pos)))
            res[name][s] = hits / n_sig
    summary = {}
    for name in cases:
        p90 = next((s for s in snrs if res[name][s] >= 0.9), None)
        errs = [e for s, e in toa[name] if s >= -6]
        summary[name] = {"pd_vs_snr": res[name], "snr_for_pd90": p90,
                         "median_toa_error_samples_snr_ge_-6": float(np.median(errs)) if errs else None}
    gain = (summary["energy"]["snr_for_pd90"] or 0) - (summary["mf"]["snr_for_pd90"] or 0)

    # Theory check for the matched filter. Under noise, |<x,p>|^2 / (sigma^2 |p|^2) ~ Exp(1) at each
    # position; with M positions the per-position Pfa is ~1e-3 / M (Bonferroni, slightly conservative
    # because neighbouring positions are correlated). Pd = Marcum-Q -> solve for the output SNR rho;
    # input SNR = rho / preamble length (unit-power preamble).
    from scipy.optimize import brentq
    from scipy.stats import ncx2
    m = N - LP + 1
    t = -np.log(1e-3 / m)
    rho = brentq(lambda r: ncx2.sf(2 * t, 2, 2 * r) - 0.9, 1e-3, 1e4)
    theory_db = 10 * np.log10(rho / LP)
    summary["mf"]["theory_snr_for_pd90"] = round(float(theory_db), 2)
    print(f"theory check: matched filter Pd 90 % expected near {theory_db:.1f} dB (measured {summary['mf']['snr_for_pd90']} dB on a 2 dB grid)")
    with open(os.path.join(out, "matched_filter.json"), "w") as f:
        json.dump({"pfa_per_frame": 1e-3, "preamble_samples": LP, "summary": summary,
                   "mf_gain_db_at_pd90": gain}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name in cases:
        ax.plot(snrs, [res[name][s] * 100 for s in snrs], marker="o", label=name)
    ax.set_xlabel("Per-sample SNR (dB)")
    ax.set_ylabel("Pd (%) at Pfa 1e-3 per frame")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Known-preamble burst: matched filter vs energy")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "matched_filter.png"), dpi=150)
    for name, v in summary.items():
        print(f"{name:>11s}: Pd>=90 % at {v['snr_for_pd90']} dB, median TOA error {v['median_toa_error_samples_snr_ge_-6']} samples")
    print(f"matched-filter gain over energy at Pd 90 %: {gain} dB")


if __name__ == "__main__":
    main()
