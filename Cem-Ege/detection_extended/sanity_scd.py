"""Sanity check for the FAM spectral-correlation estimator (Part A.4).

Synthetic complex-baseband BPSK, rectangular pulses, symbol rate Rs = 1/8
cycles/sample, carrier offset fc = 0.05 cycles/sample, high SNR. Theory:
  non-conjugate SCD: features at alpha = +-k Rs  (k = 1, 2, ...) -> strongest +-0.125
  conjugate SCD:     features at alpha = 2 fc + k Rs           -> 0.1, 0.1 +- 0.125
White noise: no feature anywhere except the excluded alpha ~ 0 (PSD).
Checked at N = 4096 (plot) and at the detector's block length N = 512.

Writes results/sanity_scd_bpsk.png and results/sanity_scd_bpsk.json.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ext.block_detectors import fam  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def bpsk(n, sps=8, fc=0.05, snr_db=20, rng=None):
    rng = rng or np.random.default_rng(0)
    sym = rng.choice([-1.0, 1.0], n // sps + 1)
    s = np.repeat(sym, sps)[:n] * np.exp(2j * np.pi * fc * np.arange(n))
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    return s + w


def alpha_profile(C, a, N, exclude0=True):
    """max over f of C, binned to cyclic resolution 1/N."""
    ab = np.round(a * N).astype(int).ravel()
    c = C.reshape(-1)
    if exclude0:
        c = np.where(np.abs(ab) >= 4, c, 0)
    grid = np.arange(-N, N + 1)
    prof = np.zeros(len(grid))
    np.maximum.at(prof, ab + N, c)
    return grid / N, prof


def top_alphas(alpha, prof, k=3, min_sep=0.01):
    out = []
    for i in np.argsort(prof)[::-1]:
        if all(abs(alpha[i] - o) > min_sep for o, _ in out):
            out.append((float(alpha[i]), float(prof[i])))
        if len(out) == k:
            break
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    res = {}
    fig, axs = plt.subplots(2, 3, figsize=(16, 6.5))
    for col, (N, Np) in enumerate([(4096, 64), (512, 32), (512, 16)]):
        x = bpsk(N, rng=rng)
        w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        for row, conj in enumerate([False, True]):
            C, a, _ = fam(x, Np, conj=conj)
            al, prof = alpha_profile(C, a, N, exclude0=not conj)
            Cn, an, _ = fam(w, Np, conj=conj)
            _, profn = alpha_profile(Cn, an, N, exclude0=not conj)
            key = f"N{N}_Np{Np}_{'conj' if conj else 'nonconj'}"
            res[key] = {"top_alpha_bpsk": top_alphas(al, prof), "max_noise": float(profn.max()),
                        "expected": [0.1, -0.025, 0.225] if conj else [0.125, -0.125, 0.25]}
            ax = axs[row, col]
            ax.plot(al, prof, lw=0.9, label="BPSK, 20 dB")
            ax.plot(al, profn, lw=0.9, alpha=0.7, label="white noise")
            for e in res[key]["expected"]:
                ax.axvline(e, color="k", ls=":", lw=0.8)
            ax.set_xlim(-0.5, 0.5)
            ax.set_title(f"{'conjugate' if conj else 'non-conjugate'} SCD, N={N}, Np={Np}")
            ax.set_xlabel("cyclic frequency alpha (cycles/sample)")
            ax.set_ylabel("max_f coherence^2")
            ax.legend(fontsize=7)
    fig.suptitle("FAM sanity check: BPSK Rs = 1/8, fc = 0.05 (dotted = theory: Rs multiples, 2fc +- k Rs)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sanity_scd_bpsk.png"), dpi=130)
    ok = True
    for key, r in res.items():
        found = [t[0] for t in r["top_alpha_bpsk"]]
        tol = 2.0 / int(key.split("_")[0][1:])
        # theory: non-conjugate features at alpha = k Rs (k != 0), conjugate at 2 fc + k Rs (any k);
        # with rectangular pulses all harmonics k are present, so any k counts.
        base = 0.1 if "_conj" in key else 0.0
        kk = np.round((np.array(found) - base) / 0.125)
        on_set = np.abs(np.array(found) - (base + kk * 0.125)) <= tol
        if "_conj" not in key:
            on_set &= kk != 0
        hit = bool(on_set.all())   # all three strongest peaks lie on the theoretical set
        r["strongest_matches_theory"] = bool(hit)
        ok &= hit
        print(key, "top alphas", [(round(a, 4), round(c, 3)) for a, c in r["top_alpha_bpsk"]],
              "noise max", round(r["max_noise"], 4), "match", hit)
    res["pass"] = bool(ok)
    with open(os.path.join(OUT, "sanity_scd_bpsk.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
