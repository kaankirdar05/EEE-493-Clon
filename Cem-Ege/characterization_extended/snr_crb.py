"""Task 2 - SNR estimators vs the Cramer-Rao bound (synthetic; cf. Pauluzzi & Beaulieu 2000).

rho = Es/N0 = SNR at the matched-filter (symbol) output. Constellations BPSK, QPSK, 16QAM
(unit mean energy), N = 256 / 1024 symbols, coherent (carrier phase known = 0, as in
Pauluzzi & Beaulieu) unless the estimator is phase-blind anyway.

Symbol level (y_k = sqrt(rho) c_k + n_k, n_k ~ CN(0, 1)):
  da_ml     data-aided ML: A = sum y c* / sum |c|^2, N0 = mean |y - A c|^2, rho = |A|^2 / N0
  dd_ml     decision-directed ML: same with c replaced by slicer decisions (3 iterations,
            initial amplitude from M2M4)
  m2m4      second/fourth moments, S = sqrt((2 M2^2 - M4) / (2 - k_a)), k_a = E|c|^4
  kurtosis  solve (kappa - k_a) rho^2 + (2 kappa - 4) rho + (kappa - 2) = 0, kappa = M4/M2^2.
            Algebraically identical to M2M4 (shown in README); run only as a numerical check.
Sample level (8 samples/symbol, per-sample noise CN(0,1)):
  ssme      split-symbol moments (Simon & Mileant), rectangular pulse, integrate-and-dump
            halves Y1, Y2: rho = (mean|Y1+Y2|^2 - mean|Y1-Y2|^2) / mean|Y1-Y2|^2
  eig_mdl   RRC (beta 0.35) pulse; 32 x 32 forward-backward sample covariance; MDL picks the
            signal-subspace dimension p; N0 = mean of the 32 - p smallest eigenvalues;
            rho = sps (mean(diag R) - N0) / N0
  da_ml_s   data-aided ML on the samples (known waveform) = sample-level reference

Bounds (per N symbols):
  CRB_DA(symbol)  = (2 rho + rho^2) / N                      (closed form, derived in README)
  CRB_NDA(symbol) = g' J^-1 g / N, J = per-symbol Fisher information of the NDA mixture model
                    p(y) = mean_m CN(y; B c_m, s2), computed by Monte Carlo (4e5 draws);
                    g = d rho / d(B, s2)
  CRB_DA(sample)  = (2 rho + rho^2 / sps) / N               (extra noise-only dimensions)
  CRB_NDA(sample) = same as CRB_NDA(symbol) with (sps - 1)/s2^2 added to J_s2s2
  CRB_NDA(split)  = same with 1/s2^2 added: the bound for SSME's own statistic (Y1, Y2)
    (valid for the rectangular-pulse SSME setting; approximately for RRC)

    python snr_crb.py [--trials 500]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import common
from amc.synthetic import _constellation, _rrc

SPS = 8
RRC = _rrc(0.35, 8, SPS)
RECT = np.ones(SPS) / np.sqrt(SPS)
RHO_MAX = 1e5   # clip of estimates (noise estimate <= 0)


def cons(name):
    p = _constellation(name).astype(complex)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


KA = {c: float(np.mean(np.abs(cons(c)) ** 4)) for c in ("BPSK", "QPSK", "16QAM")}


# ---------------------------------------------------------------- bounds
def fim_nda(points, rho, n_mc=400_000, rng=None, extra_noise_dims=0):
    """Per-symbol Fisher information for (B, s2) with s2 = 1, B = sqrt(rho), coherent NDA."""
    rng = rng or np.random.default_rng(123)
    B, s2 = np.sqrt(rho), 1.0
    c = rng.choice(points, n_mc)
    y = B * c + np.sqrt(s2 / 2) * (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc))
    d = y[:, None] - B * points[None, :]
    ll = -np.abs(d) ** 2 / s2
    w = np.exp(ll - ll.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    sB = (w * 2 * np.real(d * np.conj(points[None, :])) / s2).sum(1)
    sS = (w * (-1 / s2 + np.abs(d) ** 2 / s2**2)).sum(1)
    J = np.array([[np.mean(sB * sB), np.mean(sB * sS)], [np.mean(sB * sS), np.mean(sS * sS)]])
    J[1, 1] += extra_noise_dims / s2**2
    return J


def crb_from_fim(J, rho, n):
    B = np.sqrt(rho)
    g = np.array([2 * B, -B**2])
    return float(g @ np.linalg.solve(J, g) / n)


def crb_da(rho, n, sps=1):
    return (2 * rho + rho**2 / sps) / n


# ---------------------------------------------------------------- estimators
def slicer(z, pts):
    return pts[np.argmin(np.abs(z[:, None] - pts[None, :]), axis=1)]


def m2m4(y, ka):
    M2 = np.mean(np.abs(y) ** 2)
    M4 = np.mean(np.abs(y) ** 4)
    S = np.sqrt(max(2 * M2**2 - M4, 0.0) / (2 - ka))
    Nn = M2 - S
    return S / Nn if Nn > S / RHO_MAX else RHO_MAX


def kurtosis_est(y, ka):
    M2 = np.mean(np.abs(y) ** 2)
    kap = np.mean(np.abs(y) ** 4) / M2**2
    if kap >= 2:
        return 0.0
    if kap <= ka:
        return RHO_MAX
    r = ((2 - kap) + np.sqrt((2 - kap) * (2 - ka))) / (kap - ka)
    return min(r, RHO_MAX)


def da_ml(y, c):
    A = np.vdot(c, y) / np.vdot(c, c)
    N0 = np.mean(np.abs(y - A * c) ** 2)
    return min(np.abs(A) ** 2 / N0, RHO_MAX)


def dd_ml(y, pts, ka):
    rho0 = m2m4(y, ka)
    M2 = np.mean(np.abs(y) ** 2)
    A = np.sqrt(M2 * rho0 / (1 + rho0)) if rho0 > 0 else np.sqrt(M2)
    for _ in range(3):
        c = slicer(y / max(A, 1e-9), pts)
        A = np.real(np.vdot(c, y)) / np.real(np.vdot(c, c))
    N0 = np.mean(np.abs(y - A * c) ** 2)
    return min(A**2 / N0, RHO_MAX)


def ssme(r):
    Y = r[: len(r) // SPS * SPS].reshape(-1, SPS)
    Y1, Y2 = Y[:, : SPS // 2].sum(1), Y[:, SPS // 2:].sum(1)
    U, V = np.mean(np.abs(Y1 + Y2) ** 2), np.mean(np.abs(Y1 - Y2) ** 2)
    return float(np.clip((U - V) / V, 0, RHO_MAX))


def mdl_order(lam, K):
    M = len(lam)
    best, kbest = np.inf, 0
    for k in range(M - 1):
        tail = lam[k:]
        g = np.exp(np.mean(np.log(tail)))
        a = np.mean(tail)
        v = -(M - k) * K * np.log(g / a) + 0.5 * k * (2 * M - k) * np.log(K)
        if v < best:
            best, kbest = v, k
    return kbest


def eig_mdl(r, m=32):
    H = sliding_window_view(r, m)
    R = H.T @ H.conj() / H.shape[0]
    J = np.eye(m)[::-1]
    R = 0.5 * (R + J @ R.conj() @ J)
    lam = np.sort(np.linalg.eigvalsh(R))[::-1]
    lam = np.maximum(lam, 1e-12)
    p = mdl_order(lam, H.shape[0])
    N0 = np.mean(lam[p:])
    P = np.mean(lam)
    return float(np.clip(SPS * (P - N0) / N0, 0, RHO_MAX)), p


def wave(c, pulse):
    up = np.zeros(len(c) * SPS, complex)
    up[::SPS] = c
    if len(pulse) == SPS:
        return np.repeat(c, SPS) * pulse[0]
    return np.convolve(up, pulse, "same")


# ---------------------------------------------------------------- experiment
def run(cname, n, snrs, trials, rng):
    pts = cons(cname)
    ka = KA[cname]
    names = ["da_ml", "dd_ml", "m2m4", "kurtosis", "ssme", "eig_mdl", "da_ml_s"]
    est = {k: np.zeros((len(snrs), trials)) for k in names}
    p_mdl = np.zeros((len(snrs), trials))
    for i, snr in enumerate(snrs):
        rho = 10 ** (snr / 10)
        B = np.sqrt(rho)
        for t in range(trials):
            c = rng.choice(pts, n)
            nz = lambda m: np.sqrt(0.5) * (rng.standard_normal(m) + 1j * rng.standard_normal(m))
            y = B * c + nz(n)
            est["da_ml"][i, t] = da_ml(y, c)
            est["dd_ml"][i, t] = dd_ml(y, pts, ka)
            est["m2m4"][i, t] = m2m4(y, ka)
            est["kurtosis"][i, t] = kurtosis_est(y, ka)
            s_rect = B * wave(c, RECT)
            est["ssme"][i, t] = ssme(s_rect + nz(len(s_rect)))
            s_rrc = B * wave(c, RRC)
            r = s_rrc + nz(len(s_rrc))
            est["eig_mdl"][i, t], p_mdl[i, t] = eig_mdl(r)
            est["da_ml_s"][i, t] = da_ml(r, s_rrc / B)
    return est, p_mdl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=500)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    rng = np.random.default_rng(2)
    snrs = list(np.arange(-10, 30.1, 2.5))
    out = {"snr_db": snrs, "trials": args.trials, "sps": SPS, "k_a": KA, "results": {}}

    # sanity check of the Monte Carlo FIM code: one-point "constellation" = DA case -> closed form
    chk = []
    for snr in (0.0, 10.0, 20.0):
        rho = 10 ** (snr / 10)
        num = crb_from_fim(fim_nda(np.array([1 + 0j]), rho), rho, 1)
        chk.append({"snr_db": snr, "mc_fim_crb": num, "closed_form": crb_da(rho, 1), "ratio": num / crb_da(rho, 1)})
    out["fim_code_check_DA"] = chk
    print("FIM code check (MC vs closed-form DA CRB):", [round(c["ratio"], 3) for c in chk])

    fig_s, ax_s = plt.subplots(3, 2, figsize=(13, 14))
    fig_p, ax_p = plt.subplots(3, 2, figsize=(13, 14))
    for ci, cname in enumerate(["BPSK", "QPSK", "16QAM"]):
        pts = cons(cname)
        J = {s: fim_nda(pts, 10 ** (s / 10)) for s in snrs}
        Js = {s: fim_nda(pts, 10 ** (s / 10), extra_noise_dims=SPS - 1) for s in snrs}
        J2 = {s: fim_nda(pts, 10 ** (s / 10), extra_noise_dims=1) for s in snrs}
        for ni, n in enumerate([256, 1024]):
            est, p_mdl = run(cname, n, snrs, args.trials, rng)
            rho = 10 ** (np.array(snrs) / 10)
            bounds = {"crb_da_symbol": [crb_da(r, n) for r in rho],
                      "crb_nda_symbol": [crb_from_fim(J[s], r, n) for s, r in zip(snrs, rho)],
                      "crb_da_sample": [crb_da(r, n, SPS) for r in rho],
                      "crb_nda_sample": [crb_from_fim(Js[s], r, n) for s, r in zip(snrs, rho)],
                      "crb_nda_split_symbol": [crb_from_fim(J2[s], r, n) for s, r in zip(snrs, rho)]}
            res = {}
            for k, v in est.items():
                mse = np.mean((v - rho[:, None]) ** 2, 1)
                res[k] = {"nmse": (mse / rho**2).tolist(), "nbias": (np.mean(v, 1) / rho - 1).tolist(),
                          "nmse_median_based": (np.median((v / rho[:, None] - 1) ** 2, 1)).tolist(),
                          "clipped_fraction": np.mean(v >= RHO_MAX, 1).tolist()}
            res["eig_mdl"]["mean_mdl_dimension"] = p_mdl.mean(1).tolist()
            key = f"{cname}_N{n}"
            out["results"][key] = {"bounds_normalised": {b: (np.array(v) / rho**2).tolist() for b, v in bounds.items()},
                                   "estimators": res}
            # print a compact check at 0, 10, 20 dB
            for s in (0.0, 10.0, 20.0):
                i = snrs.index(s)
                line = " ".join(f"{k}={res[k]['nmse'][i] / (bounds['crb_da_symbol'][i] / rho[i] ** 2):.2f}"
                                for k in ("da_ml", "dd_ml", "m2m4"))
                line2 = (f"ssme/CRB_split={res['ssme']['nmse'][i] / (bounds['crb_nda_split_symbol'][i] / rho[i] ** 2):.2f} "
                         f"eig/CRB_NDA_sample={res['eig_mdl']['nmse'][i] / (bounds['crb_nda_sample'][i] / rho[i] ** 2):.2f}")
                print(f"{key:12s} {s:4.0f} dB  NMSE/CRB_DA_sym: {line} | NMSE/CRB_NDA_sym "
                      f"m2m4={res['m2m4']['nmse'][i] / (bounds['crb_nda_symbol'][i] / rho[i] ** 2):.2f} | "
                      f"{line2} | bias% m2m4 {res['m2m4']['nbias'][i] * 100:+.1f} "
                      f"ssme {res['ssme']['nbias'][i] * 100:+.1f} eig {res['eig_mdl']['nbias'][i] * 100:+.1f} "
                      f"dd {res['dd_ml']['nbias'][i] * 100:+.1f}")
            a = ax_s[ci, ni]
            a.semilogy(snrs, np.array(bounds["crb_da_symbol"]) / rho**2, "k-", lw=2.5, label="CRB DA (symbol)")
            a.semilogy(snrs, np.array(bounds["crb_nda_symbol"]) / rho**2, "k--", lw=2.5, label="CRB NDA (symbol)")
            for k in ("da_ml", "dd_ml", "m2m4"):
                a.semilogy(snrs, res[k]["nmse"], marker=".", label=k)
            a.set_title(f"symbol level: {cname}, N = {n}")
            b = ax_p[ci, ni]
            b.semilogy(snrs, np.array(bounds["crb_da_sample"]) / rho**2, "k-", lw=2.5, label="CRB DA (sample)")
            b.semilogy(snrs, np.array(bounds["crb_nda_sample"]) / rho**2, "k--", lw=2.5, label="CRB NDA (sample)")
            b.semilogy(snrs, np.array(bounds["crb_nda_symbol"]) / rho**2, "k:", lw=1.5, label="CRB NDA (symbol)")
            b.semilogy(snrs, np.array(bounds["crb_nda_split_symbol"]) / rho**2, "-.", color="grey", lw=1.5,
                       label="CRB NDA (2 half-symbol samples; SSME's data)")
            for k in ("ssme", "eig_mdl", "da_ml_s", "m2m4"):
                b.semilogy(snrs, res[k]["nmse"], marker=".", label=k)
            b.set_title(f"sample level (8 sps): {cname}, N = {n}")
            for z in (a, b):
                z.set_xlabel("Es/N0 (dB)"); z.set_ylabel("NMSE = E[(rho_hat - rho)^2] / rho^2")
                z.grid(True, which="both", alpha=0.3); z.legend(fontsize=7); z.set_ylim(1e-4, 1e2)
    fig_s.tight_layout(); fig_s.savefig(os.path.join(common.RESULTS, "snr_crb_symbol_level.png"), dpi=110)
    fig_p.tight_layout(); fig_p.savefig(os.path.join(common.RESULTS, "snr_crb_sample_level.png"), dpi=110)
    out["runtime_s"] = time.perf_counter() - t0
    with open(os.path.join(common.RESULTS, "snr_crb.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
