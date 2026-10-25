"""Sanity checks, run BEFORE any benchmark number is reported.

 S1  H0 theory: noise-only frames reproduce the closed-form values of the statistics (spectrogram cell
     law, Pbar std, spectral kurtosis, envelope kurtosis, Gamma-law duty false-alarm, power-excess std,
     bispectrum floor 1/K)
 S2  dechirp implementation == ext.chirp.dechirp_search, and known-answer chirps are recovered
 S3  bispectrum: quadratic-frequency-coupled tones give bicoherence ~ 1
 S4  parameter recovery on NOISELESS frames (INR 30 dB, assumed floor 1): bandwidth, duty, period,
     sweep rate / slope, number of lines, line spacing, frequency; repeated with noise at INR 10 dB
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import jcommon as common
import dataset
import features
from jcommon import N

OUT = os.path.join(common.HERE, "results")


def s1_h0():
    rng = np.random.default_rng(dataset.ROLE_SANE)
    from wp4.spectrum import spectrogram
    x = common.cnoise(rng)
    S = spectrogram(x.astype(np.complex64))
    res = {"cell_mean": float(S.mean()), "cell_var": float(S.var()), "cell_theory": [1.0, 1.0],
           "pbar_std": float(S.mean(0).std()), "pbar_std_theory": 1 / np.sqrt(512)}
    tasks = [dict(role=dataset.ROLE_SANE, kind="none", inr=0.0, sig=False, idx=i, cal_err=0.0, bispec=True) for i in range(240)]
    X, nm, _ = dataset.run(tasks, label="S1")
    col = {n: X[:, i] for i, n in enumerate(nm)}
    res["pow_excess_db_std"] = float(col["pow_excess_db"].std())
    res["pow_excess_db_std_theory"] = float(10 / np.log(10) / np.sqrt(N))
    res["pow_excess_db_mean"] = float(col["pow_excess_db"].mean())
    res["sk_mean"] = float(col["sk_mean"].mean())
    res["sk_theory"] = 2.0
    res["env_kurt"] = float(col["env_kurt"].mean())
    res["duty_mean"] = float(col["duty"].mean())
    res["duty_theory"] = 1e-3
    res["duty_ci_halfwidth"] = float(1.96 * np.sqrt(1e-3 / (240 * 8192)))
    res["bw_frac_max"] = float(col["bw_frac"].max())
    res["n_peaks_max"] = int(col["n_peaks"].max())
    K = 32768 // 64
    res["bis_x_bic_mean"] = float(col["bis_x_bic_mean"].mean())
    res["bis_x_bic_mean_theory_1_over_K"] = 1.0 / K
    res["chirp_G_noise_mean"] = {k: float(col[k].mean()) for k in ["chirp_G256", "chirp_G1024", "chirp_G0_256", "chirp_G0_1024"]}
    return res


def s2_dechirp():
    from ext.chirp import dechirp_search, chirp
    rng = np.random.default_rng(7)
    out = {}
    for L in (256, 1024):
        ks, _ = features.KG[L]
        d = ks[1] - ks[0]
        k0 = ks[len(ks) // 2 + 5 * (L // 256)]
        x = chirp(L, 0.05, k0) + 0.3 * common.cnoise(rng, L)
        G = features.dechirp_gains(np.concatenate([x, x]), L, ks, 2)[0]
        peaks, _ = dechirp_search(x, ks, nfft=2 * L)
        E = np.sum(np.abs(x) ** 2)
        out[f"L{L}_max_rel_diff_vs_ext"] = float(np.max(np.abs(G - peaks / E)) / G.max())
        # known answers (noiseless)
        res = {}
        for name, kk in [("on_grid", k0), ("half_step", k0 + d / 2)]:
            x = chirp(L, 0.05, kk)
            G = features.dechirp_gains(np.concatenate([x, x]), L, ks, 2)[0]
            res[name] = dict(Gmax=float(G.max()), k_err_in_grid_steps=float((ks[np.argmax(G)] - kk) / d))
        out[f"L{L}_noiseless"] = res
    return out


def s3_bispec():
    import bispectrum as bs
    rng = np.random.default_rng(3)
    n = 32768
    t = np.arange(n)
    f1, f2 = 5 / 64, 9 / 64 + 0.3 / 64
    f3 = f1 + f2
    x = sum(np.exp(1j * (2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))) for f in (f1, f2, f3))
    x = x + 0.05 * common.cnoise(rng, n)
    B, b2, P, K = bs.bispectrum(bs.dc.normalize(x[None]))
    coupled = float(b2[0][bs.MASK].max())
    f3b = 0.2337
    xb = sum(np.exp(1j * (2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))) for f in (f1, f2, f3b)) + 0.05 * common.cnoise(rng, n)
    B, b2, P, K = bs.bispectrum(bs.dc.normalize(xb[None]))
    uncoupled = float(b2[0][bs.MASK].max())
    return dict(bic_max_coupled_tones=coupled, bic_max_uncoupled_tones=uncoupled, K=int(K), floor_mean_theory=1 / K)


def s4_recovery(inr_db, noise, n=60):
    kinds = common.KINDS
    out = {}
    tasks = [dict(role=dataset.ROLE_SANE, kind=k, inr=inr_db, sig=False, idx=i, noise=noise, cal_err=0.0, bispec=False, tag=int(noise))
             for k in kinds for i in range(n)]
    X, nm, metas = dataset.run(tasks, label=f"S4 inr={inr_db} noise={noise}")
    col = {nme: X[:, i] for i, nme in enumerate(nm)}
    kidx = np.repeat(np.arange(len(kinds)), n)

    def stats(e):
        e = np.abs(np.asarray(e, float))
        return dict(median=float(np.median(e)), p90=float(np.quantile(e, 0.9)), mean=float(e.mean()), n=int(len(e)))

    for ki, kind in enumerate(kinds):
        idx = np.flatnonzero(kidx == ki)
        par = [metas[i]["par"] for i in idx]
        r = {}
        c = {k: v[idx] for k, v in col.items()}
        if kind == "cw":
            r["freq_err_bins"] = stats((c["aux_f_peak"] - np.array([p["f"] for p in par])) * 512)
            r["n_lines_correct_frac"] = float(np.mean(c["n_peaks"] == 1))
            r["bw_frac_measured_median"] = float(np.median(c["bw_frac"]))
        elif kind == "comb":
            nl = np.array([p["n_lines"] for p in par])
            r["n_lines_correct_frac"] = float(np.mean(c["n_peaks"] == nl))
            r["n_lines_abs_err_mean"] = float(np.mean(np.abs(c["n_peaks"] - nl)))
            ok = c["aux_peak_spacing"] > 0
            r["spacing_rel_err"] = stats((c["aux_peak_spacing"][ok] - np.array([p["spacing"] for p in par])[ok]) / np.array([p["spacing"] for p in par])[ok])
        elif kind in ("spot", "barrage"):
            bw = np.array([p["bw"] for p in par])
            r["bw_err_frac_of_fs"] = stats(c["bw_frac"] - bw)
            r["bw_err_bins"] = stats((c["bw_frac"] - bw) * 512)
            r["bw_half_err_bins"] = stats((c["bw_half"] - bw) * 512)
            r["bw_half_rel_err"] = stats((c["bw_half"] - bw) / bw)
            r["bw_rel_err"] = stats((c["bw_frac"] - bw) / bw)
            r["center_err_bins"] = stats((c["aux_f_center"] - np.array([p["fc"] for p in par])) * 512)
        elif kind == "sweep":
            bw = np.array([p["bw"] for p in par])
            k = np.array([abs(p["k"]) for p in par])
            r["bw_err_frac_of_fs"] = stats(c["bw_frac"] - bw)
            r["bw_rel_err"] = stats((c["bw_frac"] - bw) / bw)
            r["bw_half_rel_err"] = stats((c["bw_half"] - bw) / bw)
            r["rate_k_rel_err(dechirp)"] = stats((c["aux_k_hat"] - k) / k)
            sl = k * 512 * 512
            m = sl <= 30
            hs = 10 ** c["hough_slope"]
            r["hough_slope_rel_err(slope<=30 bins/frame)"] = stats((hs[m] - sl[m]) / sl[m])
            P = np.array([p["period"] for p in par])
            mm = P >= 2048
            lag = 10 ** c["bin_acf_lag"]
            ok = mm & (c["bin_acf_lag"] > 0)
            r["period_rel_err(P>=4 frames)"] = stats((lag[ok] - P[ok]) / P[ok]) if ok.any() else None
            r["period_found_frac(P>=4 frames)"] = float(ok.sum() / max(mm.sum(), 1))
            r["n_P_ge_4frames"] = int(mm.sum())
            r["n_hough_slope_le_30"] = int(m.sum())
        else:   # pnoise, pcw
            duty = np.array([p["duty"] for p in par])
            r["duty_err_abs"] = stats(c["duty"] - duty)
            r["duty_err_rel"] = stats((c["duty"] - duty) / duty)
            P = np.array([p["period"] for p in par])
            lag = 10 ** c["per_lag"]
            ok = c["per_lag"] > 0
            r["period_rel_err"] = stats((lag[ok] - P[ok]) / P[ok]) if ok.any() else None
            r["period_found_frac"] = float(ok.mean())
            if kind == "pnoise":
                bw = np.array([p["bw"] for p in par])
                r["bw_err_frac_of_fs"] = stats(c["bw_frac"] - bw)
                r["bw_half_err_frac_of_fs"] = stats(c["bw_half"] - bw)
        out[kind] = r
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "s4":
        res = json.load(open(os.path.join(OUT, "sanity.json")))
        for k in ("S4_noiseless_inr30", "S4_noisy_inr10", "S4_noisy_inr0"):
            res.pop(k, None)
        res["S4_noiseless_inr30"] = s4_recovery(30.0, noise=False)
        res["S4_noisy_inr10"] = s4_recovery(10.0, noise=True)
        res["S4_noisy_inr0"] = s4_recovery(0.0, noise=True)
        json.dump(res, open(os.path.join(OUT, "sanity.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        return
    res = {}
    print("S1 H0 theory"); res["S1_h0"] = s1_h0()
    print(json.dumps(res["S1_h0"], indent=1))
    print("S2 dechirp"); res["S2_dechirp"] = s2_dechirp()
    print(json.dumps(res["S2_dechirp"], indent=1))
    print("S3 bispectrum"); res["S3_bispectrum"] = s3_bispec()
    print(json.dumps(res["S3_bispectrum"], indent=1))
    res["S4_noiseless_inr30"] = s4_recovery(30.0, noise=False)
    res["S4_noisy_inr10"] = s4_recovery(10.0, noise=True)
    res["S4_noisy_inr0"] = s4_recovery(0.0, noise=True)
    with open(os.path.join(OUT, "sanity.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["S4_noiseless_inr30"], indent=1))


if __name__ == "__main__":
    main()
