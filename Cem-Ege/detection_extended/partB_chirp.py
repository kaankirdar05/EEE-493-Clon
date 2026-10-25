"""Part B - linear-chirp detection: spectrogram vs (S)PWVD Radon vs FrFT / dechirp.

    python partB_chirp.py sanity      # WVD concentration on the IF line, FrFT peak angle
    python partB_chirp.py bench       # Pd at Pfa 1e-2 vs SNR + chirp-rate error

Synthetic model (frame N = 4096, unit-power complex AWGN):
    s(n) = A exp(j(2 pi f0 n' + pi k n'^2 + phi)),  n' = n - N/2
    f0 ~ U[-0.2, 0.2], sweep B = |k| N ~ U[0.05, 0.5] cycles/sample, random sign,
    phi ~ U[0, 2pi). SNR = A^2 (per-sample, full band). All parameters unknown
    to every detector (each maximises over its own parameter grid); thresholds =
    empirical (1 - 1e-2) quantile of the statistic over 2000 noise-only frames.

Statistics
    radiometer   sum |x|^2 (known noise power): the no-structure baseline
    spec_max     max cell of a 128-sample Hann spectrogram (hop 16, 512 bins)
    spec_radon   max line integral of that spectrogram over (f0, k)
    pwvd_radon   max line integral of the pseudo-WVD (Hann lag window, 255 lags of the
                 2x-upsampled signal)
    spwvd_radon  same with 64-sample Hann time smoothing (SPWVD)
    frft         max over k (2049 rates, step 2/N^2) and f of |FFT(x e^{-j pi k n'^2})|^2 / N
                 = |FrFT_alpha|^2 peak search over alpha = arccot(-kN)
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ext.chirp import (chirp, dechirp_search, frft_angle, frft_mag, k_grid_tf, line_integrals, pwvd,  # noqa: E402
                       stft_power)

OUT = os.path.join(HERE, "results", "partB")
N = 4096
M = 512
HOP = 16
KS_TF = k_grid_tf(N, M, 257)                       # 257 rates for the TF line search
KS_FR = np.arange(-0.5 / N, 0.5 / N + 1e-15, 2 / N**2)   # 2049 rates for dechirp
METHODS = ["radiometer", "spec_max", "spec_radon", "pwvd_radon", "spwvd_radon", "frft"]
SNRS = list(range(-32, -6, 2))


def draw(rng):
    f0 = rng.uniform(-0.2, 0.2)
    k = rng.choice([-1, 1]) * rng.uniform(0.05, 0.5) / N
    return f0, k, rng.uniform(0, 2 * np.pi)


def statistics(x):
    out, khat = {}, {}
    out["radiometer"] = float(np.mean(np.abs(x) ** 2))
    S = stft_power(x, M, HOP, 128)
    out["spec_max"] = float(S.max())
    for name, W in [("spec_radon", S), ("pwvd_radon", pwvd(x, M, HOP, 255)), ("spwvd_radon", pwvd(x, M, HOP, 255, g_len=64))]:
        li = line_integrals(W, N, KS_TF, HOP)
        j = np.unravel_index(li.argmax(), li.shape)
        out[name], khat[name] = float(li[j]), float(KS_TF[j[0]])
    pk, _ = dechirp_search(x, KS_FR)
    j = int(pk.argmax())
    # parabolic refinement of the rate estimate on the k grid
    if 0 < j < len(pk) - 1:
        a, b, c = pk[j - 1], pk[j], pk[j + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        d = 0.0
    out["frft"], khat["frft"] = float(pk[j]), float(KS_FR[j] + d * (KS_FR[1] - KS_FR[0]))
    return out, khat


def trial(args):
    seed, snr_db = args
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    if snr_db is None:
        st, kh = statistics(w)
        return {"snr": None, "stat": st}
    f0, k, phi = draw(rng)
    x = w + chirp(N, f0, k, 10 ** (snr_db / 20), phi)
    st, kh = statistics(x)
    return {"snr": snr_db, "stat": st, "k": k, "f0": f0, "khat": kh}


# ------------------------------------------------------------------ sanity
def sanity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    # on the dechirp search grid (f0 on the 2N-point FFT grid, k on the 2/N^2 rate grid) so the
    # coherent-gain check is exact; an off-grid chirp (f0 = 0.07, k = 0.3/N) is reported separately.
    f0, k = 576 / (2 * N), 1228 / N**2
    x = chirp(N, f0, k)
    off, _ = dechirp_search(chirp(N, 0.07, 0.3 / N), KS_FR)
    n_c = np.arange(HOP // 2, N, HOP) - N / 2
    if_bin = (f0 + k * n_c) * M + M / 2
    res = {"chirp": {"N": N, "f0": f0, "k": k, "B": k * N}}
    tfs = {"spectrogram": stft_power(x, M, HOP, 128), "pwvd": pwvd(x, M, HOP, 255), "spwvd": pwvd(x, M, HOP, 255, g_len=64)}
    inner = slice(16, -16)   # away from frame edges where the lag window is truncated
    for name, W in tfs.items():
        am = W.argmax(1)
        err = np.abs(am - if_bin)[inner]
        Wp = np.maximum(W, 0)[inner]
        near = np.abs(np.arange(M)[None, :] - if_bin[inner, None]) <= 2
        res[name] = {"max_abs_peak_minus_IF_bins": float(err.max()),
                     "mean_abs_peak_minus_IF_bins": float(err.mean()),
                     "energy_fraction_within_2_bins_of_IF": float((Wp * near).sum() / Wp.sum())}
    # FrFT: magnitude peak vs rotation angle
    a_true = frft_angle(k, N)
    alphas = np.linspace(a_true - 0.05, a_true + 0.05, 401)
    pk = np.array([frft_mag(x, a).max() for a in alphas])
    a_hat = float(alphas[pk.argmax()])
    dpk, _ = dechirp_search(x, KS_FR)
    res["frft"] = {"alpha_theory_rad": float(a_true), "alpha_peak_rad": a_hat, "alpha_step_rad": float(alphas[1] - alphas[0]),
                   "dechirp_khat_over_k": float(KS_FR[dpk.argmax()] / k),
                   "dechirp_peak_over_N": float(dpk.max() / N),
                   "dechirp_peak_over_N_offgrid_chirp": float(off.max() / N)}
    # Radon-Wigner identity: line integral of the FULL (non-pseudo) WVD peaks at the same rate as dechirp
    rng = np.random.default_rng(5)
    xn = chirp(N, f0, k, 10 ** (-15 / 20), 0.3) + (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    Mf, Lf = 8192, 4095
    Wf = pwvd(xn, Mf, 1, Lf, lag_win="rect", agg=False)       # full WVD at every sample
    dk = 2 / N**2
    ks = k + dk * np.arange(-6, 7)
    n_all = np.arange(N) - N / 2
    rw = np.array([max(Wf[np.arange(N), (np.rint((f0 + d / Mf + kk * n_all) * Mf + Mf / 2).astype(int)) % Mf].sum()
                       for d in range(-8, 9)) for kk in ks])
    dc, _ = dechirp_search(xn, ks)
    res["radon_wigner_vs_dechirp_at_-15dB"] = {"k_offsets_in_2_over_N2": list(range(-6, 7)),
                                               "argmax_full_wvd_line": int(rw.argmax()) - 6,
                                               "argmax_dechirp": int(dc.argmax()) - 6,
                                               "corr_profiles": float(np.corrcoef(rw, dc)[0, 1])}
    ok = (res["pwvd"]["max_abs_peak_minus_IF_bins"] <= 1.0 and res["pwvd"]["energy_fraction_within_2_bins_of_IF"] > 0.8
          and abs(a_hat - a_true) <= 2 * res["frft"]["alpha_step_rad"] and abs(res["frft"]["dechirp_peak_over_N"] - 1) < 0.05
          and res["radon_wigner_vs_dechirp_at_-15dB"]["argmax_full_wvd_line"] == res["radon_wigner_vs_dechirp_at_-15dB"]["argmax_dechirp"])
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "sanity_chirp.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, W) in zip(axs, tfs.items()):
        ax.imshow(W.T if name != "spectrogram" else 10 * np.log10(W.T + 1e-6), aspect="auto", origin="lower",
                  extent=[n_c[0], n_c[-1], -0.5, 0.5], cmap="viridis")
        ax.plot(n_c, f0 + k * n_c, "r--", lw=0.8, label="true IF")
        ax.set_title(name)
        ax.set_xlabel("n' (samples)")
        ax.set_ylabel("f (cycles/sample)")
        ax.legend(fontsize=7)
    axs[3].plot(alphas, pk)
    axs[3].axvline(a_true, color="r", ls="--", label="theory arccot(-kN)")
    axs[3].set_xlabel("FrFT angle alpha (rad)")
    axs[3].set_ylabel("max_u |X_alpha(u)|")
    axs[3].legend(fontsize=7)
    axs[3].set_title("FrFT peak vs angle")
    fig.suptitle(f"Clean chirp f0={f0}, B=kN={k * N}: TF concentration and FrFT angle")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sanity_chirp.png"), dpi=120)


# ------------------------------------------------------------------ benchmark
def bench(n_h0=2000, n_h1=300, workers=20, pfa=1e-2):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(10_000_000 + i, None) for i in range(n_h0)]
    jobs += [(20_000_000 + 1000 * (s + 100) + i, s) for s in SNRS for i in range(n_h1)]
    with Pool(workers) as p:
        res = p.map(trial, jobs, chunksize=4)
    h0 = [r for r in res if r["snr"] is None]
    thr = {m: float(np.quantile([r["stat"][m] for r in h0], 1 - pfa)) for m in METHODS}
    table = []
    for s in SNRS:
        rs = [r for r in res if r["snr"] == s]
        row = {"snr_db": s, "n": len(rs)}
        for m in METHODS:
            row[f"pd_{m}"] = float(np.mean([r["stat"][m] > thr[m] for r in rs]))
        for m in ["spec_radon", "pwvd_radon", "spwvd_radon", "frft"]:
            e = np.array([abs(r["khat"][m] - r["k"]) * N for r in rs])       # sweep-bandwidth error (cycles/sample)
            rel = np.array([abs(r["khat"][m] - r["k"]) / abs(r["k"]) for r in rs])
            row[f"B_err_median_{m}"] = float(np.median(e))
            row[f"B_err_rmse_{m}"] = float(np.sqrt(np.mean(e**2)))
            row[f"k_relerr_median_{m}"] = float(np.median(rel))
        table.append(row)
    # SNR at Pd = 0.5 by linear interpolation
    snr50 = {}
    for m in METHODS:
        pd = np.array([r[f"pd_{m}"] for r in table])
        i = np.flatnonzero(pd >= 0.5)
        if len(i) and i[0] > 0:
            j = i[0]
            snr50[m] = float(SNRS[j - 1] + (0.5 - pd[j - 1]) / (pd[j] - pd[j - 1]) * (SNRS[j] - SNRS[j - 1]))
        else:
            snr50[m] = None
    out = {"N": N, "pfa": pfa, "n_h0": n_h0, "n_h1_per_snr": n_h1, "thresholds": thr,
           "measured_pfa_on_h0": {m: float(np.mean([r["stat"][m] > thr[m] for r in h0])) for m in METHODS},
           "snr_at_pd50": snr50, "table": table,
           "k_grid": {"tf_lines": len(KS_TF), "dechirp": len(KS_FR), "dechirp_step_1_over_N2": 2.0}}
    with open(os.path.join(OUT, "chirp_bench.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    plot(out)
    print("SNR @ Pd 0.5:", snr50)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = out["table"]
    s = [r["snr_db"] for r in t]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for m in METHODS:
        axs[0].plot(s, [r[f"pd_{m}"] for r in t], marker="o", ms=3, label=m)
    axs[0].set_xlabel("per-sample SNR (dB), N = 4096")
    axs[0].set_ylabel(f"Pd at Pfa {out['pfa']:g}")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[0].set_title("Linear chirp detection, unknown f0 and rate")
    for m in ["spec_radon", "pwvd_radon", "spwvd_radon", "frft"]:
        axs[1].semilogy(s, [r[f"B_err_median_{m}"] for r in t], marker="o", ms=3, label=m)
    axs[1].set_xlabel("per-sample SNR (dB)")
    axs[1].set_ylabel("median |B_hat - B| (cycles/sample)")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Sweep-bandwidth (chirp-rate x N) error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chirp_bench.png"), dpi=130)


if __name__ == "__main__":
    {"sanity": sanity, "bench": bench}[sys.argv[1]]()
