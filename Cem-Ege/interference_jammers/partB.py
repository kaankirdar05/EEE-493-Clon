"""Part B - 'is a jammer present?' with a legitimate QPSK signal in the band, at a fixed false-alarm rate.

Legitimate signal: QPSK, RRC 0.35, 1-5 % of fs, random centre, SNR 10 dB per sample, actual level
U(-0.5, +0.5) dB around the nominal value (the detector knows centre + bandwidth + nominal power).
Receiver noise power known to +-0.25 dB (same convention as Part A).

Detectors (all thresholds = empirical (1 - Pfa/m) quantile of the statistic on JAMMER-FREE frames with the signal
present, seeds role CAL; realised Pfa checked on fresh jammer-free frames, role TEST):
    ED        energy detector in the signal's own band (power of FFT bins in [fc +- bw/2] / expected power - 1)
    OOB-pow   mean spectrogram power outside the signal band (guard bw/2 + 4 bins): out-of-band occupancy
    OOB-lines max/median of the time-averaged out-of-band spectrum: narrow spectral lines / spot noise
    OOB-time  mean of the top-5 % / median of the 64-sample block power of the out-of-band-filtered frame: duty-cycle anomaly
    OOB       any of the three OOB statistics, each at Pfa/3
    fused     ED or any OOB statistic, each at Pfa/4
Jammer families: noise (barrage 30-100 % fs), sweep (sawtooth chirp), pulsed (gated noise), plus co-channel spot noise
(centred on the signal, 1.5 x its bandwidth) as the case that is invisible out of band.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import multiprocessing as mp
import sys
import time

import numpy as np

import dataset
import jcommon as common
from jcommon import N
from wp4.spectrum import spectrogram

OUT = os.path.join(common.HERE, "results")
PFA = 0.01
SIG_SNR = 10.0
SIG_UNC_DB = 0.5
JSRS = [-30, -25, -20, -15, -10, -5, 0, 5, 10]
FAMS = {"noise": "barrage", "sweep": "sweep", "pulsed": "pnoise", "cochannel_spot": "spot_cochannel"}
N_H1, N_CAL, N_H0 = 100, 4000, 2000
STATS = ["ED", "OOB-pow", "OOB-lines", "OOB-time"]


def stats_frame(x, fc, bw, cal_db):
    xn = x / np.sqrt(10 ** (cal_db / 10))
    X = np.fft.fft(xn.astype(np.complex128))
    f = np.fft.fftfreq(N)
    inb = np.abs(f - fc) <= bw / 2
    p_in = np.sum(np.abs(X[inb]) ** 2) / N ** 2
    p_ref = 10 ** (SIG_SNR / 10) + bw
    ed = p_in / p_ref - 1
    guard = bw / 2 + 4 / 512
    S = spectrogram(xn.astype(np.complex64), 512)
    fb = (np.arange(512) - 256) / 512
    oob = np.abs(fb - fc) > guard
    o_pow = 10 * np.log10(S[:, oob].mean())
    Pb = S[:, oob].mean(0)
    o_pk = Pb.max() / np.median(Pb)
    mask = np.abs(f - fc) > guard
    y = np.fft.ifft(X * mask)
    pw = np.mean(np.abs(y[: (N // 64) * 64]).reshape(-1, 64) ** 2, axis=1)
    o_top = np.sort(pw)[-len(pw) // 20:].mean() / np.median(pw)
    return [ed, o_pow, o_pk, o_top]


def task(t):
    kind, jsr, idx, role = t
    ki = list(FAMS.values()).index(kind) if kind != "none" else 9
    rng = np.random.default_rng([role, 50 + ki, int(jsr + 100), idx])
    gain = rng.uniform(-SIG_UNC_DB, SIG_UNC_DB)
    if kind == "none":
        x, meta = common.make_frame("none", rng, with_sig=True, sig_snr_db=SIG_SNR, sig_gain_db=gain)
    else:
        x, meta = common.make_frame(kind, rng, with_sig=True, sig_snr_db=SIG_SNR, sig_gain_db=gain, jsr_db=jsr)
    cal = rng.uniform(-dataset.CAL_ERR_DB, dataset.CAL_ERR_DB)
    return stats_frame(x, meta["sig"]["fc"], meta["sig"]["bw"], cal)


def run(tasks, label):
    t0 = time.time()
    with mp.Pool(dataset.NPROC) as pool:
        out = pool.map(task, tasks, chunksize=8)
    print(f"  {label}: {len(tasks)} frames {time.time() - t0:.0f}s", flush=True)
    return np.array(out)


def gen():
    path = os.path.join(OUT, "cache_partB.npz")
    if os.path.exists(path):
        return
    cal = run([("none", 0, i, dataset.ROLE_CAL) for i in range(N_CAL)], "H0 cal")
    h0 = run([("none", 0, i, dataset.ROLE_TEST) for i in range(N_H0)], "H0 test")
    t = [(k, j, i, dataset.ROLE_TEST) for k in FAMS.values() for j in JSRS for i in range(N_H1)]
    h1 = run(t, "H1")
    np.savez(path, cal=cal, h0=h0, h1=h1)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(OUT, "cache_partB.npz"))
    cal, h0, h1 = d["cal"], d["h0"], d["h1"].reshape(len(FAMS), len(JSRS), N_H1, 4)

    def thr_for(cols, pfa_each):
        return {c: float(np.quantile(cal[:, c], 1 - pfa_each)) for c in cols}

    def make(cols):
        th = thr_for(cols, PFA / len(cols))
        return th, (lambda A: np.any(np.stack([A[..., c] > th[c] for c in cols], -1), -1))

    dets = {"ED": [0], "OOB-pow": [1], "OOB-lines": [2], "OOB-time": [3], "OOB (3 stats, Bonferroni)": [1, 2, 3], "fused ED+OOB (4 stats)": [0, 1, 2, 3]}
    single_at_pfa = {"ED": 1.0, "OOB-pow": 1.0, "OOB-lines": 1.0, "OOB-time": 1.0}
    R = {"pfa_target": PFA, "n_cal": int(len(cal)), "n_h0_test": int(len(h0)), "n_per_cell": N_H1, "jsr_db": JSRS,
         "sig_snr_db": SIG_SNR, "sig_level_uncertainty_db": SIG_UNC_DB, "cal_error_db": dataset.CAL_ERR_DB, "detectors": {}}
    for name, cols in dets.items():
        if len(cols) == 1:
            th = {cols[0]: float(np.quantile(cal[:, cols[0]], 1 - PFA))}       # single statistic: full Pfa
            fn = (lambda A, c=cols[0], th=th: A[..., c] > th[c])
        else:
            th, fn = make(cols)
        pfa_fresh = float(fn(h0).mean())
        k = int(fn(h0).sum())
        pd = {fam: [float(fn(h1[fi, ji]).mean()) for ji in range(len(JSRS))] for fi, fam in enumerate(FAMS)}
        R["detectors"][name] = dict(thresholds={str(c): v for c, v in th.items()}, pfa_fresh=pfa_fresh, fa_count=k, pd=pd)
        # JSR at which Pd first reaches 50 % / 90 % (linear interpolation)
        for fam in FAMS:
            for lev in (0.5, 0.9):
                p = np.array(pd[fam]); j = np.array(JSRS, float)
                if p.max() < lev:
                    v = None
                elif p[0] >= lev:
                    v = float(j[0])
                else:
                    i = np.flatnonzero(p >= lev)[0]
                    v = float(j[i - 1] + (lev - p[i - 1]) / (p[i] - p[i - 1]) * (j[i] - j[i - 1]))
                R["detectors"][name].setdefault(f"jsr_at_pd{int(lev * 100)}", {})[fam] = v
    with open(os.path.join(OUT, "partB_results.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(OUT, "partB_pd_vs_jsr.csv"), "w") as f:
        f.write("detector,family,jsr_db,pd,n\n")
        for name in dets:
            for fam in FAMS:
                for ji, j in enumerate(JSRS):
                    f.write(f"{name},{fam},{j},{R['detectors'][name]['pd'][fam][ji]:.4f},{N_H1}\n")

    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#0b0b0b"]
    ls = ["-", "-", "-", "-", "--", "-"]
    lw = [2.2, 1.2, 1.2, 1.2, 2.2, 2.6]
    mk = ["o", "s", "^", "D", "v", "P"]
    ink, mut, grid_c, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": mut, "xtick.color": mut, "ytick.color": mut,
                         "axes.facecolor": surf, "figure.facecolor": surf, "axes.titlecolor": ink})
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    titles = {"noise": "noise (barrage 30-100 % fs)", "sweep": "sweep (sawtooth chirp)", "pulsed": "pulsed noise (duty 5-50 %)",
              "cochannel_spot": "co-channel spot noise (1.5 x signal bw)"}
    for ax, fam in zip(axs, FAMS):
        for di, name in enumerate(dets):
            ax.plot(JSRS, [100 * v for v in R["detectors"][name]["pd"][fam]], color=cols[di], ls=ls[di], lw=lw[di], marker=mk[di], ms=4, label=name)
        ax.axhline(100 * PFA, color=mut, ls=":", lw=1)
        ax.set_title(titles[fam], fontsize=10, loc="left"); ax.set_xlabel("jammer-to-signal ratio JSR (dB)")
        ax.grid(color=grid_c, lw=0.6); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(-2, 102)
    axs[0].set_ylabel(f"Pd (%)   [Pfa = {PFA:.0%}]"); axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(f"Part B - jammer-present detection with a QPSK signal (SNR {SIG_SNR:.0f} dB) in the band; fresh seeds, {N_H1} frames per point, "
                 f"thresholds from {len(cal)} jammer-free frames", fontsize=11, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "partB_overview.png"), dpi=130, bbox_inches="tight")
    print(json.dumps({n: dict(pfa_fresh=v["pfa_fresh"], jsr50=v["jsr_at_pd50"]) for n, v in R["detectors"].items()}, indent=1))


if __name__ == "__main__":
    gen() if sys.argv[1] == "gen" else report()
