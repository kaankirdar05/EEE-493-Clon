"""Part A1 - OFDM vs single-carrier with fourth-order cumulants + the cyclic-prefix statistic.

Classes: OFDM (N in {64..1024}, cp = N/4..N/32, QPSK / 16QAM subcarriers, 75 % used, CFO up to +-0.3 spacings) and single-carrier
BPSK / QPSK / 16QAM / 64QAM (RRC 0.35, 2-8 samples/symbol, CFO up to +-0.01). Frames of 8192 samples, random phase/timing, AWGN
(SNR = per-sample SNR, -5..25 dB).

Features
  c40_peak  max over frequency of |FFT(x^4)|/L / ps^2   (4th-order line search: the zero-lag |C40| is destroyed by any CFO)
  c42       (E|x|^4 - |E x^2|^2 - 2 (E|x|^2)^2) / ps^2   ps = blind signal power = total power - spectral noise floor
  z_cp      max over the 20 (N, cp) candidates of ofdm_params' standardised cyclic-prefix correlation
  snr_hat   10 log10(ps / noise floor)
Classifiers: sklearn HistGradientBoosting (depth 3) trained on a SEPARATE seed (1) from the test seed (2).

    python a1_ofdm_vs_sc.py --check     (sanity checks: Gaussian cumulants, OFDM cumulants vs 1/N_used theory, SC theory)
    python a1_ofdm_vs_sc.py             (train on seed 1, test on seed 2, write results/a1_*.json/png)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from joblib import Parallel, delayed

from ofdmlib import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SC_MODS = ["BPSK", "QPSK", "16QAM", "64QAM"]
CLASSES5 = ["OFDM"] + SC_MODS
SNRS = [-5, 0, 5, 10, 15, 20, 25]
NS = (64, 128, 256, 512, 1024)
CPD = (4, 8, 16, 32)
FEATS = ["c40_peak", "c40_0", "c42", "snr_hat", "c40_peak_raw", "c42_raw"]


def one_frame(seed, i, snr):
    rng = np.random.default_rng([seed, i])
    lab = int(rng.integers(0, 5)) if False else (0 if rng.random() < 0.5 else 1 + int(rng.integers(0, 4)))
    if lab == 0:
        n = int(rng.choice(NS)); cp = n // int(rng.choice(CPD)); mod = str(rng.choice(["QPSK", "16QAM"]))
        x, meta = make_ofdm(n, cp, mod, snr, rng)
        extra = {"n": n, "cp": cp, "mod": mod}
    else:
        x, meta = make_sc(SC_MODS[lab - 1], snr, rng)
        extra = {"sps": meta["sps"]}
    f = cumulants(x)
    z = op.estimate(x)[1]
    f["z_cp"] = float(z.max())
    return lab, snr, [f[k] for k in FEATS] + [f["z_cp"]], extra


def gen(seed, snrs, per_snr=None, n_train=None):
    jobs = []
    if per_snr is not None:
        for s in snrs:
            for j in range(per_snr):
                jobs.append((seed, len(jobs), s))
    else:
        r = np.random.default_rng([seed, 999])
        for j in range(n_train):
            jobs.append((seed, j, float(r.uniform(-5, 25))))
    out = Parallel(n_jobs=20, batch_size=16)(delayed(one_frame)(*j) for j in jobs)
    y = np.array([o[0] for o in out]); snr = np.array([o[1] for o in out]); F = np.array([o[2] for o in out])
    extra = [o[3] for o in out]
    return y, snr, F, extra


def check():
    rng = np.random.default_rng(0)
    res = {}
    # 1. Gaussian noise: all cumulants 0
    g = [cumulants((rng.standard_normal(L_FRAME) + 1j * rng.standard_normal(L_FRAME)) / np.sqrt(2)) for _ in range(300)]
    c42 = np.array([a["c42_raw"] for a in g]); c40 = np.array([a["c40_0_raw"] for a in g]); pk = np.array([a["c40_peak_raw"] for a in g])
    print(f"[Gaussian noise, L={L_FRAME}, 300 trials]  C42: mean {c42.mean():+.4f} std {c42.std():.4f} | |C40| (zero lag): mean {c40.mean():.4f} | "
          f"C40 line-search peak (a max over {2*L_FRAME} bins -> positive noise floor): mean {pk.mean():.3f}")
    res["gaussian"] = {"c42_mean": float(c42.mean()), "c42_std": float(c42.std()), "c40_0_mean": float(c40.mean()), "c40_peak_mean": float(pk.mean())}
    # 2. OFDM: C42 = c42(s)/N_used, C40 -> 0 (pilot-free, noiseless, no CFO)
    print("\n[OFDM noiseless, pilots off, CFO off; 300 frames per row]   theory C42 = c42(s)/N_used (QPSK -1, 16QAM -0.68); |C40| -> 0")
    print(f"{'N':>5s} {'N_used':>6s} | {'QPSK C42':>9s} {'theory':>8s} | {'16QAM C42':>9s} {'theory':>8s} | {'|C40|0 QPSK':>11s} {'peak':>6s}")
    res["ofdm"] = {}
    for n in NS:
        used, _, _ = layout(n); nu = int(used.sum())
        row = {}
        for mod, c42s in (("QPSK", -1.0), ("16QAM", -0.68)):
            v = [cumulants(make_ofdm(n, n // 8, mod, 100, rng, cfo=False, pilots=False, noise=False)[0]) for _ in range(300)]
            row[mod] = (float(np.mean([a["c42_raw"] for a in v])), c42s / nu, float(np.mean([a["c40_0_raw"] for a in v])), float(np.mean([a["c40_peak_raw"] for a in v])))
        print(f"{n:5d} {nu:6d} | {row['QPSK'][0]:+9.4f} {row['QPSK'][1]:+8.4f} | {row['16QAM'][0]:+9.4f} {row['16QAM'][1]:+8.4f} | {row['QPSK'][2]:11.4f} {row['QPSK'][3]:6.3f}")
        res["ofdm"][n] = row
    # 3. single carrier, sample-domain theory: C42 = c42(s) * T sum h^4 / (sum h^2)^2
    print("\n[single-carrier, noiseless, no CFO, 300 frames]  C42_sample = c42(s) * T*sum(h^4)/(sum h^2)^2 + 2 Var_phase(E|x|^2)/mean^2  (RRC 0.35; 2nd term: time-averaged moments of a cyclostationary signal)")
    from amc.synthetic import _rrc
    res["sc"] = {}
    c42s = {"QPSK": -1.0, "16QAM": -0.68, "64QAM": -0.619}
    for sps in (2, 4, 8):
        h = _rrc(0.35, 8, sps); fac = sps * np.sum(h ** 4) / np.sum(h ** 2) ** 2
        sig = np.array([np.sum(h[np.arange(p, len(h), sps)] ** 2) for p in range(sps)])      # E|x_n|^2 over the symbol phase
        cyc = 2 * (np.mean(sig ** 2) / np.mean(sig) ** 2 - 1)                                  # cyclostationarity term of time-averaged moments
        line = []
        for mod, cs in c42s.items():
            vals = []
            for _ in range(300):
                nsym = L_FRAME // sps + 64
                up = np.zeros(nsym * sps, complex); up[::sps] = rng.choice(MODS[mod], nsym)
                s = np.convolve(up, h, "same")[8 * sps:8 * sps + L_FRAME]
                vals.append(cumulants(s)["c42_raw"])
            line.append(f"{mod}: {np.mean(vals):+.3f} (theory {cs * fac + cyc:+.3f})")
            res["sc"][f"{mod}_sps{sps}"] = (float(np.mean(vals)), float(cs * fac + cyc))
        print(f"  sps={sps}:  " + "   ".join(line))
    json.dump(res, open(os.path.join(RES, "a1_sanity.json"), "w"), indent=1)
    ok = abs(res["gaussian"]["c42_mean"]) < 0.01 and all(abs(res["ofdm"][n]["QPSK"][0] - res["ofdm"][n]["QPSK"][1]) < 0.01 for n in NS)
    return ok


def acc_by_snr(pred, y, snr):
    return {int(s): float(np.mean(pred[snr == s] == y[snr == s])) for s in sorted(set(snr.tolist()))}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    ytr, str_, Ftr, _ = gen(1, None, n_train=20000)
    print(f"train frames {len(ytr)} ({time.time() - t0:.0f} s)", flush=True)
    yte, ste, Fte, ex = gen(2, SNRS, per_snr=1000)
    print(f"test frames {len(yte)} ({time.time() - t0:.0f} s)", flush=True)
    col = {k: i for i, k in enumerate(FEATS + ["z_cp"])}
    sets = {"cumulants only (c40_peak, c42, snr_hat)": ["c40_peak", "c42", "snr_hat"],
            "cumulants only, no SNR estimate (raw-normalised)": ["c40_peak_raw", "c42_raw"],
            "CP statistic only (z_cp)": ["z_cp"],
            "cumulants + CP statistic": ["c40_peak", "c42", "snr_hat", "z_cp"]}
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "per_snr_test": 1000, "snrs": SNRS, "test_seed": 2, "train_seed": 1,
           "binary": {}, "five_class": {}}
    bin_tr, bin_te = (ytr == 0).astype(int), (yte == 0).astype(int)
    for name, fs in sets.items():
        idx = [col[k] for k in fs]
        m = HGB(max_depth=3, max_iter=150, learning_rate=0.1, random_state=0).fit(Ftr[:, idx], bin_tr)
        p = m.predict(Fte[:, idx])
        out["binary"][name] = {"acc_all": float(np.mean(p == bin_te)), "acc_vs_snr": acc_by_snr(p, bin_te, ste),
                                "ofdm_recall_vs_snr": {int(s): float(np.mean(p[(ste == s) & (bin_te == 1)] == 1)) for s in SNRS},
                                "sc_recall_vs_snr": {int(s): float(np.mean(p[(ste == s) & (bin_te == 0)] == 0)) for s in SNRS}}
        print(f"binary  {name:>50s}: " + " ".join(f"{s:+d}:{out['binary'][name]['acc_vs_snr'][s] * 100:5.1f}" for s in SNRS) + f"  | all {out['binary'][name]['acc_all'] * 100:.1f}", flush=True)
    idx = [col[k] for k in sets["cumulants + CP statistic"]]
    m5 = HGB(max_depth=3, max_iter=150, learning_rate=0.1, random_state=0).fit(Ftr[:, idx], ytr)
    p5 = m5.predict(Fte[:, idx])
    out["five_class"]["acc_vs_snr"] = acc_by_snr(p5, yte, ste)
    out["five_class"]["acc_all"] = float(np.mean(p5 == yte))
    out["five_class"]["ofdm_vs_sc_from_5class_vs_snr"] = acc_by_snr((p5 == 0).astype(int), bin_te, ste)
    mk = ste == 10
    cm = np.zeros((5, 5), int)
    for a, b in zip(yte[mk], p5[mk]):
        cm[a, b] += 1
    out["five_class"]["confusion_10dB"] = {"classes": CLASSES5, "rows_true_cols_pred": cm.tolist()}
    print("5-class confusion at 10 dB (rows = true, cols = predicted):", CLASSES5)
    print(cm)
    # binary confusion at 10 dB for the combined model, by OFDM (N, cp) and SC mod/sps
    idxb = [col[k] for k in sets["cumulants + CP statistic"]]
    mb = HGB(max_depth=3, max_iter=150, learning_rate=0.1, random_state=0).fit(Ftr[:, idxb], bin_tr)
    pb = mb.predict(Fte[:, idxb])
    r = {}
    for s in (0, 10, 20):
        sel = (ste == s) & (yte == 0)
        r[f"snr{s}_ofdm_recall_by_N"] = {int(n): float(np.mean(pb[sel & np.array([e.get('n') == n for e in ex])] == 1)) for n in NS}
        r[f"snr{s}_ofdm_recall_by_cp_ratio"] = {f"N/{d}": float(np.mean(pb[sel & np.array([e.get('n') is not None and e['n'] // e['cp'] == d for e in ex])] == 1)) for d in CPD}
        r[f"snr{s}_sc_false_alarm_by_sps"] = {int(q): float(np.mean(pb[(ste == s) & (yte > 0) & np.array([e.get('sps') == q for e in ex])] == 1)) for q in (2, 3, 4, 6, 8)}
        r[f"snr{s}_sc_false_alarm_by_mod"] = {SC_MODS[k - 1]: float(np.mean(pb[(ste == s) & (yte == k)] == 1)) for k in range(1, 5)}
    out["binary_breakdown_combined"] = r
    # limitation probe: frames that contain NO signal (white Gaussian noise) - what do the trained OFDM-vs-SC models say?
    rn = np.random.default_rng([2, 4242]); NF = []
    for _ in range(500):
        w = (rn.standard_normal(L_FRAME) + 1j * rn.standard_normal(L_FRAME)) / np.sqrt(2)
        f = cumulants(w); z = op.estimate(w)[1]; f["z_cp"] = float(z.max()); NF.append([f[k] for k in FEATS] + [f["z_cp"]])
    NF = np.array(NF)
    r["noise_only_called_OFDM_cumulants_plus_cp"] = float(np.mean(mb.predict(NF[:, idxb]) == 1))
    idx_c = [col[k] for k in sets["cumulants only (c40_peak, c42, snr_hat)"]]
    mc = HGB(max_depth=3, max_iter=150, learning_rate=0.1, random_state=0).fit(Ftr[:, idx_c], bin_tr)
    r["noise_only_called_OFDM_cumulants_only"] = float(np.mean(mc.predict(NF[:, idx_c]) == 1))
    print("pure-noise frames called OFDM: cumulants+CP", r["noise_only_called_OFDM_cumulants_plus_cp"], " cumulants only", r["noise_only_called_OFDM_cumulants_only"])
    for k, v in r.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    # feature statistics (medians) by class for the 10 dB frames
    stats = {}
    for k, nm in enumerate(CLASSES5):
        sel = (ste == 10) & (yte == k)
        stats[nm] = {f: float(np.median(Fte[sel, col[f]])) for f in ["c40_peak", "c42", "z_cp", "snr_hat"]}
    out["median_features_10dB"] = stats
    print("median features at 10 dB:", json.dumps(stats))
    json.dump(out, open(os.path.join(RES, "a1_ofdm_vs_sc.json"), "w"), indent=1)
    np.savez(os.path.join(RES, "a1_test_features.npz"), y=yte, snr=ste, F=Fte, feats=np.array(FEATS + ["z_cp"]))
    plot(out, Fte, yte, ste, col)


def plot(out, F, y, snr, col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, r in out["binary"].items():
        ax[0].plot(SNRS, [r["acc_vs_snr"][s] * 100 for s in SNRS], marker="o", label=f"{name} ({r['acc_all'] * 100:.1f} %)")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("OFDM vs single-carrier accuracy (%)"); ax[0].set_ylim(40, 101); ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_title(f"A1: OFDM vs single carrier ({out['n_test']} test frames, 1000/SNR)", fontsize=9)
    sel = snr == 10
    cols = {0: "k", 1: "tab:red", 2: "tab:orange", 3: "tab:blue", 4: "tab:green"}
    for k, nm in enumerate(CLASSES5):
        s = sel & (y == k)
        ax[1].scatter(F[s, col["c42"]], F[s, col["c40_peak"]], s=5, alpha=0.4, color=cols[k], label=nm)
    ax[1].set_xlabel("C42 (signal-power normalised)"); ax[1].set_ylabel("C40 line-search peak"); ax[1].set_title("features at 10 dB", fontsize=9); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    cm = np.array(out["five_class"]["confusion_10dB"]["rows_true_cols_pred"], float)
    cm = cm / cm.sum(1, keepdims=True)
    ax[2].imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax[2].text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", color="w" if cm[i, j] > 0.5 else "k", fontsize=9)
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(CLASSES5, fontsize=8); ax[2].set_yticks(range(5)); ax[2].set_yticklabels(CLASSES5, fontsize=8)
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true"); ax[2].set_title("5-class confusion at 10 dB (%)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "a1_ofdm_vs_sc.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.check:
        sys.exit(0 if check() else 1)
    main()
