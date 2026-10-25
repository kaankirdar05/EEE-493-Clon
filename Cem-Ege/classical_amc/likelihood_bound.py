"""Likelihood-based modulation classification (ALRT / HLRT): the classical upper bound.

The likelihood-ratio classifier is optimal when the receiver knows everything
except the symbols: symbol timing, noise power (SNR) and the candidate
constellations, with only the carrier phase unknown. RadioML frames do not give
us those (pulse shaping, unknown timing and offsets), so this script tests it on
symbol-level synthetic frames where the assumptions hold:

    r_n = s_n e^{j theta} + w_n,  s_n from the class constellation (unit power),
    theta ~ U[0, 2 pi), w_n ~ CN(0, sigma^2), sigma^2 known, perfect timing.

Classifiers compared on the SAME frames:
    alrt          average likelihood over a phase grid (optimal, Bayesian phase)
    hlrt          maximum likelihood over the phase grid (hybrid LRT)
    features_hgb  the job-5.1 features + gradient-boosted trees, trained on
                  separate frames of the same kind (shows the gap to optimal)
    cumulant_theory  nearest theoretical (|C40|, C42), training-free

    python likelihood_bound.py                     # 5 classes, N = 128 symbols
    python likelihood_bound.py --symbols 512 --per-snr 300

Outputs in results/likelihood_bound/: metrics.json, acc_vs_snr.csv/png, confusion_alrt.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.special import logsumexp

from amc.features import FEATURE_NAMES, extract_features
from amc.synthetic import _constellation

CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]


def constellations(classes):
    out = {}
    for c in classes:
        pts = _constellation(c)
        out[c] = (pts / np.sqrt(np.mean(np.abs(pts) ** 2))).astype(np.complex128)
    return out


def make_frames(classes, snrs, per_snr, n_sym, rng):
    cons = constellations(classes)
    xs, ys, ss = [], [], []
    for k, c in enumerate(classes):
        for s in snrs:
            sig = rng.choice(cons[c], size=(per_snr, n_sym))
            theta = rng.uniform(0, 2 * np.pi, size=(per_snr, 1))
            sigma2 = 10 ** (-s / 10)
            w = np.sqrt(sigma2 / 2) * (rng.standard_normal((per_snr, n_sym)) + 1j * rng.standard_normal((per_snr, n_sym)))
            xs.append(sig * np.exp(1j * theta) + w)
            ys.append(np.full(per_snr, k))
            ss.append(np.full(per_snr, s))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)


def log_likelihoods(r, sigma2, cons, n_phase=32, batch=256):
    """(frames, classes, phases) log p(r | class, theta), known sigma^2 per frame."""
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, n_phase, endpoint=False))
    out = np.empty((len(r), len(cons), n_phase))
    for b in range(0, len(r), batch):
        rb = r[b:b + batch][:, None, :, None]                      # (B, 1, N, 1)
        s2 = sigma2[b:b + batch][:, None, None, None]
        for k, pts in enumerate(cons.values()):
            ref = (phases[:, None] * pts[None, :])[None, :, None, :]  # (1, P, 1, M)
            d = -np.abs(rb - ref) ** 2 / s2                       # (B, P, N, M)
            ll = logsumexp(d, axis=-1) - np.log(len(pts))          # (B, P, N)
            out[b:b + batch, k] = ll.sum(-1)
    return out


def cumulant_theory_predict(F, classes):
    cons = constellations(classes)
    theory = []
    for c in classes:
        x = cons[c]
        m20, m40, m42 = np.mean(x**2), np.mean(x**4), np.mean(np.abs(x) ** 4)
        theory.append([abs(m40 - 3 * m20**2), m42 - abs(m20) ** 2 - 2])
    theory = np.array(theory)
    v = F[:, [FEATURE_NAMES.index("abs_C40"), FEATURE_NAMES.index("C42")]]
    return np.argmin(((v[:, None, :] - theory[None]) ** 2).sum(-1), axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", type=int, default=128)
    p.add_argument("--per-snr", type=int, default=200, help="test frames per (class, SNR)")
    p.add_argument("--train-per-snr", type=int, default=400, help="training frames per (class, SNR) for features_hgb")
    p.add_argument("--snr-min", type=int, default=-10)
    p.add_argument("--snr-max", type=int, default=20)
    p.add_argument("--phases", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results", "likelihood_bound"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import confusion_matrix

    rng = np.random.default_rng(args.seed)
    snrs = list(range(args.snr_min, args.snr_max + 1, 2))
    cons = constellations(CLASSES)
    x, y, snr = make_frames(CLASSES, snrs, args.per_snr, args.symbols, rng)
    xtr, ytr, _ = make_frames(CLASSES, snrs, args.train_per_snr, args.symbols, rng)
    sigma2 = 10 ** (-snr / 10)

    preds, timing = {}, {}
    t0 = time.perf_counter()
    LL = log_likelihoods(x, sigma2, cons, args.phases)
    t_ll = time.perf_counter() - t0
    preds["alrt"] = np.argmax(logsumexp(LL, axis=2), axis=1)
    preds["hlrt"] = np.argmax(LL.max(axis=2), axis=1)
    timing["alrt"] = timing["hlrt"] = t_ll / len(x) * 1e3

    t0 = time.perf_counter()
    Ftr, Fte = extract_features(xtr), extract_features(x)
    t_feat = (time.perf_counter() - t0) / (len(xtr) + len(x))
    clf = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=args.seed).fit(Ftr, ytr)
    preds["features_hgb"] = clf.predict(Fte)
    timing["features_hgb"] = t_feat * 1e3
    preds["cumulant_theory"] = cumulant_theory_predict(Fte, CLASSES)
    timing["cumulant_theory"] = t_feat * 1e3

    metrics = {"classes": CLASSES, "symbols_per_frame": args.symbols, "test_frames": int(len(y)),
               "assumptions": "perfect timing, known SNR, unknown uniform carrier phase, AWGN, no pulse shaping",
               "models": {}}
    rows = ["snr_db," + ",".join(preds)]
    for s in snrs:
        rows.append(f"{s}," + ",".join(f"{np.mean(pr[snr == s] == y[snr == s]):.4f}" for pr in preds.values()))
    for name, pr in preds.items():
        acc_snr = {int(s): float(np.mean(pr[snr == s] == y[snr == s])) for s in snrs}
        reach = next((s for s in snrs if acc_snr[s] >= 0.85), None)
        metrics["models"][name] = {"accuracy_all": float(np.mean(pr == y)),
                                   "accuracy_snr_ge_10": float(np.mean(pr[snr >= 10] == y[snr >= 10])),
                                   "snr_first_85": reach, "ms_per_frame": timing[name],
                                   "accuracy_vs_snr": acc_snr}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "acc_vs_snr.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name in preds:
        ax.plot(snrs, [metrics["models"][name]["accuracy_vs_snr"][s] * 100 for s in snrs], marker="o", label=name)
    ax.axhline(85, color="gray", ls="--", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Likelihood bound vs features ({', '.join(CLASSES)}; N={args.symbols})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "acc_vs_snr.png"), dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y, preds["alrt"])
    print(f"{len(y)} test frames, {args.symbols} symbols/frame, classes {CLASSES}")
    for name, m in metrics["models"].items():
        print(f"  {name:>16s}: all {m['accuracy_all'] * 100:5.1f} %  SNR>=10 {m['accuracy_snr_ge_10'] * 100:5.1f} %  "
              f"85 % first at {m['snr_first_85']} dB  {m['ms_per_frame']:.3f} ms/frame")
    print("ALRT confusion (rows true):\n", cm)
    print(f"Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
