"""The two classical classifiers named in the requirements document that were only partly covered.

1. Azzouz & Nandi (1995/96) style decision tree with a HAND-DESIGNED structure.
   Each node tests ONE classic feature and splits one group of classes; its threshold is
   chosen the way Azzouz & Nandi chose theirs: the value that minimises the number of wrong
   left/right decisions on training frames at the design SNR (>= 10 dB). (Their published
   numbers depend on their own feature normalisation and sample rate, so they are not
   copied.) Classes: the 7-class RadioML subset. Features: the 19-feature cache
   (sigma_af = std of |inst. frequency|, sigma_aa = std of |centred amplitude|, power-law
   lines x^1, x^2, x^4).

        N1 sigma_af  : {FM, GMSK}               | {AM-DSB-WC, BPSK, QPSK, 8PSK, 16QAM}
        N2 line_x2   : {GMSK}                   | {FM}            (MSK-type x^2 lines)
        N3 line_x1   : {AM-DSB-WC}              | {BPSK, QPSK, 8PSK, 16QAM}   (carrier line)
        N4 line_x2   : {BPSK}                   | {QPSK, 8PSK, 16QAM}
        N5 line_x4   : {QPSK, 16QAM}            | {8PSK}
        N6 sigma_aa  : {16QAM}                  | {QPSK}

2. Swami & Sadler (2000) HIERARCHICAL cumulant classifier, training-free.
   Cumulants on symbol-spaced samples (practical_alrt.to_symbols, symbol rate known = 8 on
   RadioML), normalised by the SIGNAL power C21 - N_hat (N_hat from the M2M4 estimator), so
   the noise does not shrink them. Thresholds are midpoints between theoretical values:
        stage 1  |C40| > (2 + 1)/2        -> BPSK
        stage 2  |C40| > (1 + 0.68)/2     -> QPSK
                 |C40| < (0.62 + 0)/2     -> 8PSK
        stage 3  among QAM: C42 > (-0.68 + -0.619)/2 -> 64QAM else 16QAM
   Classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM; same frames / split as practical_alrt.

Sanity checks (run first, --check): on noiseless ideal constellations the hierarchical
classifier is 100 % correct; on the training split the hand tree's node errors are printed.

    python requirements_methods.py --check
    python requirements_methods.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from amc.features import FEATURE_NAMES
from amc.synthetic import _constellation

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}
CLASSES7 = ["AM-DSB-WC", "FM", "BPSK", "QPSK", "8PSK", "16QAM", "GMSK"]
TREE = [  # (node, feature, left group, right group); "left" = feature value ABOVE threshold
    ("N1", "sigma_aa", {"FM", "GMSK"}, {"AM-DSB-WC", "BPSK", "QPSK", "8PSK", "16QAM"}),   # constant envelope; sigma_af gave 25 % node error
    ("N2", "line_x2", {"GMSK"}, {"FM"}),
    ("N3", "line_x1", {"AM-DSB-WC"}, {"BPSK", "QPSK", "8PSK", "16QAM"}),
    ("N4", "line_x2", {"BPSK"}, {"QPSK", "8PSK", "16QAM"}),
    ("N5", "abs_C40", {"QPSK", "16QAM"}, {"8PSK"}),   # |C40|: 8PSK ~0; line_x4 gave 21 % node error
    ("N6", "sigma_aa", {"16QAM"}, {"QPSK"}),
]


# ------------------------------------------------------------------ Azzouz-Nandi style tree

def fit_tree(F, y, snr, design_snr=10):
    names = np.array(CLASSES7)[y]
    sel = snr >= design_snr
    thr, info = {}, {}
    for node, feat, left, right in TREE:
        m = sel & np.isin(names, list(left | right))
        v, is_left = F[m, FI[feat]], np.isin(names[m], list(left))
        cand = np.unique(np.quantile(v, np.linspace(0.005, 0.995, 400)))
        err = np.array([np.mean((v > c) != is_left) for c in cand])
        # a node may need "below threshold = left group": allow both directions
        k_up, k_dn = int(np.argmin(err)), int(np.argmin(1 - err))
        up = err[k_up] <= 1 - err[k_dn]
        k = k_up if up else k_dn
        thr[node] = (float(cand[k]), bool(up))
        info[node] = {"feature": feat, "threshold": float(cand[k]), "left_if_above": bool(up),
                      "train_error": float(err[k] if up else 1 - err[k])}
    return thr, info


def predict_tree(F, thr):
    def left(f, node, feat):
        t, up = thr[node]
        return (f[FI[feat]] > t) == up
    out = []
    for f in F:
        if left(f, "N1", "sigma_aa"):
            out.append("GMSK" if left(f, "N2", "line_x2") else "FM")
        elif left(f, "N3", "line_x1"):
            out.append("AM-DSB-WC")
        elif left(f, "N4", "line_x2"):
            out.append("BPSK")
        elif left(f, "N5", "abs_C40"):
            out.append("16QAM" if left(f, "N6", "sigma_aa") else "QPSK")
        else:
            out.append("8PSK")
    return np.array([CLASSES7.index(c) for c in out])


# ------------------------------------------------------------------ Swami-Sadler hierarchy

DIGITAL = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]


def theory():
    t = {}
    for c in DIGITAL:
        s = _constellation(c).astype(complex)
        s /= np.sqrt(np.mean(np.abs(s) ** 2))
        m20, m40, m42 = np.mean(s**2), np.mean(s**4), np.mean(np.abs(s) ** 4)
        t[c] = (abs(m40 - 3 * m20**2), float(m42 - abs(m20) ** 2 - 2))
    return t


TH = theory()


def m2m4_noise(z):
    m2, m4 = np.mean(np.abs(z) ** 2), np.mean(np.abs(z) ** 4)
    s = np.sqrt(max(2 * m2**2 - m4, 0.0))
    return max(m2 - s, 0.0)


def hier_classify(z, noise=None):
    z = z - z.mean()
    n = m2m4_noise(z) if noise is None else noise
    c21 = np.mean(np.abs(z) ** 2)
    ps = max(c21 - n, 1e-3 * c21)                      # signal power
    m20, m40, m42 = np.mean(z**2), np.mean(z**4), np.mean(np.abs(z) ** 4)
    c40 = abs(m40 - 3 * m20**2) / ps**2
    c42 = (m42 - abs(m20) ** 2 - 2 * c21**2) / ps**2     # Gaussian noise adds nothing to C42
    if c40 > (TH["BPSK"][0] + TH["QPSK"][0]) / 2:
        return 0
    if c40 > (TH["QPSK"][0] + TH["16QAM"][0]) / 2:
        return 1
    if c40 < (TH["64QAM"][0] + TH["8PSK"][0]) / 2:
        return 2
    return 4 if c42 > (TH["16QAM"][1] + TH["64QAM"][1]) / 2 else 3


def check():
    rng = np.random.default_rng(0)
    ok = 0
    for k, c in enumerate(DIGITAL):
        s = _constellation(c).astype(complex)
        s /= np.sqrt(np.mean(np.abs(s) ** 2))
        for _ in range(20):
            ok += hier_classify(rng.choice(s, 4000) * np.exp(1j * rng.uniform(0, 6.28)), noise=0.0) == k
    print("theory |C40|, C42:", {c: (round(a, 3), round(b, 3)) for c, (a, b) in TH.items()})
    print(f"hierarchical classifier on noiseless ideal symbols: {ok}/100 correct (expect 100)")
    return ok == 100


def summary(pred, y, snr):
    snrs = sorted(set(snr.tolist()))
    acc = {int(s): float(np.mean(pred[snr == s] == y[snr == s])) for s in snrs}
    return {"accuracy_all": float(np.mean(pred == y)), "accuracy_snr_ge_10": float(np.mean(pred[snr >= 10] == y[snr >= 10])),
            "snr_first_85": next((s for s in snrs if acc[s] >= 0.85), None), "accuracy_vs_snr": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    from sklearn.model_selection import train_test_split
    out = {}

    d = np.load(os.path.join("results", "radioml_hgb", "features.npz"))
    F, y, snr = d["F"], d["y"].astype(int), d["snr"].astype(int)
    assert [str(c) for c in d["classes"]] == CLASSES7
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y * 1000 + (snr + 100), random_state=0)
    thr, info = fit_tree(F[tr], y[tr], snr[tr])
    out["azzouz_nandi_tree"] = {"nodes": info, **summary(predict_tree(F[te], thr), y[te], snr[te])}
    r = out["azzouz_nandi_tree"]
    print("Azzouz-Nandi style tree (7 classes, same split as 5.1):", {k: round(v["train_error"], 3) for k, v in info.items()})
    print(f"   all {r['accuracy_all'] * 100:.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  85 % at {r['snr_first_85']} dB"
          "   (learned depth-6 tree: 63.8 / 99.4, boosted trees 71.4 / 99.95)")

    from practical_alrt import load, to_symbols
    x, y5, snr5, _ = load("radioml", 5)
    tr5, te5 = train_test_split(np.arange(len(y5)), test_size=0.5, stratify=y5 * 100 + (snr5 + 50), random_state=0)
    pred = np.array([hier_classify(to_symbols(x[i], 8.0)) for i in te5])
    out["swami_sadler_hierarchical"] = summary(pred, y5[te5], snr5[te5])
    r = out["swami_sadler_hierarchical"]
    print("Swami-Sadler hierarchical (5 digital classes, symbol rate known, training-free):")
    print(f"   all {r['accuracy_all'] * 100:.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  85 % at {r['snr_first_85']} dB"
          "   (same frames: features+HGB 66.2 / 87.1, ALRT 70.4 / 93.3, sample-domain cumulant theory weak)")
    with open(os.path.join("results", "requirements_methods.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
