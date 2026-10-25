"""Hilbert–Huang transform (EMD) features for modulation classification.

Empirical mode decomposition splits a real signal into intrinsic mode functions
(IMFs) by repeated "sifting" (subtract the mean of the cubic-spline envelopes
through the local maxima and minima). The Hilbert transform of each IMF gives an
instantaneous amplitude and frequency; their statistics are classic HHT features.

Here EMD is applied to the real and imaginary parts of the frame. Features per
part, for the first 4 IMFs: energy fraction, mean / std of instantaneous
frequency, kurtosis of instantaneous amplitude (4 x 4 x 2 = 32 features).

Sanity check (run as a script with --check): EMD of the sum of two sines with well
separated frequencies returns the faster sine as IMF 1 and the slower as IMF 2.

Benchmark: RadioML 2018.01A, 7 classes (AM-DSB-WC, FM, BPSK, QPSK, 8PSK, 16QAM, GMSK),
-20..30 dB, 300 frames per (class, SNR), 75/25 split stratified by class x SNR;
boosted trees on the 19 baseline features vs HHT features vs both, same frames.

    python hht_features.py --check
    python hht_features.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema, hilbert
from scipy.stats import kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical_amc"))


def _envelope_mean(x):
    n = np.arange(len(x))
    mx = argrelextrema(x, np.greater)[0]
    mn = argrelextrema(x, np.less)[0]
    if len(mx) < 2 or len(mn) < 2:
        return None
    # mirror the end points so the splines do not swing at the edges
    mx = np.concatenate([[0], mx, [len(x) - 1]])
    mn = np.concatenate([[0], mn, [len(x) - 1]])
    upper = CubicSpline(mx, x[mx])(n)
    lower = CubicSpline(mn, x[mn])(n)
    return (upper + lower) / 2


def emd(x, n_imfs=4, max_sift=10, sd_stop=0.2):
    imfs, r = [], x.astype(float).copy()
    for _ in range(n_imfs):
        h = r.copy()
        for _ in range(max_sift):
            m = _envelope_mean(h)
            if m is None:
                break
            h_new = h - m
            sd = np.sum((h - h_new) ** 2) / max(np.sum(h**2), 1e-30)
            h = h_new
            if sd < sd_stop:
                break
        imfs.append(h)
        r = r - h
        if _envelope_mean(r) is None:
            break
    while len(imfs) < n_imfs:
        imfs.append(np.zeros_like(x, dtype=float))
    return np.array(imfs)


def hht_features(frame, n_imfs=4):
    out = []
    for part in (frame.real, frame.imag):
        part = part / max(np.std(part), 1e-12)
        imfs = emd(part, n_imfs)
        tot = max(np.sum(imfs**2), 1e-30)
        for imf in imfs:
            a = hilbert(imf)
            amp = np.abs(a)
            f = np.diff(np.unwrap(np.angle(a))) / (2 * np.pi)
            out += [np.sum(imf**2) / tot, f.mean(), f.std(), kurtosis(amp) if amp.std() > 0 else 0.0]
    return np.nan_to_num(np.array(out, dtype=np.float32))


def check():
    n = np.arange(2048)
    fast, slow = np.sin(2 * np.pi * 0.12 * n), 0.8 * np.sin(2 * np.pi * 0.01 * n)
    imfs = emd(fast + slow, 2)
    c1 = np.corrcoef(imfs[0][200:-200], fast[200:-200])[0, 1]
    c2 = np.corrcoef(imfs[1][200:-200], slow[200:-200])[0, 1]
    print(f"EMD check: corr(IMF1, fast sine) = {c1:.3f}, corr(IMF2, slow sine) = {c2:.3f}  (expect both > 0.95)")
    return c1 > 0.95 and c2 > 0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--per-group", type=int, default=300)
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)

    from amc.features import extract_features
    from amc.radioml import iter_radioml2018
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    classes = ["AM-DSB-WC", "FM", "BPSK", "QPSK", "8PSK", "16QAM", "GMSK"]
    xs, ys, ss = [], [], []
    for x, k, s in iter_radioml2018(r"D:\data\radioml2018\GOLD_XYZ_OSC.0001_1024.hdf5", classes=classes,
                                    per_group=args.per_group, order="hdf5", seed=11):
        xs.append(x); ys.append(np.full(len(x), k)); ss.append(np.full(len(x), s))
    x, y, snr = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)
    F19 = extract_features(x)
    t0 = time.perf_counter()
    H = np.array([hht_features(xi.astype(complex)) for xi in x])
    hht_ms = (time.perf_counter() - t0) / len(x) * 1e3
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y * 1000 + (snr + 100), random_state=0)
    res = {}
    for name, F in [("19 features", F19), ("HHT only", H), ("19 + HHT", np.hstack([F19, H]))]:
        p = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0).fit(F[tr], y[tr]).predict(F[te])
        st, yt = snr[te], y[te]
        acc = {int(s): float(np.mean(p[st == s] == yt[st == s])) for s in sorted(set(st.tolist()))}
        res[name] = {"accuracy_all": float(np.mean(p == yt)), "accuracy_snr_ge_10": float(np.mean(p[st >= 10] == yt[st >= 10])),
                     "snr_first_85": next((s for s in sorted(acc) if acc[s] >= 0.85), None), "accuracy_vs_snr": acc}
        print(f"{name:>12s}: all {res[name]['accuracy_all'] * 100:5.1f} %  SNR>=10 {res[name]['accuracy_snr_ge_10'] * 100:5.1f} %  85 % at {res[name]['snr_first_85']} dB")
    print(f"HHT feature extraction: {hht_ms:.1f} ms/frame")
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "hht_features.json"), "w") as f:
        json.dump({"classes": classes, "per_group": args.per_group, "hht_ms_per_frame": hht_ms, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
