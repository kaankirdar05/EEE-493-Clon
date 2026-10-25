"""Practical likelihood-ratio classification (ALRT / HLRT) on realistic frames.

likelihood_bound.py gives the likelihood classifier everything (timing, SNR, no
pulse shaping). Here it gets nothing: every nuisance parameter is estimated
blindly per frame, as a real receiver would have to:

  1. symbol rate      inst_freq_s3 estimator from symbol_rate.py (or a fixed value)
  2. resample         to exactly 8 samples/symbol (FFT resampling)
  3. matched filter   RRC, roll-off 0.35 (the true roll-off / pulse is unknown)
  4. timing           the one of 8 sampling phases with the most energy
  5. carrier offset   strongest line of z^4 or z^8 (whichever is more prominent), / 4 or / 8
  6. likelihood       unknown phase: averaged (ALRT) or maximised (HLRT) over 16 phases;
                      unknown noise level: maximised over a grid of 8 noise variances,
                      with signal amplitude sqrt(1 - sigma^2) after unit-power normalisation

Compared on the SAME test frames with the 19-feature + boosted-trees classifier
(trained on separate frames from the same source). Classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM.

Sources:
    radioml   RadioML 2018.01A frames (1,024 samples, 8 samples/symbol, -10..20 dB)
    torchsig  TorchSig's constellation_modulator (1,024 samples, random 3-20 samples/symbol,
              SRRC or rectangular pulses), from gen_alrt_torchsig in extra_methods/

    python practical_alrt.py --source radioml
    python practical_alrt.py --source torchsig
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.signal import resample
from scipy.special import logsumexp

from amc.features import extract_features
from amc.synthetic import _constellation, _rrc
from symbol_rate import ESTIMATORS

CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
SPS = 8
MF = _rrc(0.35, 8, SPS)


def constellations():
    out = []
    for c in CLASSES:
        p = _constellation(c).astype(complex)
        out.append(p / np.sqrt(np.mean(np.abs(p) ** 2)))
    return out


CONS = constellations()
PHASES = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
SIG2 = np.geomspace(0.005, 0.9, 8)


def to_symbols(x, sps_hat):
    L = int(round(len(x) * SPS / sps_hat))
    y = resample(x.astype(complex), L)
    y = np.convolve(y, MF, "same")
    k = int(np.argmax([np.mean(np.abs(y[i::SPS]) ** 2) for i in range(SPS)]))
    z = y[k::SPS][4:-4]
    best, cfo = -1.0, 0.0
    for p in (4, 8):
        P = np.abs(np.fft.fft(z**p, 8192))
        ratio = P.max() / np.mean(P)
        if ratio > best:
            best, cfo = ratio, np.fft.fftfreq(8192)[np.argmax(P)] / p
    z = z * np.exp(-2j * np.pi * cfo * np.arange(len(z)))
    return z / np.sqrt(np.mean(np.abs(z) ** 2))


def loglik(z):
    """(classes, phases, sigma^2 grid) log-likelihoods of symbol vector z."""
    out = np.empty((len(CONS), len(PHASES), len(SIG2)))
    for k, pts in enumerate(CONS):
        for j, s2 in enumerate(SIG2):
            ref = np.sqrt(1 - s2) * PHASES[:, None] * pts[None, :]          # (P, M)
            d = -np.abs(z[None, :, None] - ref[:, None, :]) ** 2 / s2         # (P, N, M)
            out[k, :, j] = (logsumexp(d, axis=2) - np.log(len(pts)) - np.log(np.pi * s2)).sum(1)
    return out


def classify(x, sps_hat):
    LL = loglik(to_symbols(x, sps_hat))
    alrt = int(np.argmax(logsumexp(LL, axis=1).max(axis=1)))     # average phase, max noise
    hlrt = int(np.argmax(LL.max(axis=(1, 2))))                    # max phase, max noise
    return alrt, hlrt


def load(source, seed):
    if source == "radioml":
        from amc.radioml import iter_radioml2018
        xs, ys, ss = [], [], []
        for x, k, s in iter_radioml2018(r"D:\data\radioml2018\GOLD_XYZ_OSC.0001_1024.hdf5", classes=CLASSES,
                                        snr_min=-10, snr_max=20, per_group=120, order="hdf5", seed=seed):
            xs.append(x); ys.append(np.full(len(x), k)); ss.append(np.full(len(x), s))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), None
    d = np.load(r"D:\data\alrt_torchsig.npz")
    return d["x"], d["y"], d["snr"], d["sps"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["radioml", "torchsig"], required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    x, y, snr, sps_true = load(args.source, args.seed)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y * 100 + (snr + 50), random_state=0)
    clf = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0)
    clf.fit(extract_features(x[tr]), y[tr])
    pred = {"features_hgb": clf.predict(extract_features(x[te]))}

    est = ESTIMATORS["inst_freq_s3"]
    variants = {"blind": lambda i: 1 / est(x[i].astype(complex))}
    if args.source == "radioml":
        variants["sps_known"] = lambda i: 8.0                       # RadioML: 8 samples/symbol throughout
    else:
        variants["sps_known"] = lambda i: float(sps_true[i])
    t0 = time.perf_counter()
    for vname, sps_fn in variants.items():
        a, h = zip(*[classify(x[i], sps_fn(i)) for i in te])
        pred[f"alrt_{vname}"], pred[f"hlrt_{vname}"] = np.array(a), np.array(h)
    ms = (time.perf_counter() - t0) / (len(te) * len(variants)) * 1e3

    snrs = sorted(set(snr[te].tolist()))
    yt, st = y[te], snr[te]
    res = {}
    print(f"{args.source}: {len(te)} test frames, classes {CLASSES}, SNR {snrs[0]}..{snrs[-1]} dB")
    for name, p in pred.items():
        acc = {int(s): float(np.mean(p[st == s] == yt[st == s])) for s in snrs}
        res[name] = {"accuracy_all": float(np.mean(p == yt)), "accuracy_snr_ge_10": float(np.mean(p[st >= 10] == yt[st >= 10])),
                     "snr_first_85": next((s for s in snrs if acc[s] >= 0.85), None), "accuracy_vs_snr": acc}
        print(f"  {name:>16s}: all {res[name]['accuracy_all'] * 100:5.1f} %  SNR>=10 {res[name]['accuracy_snr_ge_10'] * 100:5.1f} %  "
              f"85 % at {res[name]['snr_first_85']} dB")
    print(f"  likelihood classifier: {ms:.1f} ms/frame")
    os.makedirs(os.path.join("results", "practical_alrt"), exist_ok=True)
    with open(os.path.join("results", "practical_alrt", f"{args.source}.json"), "w") as f:
        json.dump({"source": args.source, "n_test": int(len(te)), "ms_per_frame_likelihood": ms, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
