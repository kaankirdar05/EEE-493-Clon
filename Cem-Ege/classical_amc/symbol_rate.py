"""Classical symbol-rate estimators (WP6 add-on; TorchSig boxes carry no symbol-rate label).

Test signals: pulse-shaped linear modulations (RRC, roll-off 0.2-0.5) and GMSK,
4096 samples per frame, samples-per-symbol 3-20 (symbol rate fs/sps), random
carrier phase, small frequency offset, AWGN. All estimators search for a
spectral line at the symbol rate alpha = Rs (cyclostationarity), with the peak
refined by parabolic interpolation:

    envelope     |FFT(|x|^2)|            (squared-envelope cyclic line; strength ~ roll-off)
    envelope4    |FFT(|x|^4)|            (4th-power envelope, stronger line for low roll-off)
    delay_mult   |FFT(x[n] x*[n-d])|, d = 1..3 summed  (delay-and-multiply)
    haar         |FFT(|Haar wavelet transform|)|   (phase/amplitude transitions)
    inst_freq    |FFT(|d phase/dn - median|)|  (phase jumps: PSK; frequency transitions: GMSK / FSK)

A frame counts as correct if |Rs_hat - Rs| / Rs < 2 %.

    python symbol_rate.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from amc.synthetic import _constellation, _rrc

LINEAR = ["BPSK", "QPSK", "8PSK", "16QAM"]
CLASSES = LINEAR + ["GMSK"]
N = 4096


def gen(name, sps, beta, snr_db, rng):
    nsym = N // sps + 40
    if name == "GMSK":
        bits = rng.choice([-1.0, 1.0], nsym)
        nrz = np.repeat(bits, sps)
        t = np.arange(-2 * sps, 2 * sps + 1) / sps
        g = np.exp(-2 * np.pi**2 * 0.3**2 * t**2 / np.log(2))
        g /= g.sum()
        s = np.exp(1j * np.cumsum(np.convolve(nrz, g, "same")) * (np.pi / 2) / sps)
    else:
        pts = _constellation(name)
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
        up = np.zeros(nsym * sps, complex)
        up[::sps] = rng.choice(pts, nsym)
        s = np.convolve(up, _rrc(beta, 8, sps), "same")
    off = rng.integers(10 * sps, 20 * sps)
    s = s[off:off + N]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    s = s * np.exp(1j * (rng.uniform(0, 2 * np.pi) + 2 * np.pi * rng.uniform(-0.01, 0.01) * np.arange(N)))
    w = np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return s + w


def line_peak(v, a_min=1 / 25, a_max=1 / 2.5):
    """Strongest spectral line of real sequence v in normalised cyclic frequency
    [a_min, a_max] (cycles/sample); parabolic interpolation of the peak."""
    v = v - v.mean()
    n = 8 * len(v)
    P = np.abs(np.fft.rfft(v * np.hanning(len(v)), n))
    f = np.arange(len(P)) / n
    sel = np.flatnonzero((f >= a_min) & (f <= a_max))
    k = sel[np.argmax(P[sel])]
    if 0 < k < len(P) - 1:
        a, b, c = np.log(P[k - 1] + 1e-30), np.log(P[k] + 1e-30), np.log(P[k + 1] + 1e-30)
        k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else k
    return k / n


def haar_transform(x, scale):
    h = np.concatenate([np.ones(scale), -np.ones(scale)]) / (2 * scale)
    return np.abs(np.convolve(x, h, "same"))


def _delay_mult(x):
    """sum over lags d = 1..3 of |x[n + d] x*[n]|^2."""
    n = len(x) - 3
    return sum(np.abs(x[d:d + n] * np.conj(x[:n])) ** 2 for d in (1, 2, 3))


def _inst_freq_dev(x, smooth=1):
    """|instantaneous frequency - its median|: removes the carrier offset so
    frequency transitions (GMSK/FSK) and phase jumps (PSK) both give a line at Rs."""
    f = np.angle(x[1:] * np.conj(x[:-1]))
    f = f - np.median(f)
    if smooth > 1:
        f = np.convolve(f, np.ones(smooth) / smooth, "same")
    return np.abs(f)


ESTIMATORS = {
    "envelope": lambda x: line_peak(np.abs(x) ** 2),
    "envelope4": lambda x: line_peak(np.abs(x) ** 4),
    "delay_mult": lambda x: line_peak(_delay_mult(x)),
    "haar": lambda x: line_peak(haar_transform(x, 1)),
    "inst_freq": lambda x: line_peak(_inst_freq_dev(x)),
    # 3-tap smoothing of the frequency before |.|: suppresses sample-to-sample
    # phase noise; needed for GMSK, whose frequency only dips at bit transitions
    "inst_freq_s3": lambda x: line_peak(_inst_freq_dev(x, smooth=3)),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-cell", type=int, default=60, help="frames per (class, SNR)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results", "symbol_rate"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    snrs = [0, 5, 10, 15, 20]

    rows = []
    for c in CLASSES:
        for s in snrs:
            for _ in range(args.per_cell):
                sps = int(rng.integers(3, 21))
                beta = float(rng.uniform(0.2, 0.5))
                x = gen(c, sps, beta, s, rng)
                est = {k: f(x) for k, f in ESTIMATORS.items()}
                rows.append({"class": c, "snr": s, "sps": sps, "beta": beta, "rs_true": 1 / sps,
                             **{k: float(v) for k, v in est.items()}})

    def ok(r, k):
        return abs(r[k] - r["rs_true"]) / r["rs_true"] < 0.02

    summary = {}
    print(f"{len(rows)} frames; success = within 2 % of true symbol rate")
    header = f"{'estimator':>11s} | all  | " + " | ".join(f"{c:>5s}" for c in CLASSES) + " | " + " | ".join(f"{s:>2d}dB" for s in snrs)
    print(header)
    for k in ESTIMATORS:
        allr = np.mean([ok(r, k) for r in rows])
        per_c = {c: float(np.mean([ok(r, k) for r in rows if r["class"] == c])) for c in CLASSES}
        per_s = {s: float(np.mean([ok(r, k) for r in rows if r["snr"] == s])) for s in snrs}
        med = float(np.median([abs(r[k] - r["rs_true"]) / r["rs_true"] for r in rows]))
        summary[k] = {"success_all": float(allr), "per_class": per_c, "per_snr": per_s, "median_rel_error": med}
        print(f"{k:>11s} | {allr * 100:4.0f} | " + " | ".join(f"{per_c[c] * 100:5.0f}" for c in CLASSES) + " | " +
              " | ".join(f"{per_s[s] * 100:4.0f}" for s in snrs))
    best = {c: max(ESTIMATORS, key=lambda k: summary[k]["per_class"][c]) for c in CLASSES}
    oracle = np.mean([ok(r, best[r["class"]]) for r in rows])
    print(f"best estimator per class: {best} -> {oracle * 100:.0f} % if the class is known (e.g. from 5.1)")
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "best_per_class": best, "success_with_class_known": float(oracle),
                   "n_frames": len(rows), "frame_len": N, "sps_range": [3, 20], "rolloff_range": [0.2, 0.5]}, f, indent=2)
    with open(os.path.join(args.out, "per_frame.csv"), "w", encoding="utf-8") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        f.writelines(",".join(str(r[k]) for k in keys) + "\n" for r in rows)


if __name__ == "__main__":
    main()
