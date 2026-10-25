"""FSK-family symbol rate at low SNR: band-limit before the FM discriminator.

The instantaneous-frequency (FM discriminator) estimators see ALL the noise in the
sampled band, but an FSK signal occupies only a fraction of it (frames here have
3-20 samples per symbol). Masking the spectrum to the signal's occupied band first
removes the out-of-band noise and raises the discriminator's input SNR by roughly
fs / B_occupied, which is what fails at 0-5 dB.

Steps per frame: Welch-free PSD (FFT of the whole frame, smoothed) -> occupied band =
contiguous region around the power centroid above the noise median + 3 dB, widened by
25 % -> FFT mask -> the same estimators as before (inst_freq_s3 and the 12-estimator
vote of characterization_extended) on the band-limited frame.

Check (--check): on a clean MSK frame the band-limited estimate equals the
un-filtered one; on pure noise the mask keeps a small band (no crash, no structure).
Benchmark: held-out TorchSig frames (D:\\data\\symbol_rate_torchsig.npz), FSK family
(2fsk, 4fsk, 2gfsk, 2msk, 2gmsk) at 0, 5, 10 dB, plus PSK/QAM to make sure nothing breaks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "classical_amc"))
sys.path.insert(0, os.path.join(HERE, "..", "characterization_extended"))


def bandlimit(x, widen=0.25, smooth=33):
    n = len(x)
    X = np.fft.fft(x)
    P = uniform_filter1d(np.abs(np.fft.fftshift(X)) ** 2, smooth, mode="wrap")
    noise = np.median(P)
    k0 = int(np.argmax(P))
    above = P > 2.0 * noise                       # +3 dB over the noise median
    lo = k0
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = k0
    while hi < n - 1 and above[hi + 1]:
        hi += 1
    w = hi - lo + 1
    if w < 8:                                     # no clear occupied band: leave the frame unfiltered
        return x, 1.0
    lo, hi = max(0, lo - int(widen * w)), min(n - 1, hi + int(widen * w))
    mask = np.zeros(n)
    mask[lo:hi + 1] = 1
    return np.fft.ifft(X * np.fft.ifftshift(mask)), (hi - lo + 1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from symbol_rate import ESTIMATORS
    est = ESTIMATORS["inst_freq_s3"]
    if args.check:
        sys.path.insert(0, HERE)
        from msk_line_pair import msk
        rng = np.random.default_rng(3)
        x = msk(4096, 8.0, rng)
        xb, frac = bandlimit(x)
        print(f"clean MSK sps 8: plain {1 / est(x):.3f}, band-limited {1 / est(xb):.3f} sps (true 8), kept {frac * 100:.0f} % of band")
        xn = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        print(f"pure noise: kept {bandlimit(xn)[1] * 100:.1f} % of band")
        return
    import sr_bank
    from blind_alrt import consensus_rates
    params = json.load(open(os.path.join(HERE, "..", "characterization_extended", "results", "consensus_params.json")))["cons_sr_dev"]

    def vote(x):
        m = sr_bank.members(x)
        names = list(m)
        R = np.array([[m[k][0] for k in names]]); P = np.array([[m[k][1] for k in names]])
        return float(consensus_rates(R, P, names, params)[0])

    d = np.load(r"D:\data\symbol_rate_torchsig.npz")
    res = {}
    for cls in ("2gfsk", "2gmsk", "2fsk", "4fsk", "2msk", "qpsk", "16qam"):
        for snr in (0, 5, 10):
            idx = np.flatnonzero((d["cls"] == cls) & (d["snr"] == snr))
            r = {"plain_if": [], "bl_if": [], "plain_vote": [], "bl_vote": []}
            for i in idx:
                x, rs = d["x"][i].astype(complex), d["rs"][i]
                xb, _ = bandlimit(x)
                ok = lambda e: abs(e - rs) / rs < 0.02
                r["plain_if"].append(ok(est(x))); r["bl_if"].append(ok(est(xb)))
                r["plain_vote"].append(ok(vote(x))); r["bl_vote"].append(ok(vote(xb)))
            res[f"{cls}@{snr}"] = {k: float(np.mean(v)) for k, v in r.items()}
            q = res[f"{cls}@{snr}"]
            print(f"{cls:>6s} {snr:2d} dB: inst-freq {q['plain_if'] * 100:4.0f} -> {q['bl_if'] * 100:4.0f} %   vote {q['plain_vote'] * 100:4.0f} -> {q['bl_vote'] * 100:4.0f} %", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "fsk_prefilter.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
