"""Low-SNR symbol-rate estimation for MSK-type signals: squared-signal line pair.

For continuous-phase FSK with modulation index h = 0.5 (MSK, GMSK), squaring the
signal doubles the phase, so the +-pi/2 phase steps per symbol become +-pi: x^2
contains two spectral LINES at 2*fc +- Rs/2, exactly one symbol rate apart.
A line is coherent energy, so it survives far lower SNR than the frequency-
transition features the general estimators use.

Estimator: P = |FFT(x^2)|^2 (zero-padded, Hann); for every candidate spacing d in
the search range score(d) = max over f of P(f) * P(f + d) (the strongest line pair
with that spacing); Rs_hat = argmax score, refined by parabolic interpolation.

Checks:
  --check   clean MSK with known Rs: estimate within 0.5 %; white noise: no structure
Benchmark (held-out TorchSig frames D:\\data\\symbol_rate_torchsig.npz, classes 2msk,
2gmsk, 2gfsk at 0 and 5 dB): success within 2 %, compared with the consensus vote of
characterization_extended (recomputed on the same frames).

    python msk_line_pair.py --check
    python msk_line_pair.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

A_MIN, A_MAX = 1 / 25, 1 / 2.5


def line_pair_rate(x, pad=8):
    n = len(x)
    y = (x * x) * np.hanning(n)
    m = pad * n
    P = np.abs(np.fft.fft(y, m)) ** 2
    P = P / np.median(P)
    d_lo, d_hi = int(np.ceil(A_MIN * m)), int(np.floor(A_MAX * m))
    scores = np.empty(d_hi - d_lo + 1)
    for j, d in enumerate(range(d_lo, d_hi + 1)):
        scores[j] = np.max(P * np.roll(P, -d))
    k = int(np.argmax(scores))
    if 0 < k < len(scores) - 1:
        a, b, c = np.log(scores[k - 1:k + 2] + 1e-30)
        den = a - 2 * b + c
        k = k + (0.5 * (a - c) / den if den != 0 else 0)
    return (d_lo + k) / m


def msk(n, sps, rng, snr_db=None):
    bits = rng.choice([-1.0, 1.0], n // int(sps) + 4)
    t = np.arange(n)
    # exact continuous-phase MSK: phase = pi/2 * (sum of completed symbols + fraction of the current one);
    # rounding symbol boundaries to whole samples would make the per-symbol step differ from pi/2
    u = t / sps
    k = u.astype(int)
    done = np.concatenate([[0.0], np.cumsum(bits)])[k]
    phase = (np.pi / 2) * (done + bits[k] * (u - k))
    x = np.exp(1j * (phase + 2 * np.pi * rng.uniform(-0.01, 0.01) * t + rng.uniform(0, 6.28)))
    if snr_db is not None:
        x = x + np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x


def check():
    rng = np.random.default_rng(1)
    errs = []
    for sps in (3.3, 5.0, 8.7, 12.0, 19.0):
        for _ in range(5):
            errs.append(abs(line_pair_rate(msk(4096, sps, rng)) - 1 / sps) * sps)
    noise = [line_pair_rate((rng.standard_normal(4096) + 1j * rng.standard_normal(4096))) for _ in range(5)]
    print(f"clean MSK, 25 frames: max relative error {max(errs):.4f}, median {np.median(errs):.5f} (expect max < 0.005)")
    print(f"white noise estimates (should be scattered, no preferred value): {np.round(noise, 3)}")
    return max(errs) < 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "characterization_extended"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical_amc"))
    import sr_bank
    from blind_alrt import consensus_rates
    params = json.load(open(os.path.join(os.path.dirname(__file__), "..", "characterization_extended", "results", "consensus_params.json")))["cons_sr_dev"]

    d = np.load(r"D:\data\symbol_rate_torchsig.npz")
    res = {}
    for cls in ("2msk", "2gmsk", "2gfsk"):
        for snr in (0, 5, 10):
            idx = np.flatnonzero((d["cls"] == cls) & (d["snr"] == snr))
            ok_lp, ok_cs = [], []
            for i in idx:
                x, rs = d["x"][i].astype(complex), d["rs"][i]
                ok_lp.append(abs(line_pair_rate(x) - rs) / rs < 0.02)
                m = sr_bank.members(x)
                names = list(m)
                R = np.array([[m[k][0] for k in names]]); Pm = np.array([[m[k][1] for k in names]])
                ok_cs.append(abs(consensus_rates(R, Pm, names, params)[0] - rs) / rs < 0.02)
            res[f"{cls}@{snr}dB"] = {"n": int(len(idx)), "line_pair": float(np.mean(ok_lp)), "consensus": float(np.mean(ok_cs))}
            print(f"{cls:>6s} {snr:2d} dB (n={len(idx)}): line pair {np.mean(ok_lp) * 100:5.1f} %   consensus {np.mean(ok_cs) * 100:5.1f} %", flush=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "msk_line_pair.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
