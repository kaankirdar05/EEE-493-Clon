"""Sanity checks of the Morlet CWT symbol-rate estimator (run BEFORE any benchmark number is reported).
 1. FFT-based Morlet CWT == direct time-domain convolution with the truncated Morlet wavelet.
 2. Tone response of the CWT magnitude = Gaussian of the wavelet's Fourier transform (theory exp(-(f-fc)^2/(2 sf^2))).
 3. Noiseless MSK and RRC-QPSK (continuous-time definitions, non-integer samples/symbol, 4096 samples) recovered within 0.5 %.
    (--cfg to choose; default: every config is reported, the tuned config is read from results/b_tuned.json if present.)
"""
import json
import os
import sys
sys.dont_write_bytecode = True
import numpy as np
import wavelet_sr as ws
from sr_synth import gen_msk, gen_qpsk

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    rng = np.random.default_rng(0)
    res = {}
    # 1
    x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    a = ws.morlet_cwt(x, 0.1, 0.05)
    b = ws.morlet_cwt_direct(x, 0.1, 0.05)
    sl = slice(200, -200)
    err = np.max(np.abs(a[sl] - b[sl])) / np.sqrt(np.mean(np.abs(a[sl]) ** 2))
    print(f"[1] FFT CWT vs direct time-domain Morlet convolution: max |diff| / rms = {err:.2e}")
    res["cwt_fft_vs_direct"] = float(err)
    # 2
    n = 4096
    fc, sf = 0.1, 0.04
    rows = []
    for ft in (0.1, 0.12, 0.14, 0.18, 0.05):
        t = np.exp(2j * np.pi * ft * np.arange(n))
        m = np.abs(ws.morlet_cwt(t, fc, sf))[300:-300].mean()
        rows.append((ft, m, np.exp(-(ft - fc) ** 2 / (2 * sf ** 2))))
    print("[2] tone response |W| vs theory exp(-(f-fc)^2/2sf^2):", ", ".join(f"f={a:.2f}: {m:.4f} / {t:.4f}" for a, m, t in rows))
    res["tone"] = max(abs(m - t) for _, m, t in rows)
    # 3
    tuned = None
    p = os.path.join(RES, "b_tuned.json")
    if os.path.exists(p):
        tj = json.load(open(p))
        tuned = (tj["tuned_cfg"][0], tj["tuned_cfg"][1], tj["tuned_cfg"][2]); mode = tj["mode"]
    cfgs = [tuned] if tuned else ws.CONFIGS
    out = {}
    for name, g in (("QPSK RRC 0.35", gen_qpsk), ("MSK", gen_msk)):
        rates = np.exp(rng.uniform(np.log(1 / 20), np.log(1 / 3), 60))
        frames = [(g(r, 4096, rng), r) for r in rates]
        for cfg in cfgs:
            rel = np.array([abs(ws.estimate(x, cfg, mode if tuned else 'l') / r - 1) for x, r in frames])
            out[(name, cfg)] = rel
    print("[3] noiseless, 60 frames per modulation, rates log-uniform in 1/20..1/3 cycles/sample (non-integer sps), no carrier offset")
    for (name, cfg), rel in out.items():
        print(f"    {name:>14s} {ws.cfg_name(cfg):>24s}: within 0.5 %: {np.mean(rel < 0.005) * 100:5.1f} %   median rel err {np.median(rel):.4f}   max {rel.max():.3f}")
    res["noiseless"] = {f"{n}|{ws.cfg_name(c)}": {"within_0.5pct": float(np.mean(r < 0.005)), "median": float(np.median(r)), "max": float(r.max())} for (n, c), r in out.items()}
    json.dump(res, open(os.path.join(RES, "b_sanity.json" if tuned else "b_sanity_prelim.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
