"""Diagnostic for the SR paper-setting result: why is the paper metric (max FFT bin of the
SR output) at chance? Mean output periodogram under H0 / H1 and the argmax-bin histogram."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sr_bench import gen, NFFT, F0
from disc.sr import bistable
res = {}
cfgs = {"paper g=1 D=0.43 h=0.01": (1.0, 0.43, 0.01), "dev-best sr_fftmax g=0.1 D=3 h=0.01": (0.1, 3.0, 0.01),
        "dev-best sr_known_f g=30 D=1 h=0.03": (30.0, 1.0, 0.03)}
for name, (g, D, h) in cfgs.items():
    r = {}
    for tag, snr in (("H0", None), ("H1 -5 dB", -5), ("H1 0 dB", 0)):
        rng = np.random.default_rng(3)
        y = gen(rng, 2000, snr)
        z = bistable(g * y + np.sqrt(D) * rng.standard_normal(y.shape), h)[:, :NFFT]
        z = z - z.mean(-1, keepdims=True)
        P = np.abs(np.fft.rfft(z, axis=-1)) ** 2 / NFFT
        kt = int(round(F0 * NFFT))
        am = P[:, 1:].argmax(-1) + 1
        r[tag] = {"mean_P_bins_1_5": P[:, 1:6].mean(0).round(4).tolist(), "mean_P_tone_bins_25_26": P[:, kt-1:kt+2].mean(0).round(5).tolist(),
                  "mean_P_bins_60_70": float(P[:, 60:70].mean()), "frac_argmax_at_tone_bin(+-1)": float(np.mean(np.abs(am - F0 * NFFT) <= 1.5)),
                  "median_argmax_bin": float(np.median(am))}
    res[name] = r
os.makedirs(os.path.join(HERE, "results", "sr_paper_setting"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", "sr_paper_setting", "diag_sr_spectrum.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
