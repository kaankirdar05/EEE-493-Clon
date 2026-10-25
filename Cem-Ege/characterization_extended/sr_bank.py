"""Bank of symbol-rate line estimators (each -> rate, prominence) evaluated on a set of frames.

Members (search range 1/25 .. 1/2.5 cycles/sample; prominence = line power / local median
background, see fsk_rate.line):
  raw-signal features (from classical_amc/symbol_rate.py):  env2 |x|^2, env4 |x|^4,
      dm (delay-and-multiply, |x[n+d] x*[n]|^2 summed over d = 1..3), haarx (|Haar(x)|, scale 1),
      ifr1 / ifr3 (|inst. freq - median|, smoothing 1 / 3 = inst_freq / inst_freq_s3)
  FM-discriminator features after band-limiting (fsk_rate.bandlimit):
      ifd_L  |f_s - median|, haar_L Haar of frequency, dsq_L squared increment,  L in SCALES
      caf_d  cyclic autocorrelation |FFT(y[n+d] y*[n])|^2 folded over +-alpha, d in CAF_LAGS
The discriminator / CAF members are computed for two band-limiting widths: 0.6 x OBW (no suffix)
and 0.4 x OBW (suffix "n", better below the FM threshold, worse at high SNR).
Every member is computed twice: local-background prominence (".l") and band-median prominence (".g").

    python sr_bank.py dev|heldout|alrt_dev|alrt_heldout|radioml
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import common
import fsk_rate as fr
from symbol_rate import _delay_mult, haar_transform

SCALES = (1, 2, 3, 5, 8)
CAF_LAGS = (1, 2, 3, 4, 6, 8)


def _ifr(x, L):
    f = np.angle(x[1:] * np.conj(x[:-1]))
    f = f - np.median(f)
    if L > 1:
        f = np.convolve(f, np.ones(L) / L, "same")
    return np.abs(f)


def members(x):
    """dict name -> (rate, prominence) for one frame, both prominence modes."""
    x = np.asarray(x, complex)
    raw = {"env2": np.abs(x) ** 2, "env4": np.abs(x) ** 4, "dm": _delay_mult(x), "haarx": haar_transform(x, 1),
           "ifr1": _ifr(x, 1), "ifr3": _ifr(x, 3)}
    ys = {"": fr.bandlimit(x), "n": fr.bandlimit(x, factor=0.4)}   # n = narrow band-limiting
    for tag, y in ys.items():
        f = fr.discriminator(y)
        for L in SCALES:
            raw[f"ifd{L}{tag}"] = fr.feat_if_dev(f, L)
            raw[f"haar{L}{tag}"] = fr.feat_haar(f, L)
            raw[f"dsq{L}{tag}"] = fr.feat_dsq(f, L)
    out = {}
    for mode in ("l", "g"):
        fr.LOCAL_BG = mode == "l"
        for k, v in raw.items():
            out[f"{k}.{mode}"] = fr.line(v)
        for tag, y in ys.items():
            for d in CAF_LAGS:
                P, n = fr.caf_spectrum(y, (d,))
                out[f"caf{d}{tag}.{mode}"] = fr.line_folded(P, n)
    fr.LOCAL_BG = True
    return out


def compute(xs, verbose=True):
    t0 = time.perf_counter()
    first = members(xs[0])
    names = list(first)
    R = np.zeros((len(xs), len(names)))
    P = np.zeros((len(xs), len(names)))
    for i, x in enumerate(xs):
        m = first if i == 0 else members(x)
        R[i] = [m[k][0] for k in names]
        P[i] = [m[k][1] for k in names]
        if verbose and i % 500 == 0:
            print(f"  {i}/{len(xs)}  {(time.perf_counter() - t0) / (i + 1) * 1e3:.0f} ms/frame", flush=True)
    return names, R, P


def load_set(name):
    if name == "dev":
        d = np.load(common.DEV_SR)
        return d["x"], {"rs": d["rs"], "cls": d["cls"], "snr": d["snr"], "pulse": d["pulse"]}
    if name == "heldout":
        d = np.load(common.HELDOUT_SR)
        return d["x"], {"rs": d["rs"], "cls": d["cls"], "snr": d["snr"], "pulse": d["pulse"]}
    if name in ("alrt_dev", "alrt_heldout"):
        d = np.load(common.DEV_ALRT if name == "alrt_dev" else common.HELDOUT_ALRT)
        return d["x"], {"rs": 1 / d["sps"], "y": d["y"], "snr": d["snr"]}
    if name == "radioml":
        from amc.radioml import iter_radioml2018
        CL = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
        xs, ys, ss = [], [], []
        for x, k, s in iter_radioml2018(common.RADIOML, classes=CL, snr_min=-10, snr_max=20, per_group=120,
                                        order="hdf5", seed=5):
            xs.append(x); ys.append(np.full(len(x), k)); ss.append(np.full(len(x), s))
        x = np.concatenate(xs)
        return x, {"rs": np.full(len(x), 1 / 8), "y": np.concatenate(ys), "snr": np.concatenate(ss)}
    raise ValueError(name)


def bank_path(name):
    return os.path.join(common.DATA, f"bank_{name}.npz")


def get_bank(name):
    p = bank_path(name)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=False)
        meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
        return list(d["names"]), d["R"], d["P"], meta
    x, meta = load_set(name)
    names, R, P = compute(x)
    np.savez(p, names=np.array(names), R=R, P=P, **{f"meta_{k}": v for k, v in meta.items()})
    return names, R, P, meta


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        print(nm)
        get_bank(nm)
