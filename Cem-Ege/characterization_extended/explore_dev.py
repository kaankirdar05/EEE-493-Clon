"""Scratch exploration on the DEV set only (seed 1). Prints per-class success of candidate estimators."""
import sys
import time

import numpy as np

import common
from symbol_rate import ESTIMATORS
import fsk_rate as fr

d = np.load(common.DEV_SR)
x, rs, cls, snr = d["x"], d["rs"], d["cls"], d["snr"]
classes = list(dict.fromkeys(cls.tolist()))
sub = None
if len(sys.argv) > 1 and sys.argv[1] == "fsk":
    sub = np.isin(cls, ["2fsk", "4fsk", "2gfsk", "2msk", "2gmsk"])
idx = np.flatnonzero(sub) if sub is not None else np.arange(len(x))

cands = {"inst_freq_s3": lambda z: ESTIMATORS["inst_freq_s3"](z)}
sel = sys.argv[2].split(",") if len(sys.argv) > 2 else list(fr.CANDIDATES)
cands.update({k: (lambda f: (lambda z: f(z)[0]))(f) for k, f in fr.CANDIDATES.items() if k in sel})
for k, f in cands.items():
    t0 = time.perf_counter()
    est = np.array([f(x[i].astype(complex)) for i in idx])
    ok = np.abs(est - rs[idx]) / rs[idx] < 0.02
    pc = {c: ok[cls[idx] == c].mean() for c in classes if (cls[idx] == c).any()}
    print(f"{k:>16s} {ok.mean() * 100:5.1f} | " + " ".join(f"{c}:{v * 100:3.0f}" for c, v in pc.items()) +
          f"  ({(time.perf_counter() - t0) / len(idx) * 1e3:.1f} ms)", flush=True)
