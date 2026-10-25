"""Part B step 1: evaluate every Morlet configuration (kappa x centre set x power; local / global prominence) on the DEV and held-out TorchSig
symbol-rate frames and cache (rate, prominence) per frame in results/b_features_{dev,heldout}.npz.  NO selection happens here (selection: b_eval.py, DEV only).

    python b_features.py dev|heldout
"""
import os
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from joblib import Parallel, delayed
import wavelet_sr as ws

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PATHS = {"dev": os.path.join(ws.CE, "data", "symbol_rate_torchsig_dev.npz"), "heldout": r"D:\data\symbol_rate_torchsig.npz"}


def chunk(xs):
    out = []
    for x in xs:
        d = ws.frame_features(x)
        out.append([v for c in ws.CONFIGS for v in d[c]])        # (rate_l, prom_l, rate_g, prom_g) per config
    return np.array(out)


def main(which, n_jobs=20):
    d = np.load(PATHS[which])
    x = d["x"]
    t0 = time.time()
    parts = np.array_split(np.arange(len(x)), 120)
    res = Parallel(n_jobs=n_jobs)(delayed(chunk)(x[p].astype(complex)) for p in parts)
    A = np.concatenate(res)                                             # (frames, configs * 4)
    names, R, P = [], [], []
    for i, c in enumerate(ws.CONFIGS):
        for j, mode in enumerate(("l", "g")):
            names.append(ws.cfg_name(c) + "." + mode)
            R.append(A[:, 4 * i + 2 * j]); P.append(A[:, 4 * i + 2 * j + 1])
    np.savez(os.path.join(RES, f"b_features_{which}.npz"), names=np.array(names), R=np.array(R).T, P=np.array(P).T,
             rs=d["rs"], cls=d["cls"], snr=d["snr"], pulse=d["pulse"])
    print(which, A.shape, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
