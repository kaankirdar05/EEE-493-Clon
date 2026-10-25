"""Extract the four new feature families (plus the 19 baseline features, as a frame-identity check)
from the SAME RadioML 2018.01A frames as the cached baseline / extended features:
per_group 1000 (7 classes) or 200 (24 classes), seed 0, order 'hdf5', SNR -20..30.

    python extract_features.py --set 7
    python extract_features.py --set 24

Output: results/features_new_{set}.npz with bis (30), fd (22), info (24), st (23), y, snr and
the per-family extraction time (ms/frame, measured inside single-threaded worker processes).
The labels, SNRs and recomputed base19 features are compared with extended/results/features_{set}.npz.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import dcommon as dc  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

FAMS = ("bis", "fd", "info", "st")


def work(x):
    import bispectrum as bs
    import fractal as fd
    import infotheory as it
    import stransform as st
    from amc.features import extract_features
    out, t = {}, {}
    for key, fn in (("base", extract_features), ("bis", bs.bispectrum_features), ("fd", fd.fractal_features),
                    ("info", it.info_features), ("st", st.st_features)):
        t0 = time.perf_counter()
        out[key] = fn(x)
        t[key] = time.perf_counter() - t0
    return out, t


def main():
    import common
    import bispectrum as bs
    import fractal as fd
    import infotheory as it
    import stransform as st
    from amc.radioml import CLASSES_HDF5, DEFAULT_SUBSET, iter_radioml2018

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["7", "24"], required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--buffer", type=int, default=26, help="groups read before each parallel pass")
    args = ap.parse_args()
    classes = DEFAULT_SUBSET if args.set == "7" else CLASSES_HDF5
    per_group = 1000 if args.set == "7" else 200
    res = {k: [] for k in ("base",) + FAMS}
    tsum = {k: 0.0 for k in ("base",) + FAMS}
    ys, ss = [], []
    buf = []
    n = 0
    t_start = time.perf_counter()
    par = Parallel(n_jobs=args.jobs)

    def flush():
        nonlocal buf
        if not buf:
            return
        X = np.concatenate(buf)
        chunks = [X[i:i + args.chunk] for i in range(0, len(X), args.chunk)]
        for o, t in par(delayed(work)(c) for c in chunks):
            for k in res:
                res[k].append(o[k])
                tsum[k] += t[k]
        buf = []

    for i, (x, k, s) in enumerate(iter_radioml2018(common.RADIOML, classes=classes, snr_min=-20, snr_max=30,
                                                   per_group=per_group, order="hdf5", seed=0), 1):
        buf.append(x)
        ys.append(np.full(len(x), k, np.int16))
        ss.append(np.full(len(x), s, np.int16))
        n += len(x)
        if i % args.buffer == 0:
            flush()
            print(f"  {i} groups, {n} frames, {time.perf_counter() - t_start:.0f} s", flush=True)
    flush()
    F = {k: np.concatenate(v) for k, v in res.items()}
    y, snr = np.concatenate(ys), np.concatenate(ss)
    ref = np.load(os.path.join(common.RESULTS, f"features_{args.set}.npz"))
    same = bool(np.array_equal(ref["y"], y) and np.array_equal(ref["snr"], snr))
    dmax = float(np.max(np.abs(ref["base"] - F["base"]))) if same else float("nan")
    ms = {k: v / n * 1e3 for k, v in tsum.items()}
    print(f"labels/SNR identical to extended/results/features_{args.set}.npz: {same}; max |base19 - cached| = {dmax}")
    print("ms/frame (worker CPU, single-threaded):", ms)
    np.savez_compressed(os.path.join(dc.RESULTS, f"features_new_{args.set}.npz"),
                        bis=F["bis"], fd=F["fd"], info=F["info"], st=F["st"], y=y, snr=snr,
                        bis_names=np.array(bs.BIS_NAMES), fd_names=np.array(fd.FD_NAMES),
                        info_names=np.array(it.INFO_NAMES), st_names=np.array(st.ST_NAMES),
                        **{f"ms_{k}": v for k, v in ms.items()},
                        labels_identical=same, base_max_abs_diff=dmax, n_jobs=args.jobs,
                        wall_seconds=time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
