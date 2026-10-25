"""Extract baseline (19), extended cumulant/cyclic (17) and Haar wavelet (24) features from
the SAME RadioML 2018.01A frames as the cached baseline features
(per_group 1000 for the 7-class set, 200 for all 24 classes, seed 0, order 'hdf5').

The recomputed 19 baseline features are compared with the cached ones to prove that the
frames (and hence the split) are identical.

    python extract_radioml_features.py --set 7
    python extract_radioml_features.py --set 24
"""

from __future__ import annotations

import argparse
import os
import time

import common
import numpy as np

from amc.features import extract_features
from amc.radioml import CLASSES_HDF5, DEFAULT_SUBSET, iter_radioml2018
from cumulants_ext import EXT_NAMES, extract_extended
from wavelet import WAV_NAMES, wavelet_features

CACHE = {"7": os.path.join(common.ROOT, "results", "radioml_hgb", "features.npz"),
         "24": os.path.join(common.ROOT, "results", "radioml_hgb_all24", "features.npz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["7", "24"], required=True)
    args = ap.parse_args()
    classes = DEFAULT_SUBSET if args.set == "7" else CLASSES_HDF5
    per_group = 1000 if args.set == "7" else 200
    B, E, W, ys, ss = [], [], [], [], []
    t = {"base": 0.0, "ext": 0.0, "wav": 0.0}
    n = 0
    for i, (x, k, s) in enumerate(iter_radioml2018(common.RADIOML, classes=classes, snr_min=-20, snr_max=30,
                                                   per_group=per_group, order="hdf5", seed=0), 1):
        for key, fn, out in (("base", extract_features, B), ("ext", extract_extended, E), ("wav", wavelet_features, W)):
            t0 = time.perf_counter()
            out.append(fn(x))
            t[key] += time.perf_counter() - t0
        ys.append(np.full(len(x), k, np.int16))
        ss.append(np.full(len(x), s, np.int16))
        n += len(x)
        if i % 26 == 0:
            print(f"  {i} groups, {n} frames", flush=True)
    B, E, W, y, snr = map(np.concatenate, (B, E, W, ys, ss))
    c = np.load(CACHE[args.set])
    same = bool(np.array_equal(c["y"], y) and np.array_equal(c["snr"], snr))
    dmax = float(np.max(np.abs(c["F"] - B))) if same else None
    print(f"labels identical to cache: {same}; max |baseline - cached| = {dmax}")
    np.savez_compressed(os.path.join(common.RESULTS, f"features_{args.set}.npz"), base=B, ext=E, wav=W, y=y, snr=snr,
                        classes=np.array(classes), ext_names=np.array(EXT_NAMES), wav_names=np.array(WAV_NAMES),
                        ms_base=t["base"] / n * 1e3, ms_ext=t["ext"] / n * 1e3, ms_wav=t["wav"] / n * 1e3,
                        cache_labels_identical=same, cache_max_abs_diff=np.nan if dmax is None else dmax)
    print({k: v / n * 1e3 for k, v in t.items()}, "ms/frame")


if __name__ == "__main__":
    main()
