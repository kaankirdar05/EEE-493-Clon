"""Shared loader and paths for running the Cem-Ege classical methods on Dataset 1 v3 (ds1_v3.h5).

Imports the existing code from ../classical_amc, ../wp4_detection, ../detection_*, ../wp6_characterization,
../characterization_extended and ../extra_methods; never edits it.
Set DS1 to the path of ds1_v3.h5 (made with ../../synthetic/make_synth_v3.py --per-snr 1000, seed 2026).
"""
from __future__ import annotations
import os, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # Cem-Ege
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
DS1 = os.environ.get("DS1", os.path.join(ROOT, "..", "synthetic", "ds1_v3.h5"))   # where the README quick start writes it

for p in ("classical_amc", "wp4_detection", "detection_discovered", "detection_extended",
          "wp6_characterization", "characterization_extended", "extra_methods"):
    q = os.path.join(ROOT, p)
    if q not in sys.path:
        sys.path.insert(0, q)


def meta():
    """Labels for all clips: classes, Y, Z, Z_inband, sps, bw99, split."""
    with h5py.File(DS1, "r") as f:
        m = {k: f[k][:] for k in ("Y", "Z", "Z_inband", "sps", "bw99", "split")}
        m["classes"] = [str(c) for c in f.attrs["classes"]]
    return m


def load_x(idx):
    """Complex clips (len(idx), 1024) for sorted or unsorted indices."""
    idx = np.asarray(idx)
    order = np.argsort(idx)
    with h5py.File(DS1, "r") as f:
        X = f["X"]
        out = np.empty((len(idx), X.shape[1]), np.complex64)
        s = idx[order]
        # read in contiguous-ish chunks
        for a in range(0, len(s), 20000):
            blk = X[s[a:a + 20000]]
            out[order[a:a + 20000]] = blk[..., 0] + 1j * blk[..., 1]
    return out


def subsample(m, split, per_cell, seed=0, classes=None):
    """Indices of `split` with at most `per_cell` clips per (class, SNR) cell."""
    rng = np.random.default_rng(seed)
    out = []
    names = m["classes"]
    for c in range(len(names)):
        if classes is not None and names[c] not in classes:
            continue
        sel = (m["split"] == split) & (m["Y"] == c)
        zz = np.where(np.isnan(m["Z"]), -999, m["Z"])
        if names[c] == "NOISE":
            ii = np.flatnonzero(sel)
            out.append(rng.choice(ii, min(len(ii), per_cell * 16), replace=False))
            continue
        for s in np.unique(zz[sel]):
            ii = np.flatnonzero(sel & (zz == s))
            out.append(rng.choice(ii, min(len(ii), per_cell), replace=False))
    return np.sort(np.concatenate(out))
