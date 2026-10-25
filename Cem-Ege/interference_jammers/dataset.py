"""Parallel frame generation + feature extraction with per-frame deterministic seeds.

Seed roles (never mixed between train / tune / report):
    ROLE_TRAIN = 1   seeds "A"   classifier training
    ROLE_TEST  = 2   seeds "B"   fresh frames that every reported number comes from
    ROLE_DEV   = 3   dev seeds   hyper-parameter selection (tree depth, boosting size)
    ROLE_CAL   = 4   gate calibration (noise-only frames)
    ROLE_SANE  = 5   sanity checks
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
import time

import numpy as np

import jcommon as common
import features

ROLE_TRAIN, ROLE_TEST, ROLE_DEV, ROLE_CAL, ROLE_SANE = 1, 2, 3, 4, 5
CAL_ERR_DB = 0.25          # receiver noise power known to +-0.25 dB (uniform)
NPROC = 12
_NAMES = None


def names():
    global _NAMES
    if _NAMES is None:
        _NAMES = features.feature_names()
    return _NAMES


def frame_task(t):
    """t = dict(role, kind, inr, sig (bool), idx, [noise], [cal_err])."""
    kind = t["kind"]
    ki = (common.KINDS + ["none"]).index(kind)
    rng = np.random.default_rng([t["role"], ki, int(round((t["inr"] + 100) * 10)), int(t["sig"]), t["idx"], t.get("tag", 0)])
    x, meta = common.make_frame(kind, rng, inr_db=t["inr"], with_sig=t["sig"], noise=t.get("noise", True))
    cal = rng.uniform(-t.get("cal_err", CAL_ERR_DB), t.get("cal_err", CAL_ERR_DB)) if t.get("cal_err", CAL_ERR_DB) > 0 else 0.0
    f = features.extract(x, cal, bispec=t.get("bispec", True))
    return np.array([f[n] for n in f]), list(f.keys()), meta


def run(tasks, nproc=NPROC, label="", chunksize=8):
    t0 = time.time()
    rows, metas, nm = [], [], None
    with mp.Pool(nproc) as pool:
        for i, (v, names_, meta) in enumerate(pool.imap(frame_task, tasks, chunksize=chunksize)):
            rows.append(v)
            metas.append(meta)
            nm = names_
            if (i + 1) % 1000 == 0:
                print(f"  {label} {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"  {label} done {len(tasks)} frames in {time.time() - t0:.0f}s", flush=True)
    return np.array(rows), nm, metas
