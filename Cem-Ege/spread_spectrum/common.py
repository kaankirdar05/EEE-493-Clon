"""Shared helpers: single-threaded BLAS in workers, process pool, repo imports, seeds."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "classical_amc"))
sys.path.insert(0, os.path.join(REPO, "wp4_detection"))
os.makedirs(RES, exist_ok=True)

NPROC = 20

# Seed blocks (never overlap): calibration of thresholds, development/tuning, final report, Pfa validation
SEED_CAL, SEED_DEV, SEED_REP, SEED_VAL = 3_000_000, 1_000_000, 2_000_000, 4_000_000


def rng_for(base, *idx):
    return np.random.default_rng([base, *idx])


def pool_map(fn, args, nproc=NPROC, chunksize=None):
    import multiprocessing as mp
    args = list(args)
    if nproc <= 1 or len(args) < 4:
        return [fn(a) for a in args]
    cs = chunksize or max(1, len(args) // (nproc * 8))
    with mp.get_context("spawn").Pool(nproc) as p:
        return p.map(fn, args, chunksize=cs)


def cplx_noise(n, rng):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
