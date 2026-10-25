"""Calibrate the scd_fam threshold on 400k white-noise blocks in parallel (same
quantile rule as Detector.calibrate, just faster). Writes results/calibration/<name>.npz."""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext.block_detectors import SCDFam  # noqa: E402

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else None


def job(seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((10000, 512)) + 1j * rng.standard_normal((10000, 512))) / np.sqrt(2)
    return SCDFam(Np=NP).block_stat(x)


if __name__ == "__main__":
    with Pool(20) as p:
        h = np.concatenate(p.map(job, range(SEED0, SEED0 + 40)))
    pf = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    name = OUTNAME or SCDFam(Np=NP).name
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "calibration")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, name + ".npz"), pfas=np.array(pf), thr=np.array([np.quantile(h, 1 - q) for q in pf]),
             n_h0=len(h))
    np.save(os.path.join(d, name + "_h0.npy"), h.astype(np.float32))
    print(name, len(h), [round(float(np.quantile(h, 1 - q)), 5) for q in pf])
