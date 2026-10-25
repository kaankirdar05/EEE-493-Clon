"""Run every built-in WP4 detector on both datasets in parallel, then compare.

    python run_all_wp4.py                  # all detectors, impaired + clean
    python run_all_wp4.py --jobs 4 --detectors energy,cfar_ca
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from wp4.detectors import all_detectors

DATASETS = {"impaired": r"D:\data\wideband\scenes.h5", "clean": r"D:\data\wideband\scenes_clean.h5"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--detectors", default=",".join(all_detectors()))
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE")
    # Calibrate once up front so parallel runs don't race on the cache files.
    calib = ("import sys\n"
             "from wp4.detectors import all_detectors\n"
             "names = sys.argv[1].split(',')\n"
             "for n, d in all_detectors().items():\n"
             "    if n in names:\n"
             "        d.calibrate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'results/calibration')\n")
    subprocess.run([sys.executable, "-c", calib, args.detectors], cwd=here, env=env, check=True)

    runs = [(d, name, path) for d in args.detectors.split(",") for name, path in DATASETS.items()]

    def run(r):
        det, name, path = r
        out = os.path.join("results", f"{det}_{name}")
        res = subprocess.run([sys.executable, "run_wp4.py", "--data", path, "--detector", det, "--out", out],
                             cwd=here, env=env, capture_output=True, text=True)
        status = "ok" if res.returncode == 0 else f"FAILED\n{res.stderr[-2000:]}"
        print(f"{det:>10s} {name:>8s}: {status}", flush=True)
        return res.returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        codes = list(ex.map(run, runs))
    subprocess.run([sys.executable, "compare_wp4.py"], cwd=here, env=env)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
