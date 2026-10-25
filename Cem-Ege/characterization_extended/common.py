"""Shared paths and imports for characterization_extended (imports existing code, never edits it)."""

import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
os.makedirs(RESULTS, exist_ok=True)

for p in (os.path.join(ROOT, "classical_amc"), os.path.join(ROOT, "extra_methods")):
    if p not in sys.path:
        sys.path.insert(0, p)

HELDOUT_SR = r"D:\data\symbol_rate_torchsig.npz"          # seed 0, held-out (report only)
DEV_SR = os.path.join(DATA, "symbol_rate_torchsig_dev.npz")  # seed 1, tuning
HELDOUT_ALRT = r"D:\data\alrt_torchsig.npz"
DEV_ALRT = os.path.join(DATA, "alrt_torchsig_dev.npz")
RADIOML = r"D:\data\radioml2018\GOLD_XYZ_OSC.0001_1024.hdf5"


def wrap(e, period=1.0):
    """Wrap an error into [-period/2, period/2)."""
    return (e + period / 2) % period - period / 2
