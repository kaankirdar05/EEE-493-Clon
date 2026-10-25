"""Single-process timing of each Part A detector's stat() on 5 impaired scenes (shared machine:
other jobs were running, so treat as upper bounds)."""
import json
import os
import sys
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_partA import make  # noqa: E402
from wp4.spectrum import spectrogram  # noqa: E402

SPECS = [("cav", "L=4"), ("acorr", "L=32"), ("entropy_spec", "m=64"), ("entropy_hist", "bins=16"), ("scd_fam", "Np=16"),
         ("cav_pw", "L=4"), ("entropy_spec_pw", "m=64")]
out = {}
with h5py.File(r"D:\data\wideband\scenes.h5", "r") as f:
    iqs = [f["iq"][i] for i in range(100, 105)]
for d, p in SPECS:
    det = make(d, [p])
    t = time.perf_counter()
    for iq in iqs:
        det.stat(spectrogram(iq, 512), iq)
    out[det.name] = (time.perf_counter() - t) / len(iqs) * 1e3
    print(det.name, f"{out[det.name]:.0f} ms/scene", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "timing_single_process.json"), "w"), indent=2)
