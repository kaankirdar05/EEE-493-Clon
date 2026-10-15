"""Cut labelled SigMF recordings into 1,024-sample training clips, saved as HDF5 (our agreed format).
Example: python sigmf_to_h5.py practice_qpsk --out real_v1.h5      (several recordings: list them all)
"""
import argparse, pathlib
from fractions import Fraction
import numpy as np, h5py
from scipy.signal import resample_poly
from sigmf import sigmffile

CLASSES = (pathlib.Path(__file__).resolve().parents[1] / "classes.txt").read_text().split()   # shared class list
N = 1024                                             # samples per clip

p = argparse.ArgumentParser()
p.add_argument("recordings", nargs="+")              # recording names without .sigmf-meta
p.add_argument("--out", required=True)
p.add_argument("--source", default="cabled")         # cabled or ota
a = p.parse_args()

X, Y = [], []
for name in a.recordings:
    rec = sigmffile.fromfile(name)
    fs = rec.get_global_info()["core:sample_rate"]
    fc = rec.get_captures()[0]["core:frequency"]
    for ann in rec.get_annotations():
        if ann["core:label"] not in CLASSES:          # e.g. "noise" files are skipped here
            continue
        x = rec.read_samples(ann["core:sample_start"], ann["core:sample_count"])
        lo, hi = ann["core:freq_lower_edge"], ann["core:freq_upper_edge"]
        # 1. move the signal's centre to 0 Hz
        x = x * np.exp(-2j * np.pi * ((lo + hi) / 2 - fc) / fs * np.arange(len(x)))
        # 2. resample so the new sample rate is 4 x the signal bandwidth (cuts away the other signals too)
        r = Fraction(4 * (hi - lo) / fs).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)
        # 3. cut into 1,024-sample clips  4. scale each clip to average power 1 and label it
        for k in range(len(x) // N):
            c = x[k * N:(k + 1) * N]
            c = c / np.sqrt(np.mean(np.abs(c) ** 2))
            X.append(np.stack([c.real, c.imag], axis=1))
            Y.append(CLASSES.index(ann["core:label"]))
    print(f"{name}: {len(X)} clips so far")

X = np.array(X, np.float32); Y = np.array(Y, np.int64)
Z = np.full(len(Y), np.nan, np.float32)               # SNR not known yet: fill from the log or estimate (job 6.2)
with h5py.File(a.out, "w") as f:
    f["X"], f["Y"], f["Z"] = X, Y, Z
    f.attrs.update(classes=CLASSES, resampling="4 x signal bandwidth", source=a.source, version="v1")
print(f"saved {len(Y)} clips to {a.out}: X {X.shape}")
