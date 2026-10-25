"""Diagnostic (dev scenes 0-39): where do `energy` and `tkeo_stft` differ?

TKEO along time per STFT bin removes the part of |X|^2 that is coherent between
frames m-1 and m+1: a constant phasor at a bin (DC offset / LO leakage, an
unmodulated carrier whose frequency puts 2*hop phase rotation near 0) gives
Psi ~ 0. Reports, at Pfa target 1e-3, the false-alarm cells per frequency region
and the emitter Pd split by emitter duration.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disc import ROOT  # noqa: E402
from disc.tkeo import TKEOSTFT  # noqa: E402
from wp4.detectors import Energy  # noqa: E402
from wp4.spectrum import box_to_cells, spectrogram  # noqa: E402


def main():
    e = Energy()
    e.calibrate([1e-3], os.path.join(ROOT, "wp4_detection", "results", "calibration"))
    t = TKEOSTFT()
    t.calibrate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], os.path.join(HERE, "results", "calibration"))
    out = {}
    for ds, path in (("impaired", r"D:\data\wideband\scenes.h5"), ("clean", r"D:\data\wideband\scenes_clean.h5")):
        f = h5py.File(path, "r")
        fs = float(f.attrs["sample_rate"])
        boxes = f["boxes"][:]
        fa = {"energy": np.zeros(512), "tkeo_stft": np.zeros(512)}
        noise_cells = np.zeros(512)
        em = []
        for i in range(40):
            iq = f["iq"][i]
            spec = spectrogram(iq, 512)
            truth = boxes[boxes["scene"] == i]
            tm = np.zeros(spec.shape, bool)
            cells = [box_to_cells(b, spec.shape, fs) for b in truth]
            for t0, t1, f0, f1 in cells:
                tm[t0:t1, f0:f1] = True
            masks = {"energy": e.mask(e.stat(spec, iq), 1e-3), "tkeo_stft": t.mask(t.stat(spec, iq), 1e-3)}
            noise_cells += (~tm).sum(0)
            for k, m in masks.items():
                fa[k] += (m & ~tm).sum(0)
            for b, (t0, t1, f0, f1) in zip(truth, cells):
                a = max((t1 - t0) * (f1 - f0), 1)
                em.append({"dur": float(b["t_stop"] - b["t_start"]), "snr": float(b["snr_db"]),
                           **{k: float(m[t0:t1, f0:f1].sum() / a) for k, m in masks.items()}})
        regions = {"DC +-3 bins": slice(253, 260), "edges 0-31 & 480-511": np.r_[0:32, 480:512], "rest": None}
        r = {}
        for name, sl in regions.items():
            if sl is None:
                sel = np.ones(512, bool)
                sel[253:260] = False
                sel[:32] = sel[480:] = False
            else:
                sel = np.zeros(512, bool)
                sel[sl] = True
            r[name] = {k: float(v[sel].sum() / noise_cells[sel].sum()) for k, v in fa.items()}
        long_ = [x for x in em if x["dur"] > 0.9]
        short = [x for x in em if x["dur"] <= 0.9]
        r["emitter_pd_long(dur>0.9)"] = {k: float(np.mean([x[k] >= 0.5 for x in long_])) for k in fa} | {"n": len(long_)}
        r["emitter_pd_short"] = {k: float(np.mean([x[k] >= 0.5 for x in short])) for k in fa} | {"n": len(short)}
        out[ds] = r
        print(ds, json.dumps(r, indent=1), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "diag_tkeo_vs_energy.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
