"""Diagnostic: block false-alarm rate on scene blocks that do not touch any truth
box (1-block margin), and the noise autocorrelation in such blocks. Separates
'coloured receiver noise' (impaired set) from 'coarse block overlaps signal
skirts' as the source of the high measured cell Pfa. Dev scenes 0-99 only."""

import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext import WP4  # noqa: E402,F401
from ext.block_detectors import CAV, AutoCorr, SpectralEntropy, Whitened  # noqa: E402
from wp4.detectors import subbands  # noqa: E402
from wp4.spectrum import box_to_cells  # noqa: E402

CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "calibration")


def main():
    dets = [CAV(), AutoCorr(), SpectralEntropy(), Whitened(CAV()), Whitened(SpectralEntropy())]
    for d in dets:
        d.calibrate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], CAL)
    out = {}
    for tag, path in [("impaired", r"D:\data\wideband\scenes.h5"), ("clean", r"D:\data\wideband\scenes_clean.h5")]:
        fa = {d.name: [] for d in dets}
        fa_sb = {d.name: np.zeros(16) for d in dets}
        n_sb = np.zeros(16)
        rho1 = []
        with h5py.File(path, "r") as f:
            B = f["boxes"][:]
            for i in range(100):
                iq = f["iq"][i]
                occ = np.zeros((32, 16), bool)
                for b in B[B["scene"] == i]:
                    t0, t1, f0, f1 = box_to_cells(b, (512, 512), 1e7)
                    occ[max(t0 // 16 - 1, 0): t1 // 16 + 2, max(f0 // 32 - 1, 0): (f1 - 1) // 32 + 2] = True
                x = subbands(iq)
                free = ~occ
                n_sb += free.sum(0)
                for d in dets:
                    hit = d.block_stat(x) > d.thr[1e-3]
                    fa[d.name].append(hit[free])
                    fa_sb[d.name] += (hit & free).sum(0)
                xs = x[free]
                rho1.append(np.abs(np.sum(xs[:, 1:] * np.conj(xs[:, :-1]), 1)) / np.sum(np.abs(xs) ** 2, 1))
        out[tag] = {k: float(np.mean(np.concatenate(v))) for k, v in fa.items()}
        out[tag]["per_subband_fa"] = {k: list(np.round(v / np.maximum(n_sb, 1), 4)) for k, v in fa_sb.items()}
        out[tag]["n_free_blocks"] = int(sum(len(v) for v in fa["cav"]))
        out[tag]["median_|rho1|_free_blocks"] = float(np.nanmedian(np.concatenate(rho1)))
        out[tag]["median_|rho1|_white_noise_N512"] = float(np.sqrt(np.pi / 4 / 512))
        print(tag, out[tag])
    with open(os.path.join(os.path.dirname(CAL), "diag_noise_blocks.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
