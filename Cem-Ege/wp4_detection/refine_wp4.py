"""Score box refinement (wavelet frequency edges, CUSUM time edges) on top of the
energy detector + segmentation, scenes 100-299 of both datasets, Pfa 1e-3.

    python refine_wp4.py            -> results/refinement.json, printed table
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np

from wp4.detectors import Energy
from wp4.refine import cusum_time_edges, wavelet_freq_edges
from wp4.scoring import score_scene, summarize
from wp4.segmentation import segment
from wp4.spectrum import spectrogram

DATASETS = {"impaired": r"D:\data\wideband\scenes.h5", "clean": r"D:\data\wideband\scenes_clean.h5"}
VARIANTS = {
    "segmentation only": lambda s, b: b,
    "+ wavelet freq edges": lambda s, b: wavelet_freq_edges(s, b),
    "+ CUSUM time edges": lambda s, b: cusum_time_edges(s, b),
    "+ both": lambda s, b: cusum_time_edges(s, wavelet_freq_edges(s, b)),
}


def main():
    det = Energy()
    det.calibrate([1e-3], os.path.join("results", "calibration"))
    out = {}
    for ds, path in DATASETS.items():
        scores = {v: [] for v in VARIANTS}
        with h5py.File(path, "r") as f:
            B = f["boxes"][:]
            for i in range(100, f["iq"].shape[0]):
                iq = f["iq"][i]
                spec = spectrogram(iq)
                mask = det.masks(spec, iq, [1e-3])[1e-3]
                boxes = segment(mask)
                truth = B[B["scene"] == i]
                for v, fn in VARIANTS.items():
                    refined = [fn(spec, b) for b in boxes]
                    scores[v].append(score_scene(mask, refined, truth, spec.shape, 1e7))
        out[ds] = {}
        print(f"\n{ds}:")
        for v in VARIANTS:
            s = summarize(scores[v])
            out[ds][v] = {k: s[k] for k in ("box_f1", "box_precision", "box_recall", "mean_iou_matched")}
            out[ds][v]["recall_per_snr"] = {f"{q['snr_lo']}..{q['snr_hi']}": q["box_recall"] for q in s["per_snr"]}
            print(f"  {v:>22s}: F1 {s['box_f1'] * 100:5.1f}  P {s['box_precision'] * 100:5.1f}  R {s['box_recall'] * 100:5.1f}  "
                  f"mean IoU {s['mean_iou_matched']:.3f}")
    with open(os.path.join("results", "refinement.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
