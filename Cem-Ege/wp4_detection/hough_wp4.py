"""Hough-transform chirp detection on the TorchSig scenes (100-299), both datasets.

Chirp family = classes whose name contains chirp, lfm or lora. Compared:
    energy + segmentation   the standard WP4 pipeline (8-frame smoothing)
    Hough                   lines in a 2-frame-smoothed energy mask (less time smear
                            of sloped lines), Pfa 1e-3, then sweep merging
    union                   best of the two boxes per emitter
Scored: recall of chirp-family emitters at IoU >= 0.5 and >= 0.3, and Hough boxes
that do not overlap any chirp emitter (IoU < 0.1) per scene (false chirp calls).

    python hough_wp4.py  -> results/hough.json
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np

from wp4.detectors import Energy
from wp4.hough import hough_boxes
from wp4.scoring import iou
from wp4.segmentation import segment
from wp4.spectrum import box_to_cells, spectrogram

DATASETS = {"impaired": r"D:\data\wideband\scenes.h5", "clean": r"D:\data\wideband\scenes_clean.h5"}
CHIRP = ("chirp", "lfm", "lora")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side-ratio", type=float, default=99.0, help="thinness rule: 99 = off (default), 0.5 = on")
    args = ap.parse_args()
    cal = os.path.join("results", "calibration")
    e8 = Energy()
    e8.calibrate([1e-3], cal)
    e2 = Energy(kt=2)
    e2.name = "energy_kt2"
    e2.calibrate([1e-3], cal)
    out = {}
    for ds, path in DATASETS.items():
        best = {"energy": [], "hough": [], "union": []}
        false_calls, n_scenes = 0, 0
        with h5py.File(path, "r") as f:
            B = f["boxes"][:]
            for i in range(100, f["iq"].shape[0]):
                iq = f["iq"][i]
                spec = spectrogram(iq)
                truth = B[B["scene"] == i]
                eboxes = segment(e8.masks(spec, iq, [1e-3])[1e-3])
                hboxes, _ = hough_boxes(e2.masks(spec, iq, [1e-3])[1e-3], max_side_ratio=args.max_side_ratio)
                chirp_cells = [box_to_cells(b, spec.shape, 1e7) for b in truth
                               if any(c in b["class_name"].decode() for c in CHIRP)]
                for tc in chirp_cells:
                    be = max([iou(p, tc) for p in eboxes], default=0.0)
                    bh = max([iou(p, tc) for p in hboxes], default=0.0)
                    best["energy"].append(be)
                    best["hough"].append(bh)
                    best["union"].append(max(be, bh))
                false_calls += sum(1 for h in hboxes if max([iou(h, tc) for tc in chirp_cells], default=0.0) < 0.1)
                n_scenes += 1
        r = {k: {"recall_iou50": float(np.mean(np.array(v) >= 0.5)), "recall_iou30": float(np.mean(np.array(v) >= 0.3)),
                 "median_best_iou": float(np.median(v))} for k, v in best.items()}
        r["n_chirp_emitters"] = len(best["energy"])
        r["hough_false_chirp_boxes_per_scene"] = false_calls / n_scenes
        out[ds] = r
        print(f"{ds}: {r['n_chirp_emitters']} chirp-family emitters")
        for k in best:
            print(f"   {k:>7s}: recall@IoU0.5 {r[k]['recall_iou50'] * 100:5.1f} %  recall@IoU0.3 {r[k]['recall_iou30'] * 100:5.1f} %  "
                  f"median best IoU {r[k]['median_best_iou']:.2f}")
        print(f"   Hough boxes not on a chirp emitter: {r['hough_false_chirp_boxes_per_scene']:.2f} per scene")
    name = "hough.json" if args.max_side_ratio >= 99 else f"hough_side{args.max_side_ratio:g}.json"
    with open(os.path.join("results", name), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
