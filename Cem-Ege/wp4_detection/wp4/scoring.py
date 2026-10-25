"""Shared WP4 scoring: any detector that outputs a (T, F) mask, and any segmenter
that outputs boxes, is scored the same way. This is what makes the energy vs
CA-CFAR (and later YOLO, job 4.4) comparison fair.

Cell level (job 4.1 / 4.2 "Pd and Pfa"):
    Pfa       = flagged cells outside every truth box / cells outside every truth box
    emitter Pd = fraction of an emitter's truth cells that are flagged;
                 the emitter counts as detected when that fraction >= `hit_fraction`
Box level (job 4.3 "IoU"):
    predicted and truth boxes are matched one-to-one (Hungarian) on IoU; a match
    with IoU >= `iou_thr` is a true positive.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .spectrum import box_to_cells


def iou(a, b) -> float:
    t = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    f = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = t * f
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def score_scene(mask, pred_boxes, truth, shape, sample_rate, hit_fraction=0.5, iou_thr=0.5):
    """Score one scene. `truth` is the scene's rows of the HDF5 `boxes` table."""
    truth_cells = [box_to_cells(b, shape, sample_rate) for b in truth]
    tmask = np.zeros(shape, dtype=bool)
    for t0, t1, f0, f1 in truth_cells:
        tmask[t0:t1, f0:f1] = True

    emitters = []
    for b, (t0, t1, f0, f1) in zip(truth, truth_cells):
        area = max((t1 - t0) * (f1 - f0), 1)
        frac = float(mask[t0:t1, f0:f1].sum() / area)
        emitters.append({"snr_db": float(b["snr_db"]), "class_name": b["class_name"].decode(),
                         "cell_pd": frac, "detected": frac >= hit_fraction})

    # box matching
    best_iou = np.zeros(len(truth_cells))
    tp = 0
    if pred_boxes and truth_cells:
        m = np.array([[iou(p, t) for t in truth_cells] for p in pred_boxes])
        r, c = linear_sum_assignment(-m)
        for i, j in zip(r, c):
            best_iou[j] = m[i, j]
            tp += m[i, j] >= iou_thr
    for e, v in zip(emitters, best_iou):
        e["iou"] = float(v)
        e["box_hit"] = bool(v >= iou_thr)

    return {
        "noise_cells": int((~tmask).sum()),
        "false_alarm_cells": int((mask & ~tmask).sum()),
        "signal_cells": int(tmask.sum()),
        "hit_cells": int((mask & tmask).sum()),
        "n_pred": len(pred_boxes),
        "n_truth": len(truth_cells),
        "tp": int(tp),
        "emitters": emitters,
    }


def summarize(scene_scores, snr_edges=(-5, 0, 5, 10, 15, 20, 25, 30)):
    """Aggregate scene scores into overall metrics and per-SNR curves."""
    noise = sum(s["noise_cells"] for s in scene_scores)
    fa = sum(s["false_alarm_cells"] for s in scene_scores)
    sig = sum(s["signal_cells"] for s in scene_scores)
    hit = sum(s["hit_cells"] for s in scene_scores)
    n_pred = sum(s["n_pred"] for s in scene_scores)
    n_truth = sum(s["n_truth"] for s in scene_scores)
    tp = sum(s["tp"] for s in scene_scores)
    em = [e for s in scene_scores for e in s["emitters"]]
    snr = np.array([e["snr_db"] for e in em])

    per_snr = []
    for lo, hi in zip(snr_edges[:-1], snr_edges[1:]):
        sel = [e for e, v in zip(em, snr) if lo <= v < hi or (hi == snr_edges[-1] and v == hi)]
        if not sel:
            continue
        per_snr.append({
            "snr_lo": lo, "snr_hi": hi, "n": len(sel),
            "emitter_pd": float(np.mean([e["detected"] for e in sel])),
            "mean_cell_pd": float(np.mean([e["cell_pd"] for e in sel])),
            "box_recall": float(np.mean([e["box_hit"] for e in sel])),
            "mean_iou": float(np.mean([e["iou"] for e in sel])),
        })
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_truth if n_truth else 0.0
    return {
        "cell_pfa": fa / noise if noise else 0.0,
        "cell_pd": hit / sig if sig else 0.0,
        "emitter_pd": float(np.mean([e["detected"] for e in em])) if em else 0.0,
        "box_precision": precision,
        "box_recall": recall,
        "box_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_iou_matched": float(np.mean([e["iou"] for e in em if e["box_hit"]])) if any(e["box_hit"] for e in em) else 0.0,
        "n_scenes": len(scene_scores), "n_emitters": len(em), "n_pred_boxes": n_pred,
        "per_snr": per_snr,
    }
