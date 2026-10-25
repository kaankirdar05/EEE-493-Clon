"""Hough-transform line detection for chirp-like emitters (LFM, chirp spread spectrum, LoRa).

A linear frequency sweep is a straight, sloped line in the spectrogram. Steps:
1. Hough accumulator over the detection mask: rho = f cos(theta) + t sin(theta),
   theta in [0, 180) degrees, 1-pixel rho bins.
2. Peaks above `min_votes`, with non-maximum suppression, excluding near-vertical
   lines (constant frequency: ordinary narrowband signals, theta ~ 0) and
   near-horizontal lines (constant time: wideband bursts, gain steps, theta ~ 90).
3. Mask pixels within 1.5 px of each line, split into segments at gaps > `max_gap`;
   segments with >= `min_len` pixels are kept and drawn into a new mask.
4. That mask goes through the normal segmentation, so a sawtooth sweep (many
   parallel segments) becomes one box.

Defaults tuned on dev scenes 0-99 (both datasets, 32 chirp-family emitters):
TorchSig chirps are mostly slow sweeps, only ~5-20 degrees from vertical, so the
constant-frequency exclusion is narrow (3 deg) and the 10-bin sweep rule does the
separation. The thinness test (max_side_ratio) rejected dense LoRa sawtooths and
is off by default (99); 0.5 enables it (fewer false boxes, much lower recall).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .segmentation import segment


def hough_lines(mask, n_theta=180, min_votes=40, exclude_deg=3, nms=(8, 6), max_lines=20):
    t, f = np.nonzero(mask)
    if len(t) == 0:
        return []
    thetas = np.deg2rad(np.arange(n_theta) * 180 / n_theta)
    diag = int(np.ceil(np.hypot(*mask.shape)))
    rho = np.round(np.outer(f, np.cos(thetas)) + np.outer(t, np.sin(thetas))).astype(int) + diag
    acc = np.zeros((2 * diag + 1, n_theta), dtype=np.int32)
    for j in range(n_theta):
        acc[:, j] = np.bincount(rho[:, j], minlength=2 * diag + 1)
    deg = np.rad2deg(thetas)
    bad = (deg < exclude_deg) | (deg > 180 - exclude_deg) | (np.abs(deg - 90) < exclude_deg)
    acc[:, bad] = 0
    peaks = (acc == ndimage.maximum_filter(acc, size=nms)) & (acc >= min_votes)
    r, c = np.nonzero(peaks)
    order = np.argsort(acc[r, c])[::-1][:max_lines]
    return [(float(r[k] - diag), float(thetas[c[k]])) for k in order]


def line_segments_mask(mask, lines, tol=3.0, max_gap=8, min_len=20, side=(4, 8), max_side_ratio=99.0,
                       min_freq_span=10):
    t, f = np.nonzero(mask)
    out = np.zeros_like(mask)
    for rho, th in lines:
        d = np.abs(f * np.cos(th) + t * np.sin(th) - rho)
        sel = d <= tol
        side_sel = (d > side[0]) & (d <= side[1])
        side_proj = -f[side_sel] * np.sin(th) + t[side_sel] * np.cos(th)
        if sel.sum() < min_len:
            continue
        tt, ff = t[sel], f[sel]
        proj = -ff * np.sin(th) + tt * np.cos(th)       # position along the line
        o = np.argsort(proj)
        tt, ff, proj = tt[o], ff[o], proj[o]
        breaks = np.flatnonzero(np.diff(proj) > max_gap) + 1
        for seg in np.split(np.arange(len(proj)), breaks):
            # judge by extent ALONG the line, not pixel count: a line crossing a wide
            # constant-frequency stripe collects many pixels over a short extent
            if proj[seg[-1]] - proj[seg[0]] < min_len:
                continue
            # a chirp must actually sweep in frequency; a near-vertical line crossing a
            # constant-frequency signal covers only that signal's few bins
            if ff[seg].max() - ff[seg].min() < min_freq_span:
                continue
            # thinness: a chirp is a thin line, so the bands 4-8 px either side must be mostly
            # empty; inside a thick region (wideband burst, wide signal) they are as full as the core
            n_side = np.sum((side_proj >= proj[seg[0]]) & (side_proj <= proj[seg[-1]]))
            core_density = len(seg) / (2 * tol + 1)
            side_density = n_side / (2 * (side[1] - side[0]))
            if side_density <= max_side_ratio * core_density:
                out[tt[seg], ff[seg]] = True
    return out


def merge_sweeps(boxes, f_overlap=0.5, max_t_gap=32):
    """Merge boxes covering the same band that follow each other in time
    (the sweeps of one sawtooth chirp emitter)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ov = min(a[3], b[3]) - max(a[2], b[2])
                if ov <= 0 or ov < f_overlap * min(a[3] - a[2], b[3] - b[2]):
                    continue
                gap = max(a[0], b[0]) - min(a[1], b[1])
                if gap <= max_t_gap:
                    boxes[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def hough_boxes(mask, **kw):
    lines = hough_lines(mask, **{k: v for k, v in kw.items() if k in ("n_theta", "min_votes", "exclude_deg", "nms", "max_lines")})
    seg_mask = line_segments_mask(mask, lines, **{k: v for k, v in kw.items() if k in ("tol", "max_gap", "min_len", "side", "max_side_ratio", "min_freq_span")})
    # thicken the 1-px segments so parallel sweeps of one emitter (sawtooth) merge into one box
    seg_mask = ndimage.binary_dilation(seg_mask, structure=np.ones((13, 5), bool))
    boxes = segment(seg_mask, close=(25, 9), open_=None, min_cells=40)
    # undo the dilation margin (6 frames, 2 bins) on each side, then merge sawtooth sweeps
    boxes = [(t0 + 6, t1 - 6, f0 + 2, f1 - 2) for t0, t1, f0, f1 in boxes if t1 - t0 > 12 and f1 - f0 > 4]
    return merge_sweeps(boxes), lines
