"""Covariance-index blind detectors found in the second literature sweep
(Sensors 2026, "Partial covariance-based detectors for cooperative spectrum sensing", doi 10.3390/s26082557,
and the Guimarães detector family it cites):

    GID    Gini index detector:            sum|r_i| / sum_{i,j}|r_i - r_j|        (inverse Gini of the entries)
    GRCR   Gerschgorin radii/centres ratio: sum_i sum_{j!=i}|R_ij| / sum_i |R_ii|
    PRIDe  Pietra-Ricci index detector:     sum|r_i| / sum|r_i - mean(r)|
    TID    Theil index detector:            Theil index of |r_i| (negated: high under signal)
    AID    Atkinson index detector:         geometric_mean(|r|) / arithmetic_mean(|r|)  (= 1 - Atkinson index, so it rises with signal)

R is the L x L sample covariance (smoothing factor L) of each sub-band block (32 time blocks x 16
sub-bands, the same grid as the other block detectors); r = vec(R). Under white noise R ~ sigma^2 I, so the
entries are very unequal (diagonal vs ~0 off-diagonal); a correlated signal fills the off-diagonal.
Thresholds are calibrated on white noise like every other WP4 detector; since calibration is
empirical, any monotonic re-scaling of a statistic gives the same detector.

    python cov_index_detectors.py --check     (fresh-noise Pfa, signal raises the statistics)
    python cov_index_detectors.py             (held-out scenes 100-299, both datasets)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wp4_detection"))
from wp4.detectors import BlockDetector  # noqa: E402

L = 4


def _cov(x):
    n = x.shape[-1] - L + 1
    H = np.stack([x[..., i:i + n] for i in range(L)], axis=-2)
    return H @ np.conj(np.swapaxes(H, -1, -2)) / n


def _stat(kind, R):
    a = np.abs(R).reshape(*R.shape[:-2], -1)                 # |r_i|
    if kind == "gid":
        diff = np.abs(a[..., :, None] - a[..., None, :]).sum((-1, -2))
        return a.sum(-1) / np.maximum(diff, 1e-30)
    if kind == "grcr":
        d = np.abs(np.diagonal(R, axis1=-2, axis2=-1)).sum(-1)
        return (np.abs(R).sum((-1, -2)) - d) / np.maximum(d, 1e-30)
    if kind == "pride":
        return a.sum(-1) / np.maximum(np.abs(a - a.mean(-1, keepdims=True)).sum(-1), 1e-30)
    if kind == "tid":
        u = a / np.maximum(a.mean(-1, keepdims=True), 1e-30)
        return -np.mean(u * np.log(np.maximum(u, 1e-30)), -1)
    if kind == "aid":
        gm = np.exp(np.mean(np.log(np.maximum(a, 1e-30)), -1))
        return gm / np.maximum(a.mean(-1), 1e-30)   # = 1 - Atkinson index: entries become more equal under signal
    raise ValueError(kind)


class CovIndex(BlockDetector):
    def __init__(self, kind):
        self.kind = kind
        self.name = f"{kind}_L{L}"

    def block_stat(self, x):
        return _stat(self.kind, _cov(x))


KINDS = ["gid", "grcr", "pride", "tid", "aid"]


def check(cache):
    from wp4.spectrum import spectrogram
    rng = np.random.default_rng(9)
    ok = True
    for k in KINDS:
        d = CovIndex(k); d.calibrate([1e-2, 1e-3], cache)
        fa = []
        for _ in range(8):
            iq = ((rng.standard_normal(512 * 512) + 1j * rng.standard_normal(512 * 512)) / np.sqrt(2)).astype(np.complex64)
            m = d.masks(spectrogram(iq), iq, [1e-2])[1e-2]; fa.append(m.mean())
        # correlated test signal: low-pass noise (strongly correlated samples) at 0 dB in every block
        z = (rng.standard_normal((200, 512)) + 1j * rng.standard_normal((200, 512)))
        sig = np.apply_along_axis(lambda v: np.convolve(v, np.ones(8) / np.sqrt(8), "same"), 1, z)
        w = (rng.standard_normal((200, 512)) + 1j * rng.standard_normal((200, 512))) / np.sqrt(2)
        s_noise, s_sig = d.block_stat(w), d.block_stat(sig + w)
        rises = np.median(s_sig) > np.quantile(s_noise, 0.99)
        print(f"{d.name:>9s}: fresh-noise Pfa at 1e-2 target = {np.mean(fa):.4f};  correlated signal raises statistic above noise 99 %: {rises}")
        ok &= abs(np.mean(fa) - 1e-2) < 4e-3 and rises
    return ok


def run(cache):
    import h5py
    from wp4.scoring import score_scene, summarize
    from wp4.segmentation import segment
    from wp4.spectrum import spectrogram
    out = {}
    dets = [CovIndex(k) for k in KINDS]
    for d in dets:
        d.calibrate([1e-3], cache)
    for ds, path in [("impaired", r"D:\data\wideband\scenes.h5"), ("clean", r"D:\data\wideband\scenes_clean.h5")]:
        f = h5py.File(path, "r"); B = f["boxes"][:]
        sc = {d.name: [] for d in dets}
        for i in range(100, f["iq"].shape[0]):
            iq = f["iq"][i]; s = spectrogram(iq)
            for d in dets:
                m = d.masks(s, iq, [1e-3])[1e-3]
                sc[d.name].append(score_scene(m, segment(m), B[B["scene"] == i], s.shape, 1e7))
        for d in dets:
            r = summarize(sc[d.name])
            out[f"{d.name}_{ds}"] = {k: r[k] for k in ("box_f1", "box_precision", "box_recall", "emitter_pd", "cell_pfa")}
            print(f"{ds:>8s} {d.name:>9s}: box F1 {r['box_f1'] * 100:5.1f}  emitter Pd {r['emitter_pd'] * 100:5.1f}  Pfa {r['cell_pfa']:.1e}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "cov_index_detectors.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cache = os.path.join(HERE, "results", "calibration_cov_index")
    if a.check:
        raise SystemExit(0 if check(cache) else 1)
    run(cache)
