"""Part A runner: calibrate, verify Pfa on fresh white noise, run on scenes.

Same loop and outputs as wp4_detection/run_wp4.py (segmentation, scoring and
plots are imported from there, unchanged), parallelised over scenes.

    python run_partA.py calibrate  --det cav [--param L=16]
    python run_partA.py verify     --det cav
    python run_partA.py run        --det cav --data D:\\data\\wideband\\scenes.h5 --out results\\cav_impaired
    python run_partA.py run ... --start 0 --scenes 100      # dev tuning (scenes 0-99)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ext import WP4  # noqa: E402
from ext.block_detectors import CAV, AutoCorr, HistEntropy, SCDFam, SpectralEntropy, Whitened  # noqa: E402

sys.path.insert(0, WP4)
from run_wp4 import plot_curves, plot_example  # noqa: E402
from wp4.scoring import score_scene, summarize  # noqa: E402
from wp4.segmentation import segment  # noqa: E402
from wp4.spectrum import box_to_cells, spectrogram  # noqa: E402

CAL_DIR = os.path.join(HERE, "results", "calibration")
FACTORY = {"cav": CAV, "acorr": AutoCorr, "entropy_spec": SpectralEntropy, "entropy_hist": HistEntropy,
           "scd_fam": SCDFam}
FACTORY.update({k + "_pw": (lambda c: (lambda **kw: Whitened(c(**kw))))(c) for k, c in list(FACTORY.items())})
PFAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def make(det: str, params: list[str]):
    kw = {}
    for p in params or []:
        k, v = p.split("=")
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return FACTORY[det](**kw)


# ------------------------------------------------------------------ worker state
_W = {}


def _init(det_name, params, thr, data, fft):
    d = make(det_name, params)
    d.thr = thr
    _W.update(det=d, f=h5py.File(data, "r"), fft=fft)


def _scene(args):
    i, truth, fs, pfas, want_example = args
    det = _W["det"]
    iq = _W["f"]["iq"][i]
    spec = spectrogram(iq, _W["fft"])
    t0 = time.perf_counter()
    masks = det.masks(spec, iq, pfas)
    t_det = time.perf_counter() - t0
    res, ex = {}, None
    for pf in pfas:
        boxes = segment(masks[pf])
        res[pf] = score_scene(masks[pf], boxes, truth, spec.shape, fs)
        if want_example and pf == 1e-3:
            ex = (spec, boxes)
    return i, res, t_det, ex


def cmd_run(a):
    det = make(a.det, a.param)
    det.calibrate(PFAS, CAL_DIR)
    out = a.out
    os.makedirs(out, exist_ok=True)
    with h5py.File(a.data, "r") as f:
        fs, fft = float(f.attrs["sample_rate"]), int(f.attrs["fft_size"])
        n = f["iq"].shape[0]
        all_boxes = f["boxes"][:]
    last = n if a.scenes is None else min(a.start + a.scenes, n)
    ids = list(range(a.start, last))
    jobs = [(i, all_boxes[all_boxes["scene"] == i], fs, PFAS, k < a.examples) for k, i in enumerate(ids)]
    per = {p: {} for p in PFAS}
    t_det = []
    with Pool(a.workers, initializer=_init, initargs=(a.det, a.param, det.thr, a.data, fft)) as pool:
        for i, res, td, ex in pool.imap_unordered(_scene, jobs):
            for p in PFAS:
                per[p][i] = res[p]
            t_det.append(td)
            if ex is not None:
                truth = all_boxes[all_boxes["scene"] == i]
                cells = [box_to_cells(b, ex[0].shape, fs) for b in truth]
                plot_example(ex[0], cells, ex[1], os.path.join(out, f"example_{i}.png"),
                             f"Scene {i}: {len(truth)} emitters ({det.name})")
    sweep = []
    for p in sorted(PFAS, reverse=True):
        s = summarize([per[p][i] for i in ids])
        s["pfa_target"] = p
        sweep.append(s)
    main_s = next(s for s in sweep if s["pfa_target"] == 1e-3)
    # single-process-equivalent time per scene (workers run 1 BLAS thread each)
    main_s["timing_ms_per_scene"] = {"detector": float(np.mean(t_det) * 1e3), "segmentation": float("nan")}
    main_s["settings"] = {"detector": det.name, "params": a.param, "data": a.data, "start": a.start,
                          "n_scenes": len(ids), "thr": det.thr}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"main": main_s, "sweep": [{k: v for k, v in s.items() if k != "per_snr"} for s in sweep]},
                  fh, indent=2)
    with open(os.path.join(out, "pd_vs_snr.csv"), "w", encoding="utf-8") as fh:
        fh.write("snr_lo,snr_hi,n,emitter_pd,box_recall,mean_iou\n")
        fh.writelines(f"{q['snr_lo']},{q['snr_hi']},{q['n']},{q['emitter_pd']:.4f},{q['box_recall']:.4f},"
                      f"{q['mean_iou']:.4f}\n" for q in main_s["per_snr"])
    plot_curves(main_s, sweep, os.path.join(out, "pd_vs_snr.png"), os.path.join(out, "roc.png"), det.name)
    print(f"{det.name} {os.path.basename(a.data)} scenes {ids[0]}-{ids[-1]}: F1 {main_s['box_f1'] * 100:.1f} "
          f"P {main_s['box_precision'] * 100:.1f} R {main_s['box_recall'] * 100:.1f} Pd {main_s['emitter_pd'] * 100:.1f} "
          f"Pfa {main_s['cell_pfa']:.1e} {main_s['timing_ms_per_scene']['detector']:.0f} ms/scene", flush=True)


def cmd_verify(a):
    """Measured Pfa on FRESH white noise (seed differs from calibration seed 12345)."""
    det = make(a.det, a.param)
    det.calibrate(PFAS, CAL_DIR)
    rng = np.random.default_rng(20261015)
    t0 = time.perf_counter()
    h0 = det.h0_samples(rng)                       # fresh block-level samples (same size as calibration)
    blk = {p: float(np.mean(h0 > det.thr[p])) for p in PFAS[:4]}
    # full pipeline (stat(spec, iq) on whole noise scenes) - checks the grid mapping too
    cells = []
    for _ in range(a.noise_scenes):
        iq = ((rng.standard_normal(512 * 512) + 1j * rng.standard_normal(512 * 512)) / np.sqrt(2)).astype(np.complex64)
        cells.append(det.stat(spectrogram(iq, 512), iq))
    cells = np.stack(cells)
    full = {p: float(np.mean(cells > det.thr[p])) for p in PFAS[:4]}
    r = {"detector": det.name, "n_fresh_blocks": int(h0.size), "n_noise_scenes": a.noise_scenes,
         "blocks_per_scene": 512, "pfa_target": PFAS[:4], "measured_block": blk, "measured_full_pipeline": full,
         "seconds": time.perf_counter() - t0}
    os.makedirs(os.path.join(HERE, "results", "pfa_check"), exist_ok=True)
    with open(os.path.join(HERE, "results", "pfa_check", f"{det.name}.json"), "w") as fh:
        json.dump(r, fh, indent=2)
    print(det.name, "block:", {p: f"{v:.2e}" for p, v in blk.items()}, "full:", {p: f"{v:.2e}" for p, v in full.items()},
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["calibrate", "verify", "run"])
    p.add_argument("--det", required=True, choices=sorted(FACTORY))
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--data", default=r"D:\data\wideband\scenes.h5")
    p.add_argument("--out", default=None)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--scenes", type=int, default=None)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--noise-scenes", type=int, default=40)
    a = p.parse_args()
    if a.cmd == "calibrate":
        d = make(a.det, a.param)
        t = time.perf_counter()
        d.calibrate(PFAS, CAL_DIR)
        print(d.name, {k: round(v, 5) for k, v in d.thr.items()}, f"{time.perf_counter() - t:.0f}s")
    elif a.cmd == "verify":
        cmd_verify(a)
    else:
        a.out = a.out or os.path.join(HERE, "results", f"{make(a.det, a.param).name}_tmp")
        cmd_run(a)


if __name__ == "__main__":
    main()
