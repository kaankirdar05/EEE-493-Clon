"""WP4 detection benchmark: detector -> segmentation (4.3) -> shared scoring.

    python run_wp4.py --data D:\\data\\wideband\\scenes.h5                       # energy detector (4.1)
    python run_wp4.py --data D:\\data\\wideband\\scenes.h5 --detector cfar_os
    python compare_wp4.py                                                      # table + overlay plots

Built-in detectors are in wp4/detectors.py; each one's thresholds are calibrated
on pure white noise (results/calibration/), so "Pfa 1e-3" means the same thing
for all of them.

Adding another detector (e.g. Kagan's CA-CFAR, job 4.2): write a function

    def my_detector(spec, pfa, **kw) -> (mask, info)
        spec : (T, F) power spectrogram from wp4.spectrum.spectrogram
        mask : (T, F) bool, True = signal present
        info : dict of diagnostics (may be empty)

and add `"my_name": FunctionDetector(my_detector)` to DETECTORS below. Its own
thresholds are used as they are (no white-noise calibration). Everything after
the mask (boxes, Pd/Pfa, IoU, plots) is shared, so the numbers are comparable.

Outputs in --out (default results/<detector>):
    metrics.json       all metrics at --pfa, plus the Pfa sweep
    pd_vs_snr.csv/png  emitter Pd, box recall and IoU per SNR bin
    roc.png            emitter Pd vs measured Pfa over the sweep, and Pfa calibration
    example_*.png      spectrograms with truth (green) and detected (red) boxes
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np

from wp4.detectors import all_detectors
from wp4.scoring import score_scene, summarize
from wp4.segmentation import segment
from wp4.spectrum import box_to_cells, spectrogram


class FunctionDetector:
    """Wrap a plain `f(spec, pfa) -> (mask, info)` detector (e.g. Kagan's CA-CFAR)."""
    coarse = False

    def __init__(self, fn):
        self.fn = fn

    def calibrate(self, pfas, cache_dir):
        pass

    def masks(self, spec, iq, pfas):
        return {p: self.fn(spec, pfa=p)[0] for p in pfas}


DETECTORS = {
    **all_detectors(),
    # "kagan_ca_cfar": FunctionDetector(kagan_ca_cfar),
}


def plot_example(spec, truth_cells, pred_boxes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(10 * np.log10(spec + 1e-12), aspect="auto", origin="lower", cmap="viridis")
    for t0, t1, f0, f1 in truth_cells:
        ax.add_patch(Rectangle((f0 - 0.5, t0 - 0.5), f1 - f0, t1 - t0, fill=False, ec="lime", lw=1.5))
    for t0, t1, f0, f1 in pred_boxes:
        ax.add_patch(Rectangle((f0 - 0.5, t0 - 0.5), f1 - f0, t1 - t0, fill=False, ec="red", lw=1.2, ls="--"))
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time frame")
    ax.set_title(title + "   (green = truth, red = detected)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_curves(summary, sweep, path_snr, path_roc, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = summary["per_snr"]
    x = [(p["snr_lo"] + p["snr_hi"]) / 2 for p in ps]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(x, [p["emitter_pd"] * 100 for p in ps], marker="o", label="Emitter Pd (>= 50 % of cells)")
    ax.plot(x, [p["box_recall"] * 100 for p in ps], marker="s", label="Box recall (IoU >= 0.5)")
    ax.plot(x, [p["mean_iou"] * 100 for p in ps], marker="^", label="Mean IoU x 100")
    ax.set_xlabel("Emitter SNR (dB, bin centre)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{name}: detection vs SNR (Pfa target {summary['pfa_target']:g})")
    fig.tight_layout()
    fig.savefig(path_snr, dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    tgt = [s["pfa_target"] for s in sweep]
    meas = [max(s["cell_pfa"], 1e-7) for s in sweep]
    axs[0].semilogx(meas, [s["emitter_pd"] * 100 for s in sweep], marker="o", label="emitter Pd")
    axs[0].semilogx(meas, [s["box_recall"] * 100 for s in sweep], marker="s", label="box recall")
    axs[0].set_xlabel("Measured cell Pfa")
    axs[0].set_ylabel("%")
    axs[0].set_ylim(0, 100)
    axs[0].grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Operating curve (all SNRs)")
    axs[1].loglog(tgt, meas, marker="o", label="measured")
    axs[1].loglog(tgt, tgt, "k--", lw=1, label="ideal")
    axs[1].set_xlabel("Target Pfa")
    axs[1].set_ylabel("Measured cell Pfa")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8)
    axs[1].set_title("False-alarm calibration")
    fig.tight_layout()
    fig.savefig(path_roc, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--detector", choices=sorted(DETECTORS), default="energy")
    p.add_argument("--pfa", type=float, default=1e-3, help="operating point for the main metrics")
    p.add_argument("--sweep", default="1e-1,1e-2,1e-3,1e-4,1e-5", help="Pfa targets for the curves")
    p.add_argument("--min-cells", type=int, default=120, help="segmentation: smallest box kept")
    p.add_argument("--start", type=int, default=100, help="first scene (0-99 were used to tune segmentation)")
    p.add_argument("--scenes", type=int, default=None, help="use only the first N scenes")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    args.out = args.out or os.path.join("results", args.detector)
    os.makedirs(args.out, exist_ok=True)
    det = DETECTORS[args.detector]
    pfas = sorted({float(v) for v in args.sweep.split(",")} | {args.pfa}, reverse=True)
    det.calibrate(pfas, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "calibration"))

    with h5py.File(args.data, "r") as f:
        fs = float(f.attrs["sample_rate"])
        fft = int(f.attrs["fft_size"])
        last = f["iq"].shape[0] if args.scenes is None else min(args.start + args.scenes, f["iq"].shape[0])
        scene_ids = range(args.start, last)
        n_scenes = len(scene_ids)
        all_boxes = f["boxes"][:]
        per_pfa = {pf: [] for pf in pfas}
        t_det = t_seg = 0.0
        for k, i in enumerate(scene_ids):
            iq = f["iq"][i]
            spec = spectrogram(iq, fft)
            truth = all_boxes[all_boxes["scene"] == i]
            t0 = time.perf_counter()
            masks = det.masks(spec, iq, pfas)  # statistic computed once, thresholded per Pfa
            t_det += time.perf_counter() - t0
            for pf in pfas:
                mask = masks[pf]
                t1 = time.perf_counter()
                boxes = segment(mask, min_cells=args.min_cells)
                t2 = time.perf_counter()
                if pf == args.pfa:
                    t_seg += t2 - t1
                    if k < args.examples:
                        cells = [box_to_cells(b, spec.shape, fs) for b in truth]
                        plot_example(spec, cells, boxes, os.path.join(args.out, f"example_{i}.png"),
                                     f"Scene {i}: {len(truth)} emitters")
                per_pfa[pf].append(score_scene(mask, boxes, truth, spec.shape, fs))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{n_scenes} scenes", flush=True)

    sweep = []
    for pf in pfas:
        s = summarize(per_pfa[pf])
        s["pfa_target"] = pf
        sweep.append(s)
    main_s = next(s for s in sweep if s["pfa_target"] == args.pfa)
    main_s["timing_ms_per_scene"] = {"detector": t_det / n_scenes * 1e3, "segmentation": t_seg / n_scenes * 1e3}
    main_s["settings"] = {k: v for k, v in vars(args).items()}

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"main": main_s, "sweep": [{k: v for k, v in s.items() if k != "per_snr"} for s in sweep]},
                  fh, indent=2)
    with open(os.path.join(args.out, "pd_vs_snr.csv"), "w", encoding="utf-8") as fh:
        fh.write("snr_lo,snr_hi,n,emitter_pd,box_recall,mean_iou\n")
        fh.writelines(f"{q['snr_lo']},{q['snr_hi']},{q['n']},{q['emitter_pd']:.4f},{q['box_recall']:.4f},{q['mean_iou']:.4f}\n"
                      for q in main_s["per_snr"])
    plot_curves(main_s, sweep, os.path.join(args.out, "pd_vs_snr.png"), os.path.join(args.out, "roc.png"), args.detector)

    print(f"\n{args.detector} detector, {n_scenes} scenes, {main_s['n_emitters']} emitters, Pfa target {args.pfa:g}")
    print(f"  measured cell Pfa {main_s['cell_pfa']:.2e}   cell Pd {main_s['cell_pd'] * 100:.1f} %   emitter Pd {main_s['emitter_pd'] * 100:.1f} %")
    print(f"  boxes: precision {main_s['box_precision'] * 100:.1f} %  recall {main_s['box_recall'] * 100:.1f} %  "
          f"F1 {main_s['box_f1'] * 100:.1f} %  mean IoU (matched) {main_s['mean_iou_matched']:.2f}")
    print("  per SNR (emitter Pd / box recall): " + "  ".join(
        f"[{q['snr_lo']},{q['snr_hi']}) {q['emitter_pd'] * 100:.0f}/{q['box_recall'] * 100:.0f}" for q in main_s["per_snr"]))
    print("  sweep (target -> measured Pfa, emitter Pd): " + "  ".join(
        f"{s['pfa_target']:g}->{s['cell_pfa']:.1e},{s['emitter_pd'] * 100:.0f}%" for s in sweep))
    print(f"  timing: detector {main_s['timing_ms_per_scene']['detector']:.1f} ms/scene, "
          f"segmentation {main_s['timing_ms_per_scene']['segmentation']:.1f} ms/scene")
    print(f"Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
