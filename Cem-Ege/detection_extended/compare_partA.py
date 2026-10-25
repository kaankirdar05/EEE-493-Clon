"""Collect Part A held-out results (scenes 100-299) next to the WP4 reference rows.

Writes results/partA_comparison.md and results/partA_comparison.json, plus
results/partA_recall_vs_snr_<dataset>.png.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WP4RES = os.path.join(os.path.dirname(HERE), "wp4_detection", "results")
REF = ["energy", "cfar_so", "cfar_os", "eigen_mme", "cyclo", "gof_ad"]
NEW = ["cav_L4", "acorr_L32", "entropy_spec", "entropy_hist", "scd_fam", "scd_fam_Np16", "cav_L4_pw", "entropy_spec_pw"]


def recall_at(per_snr, lo):
    return next((q["box_recall"] for q in per_snr if q["snr_lo"] == lo), float("nan"))


def row(name, m, ref):
    s = m["main"]
    return {"detector": name, "reference": ref, "box_f1": s["box_f1"] * 100, "precision": s["box_precision"] * 100,
            "recall": s["box_recall"] * 100, "recall_0_5": recall_at(s["per_snr"], 0) * 100,
            "recall_10_15": recall_at(s["per_snr"], 10) * 100, "emitter_pd": s["emitter_pd"] * 100,
            "mean_iou": s["mean_iou_matched"], "measured_pfa": s["cell_pfa"],
            "ms_per_scene": s["timing_ms_per_scene"]["detector"], "per_snr": s["per_snr"]}


def main():
    tp = os.path.join(RES, "timing_single_process.json")
    single = json.load(open(tp)) if os.path.exists(tp) else {}
    out = {}
    lines = ["# Part A - additional block detectors vs WP4 reference (scenes 100-299, Pfa target 1e-3)\n",
             "Reference rows (italic) are read from wp4_detection/results. All block detectors decide per "
             "16-frame x 32-bin block. ms/scene for the new rows is single-process time of stat() on 5 scenes "
             "(time_partA.py) on a machine shared with other jobs: indicative upper bounds only.\n"]
    for ds in ("impaired", "clean"):
        rows = []
        for d in REF:
            p = os.path.join(WP4RES, f"{d}_{ds}", "metrics.json")
            if os.path.exists(p):
                rows.append(row(d, json.load(open(p, encoding="utf-8")), True))
        for d in NEW:
            p = os.path.join(RES, f"{d}_{ds}", "metrics.json")
            if os.path.exists(p):
                rows.append(row(d, json.load(open(p, encoding="utf-8")), False))
                if d in single:
                    rows[-1]["ms_per_scene"] = single[d]
        out[ds] = rows
        lines += [f"\n## {ds}\n",
                  "| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            nm = f"*{r['detector']}*" if r["reference"] else f"**{r['detector']}**"
            lines.append(f"| {nm} | {r['box_f1']:.1f} | {r['precision']:.1f} | {r['recall']:.1f} | {r['recall_0_5']:.0f} | "
                         f"{r['recall_10_15']:.0f} | {r['emitter_pd']:.1f} | {r['mean_iou']:.2f} | {r['measured_pfa']:.1e} | "
                         f"{r['ms_per_scene']:.0f} |")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for r in rows:
            ps = r["per_snr"]
            ax.plot([(q["snr_lo"] + q["snr_hi"]) / 2 for q in ps], [q["box_recall"] * 100 for q in ps], marker="o", ms=3,
                    ls=":" if r["reference"] else "-", label=r["detector"] + (" (ref)" if r["reference"] else ""))
        ax.set_xlabel("Emitter SNR (dB, bin centre)")
        ax.set_ylabel("Box recall, IoU >= 0.5 (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"Part A block detectors vs reference, {ds}, Pfa target 1e-3")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"partA_recall_vs_snr_{ds}.png"), dpi=140)
        plt.close(fig)
    with open(os.path.join(RES, "partA_comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "partA_comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
