"""Held-out comparison table (scenes 100-299, Pfa target 1e-3) next to the reference
rows, plus the fresh-noise Pfa check table. Writes results/heldout_comparison.md/.json
and results/heldout_recall_vs_snr_<dataset>.png."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.dirname(HERE)
REF = [("energy", os.path.join(ROOT, "wp4_detection", "results")),
       ("eigen_mme", os.path.join(ROOT, "wp4_detection", "results")),
       ("cav_L4", os.path.join(ROOT, "detection_extended", "results")),
       ("scd_fam_Np16", os.path.join(ROOT, "detection_extended", "results"))]
SKIP = {"calibration", "dev_tuning", "sanity", "sr_paper_setting", "synthetic", "pfa_check"}


def recall_at(per_snr, lo):
    return next((q["box_recall"] for q in per_snr if q["snr_lo"] == lo), float("nan"))


def row(name, m, ref):
    s = m["main"]
    return {"detector": name, "reference": ref, "box_f1": s["box_f1"] * 100, "precision": s["box_precision"] * 100,
            "recall": s["box_recall"] * 100, "recall_0_5": recall_at(s["per_snr"], 0) * 100,
            "recall_10_15": recall_at(s["per_snr"], 10) * 100, "emitter_pd": s["emitter_pd"] * 100,
            "measured_pfa": s["cell_pfa"], "ms_per_scene": s["timing_ms_per_scene"]["detector"],
            "per_snr": s["per_snr"]}


def main():
    out, lines = {}, ["# Held-out results, scenes 100-299, Pfa target 1e-3\n",
                      "Reference rows in italics (read from wp4_detection/results and detection_extended/results). "
                      "ms/scene = detector stat() time inside a pool worker (1 BLAS thread) on a machine shared with "
                      "other jobs: indicative only.\n"]
    new = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(os.path.join(RES, "*_*"))
                  if os.path.isdir(p) and os.path.basename(p) not in SKIP
                  and os.path.exists(os.path.join(p, "metrics.json"))})
    for ds in ("impaired", "clean"):
        rows = []
        for d, base in REF:
            p = os.path.join(base, f"{d}_{ds}", "metrics.json")
            if os.path.exists(p):
                rows.append(row(d, json.load(open(p, encoding="utf-8")), True))
        for d in new:
            p = os.path.join(RES, f"{d}_{ds}", "metrics.json")
            if os.path.exists(p):
                rows.append(row(d, json.load(open(p, encoding="utf-8")), False))
        out[ds] = rows
        lines += [f"\n## {ds}\n",
                  "| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | measured Pfa | ms/scene |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            nm = f"*{r['detector']} (ref)*" if r["reference"] else f"**{r['detector']}**"
            lines.append(f"| {nm} | {r['box_f1']:.1f} | {r['precision']:.1f} | {r['recall']:.1f} | {r['recall_0_5']:.0f} | "
                         f"{r['recall_10_15']:.0f} | {r['emitter_pd']:.1f} | {r['measured_pfa']:.1e} | {r['ms_per_scene']:.0f} |")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for r in rows:
            ps = r["per_snr"]
            ax.plot([(q["snr_lo"] + q["snr_hi"]) / 2 for q in ps], [q["box_recall"] * 100 for q in ps], marker="o", ms=3,
                    ls=":" if r["reference"] else "-", lw=2 if r["detector"] == "energy" else 1.2,
                    label=r["detector"] + (" (ref)" if r["reference"] else ""))
        ax.set_xlabel("Emitter SNR (dB, bin centre)")
        ax.set_ylabel("Box recall, IoU >= 0.5 (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"Discovered detectors vs reference, {ds}, Pfa target 1e-3")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"heldout_recall_vs_snr_{ds}.png"), dpi=140)
        plt.close(fig)
    # fresh-noise Pfa check
    pf = sorted(glob.glob(os.path.join(RES, "pfa_check", "*.json")))
    if pf:
        lines += ["\n## Pfa on fresh white noise (seed differs from calibration)\n",
                  "Block level (fresh h0 sample, same size as calibration) / full stat(spec, iq) pipeline on 40 noise scenes.\n",
                  "| detector | 1e-1 | 1e-2 | 1e-3 | 1e-4 |", "|---|---|---|---|---|"]
        for p in pf:
            r = json.load(open(p))
            b, f = r["measured_block"], r["measured_full_pipeline"]
            cells = [f"{b[k]:.1e} / {f[k]:.1e}" for k in sorted(b, key=lambda k: -float(k))]
            lines.append(f"| {r['detector']} | " + " | ".join(cells) + " |")
    open(os.path.join(RES, "heldout_comparison.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(RES, "heldout_comparison.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
