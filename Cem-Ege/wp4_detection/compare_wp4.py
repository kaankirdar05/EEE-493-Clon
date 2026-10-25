"""Collect results/<detector>_<dataset>/metrics.json into one comparison.

Writes results/comparison.md (tables) and, per dataset, overlay plots:
    compare_recall_vs_snr_<dataset>.png   box recall vs SNR at Pfa 1e-3, one line per detector
    compare_operating_<dataset>.png       emitter Pd and box F1 vs MEASURED cell Pfa (fair comparison)
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["energy", "energy_mt", "fcme", "lad", "psd_occupancy", "cfar_ca", "cfar_go", "cfar_so", "cfar_os", "cfar_2d",
         "spectral_kurtosis", "eigen_mme", "cyclo", "gof_ad"]
COARSE = {"eigen_mme", "cyclo", "gof_ad", "psd_occupancy"}
MARKERS = "osD^vP*Xh<>p8d"


def style(det):
    i = ORDER.index(det) if det in ORDER else len(ORDER)
    return {"color": plt.cm.tab20(i % 20), "marker": MARKERS[i % len(MARKERS)], "ls": "--" if det in COARSE else "-"}


def load(dataset):
    out = {}
    for path in glob.glob(os.path.join("results", f"*_{dataset}", "metrics.json")):
        det = os.path.basename(os.path.dirname(path))[: -len(dataset) - 1]
        with open(path, encoding="utf-8") as f:
            out[det] = json.load(f)
    return dict(sorted(out.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def recall_at(per_snr, lo):
    for q in per_snr:
        if q["snr_lo"] == lo:
            return q["box_recall"]
    return float("nan")


def main():
    lines = ["# WP4 detector comparison (scenes 100-299, Pfa target 1e-3 unless noted)\n",
             "Thresholds are calibrated on white noise, so on pure noise every detector has the target Pfa. "
             "On scenes the measured Pfa is higher because of unlabelled real energy (see README). "
             "eigen_mme and cyclo decide per 16-frame x 32-bin block, so their boxes are coarse.\n"]
    for ds in ("impaired", "clean"):
        res = load(ds)
        if not res:
            continue
        lines += [f"\n## {ds}\n",
                  "| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for det, m in res.items():
            s = m["main"]
            lines.append(
                f"| {det} | {s['box_f1'] * 100:.1f} | {s['box_precision'] * 100:.1f} | {s['box_recall'] * 100:.1f} | "
                f"{recall_at(s['per_snr'], 0) * 100:.0f} | {recall_at(s['per_snr'], 10) * 100:.0f} | "
                f"{s['emitter_pd'] * 100:.1f} | {s['mean_iou_matched']:.2f} | {s['cell_pfa']:.1e} | "
                f"{s['timing_ms_per_scene']['detector']:.0f} |")

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for det, m in res.items():
            ps = m["main"]["per_snr"]
            ax.plot([(q["snr_lo"] + q["snr_hi"]) / 2 for q in ps], [q["box_recall"] * 100 for q in ps],
                    label=det, **style(det))
        ax.set_xlabel("Emitter SNR (dB, bin centre)")
        ax.set_ylabel("Box recall, IoU >= 0.5 (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"WP4 detectors, {ds} scenes, Pfa target 1e-3")
        fig.tight_layout()
        fig.savefig(os.path.join("results", f"compare_recall_vs_snr_{ds}.png"), dpi=150)
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for det, m in res.items():
            sw = sorted(m["sweep"], key=lambda s: s["cell_pfa"])
            axs[0].semilogx([max(s["cell_pfa"], 1e-6) for s in sw], [s["emitter_pd"] * 100 for s in sw], label=det, **style(det))
            axs[1].semilogx([max(s["cell_pfa"], 1e-6) for s in sw], [s["box_f1"] * 100 for s in sw], label=det, **style(det))
        for a, yl in zip(axs, ["Emitter Pd (%)", "Box F1 (%)"]):
            a.set_xlabel("Measured cell Pfa on scenes")
            a.set_ylabel(yl)
            a.set_ylim(0, 100)
            a.grid(alpha=0.3, which="both")
        axs[1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Operating curves, {ds} scenes (Pfa sweep 1e-1 ... 1e-5)")
        fig.tight_layout()
        fig.savefig(os.path.join("results", f"compare_operating_{ds}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join("results", "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
