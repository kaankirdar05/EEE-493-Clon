"""Figures and compact CSV tables from the results/*.json files (run after the run_*.py scripts).

    python make_figures.py        -> results/figures/*.png and results/tables/*.csv
"""
from __future__ import annotations

import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ds1common as dc

R = dc.RESULTS
FIG = os.path.join(R, "figures")
TAB = os.path.join(R, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
SIGNAL = ["AM", "NBFM", "WBFM", "OOK", "2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "PI4DQPSK", "8PSK", "16QAM", "OFDM"]
SNRS = list(range(-10, 21, 2))


def load(path):
    p = os.path.join(R, path)
    return json.load(open(p)) if os.path.exists(p) else None


def heatmap(M, rows, cols, title, path, fmt="{:.0f}", scale=100, vmin=0, vmax=1, cmap="Blues", xlabel="SNR (dB, Z label)"):
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(cols), 0.9 + 0.38 * len(rows)))
    im = ax.imshow(np.array(M, float), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), [str(c) for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i][j]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.text(j, i, fmt.format(v * scale), ha="center", va="center", fontsize=6.5,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "black")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(TAB, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def identification():
    d = load("identification/classifiers.json")
    if not d:
        return
    classes = d["classes"]
    res = {r["method"]: r for r in d["results"]}
    # accuracy vs SNR, all models
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in res.items():
        s = sorted(int(k) for k in r["accuracy_vs_snr"])
        ax.plot(s, [r["accuracy_vs_snr"][str(k)] * 100 for k in s], marker="o", ms=3,
                ls=":" if name == "cumulant_theory" else "-", label=name)
    ax.axhline(85, color="gray", ls="--", lw=1, label="85 % target")
    ax.set(xlabel="SNR (dB, Z label)", ylabel="accuracy (%)", ylim=(0, 100),
           title="Dataset 1 v3, 14 classes: classical classifiers on 19 features (test split)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "id_acc_vs_snr_all_models.png"), dpi=140); plt.close(fig)
    write_csv("id_classifier_comparison.csv",
              ["model", "acc_all_clips", "acc_signals_all_snr", "acc_snr_ge_10", "acc_snr_le_0", "noise_correct",
               "sustained_85_from_db", "fit_s", "predict_ms_per_clip", "n_train", "note"],
              [[k, f"{r['accuracy_all_clips']:.4f}", f"{r['accuracy_signals_all_snr']:.4f}", f"{r['accuracy_snr_ge_10']:.4f}",
                f"{r['accuracy_snr_le_0']:.4f}", "" if r["noise_clips_correct"] is None else f"{r['noise_clips_correct']:.4f}",
                r["sustained_snr_ge_85"], f"{r['fit_seconds']:.1f}", f"{r['predict_ms_per_clip']:.4f}", r["n_train"], r["note"]]
               for k, r in res.items()])
    write_csv("id_acc_vs_snr.csv", ["model", "snr_db", "accuracy"],
              [[k, s, f"{v:.4f}"] for k, r in res.items() for s, v in sorted((int(a), b) for a, b in r["accuracy_vs_snr"].items())])
    for name in ("hgb", "rf"):
        if name not in res:
            continue
        r = res[name]
        M = [[r["per_class_vs_snr"][c].get(str(s)) for s in SNRS] for c in SIGNAL]
        heatmap(M, SIGNAL, SNRS, f"{name}: accuracy (%) per class and SNR, test split", os.path.join(FIG, f"id_{name}_class_snr.png"))
        write_csv(f"id_{name}_class_snr.csv", ["class"] + [str(s) for s in SNRS],
                  [[c] + [f"{r['per_class_vs_snr'][c][str(s)]:.4f}" for s in SNRS] for c in SIGNAL])
        for tag in ("snr_ge_10", "all"):
            cm = np.array(d["confusion"][name][tag], float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            heatmap(cmn.tolist(), classes, classes, f"{name}: confusion matrix ({'Z >= 10 dB and NOISE' if tag != 'all' else 'all SNRs'}), row-normalised %",
                    os.path.join(FIG, f"id_{name}_confusion_{tag}.png"), xlabel="predicted class")
            write_csv(f"id_{name}_confusion_{tag}.csv", ["true \\ predicted"] + classes,
                      [[classes[i]] + [int(v) for v in cm[i]] for i in range(len(classes))])


def extended_and_rules():
    curves = {}
    d = load("identification/extended.json")
    if d:
        for r in d["results"]:
            curves["HGB: " + r["method"]] = r
            if r["method"] == "19 features + 24 Haar wavelet":
                M = [[r["per_class_vs_snr"][c].get(str(s)) for s in SNRS] for c in SIGNAL]
                heatmap(M, SIGNAL, SNRS, "HGB on 19 + 24 Haar wavelet features: accuracy (%) per class and SNR",
                        os.path.join(FIG, "id_hgb_wavelet_class_snr.png"))
                write_csv("id_hgb_wavelet_class_snr.csv", ["class"] + [str(s) for s in SNRS],
                          [[c] + [f"{r['per_class_vs_snr'][c][str(s)]:.4f}" for s in SNRS] for c in SIGNAL])
                cm = np.array(d["confusion"][r["method"]]["snr_ge_10"], float)
                heatmap((cm / np.maximum(cm.sum(1, keepdims=True), 1)).tolist(), list(d["classes"]), list(d["classes"]),
                        "HGB on 19 + 24 wavelet features: confusion (Z >= 10 dB and NOISE), row-normalised %",
                        os.path.join(FIG, "id_hgb_wavelet_confusion_snr_ge_10.png"), xlabel="predicted class")
    d = load("identification/rules.json")
    if d:
        for r in d.values():
            curves[r["method"]] = r
    d = load("identification/likelihood.json")
    if d:
        for k, r in d["results"].items():
            curves[k] = {"accuracy_vs_snr": r["accuracy_vs_snr"], "method": k}
    if not curves:
        return
    rows = []
    for k, r in curves.items():
        for s, v in r["accuracy_vs_snr"].items():
            rows.append([k, int(s), f"{v:.4f}"])
    write_csv("id_extra_methods_acc_vs_snr.csv", ["method", "snr_db", "accuracy"], rows)
    lk = load("identification/likelihood.json")
    if lk:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, r in lk["results"].items():
            if k.startswith("HLRT"):
                continue
            s = sorted(int(a) for a in r["accuracy_vs_snr"])
            ax.plot(s, [r["accuracy_vs_snr"][str(a)] * 100 for a in s], marker="o", ms=3, label=k)
        ax.axhline(85, color="gray", ls="--", lw=1)
        ax.set(xlabel="SNR (dB, Es/N0)", ylabel="accuracy (%)", ylim=(0, 100),
               title="BPSK / QPSK / 8PSK / 16QAM: likelihood classifier vs features")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "id_likelihood_vs_features.png"), dpi=140); plt.close(fig)


def detection():
    d = load("detection/detection.json")
    if not d:
        return
    rows = []
    for name, r in d["detectors"].items():
        for pfa, p in r["pfa"].items():
            rows.append([name, pfa, f"{p['measured_pfa_on_test_noise']:.4f}", f"{p['pd_all_signals']:.4f}",
                         p["snr_pd90_sustained_all"]] + [p["snr_pd90_sustained"][c] for c in SIGNAL] + [f"{r['ms_per_clip']:.3f}"])
    write_csv("det_summary.csv", ["detector", "target_pfa", "measured_pfa_test_noise", "pd_all_signals",
                                  "pd90_from_db_all"] + [f"pd90_from_db_{c}" for c in SIGNAL] + ["ms_per_clip"], rows)
    best = max(d["detectors"], key=lambda k: d["detectors"][k]["pfa"]["0.01"]["pd_all_signals"])
    for name in sorted({best, "energy", "cav", "eigen_mme", "cyclo"} & set(d["detectors"])):
        p = d["detectors"][name]["pfa"]["0.01"]
        M = [[p["pd_class_snr"][c][str(s)] for s in SNRS] for c in SIGNAL]
        heatmap(M, SIGNAL, SNRS, f"{name}: Pd (%) at Pfa 1e-2 per class and SNR", os.path.join(FIG, f"det_{name}_class_snr.png"))
        write_csv(f"det_{name}_pfa1e-2_class_snr.csv", ["class"] + [str(s) for s in SNRS],
                  [[c] + [f"{p['pd_class_snr'][c][str(s)]:.4f}" for s in SNRS] for c in SIGNAL])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 20))[:len(d["detectors"])])
    for name, r in d["detectors"].items():
        p = r["pfa"]["0.01"]["pd_vs_snr_all_classes"]
        s = sorted(int(k) for k in p)
        ax.plot(s, [p[str(k)] * 100 for k in s], marker="o", ms=3, label=name)
    ax.set(xlabel="SNR (dB, Z label)", ylabel="Pd (%), all 13 signal classes", ylim=(0, 100),
           title="Detection on Dataset 1 v3 at Pfa 1e-2 (test split)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "det_pd_vs_snr_all.png"), dpi=140); plt.close(fig)


def characterization():
    d = load("characterization/wp6.json")
    if d:
        rows = []
        for k, s in d["estimators"].items():
            rows.append([k, f"{s['median_abs']:.4f}", f"{s['p90_abs']:.4f}", f"{s['median_signed']:+.4f}", f"{s['invalid_fraction']:.4f}"] +
                        [f"{s['per_class_median_abs'][c]:.4f}" for c in SIGNAL])
        write_csv("char_wp6_summary.csv", ["estimator", "median_abs", "p90_abs", "median_signed", "invalid_fraction"] +
                  [f"median_abs_{c}" for c in SIGNAL], rows)
        for key, title, sc, fmt, vmax in (("fc_centroid", "carrier error, % of true bandwidth (median |err|)", 100, "{:.1f}", 0.2),
                                          ("bw_obw99", "99 % bandwidth, relative error % (median |err|)", 100, "{:.0f}", 1.0),
                                          ("snr_psd_trueband", "SNR error, dB (median |err|, in-band, true band)", 1, "{:.1f}", 6.0),
                                          ("snr_psd_obw", "SNR error, dB (median |err|, in-band, estimated band)", 1, "{:.1f}", 6.0)):
            s = d["estimators"][key]["per_class_snr_median_abs"]
            M = [[s[c][str(t)] for t in SNRS] for c in SIGNAL]
            heatmap(M, SIGNAL, SNRS, f"{key}: {title}", os.path.join(FIG, f"char_{key}_class_snr.png"),
                    fmt=fmt, scale=sc, vmax=vmax, cmap="Oranges")
            write_csv(f"char_{key}_class_snr.csv", ["class"] + [str(t) for t in SNRS],
                      [[c] + [f"{s[c][str(t)]:.4f}" for t in SNRS] for c in SIGNAL])
    d = load("characterization/symbol_rate.json")
    if d:
        sym = list(next(iter(d["results"].values()))["per_class"])
        rows = [[k, f"{r['all']:.4f}", f"{r['in_search_range_only']:.4f}"] + [f"{r['per_class'][c]:.4f}" for c in sym]
                for k, r in d["results"].items()]
        write_csv("char_symbol_rate_summary.csv", ["estimator", "within_2pct_all", "within_2pct_in_range"] + sym, rows)
        for k, r in d["results"].items():
            if k.startswith("consensus, 12") or k.startswith("inst_freq"):
                M = [[r["per_class_snr"][c][str(t)] for t in SNRS] for c in sym]
                tag = "consensus" if k.startswith("consensus") else "inst_freq_s3"
                heatmap(M, sym, SNRS, f"symbol rate within 2 % ({k}), % per class and SNR",
                        os.path.join(FIG, f"char_symbol_rate_{tag}_class_snr.png"))
                write_csv(f"char_symbol_rate_{tag}_class_snr.csv", ["class"] + [str(t) for t in SNRS],
                          [[c] + [f"{r['per_class_snr'][c][str(t)]:.4f}" for t in SNRS] for c in sym])


if __name__ == "__main__":
    identification(); extended_and_rules(); detection(); characterization()
    print("figures:", sorted(os.listdir(FIG)))
    print("tables:", sorted(os.listdir(TAB)))
