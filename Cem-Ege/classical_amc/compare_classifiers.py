"""Compare classical (non-deep-learning) modulation classifiers on cached 5.1 features.

Usage (run from this folder with the project venv):
    python compare_classifiers.py --features results/radioml_hgb/features.npz        # -> results/compare_7
    python compare_classifiers.py --features results/radioml_hgb_all24/features.npz  # -> results/compare_24

Uses exactly the same train/test split as train_feature_classifier.py, so numbers are
directly comparable with results/<model>/metrics.json.

Outputs (in --out):
    comparison.csv              one row per model
    metrics.json                everything, incl. accuracy per SNR, per-class accuracy, top confusions
    acc_vs_snr_<model>.csv      snr_db,accuracy (same format as acc_vs_snr.csv)
    acc_vs_snr_all_models.png   one line per model, 85 % target dashed
    tree_rules.txt              rules of the depth-6 decision tree (sklearn export_text)

Feature extraction (sec_per_frame in features.npz, ~0.5 ms/frame) is shared by all models
and is NOT included in the per-model prediction time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from amc.features import FEATURE_NAMES

TARGET = 0.85


# ---------------------------------------------------------------- theoretical cumulants

def _const_psk(m):
    return np.exp(2j * np.pi * np.arange(m) / m)


def _const_ask(m):
    return np.arange(-(m - 1), m, 2).astype(complex)


def _const_qam(m):
    """Square QAM, or cross QAM (32/128) = smallest square grid with the corners removed."""
    k = int(round(np.sqrt(m)))
    if k * k == m:
        lv = np.arange(-(k - 1), k, 2)
        return (lv[:, None] + 1j * lv[None, :]).ravel()
    side = int(2 ** np.ceil(np.log2(m) / 2))          # 32 -> 6x6 = 36, 128 -> 12x12 = 144
    lv = np.arange(-(side - 1), side, 2)
    pts = (lv[:, None] + 1j * lv[None, :]).ravel()
    n_cut = (side * side - m) // 4                     # points removed per corner
    # remove the n_cut largest-magnitude points of each quadrant (the corners)
    keep = np.ones(len(pts), bool)
    for sr in (1, -1):
        for si in (1, -1):
            q = np.where((np.sign(pts.real) == sr) & (np.sign(pts.imag) == si))[0]
            far = q[np.argsort(-np.abs(pts[q]), kind="stable")[:n_cut]]
            keep[far] = False
    pts = pts[keep]
    assert len(pts) == m, (m, len(pts))
    return pts


def theoretical_cumulants(points: np.ndarray) -> tuple[float, float]:
    """(|C40|, C42) of an equiprobable constellation after the same normalisation as
    amc.features (zero mean, unit power)."""
    x = points.astype(complex)
    x = x - x.mean()
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    m20 = np.mean(x * x)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x ** 4)
    m42 = np.mean(np.abs(x) ** 4)
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - abs(m20) ** 2 - 2 * m21 ** 2
    return float(abs(c40)), float(np.real(c42))


THEORY_CONSTELLATIONS = {
    "4ASK": lambda: _const_ask(4), "8ASK": lambda: _const_ask(8),
    "BPSK": lambda: _const_psk(2), "QPSK": lambda: _const_psk(4), "8PSK": lambda: _const_psk(8),
    "16PSK": lambda: _const_psk(16), "32PSK": lambda: _const_psk(32),
    "16QAM": lambda: _const_qam(16), "32QAM": lambda: _const_qam(32), "64QAM": lambda: _const_qam(64),
    "128QAM": lambda: _const_qam(128), "256QAM": lambda: _const_qam(256),
}
# Not included: OOK (after the per-frame mean removal in amc.features it is antipodal, i.e.
# identical to BPSK), APSK (ring radii are not documented for RadioML), OQPSK (offset is a
# sample-level effect, symbol constellation = QPSK), analogue classes, FM, GMSK.


class CumulantTheoryClassifier:
    """Training-free Swami & Sadler style classifier: nearest theoretical (|C40|, C42)."""

    def __init__(self, classes):
        self.names = [c for c in classes if c in THEORY_CONSTELLATIONS]
        self.idx = np.array([classes.index(c) for c in self.names])
        self.table = np.array([theoretical_cumulants(THEORY_CONSTELLATIONS[c]()) for c in self.names])
        self.cols = [FEATURE_NAMES.index("abs_C40"), FEATURE_NAMES.index("C42")]

    def fit(self, X, y):
        return self

    def predict(self, X):
        z = X[:, self.cols].astype(np.float64)
        d = ((z[:, None, :] - self.table[None, :, :]) ** 2).sum(-1)
        return self.idx[np.argmin(d, axis=1)]   # ties -> first class in list order


# ---------------------------------------------------------------- models

def make_model(name, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True,
                                              validation_fraction=0.1, random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "svm":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale"))
    if name == "tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=6, random_state=seed)
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1))
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()
    if name == "nearest_centroid":
        from sklearn.neighbors import NearestCentroid
        return make_pipeline(StandardScaler(), NearestCentroid())
    raise ValueError(name)


MODELS = ["hgb", "rf", "svm", "tree", "knn", "lda", "nearest_centroid", "cumulant_theory"]


def first_snr_reaching(snrs, accs, target=TARGET):
    for s, a in zip(snrs, accs):
        if a >= target:
            return s
    return None


def sustained_snr_reaching(snrs, accs, target=TARGET):
    """Lowest SNR from which accuracy stays >= target at every higher SNR."""
    best = None
    for s, a in zip(reversed(snrs), reversed(accs)):
        if a < target:
            break
        best = s
    return best


def evaluate(pred, yt, st, classes, eval_classes):
    from sklearn.metrics import confusion_matrix
    snr_levels = sorted(np.unique(st).tolist())
    acc_snr = [float(np.mean(pred[st == s] == yt[st == s])) for s in snr_levels]
    high = st >= 10
    cm = confusion_matrix(yt[high], pred[high], labels=list(range(len(classes))))
    per_class = {classes[i]: float(cm[i, i] / cm[i].sum()) for i in eval_classes if cm[i].sum()}
    off = [(int(cm[i, j]) / max(int(cm[i].sum()), 1), classes[i], classes[j])
           for i in eval_classes for j in range(len(classes)) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    return {
        "accuracy_all": float(np.mean(pred == yt)),
        "accuracy_snr_ge_10": float(np.mean(pred[high] == yt[high])),
        "first_snr_ge_85": first_snr_reaching(snr_levels, acc_snr),
        "sustained_snr_ge_85": sustained_snr_reaching(snr_levels, acc_snr),
        "accuracy_vs_snr": dict(zip(map(str, snr_levels), acc_snr)),
        "per_class_accuracy_snr_ge_10": per_class,
        "top_confusions_snr_ge_10": [{"true": t, "pred": p, "fraction_of_true": round(f, 4)}
                                     for f, t, p in off[:8]],
    }, snr_levels, acc_snr


def plot_all(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (snrs, accs) in curves.items():
        ls = ":" if name == "cumulant_theory" else "-"
        ax.plot(snrs, np.array(accs) * 100, marker="o", ms=3, ls=ls, label=name)
    ax.axhline(85, color="gray", ls="--", lw=1, label="85 % target")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import export_text

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="features.npz written by train_feature_classifier.py")
    p.add_argument("--out", default=None, help="default results/compare_<n_classes>")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--svm-max-train", type=int, default=40000)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.features, allow_pickle=False)
    F, y, snr = d["F"], d["y"], d["snr"]
    classes = [str(c) for c in d["classes"]]
    feat_ms = float(d["sec_per_frame"]) * 1e3
    args.out = args.out or os.path.join("results", f"compare_{len(classes)}")
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(y)} frames, {F.shape[1]} features, {len(classes)} classes: {classes}")

    strata = y.astype(np.int32) * 1000 + (snr.astype(np.int32) + 100)
    tr, te = train_test_split(np.arange(len(y)), test_size=args.test_size,
                              stratify=strata, random_state=args.seed)
    yt, st = y[te], snr[te]

    results, curves = {}, {}
    for name in args.models.split(","):
        tr_m, te_m, note = tr, te, ""
        if name == "svm" and len(tr) > args.svm_max_train:
            tr_m = train_test_split(tr, train_size=args.svm_max_train, stratify=strata[tr],
                                    random_state=args.seed)[0]
            note = f"training subsampled {len(tr)} -> {len(tr_m)} (stratified)"
        if name == "cumulant_theory":
            model = CumulantTheoryClassifier(classes)
            keep = np.isin(y[te], model.idx)
            te_m = te[keep]
            note = ("training-free; evaluated ONLY on test frames of " + ",".join(model.names) +
                    "; predictions restricted to those classes; theory (|C40|,C42): " +
                    "; ".join(f"{c}=({a:.3f},{b:.3f})" for c, (a, b) in zip(model.names, model.table)))
        else:
            model = make_model(name, args.seed)
        n_fit = 0 if name == "cumulant_theory" else len(tr_m)
        print(f"\n[{name}] fit on {n_fit} frames ... {note}", flush=True)
        t0 = time.perf_counter()
        model.fit(F[tr_m], y[tr_m])
        fit_s = time.perf_counter() - t0 if name != "cumulant_theory" else 0.0
        t0 = time.perf_counter()
        pred = model.predict(F[te_m])
        pred_ms = (time.perf_counter() - t0) / len(te_m) * 1e3

        eval_classes = sorted(np.unique(y[te_m]).tolist())
        res, snrs, accs = evaluate(pred, y[te_m], snr[te_m], classes, eval_classes)
        res.update({"n_train": int(len(tr_m)) if name != "cumulant_theory" else 0,
                    "n_test": int(len(te_m)), "fit_seconds": fit_s,
                    "prediction_ms_per_frame": pred_ms, "note": note,
                    "evaluated_classes": [classes[i] for i in eval_classes]})
        results[name] = res
        curves[name] = (snrs, accs)
        with open(os.path.join(args.out, f"acc_vs_snr_{name}.csv"), "w", encoding="utf-8") as f:
            f.write("snr_db,accuracy\n")
            f.writelines(f"{s},{a:.4f}\n" for s, a in zip(snrs, accs))
        if name == "tree":
            with open(os.path.join(args.out, "tree_rules.txt"), "w", encoding="utf-8") as f:
                f.write(f"# DecisionTreeClassifier(max_depth=6) on {len(classes)} classes; "
                        f"learned-threshold analogue of the Azzouz-Nandi decision tree\n")
                f.write(f"# classes: {classes}\n\n")
                f.write(export_text(model, feature_names=FEATURE_NAMES, class_names=classes,
                                    max_depth=6, decimals=3))
        print(f"  all SNRs {res['accuracy_all'] * 100:.1f} %, SNR>=10 {res['accuracy_snr_ge_10'] * 100:.1f} %, "
              f"85 % first at {res['first_snr_ge_85']} dB (sustained from {res['sustained_snr_ge_85']}), fit {fit_s:.1f} s, predict {pred_ms:.4f} ms/frame")
        print("  top confusions (SNR>=10): " + ", ".join(
            f"{c['true']}->{c['pred']} {c['fraction_of_true'] * 100:.0f}%" for c in res["top_confusions_snr_ge_10"][:4]))

    with open(os.path.join(args.out, "comparison.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc_all_snr", "acc_snr_ge_10", "first_snr_ge_85_db",
                    "sustained_snr_ge_85_db", "fit_s",
                    "predict_ms_per_frame", "feature_ms_per_frame_shared", "n_train", "n_test", "note"])
        for name, r in results.items():
            w.writerow([name, f"{r['accuracy_all']:.4f}", f"{r['accuracy_snr_ge_10']:.4f}",
                        "" if r["first_snr_ge_85"] is None else r["first_snr_ge_85"],
                        "" if r["sustained_snr_ge_85"] is None else r["sustained_snr_ge_85"],
                        f"{r['fit_seconds']:.2f}", f"{r['prediction_ms_per_frame']:.5f}",
                        f"{feat_ms:.3f}", r["n_train"], r["n_test"], r["note"]])
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"features_file": os.path.abspath(args.features), "classes": classes,
                   "features": FEATURE_NAMES, "feature_extraction_ms_per_frame_shared": feat_ms,
                   "split": {"test_size": args.test_size, "seed": args.seed,
                             "stratify": "y*1000 + (snr+100)", "n_train": int(len(tr)), "n_test": int(len(te))},
                   "models": results}, f, indent=2)
    plot_all(curves, os.path.join(args.out, "acc_vs_snr_all_models.png"),
             f"Classical AMC on RadioML 2018.01A, {len(classes)} classes "
             f"(cumulant_theory: digital subset only)")
    print(f"\nResults written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
