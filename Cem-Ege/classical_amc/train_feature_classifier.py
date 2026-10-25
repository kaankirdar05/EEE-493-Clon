"""Job 5.1 - feature-based modulation classifier (cumulants + classic features -> tree / SVM).

Usage (run from this folder with the project venv):
    python train_feature_classifier.py check-labels --data GOLD_XYZ_OSC.0001_1024.hdf5
    python train_feature_classifier.py train        --data GOLD_XYZ_OSC.0001_1024.hdf5
    python train_feature_classifier.py smoke        # no dataset needed

Outputs (in --out, default results/<model>):
    metrics.json            overall accuracy, accuracy per SNR, per-class accuracy, timing
    acc_vs_snr.csv          snr_db,accuracy  (same format the 5.2 CNN should write, for the M3 plot)
    acc_vs_snr.png
    confusion_all.png       row-normalised confusion matrix, all SNRs
    confusion_high_snr.png  same, SNR >= 10 dB
    features.npz            cached features, so re-training skips the slow part
    model.joblib            trained classifier
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np

from amc import radioml
from amc.features import FEATURE_NAMES, extract_features


# ---------------------------------------------------------------- features

def build_features(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Extract features group by group; returns (F, y, snr, classes, seconds per frame)."""
    classes = args.classes or radioml.DEFAULT_SUBSET
    feats, ys, snrs = [], [], []
    t_feat, n_frames = 0.0, 0
    groups = radioml.iter_radioml2018(
        args.data, classes=classes, snr_min=args.snr_min, snr_max=args.snr_max,
        per_group=args.per_group, order=args.order, seed=args.seed,
    )
    for i, (x, k, s) in enumerate(groups, 1):
        t0 = time.perf_counter()
        feats.append(extract_features(x))
        t_feat += time.perf_counter() - t0
        n_frames += len(x)
        ys.append(np.full(len(x), k, dtype=np.int16))
        snrs.append(np.full(len(x), s, dtype=np.int16))
        if i % 10 == 0:
            print(f"  {i} groups, {n_frames} frames ({classes[k]} @ {s} dB)", flush=True)
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(snrs), classes, t_feat / n_frames


def load_or_build_features(args):
    cache = os.path.join(args.out, "features.npz")
    key = json.dumps({"data": os.path.abspath(args.data), "classes": args.classes or radioml.DEFAULT_SUBSET,
                      "per_group": args.per_group, "snr": [args.snr_min, args.snr_max],
                      "order": args.order, "seed": args.seed, "features": FEATURE_NAMES})
    if os.path.exists(cache) and not args.rebuild:
        d = np.load(cache, allow_pickle=False)
        if str(d["key"]) == key:
            print(f"Using cached features from {cache} (pass --rebuild to recompute)")
            return d["F"], d["y"], d["snr"], [str(c) for c in d["classes"]], float(d["sec_per_frame"])
        print("Cached features were built with different settings; rebuilding.")
    print("Extracting features ...")
    F, y, snr, classes, spf = build_features(args)
    np.savez_compressed(cache, F=F, y=y, snr=snr, classes=np.array(classes), sec_per_frame=spf,
                        key=np.array(key))
    return F, y, snr, classes, spf


# ---------------------------------------------------------------- model

def make_model(name: str, seed: int):
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True,
                                              validation_fraction=0.1, random_state=seed)
    if name == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale"))
    raise ValueError(name)


# ---------------------------------------------------------------- plots

def plot_acc_vs_snr(snrs, accs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(snrs, np.array(accs) * 100, marker="o")
    ax.axhline(85, color="gray", ls="--", lw=1, label="85 % target")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + 0.75 * len(classes), 0.8 + 0.7 * len(classes)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def cmd_train(args):
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib

    args.out = args.out or os.path.join("results", args.model)
    os.makedirs(args.out, exist_ok=True)
    F, y, snr, classes, feat_spf = load_or_build_features(args)
    print(f"{len(y)} frames, {F.shape[1]} features, classes: {classes}")

    strata = y.astype(np.int32) * 1000 + (snr.astype(np.int32) + 100)
    tr, te = train_test_split(np.arange(len(y)), test_size=args.test_size,
                              stratify=strata, random_state=args.seed)
    if args.model == "svm" and len(tr) > args.svm_max_train:
        print(f"SVM: subsampling training set {len(tr)} -> {args.svm_max_train} (O(n^2) training)")
        tr = train_test_split(tr, train_size=args.svm_max_train, stratify=strata[tr],
                              random_state=args.seed)[0]

    model = make_model(args.model, args.seed)
    t0 = time.perf_counter()
    model.fit(F[tr], y[tr])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(F[te])
    pred_spf = (time.perf_counter() - t0) / len(te)

    yt, st = y[te], snr[te]
    snr_levels = sorted(np.unique(st).tolist())
    acc_snr = [float(np.mean(pred[st == s] == yt[st == s])) for s in snr_levels]
    high = st >= 10
    labels = list(range(len(classes)))
    cm_all = confusion_matrix(yt, pred, labels=labels)
    cm_high = confusion_matrix(yt[high], pred[high], labels=labels) if high.any() else None

    metrics = {
        "model": args.model,
        "classes": classes,
        "features": FEATURE_NAMES,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "accuracy_all": float(np.mean(pred == yt)),
        "accuracy_snr_ge_10": float(np.mean(pred[high] == yt[high])) if high.any() else None,
        "accuracy_vs_snr": dict(zip(map(str, snr_levels), acc_snr)),
        "per_class_accuracy_snr_ge_10": (
            {c: float(cm_high[i, i] / max(cm_high[i].sum(), 1)) for i, c in enumerate(classes)}
            if cm_high is not None else None),
        "timing": {
            "feature_extraction_ms_per_frame": feat_spf * 1e3,
            "prediction_ms_per_frame": pred_spf * 1e3,
            "fit_seconds": fit_s,
        },
        "data": {"path": args.data, "per_group": args.per_group, "snr_range": [args.snr_min, args.snr_max],
                 "class_order": args.order, "seed": args.seed},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "acc_vs_snr.csv"), "w", encoding="utf-8") as f:
        f.write("snr_db,accuracy\n")
        f.writelines(f"{s},{a:.4f}\n" for s, a in zip(snr_levels, acc_snr))
    joblib.dump({"model": model, "classes": classes, "features": FEATURE_NAMES},
                os.path.join(args.out, "model.joblib"))

    name = {"hgb": "Gradient-boosted trees", "svm": "RBF SVM"}[args.model]
    plot_acc_vs_snr(snr_levels, acc_snr, os.path.join(args.out, "acc_vs_snr.png"),
                    f"5.1 feature classifier ({name})")
    plot_confusion(cm_all, classes, os.path.join(args.out, "confusion_all.png"), "All SNRs")
    if cm_high is not None:
        plot_confusion(cm_high, classes, os.path.join(args.out, "confusion_high_snr.png"), "SNR >= 10 dB")

    print(f"\nAccuracy, all SNRs:   {metrics['accuracy_all'] * 100:.1f} %")
    if high.any():
        print(f"Accuracy, SNR >= 10:  {metrics['accuracy_snr_ge_10'] * 100:.1f} %")
        for c, a in metrics["per_class_accuracy_snr_ge_10"].items():
            print(f"   {c:>10s}: {a * 100:5.1f} %")
    print("Accuracy vs SNR: " + "  ".join(f"{s}:{a * 100:.0f}" for s, a in zip(snr_levels, acc_snr)))
    print(f"Timing: features {feat_spf * 1e3:.3f} ms/frame, prediction {pred_spf * 1e3:.4f} ms/frame")
    print(f"Results written to {os.path.abspath(args.out)}")


def check_labels(path: str, order: str, per_group: int = 200) -> bool:
    """Check the class-order mapping using power-law spectral lines at high SNR.

    BPSK has a line in x^2; QPSK has none in x^2 but one in x^4; 8PSK has neither
    (its x^8 line is too weak after pulse shaping to test reliably). If the one-hot
    columns are mapped to the wrong names, these signatures break.
    """
    names = ["BPSK", "QPSK", "8PSK"]
    with __import__("h5py").File(path, "r") as f:
        top = int(np.max(f["Z"][:]))
    lines = {}
    for x, k, s in radioml.iter_radioml2018(path, classes=names, snr_min=top, snr_max=top,
                                            per_group=per_group, order=order):
        F = extract_features(x)
        lines[names[k]] = {p: float(np.median(F[:, FEATURE_NAMES.index(f"line_x{p}")])) for p in (2, 4, 8)}
    for n in names:
        print(f"   {n:>5s} @ {top} dB   log10 line strength  x2={lines[n][2]:.2f}  x4={lines[n][4]:.2f}  x8={lines[n][8]:.2f}")
    b, q, e = lines["BPSK"], lines["QPSK"], lines["8PSK"]
    return (b[2] - max(q[2], e[2]) > 0.5) and (q[4] - q[2] > 0.5) and (q[4] - e[4] > 0.5)


def cmd_check_labels(args):
    print(f"Checking class order '{args.order}' ...")
    if check_labels(args.data, args.order):
        print("PASS: BPSK / QPSK / 8PSK signatures match this class order.")
        return
    other = "classes-txt" if args.order == "hdf5" else "hdf5"
    print(f"FAIL with '{args.order}'. Trying '{other}' ...")
    if check_labels(args.data, other):
        print(f"PASS with '{other}': use --order {other}.")
    else:
        print("Neither order passes; inspect the file before trusting any labels.")
    raise SystemExit(1)


def cmd_smoke(args):
    """End-to-end test on a small fake file with the real RadioML layout."""
    from amc import synthetic

    tmp = tempfile.mkdtemp(prefix="amc_smoke_")
    try:
        path = os.path.join(tmp, "fake_radioml.hdf5")
        print(f"Writing fake RadioML-layout file to {path} ...")
        radioml.write_fake_radioml(path, frames_per_group=args.per_group,
                                   snrs=list(range(-10, 21, 4)), generator=synthetic.generate)
        print("check-labels on fake file:")
        assert check_labels(path, "hdf5", per_group=50), "label check failed on fake data"
        assert not check_labels(path, "classes-txt", per_group=50), "label check cannot tell orders apart"
        print("Label check correctly accepts 'hdf5' and rejects 'classes-txt'.\n")
        args.data, args.snr_min, args.snr_max, args.rebuild = path, -10, 20, True
        args.out = args.out or os.path.join("results", "smoke")
        cmd_train(args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_data=True):
        if need_data:
            sp.add_argument("--data", required=True, help="path to GOLD_XYZ_OSC.0001_1024.hdf5")
        sp.add_argument("--order", default="hdf5",
                        help="class order: 'hdf5' (default, verified), 'classes-txt', or a text file with one class per line")

    t = sub.add_parser("train", help="extract features, train, evaluate")
    common(t)
    s = sub.add_parser("smoke", help="run the whole pipeline on fake data (no download needed)")
    common(s, need_data=False)
    for sp in (t, s):
        sp.add_argument("--classes", type=lambda v: v.split(","), default=None,
                        help=f"comma-separated subset (default: {','.join(radioml.DEFAULT_SUBSET)})")
        sp.add_argument("--per-group", type=int, default=1000, help="frames per (class, SNR) pair")
        sp.add_argument("--snr-min", type=int, default=-20)
        sp.add_argument("--snr-max", type=int, default=30)
        sp.add_argument("--model", choices=["hgb", "svm"], default="hgb")
        sp.add_argument("--svm-max-train", type=int, default=40000)
        sp.add_argument("--test-size", type=float, default=0.25)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", default=None)
        sp.add_argument("--rebuild", action="store_true", help="ignore cached features")
    s.set_defaults(per_group=150)

    c = sub.add_parser("check-labels", help="verify the class-order mapping of the HDF5 file")
    common(c)

    args = p.parse_args()
    {"train": cmd_train, "smoke": cmd_smoke, "check-labels": cmd_check_labels}[args.cmd](args)


if __name__ == "__main__":
    main()
