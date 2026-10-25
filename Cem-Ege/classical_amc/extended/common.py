"""Shared helpers: paths, splits, metrics, plots. Importing this puts classical_amc/ on sys.path
so that `amc.*`, `practical_alrt` and `symbol_rate` can be imported unchanged."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # classical_amc/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

RESULTS = os.path.join(HERE, "results")
RADIOML = r"D:\data\radioml2018\GOLD_XYZ_OSC.0001_1024.hdf5"
TORCHSIG = r"D:\data\alrt_torchsig.npz"
os.makedirs(RESULTS, exist_ok=True)


def standard_split(y, snr, test_size=0.25, seed=0):
    """The split used by train_feature_classifier.py (stratified by class x SNR)."""
    from sklearn.model_selection import train_test_split
    strata = y.astype(np.int32) * 1000 + (snr.astype(np.int32) + 100)
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=strata, random_state=seed)


def alrt_split(y, snr):
    """The 50/50 split used by practical_alrt.py."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=0.5, stratify=y * 100 + (snr + 50), random_state=0)


def summarize(pred, y, snr, ms_per_frame=None, extra=None):
    snrs = sorted(set(np.asarray(snr).tolist()))
    acc = {int(s): float(np.mean(pred[snr == s] == y[snr == s])) for s in snrs}
    hi = snr >= 10
    out = {"accuracy_all": float(np.mean(pred == y)),
           "accuracy_snr_ge_10": float(np.mean(pred[hi] == y[hi])) if hi.any() else None,
           "snr_first_85": next((s for s in snrs if acc[s] >= 0.85), None),
           "ms_per_frame": ms_per_frame,
           "accuracy_vs_snr": acc}
    if extra:
        out.update(extra)
    return out


def fmt_row(name, r):
    ms = f"{r['ms_per_frame']:.3f}" if r.get("ms_per_frame") is not None else "-"
    return (f"  {name:>34s}: all {r['accuracy_all'] * 100:5.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:5.1f} %  "
            f"85 % at {r['snr_first_85']} dB  {ms} ms/frame")


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def plot_overview(results: dict, title: str, path: str, highlight=None):
    """Accuracy vs SNR for every method in `results` (name -> summarize() dict)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = ["-", "--", "-.", ":"]
    for i, (name, r) in enumerate(results.items()):
        acc = r["accuracy_vs_snr"]
        s = sorted(int(k) for k in acc)
        lw = 2.4 if highlight and name in highlight else 1.3
        ax.plot(s, [acc[k] * 100 if k in acc else acc[str(k)] * 100 for k in s], styles[(i // 20) % 4],
                color=plt.cm.tab20(i % 20), marker="o", ms=3, lw=lw,
                label=f"{name} ({r['accuracy_all'] * 100:.1f} / {r['accuracy_snr_ge_10'] * 100:.1f})")
    ax.axhline(85, color="gray", ls="--", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", title="method (all / SNR>=10 %)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
