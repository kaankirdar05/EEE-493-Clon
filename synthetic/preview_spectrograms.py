"""Save a grid of example spectrogram images: one row per class, one column per SNR level.
Example: python preview_spectrograms.py ds1_v2_spec.h5 --snr -10 0 10 20
"""
import argparse
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--snr", nargs="+", type=float, default=[-10, 0, 10, 20])
p.add_argument("--out", default="spectrogram_examples.png"); a = p.parse_args()
f = h5py.File(a.file, "r")
classes = [c.decode() if isinstance(c, bytes) else str(c) for c in f.attrs["classes"]]
Y, Z = f["Y"][:], f["Z"][:]
fig, ax = plt.subplots(len(classes), len(a.snr), figsize=(1.35 * len(a.snr) + 0.9, 1.25 * len(classes)))
for r, name in enumerate(classes):
    for c, snr in enumerate(a.snr):
        hit = np.where((Y == r) & ((Z == snr) | np.isnan(Z)))[0]
        ax[r, c].imshow(f["S"][hit[0]], cmap="viridis", vmin=0, vmax=255, aspect="auto")
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(f"{snr:g} dB", fontsize=9)
    ax[r, 0].set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
fig.suptitle("Spectrogram images (x = frequency, y = time)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.975]); plt.savefig(a.out, dpi=130); print("saved", a.out)
