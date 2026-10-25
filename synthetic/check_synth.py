"""Checks a synthetic dataset file before it is shared, and makes the pictures for the report.

  python check_synth.py ds1_v2.h5               format checks + examples.png + snr_check
  python check_synth.py ds1_v2.h5 --baseline    also a quick classical baseline: accuracy_vs_snr.png
"""
import argparse
import numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--baseline", action="store_true"); a = p.parse_args()
f = h5py.File(a.file, "r")
classes = [c.decode() if isinstance(c, bytes) else c for c in f.attrs["classes"]]
Y, Z, Zi, BW, split = f["Y"][:], f["Z"][:], f["Z_inband"][:], f["bw99"][:], f["split"][:]
N = len(Y)

# 1. format checks (the hand-over checklist)
ok = lambda cond, msg: print(("PASS  " if cond else "FAIL  ") + msg)
ok(f["X"].shape == (N, 1024, 2) and f["X"].dtype == np.float32, f"X float32 {f['X'].shape}")
ok(Y.dtype == np.int64 and Z.dtype == np.float32, "Y int64, Z float32")
ok(all(k in f.attrs for k in ["classes", "sample_rate", "source", "version", "snr_definition"]), "attributes present")
idx = np.sort(np.random.default_rng(0).choice(N, min(N, 5000), replace=False))
xs = f["X"][idx]; pw = np.mean(xs[..., 0] ** 2 + xs[..., 1] ** 2, axis=1)
ok(np.allclose(pw, 1, atol=1e-3) and not np.isnan(xs).any(), "every clip has power 1 and no NaN (5,000 checked)")
counts = np.bincount(Y, minlength=len(classes))
ok(counts.min() == counts.max(), "classes balanced: " + ", ".join(f"{c} {n}" for c, n in zip(classes, counts)))
print("split sizes (train / val / test):", np.bincount(split))

# 2. SNR check: measure in-band SNR from the noisy spectrum and compare with the label
def measured_inband(x, bw):
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
    fr = np.fft.fftshift(np.fft.fftfreq(len(x)))
    inband = np.abs(fr) <= bw / 2 + 0.003
    noise = np.median(P[np.abs(fr) > bw / 2 + 0.02]) / np.log(2)   # median of exponential -> mean
    return 10 * np.log10((P[inband].sum() - noise * inband.sum()) / (noise * inband.sum()))
print("\nSNR check at label 10 dB (measured in-band SNR from the spectrum vs the stored Z_inband):")
for c, name in enumerate(classes):
    sel = np.where((Y == c) & (Z == 10) & (BW < 0.5))[0][:200]
    if name == "NOISE" or len(sel) == 0: continue
    m = [measured_inband(f["X"][i, :, 0] + 1j * f["X"][i, :, 1], BW[i]) for i in sel]
    print(f"  {name:9s} stored {np.mean(Zi[sel]):5.1f} dB   measured {np.median(m):5.1f} dB   (Z label {np.mean(Z[sel]):.0f} dB)")

# 3. one example per class at the highest SNR: spectrum and I/Q scatter
top = np.nanmax(Z)
fig, ax = plt.subplots(2, len(classes), figsize=(2.2 * len(classes), 4.6))
for c, name in enumerate(classes):
    i = np.where((Y == c) & ((Z == top) | np.isnan(Z)))[0][0]
    x = f["X"][i, :, 0] + 1j * f["X"][i, :, 1]
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(1024)))) ** 2
    ax[0, c].plot(np.linspace(-0.5, 0.5, 1024), 10 * np.log10(P / P.max() + 1e-12), lw=0.6); ax[0, c].set_ylim(-60, 3)
    ax[0, c].set_title(name, fontsize=10); ax[0, c].set_xticks([-0.5, 0, 0.5]); ax[0, c].tick_params(labelsize=7)
    ax[1, c].plot(x.real, x.imag, ".", ms=1); ax[1, c].set_xticks([]); ax[1, c].set_yticks([])
ax[0, 0].set_ylabel("spectrum (dB)"); ax[1, 0].set_ylabel("I/Q samples")
plt.suptitle(f"One clip per class at {top:.0f} dB (frequency as a fraction of the sample rate)", fontsize=10)
plt.tight_layout(); plt.savefig("examples.png", dpi=110); print("\nsaved examples.png")

# 4. optional quick baseline: simple features + random forest (a sanity check, not our final classifier)
if a.baseline:
    from sklearn.ensemble import RandomForestClassifier
    def feats(x):
        x = x - x.mean(); c21 = np.mean(np.abs(x) ** 2)
        c20 = np.mean(x ** 2); c40 = np.mean(x ** 4) - 3 * c20 ** 2
        c42 = np.mean(np.abs(x) ** 4) - np.abs(c20) ** 2 - 2 * c21 ** 2
        amp = np.abs(x); ph = np.diff(np.unwrap(np.angle(x)))
        P = np.abs(np.fft.fft(x)) ** 2; P /= P.sum(); Ps = np.sort(P)[::-1]
        return [np.abs(c20) / c21, np.abs(c40) / c21 ** 2, c42 / c21 ** 2, amp.std() / amp.mean(),
                np.mean((amp - amp.mean()) ** 4) / amp.var() ** 2, ph.std(), np.mean(np.abs(np.diff(ph))),
                np.searchsorted(np.cumsum(Ps), 0.9) / len(P), np.exp(np.mean(np.log(P + 1e-15))) / np.mean(P)]
    rng = np.random.default_rng(1)
    tr = np.sort(rng.choice(np.where(split == 0)[0], min(40000, (split == 0).sum()), replace=False))
    te = np.where(split == 2)[0]
    F = lambda ids: np.array([feats(f["X"][i, :, 0] + 1j * f["X"][i, :, 1]) for i in ids])
    clf = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(F(tr), Y[tr])
    pred = clf.predict(F(te))
    snrs = np.unique(Z[~np.isnan(Z)])
    acc = [np.mean(pred[(Z[te] == s)] == Y[te][Z[te] == s]) for s in snrs]
    print("baseline accuracy by SNR:", ", ".join(f"{s:.0f} dB {100 * v:.0f}%" for s, v in zip(snrs, acc)))
    plt.figure(figsize=(6, 3.6)); plt.plot(snrs, 100 * np.array(acc), "o-")
    plt.xlabel("SNR label Z (dB)"); plt.ylabel("accuracy (%)"); plt.ylim(0, 100); plt.grid(alpha=0.3)
    plt.title("Quick classical baseline (features + random forest)", fontsize=10)
    plt.tight_layout(); plt.savefig("accuracy_vs_snr.png", dpi=110); print("saved accuracy_vs_snr.png")
