"""Open a SigMF recording, print its description and save a spectrum + waterfall picture.
Example:  python check_sigmf.py fm      (the name without .sigmf-meta)
"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from sigmf import sigmffile

rec = sigmffile.fromfile(sys.argv[1])
fs = rec.get_global_info()["core:sample_rate"]
fc = rec.get_captures()[0]["core:frequency"]
x = rec.read_samples(0, min(rec.sample_count, int(fs)))   # first second only
print("samples:", rec.sample_count, " rate:", fs, " centre:", fc)
for ann in rec.get_annotations():
    print("signal:", ann["core:label"], ann["core:freq_lower_edge"], "-", ann["core:freq_upper_edge"], "Hz")

f, pxx = signal.welch(x, fs, nperseg=1024, return_onesided=False)
f, pxx = np.fft.fftshift(f), np.fft.fftshift(pxx)
fw, t, sxx = signal.spectrogram(x, fs, nperseg=1024, noverlap=0, return_onesided=False)
fig, ax = plt.subplots(2, 1, figsize=(8, 7))
ax[0].plot((fc + f) / 1e6, 10 * np.log10(pxx)); ax[0].set(xlabel="MHz", ylabel="dB", title="Spectrum")
ax[1].imshow(10 * np.log10(np.fft.fftshift(sxx, axes=0).T), aspect="auto", origin="lower",
             extent=[(fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6, 0, t[-1]])
ax[1].set(xlabel="MHz", ylabel="seconds", title="Waterfall")
plt.tight_layout(); plt.savefig(sys.argv[1] + "_check.png", dpi=120)
print("saved", sys.argv[1] + "_check.png")
