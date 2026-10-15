"""Measure the in-band SNR of each labelled signal in a SigMF recording (the 'answer key' for our dataset).
Noise level comes from a signal-off recording if given (best), otherwise from the empty part of the spectrum.
Example: python measure_snr.py practice_qpsk [--noise practice_noise]
"""
import argparse
import numpy as np
from scipy import signal
from sigmf import sigmffile

def psd(name):
    rec = sigmffile.fromfile(name)
    fs = rec.get_global_info()["core:sample_rate"]; fc = rec.get_captures()[0]["core:frequency"]
    f, p = signal.welch(rec.read_samples(), fs, nperseg=4096, return_onesided=False)
    return rec, fc + np.fft.fftshift(f), np.fft.fftshift(p)

ap = argparse.ArgumentParser(); ap.add_argument("recording"); ap.add_argument("--noise"); a = ap.parse_args()
rec, f, p = psd(a.recording)
for ann in rec.get_annotations():
    band = (f >= ann["core:freq_lower_edge"]) & (f <= ann["core:freq_upper_edge"])
    if a.noise:                                   # method A: same settings, generator switched off
        n = psd(a.noise)[2]
    else:                                         # method B: median of the bins outside every signal
        empty = np.ones_like(band)
        for other in rec.get_annotations():
            empty &= ~((f >= other["core:freq_lower_edge"] - 50e3) & (f <= other["core:freq_upper_edge"] + 50e3))
        n = np.full_like(p, np.median(p[empty]))
    s = np.sum(p[band] - n[band]); noise = np.sum(n[band])
    print(f'{ann["core:label"]}: in-band SNR = {10 * np.log10(s / noise):.1f} dB')
