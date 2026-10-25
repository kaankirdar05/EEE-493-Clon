"""First test of the classical baseline on REAL over-the-air data.

Recording: Allen Telescope Array RFI survey (Zenodo 8242048, CC-BY-4.0), file
2023-08-03-15-38-39_atarfi.sigmf: ADALM-Pluto + Maia SDR, 61.44 Msps, ci16_le, ~753.5 MHz.
Ground truth from the author's description (destevez.net, Aug 2023): LTE downlink carriers at
742.5 MHz (5 MHz channel), 751 MHz (10 MHz) and 763 MHz (10 MHz); an unidentified bursty
signal at 772.25 MHz (bandwidth unknown -> not scored). LTE occupies 4.5 MHz of a 5 MHz channel
and 9 MHz of a 10 MHz channel; the carriers are on for the whole recording.

Tests (20 segments of 262,144 samples spread over the file, 512 x 512 spectrogram each):
  1. Noise model: in carrier-free bins away from the band edges, are spectrogram cells
     Exp-distributed as the detectors assume, and does the energy detector's false-alarm
     rate there match its target?
  2. Detection: per detector, fraction of (segment, carrier) pairs detected (>= 50 % of the
     carrier's cells flagged), mean box IoU with the carrier, flagged-cell rate outside all
     known signals (an upper bound on false alarms: unknown real signals may exist).
  3. Characterization: carrier (power centroid) and 99 % occupied bandwidth of each carrier
     from a Welch PSD of the whole file vs the published values.

    python analyze_atarfi.py <path to extracted .sigmf-data> <path to .sigmf-meta>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wp4_detection"))
from wp4.detectors import all_detectors          # noqa: E402
from wp4.scoring import iou                      # noqa: E402
from wp4.segmentation import segment             # noqa: E402
from wp4.spectrum import spectrogram             # noqa: E402

CARRIERS = [(742.5e6, 4.5e6), (751.0e6, 9.0e6), (763.0e6, 9.0e6)]   # centre, occupied BW
UNKNOWN = [(772.25e6, 2.0e6)]                                         # excluded from scoring
N_SEG, SEG = 20, 512 * 512
DETS = ["energy", "energy_mt", "fcme", "lad", "cfar_ca", "cfar_os", "cfar_so", "cfar_2d"]


def load(data_path, meta_path):
    meta = json.load(open(meta_path))
    fs = float(meta["global"]["core:sample_rate"])
    fc = float(meta["captures"][0]["core:frequency"])
    dt = meta["global"]["core:datatype"]
    assert dt in ("ci16_le", "ci16"), dt
    raw = np.memmap(data_path, dtype="<i2", mode="r")
    return raw, fs, fc


def segment_iq(raw, k, n_total):
    start = int(k * (n_total - SEG) / (N_SEG - 1))
    s = np.asarray(raw[2 * start: 2 * (start + SEG)], dtype=np.float32)
    return (s[0::2] + 1j * s[1::2]).astype(np.complex64)


def freq_cells(f_lo, f_hi, fs, fc, F=512):
    b = lambda f: int(np.clip(np.round((f - fc + fs / 2) / fs * F), 0, F))
    return b(f_lo), b(f_hi)


def main(data_path, meta_path):
    raw, fs, fc = load(data_path, meta_path)
    n_total = len(raw) // 2
    print(f"fs {fs / 1e6:.2f} Msps, centre {fc / 1e6:.3f} MHz, {n_total} samples ({n_total / fs:.2f} s)")
    truth = [freq_cells(c - b / 2, c + b / 2, fs, fc) for c, b in CARRIERS]
    unk = [freq_cells(c - b / 2, c + b / 2, fs, fc) for c, b in UNKNOWN]
    known_mask = np.zeros(512, bool)
    for f0, f1 in truth + unk:
        known_mask[max(f0 - 4, 0):min(f1 + 4, 512)] = True
    edge = np.zeros(512, bool); edge[:40] = True; edge[-40:] = True      # receiver roll-off
    quiet = ~known_mask & ~edge

    dets = all_detectors()
    for n in DETS:
        dets[n].calibrate([1e-3], os.path.join(os.path.dirname(__file__), "..", "wp4_detection", "results", "calibration"))
    hits = {n: [] for n in DETS}; ious = {n: [] for n in DETS}; fa = {n: [] for n in DETS}; ks = []
    from scipy.stats import kstest
    for k in range(N_SEG):
        iq = segment_iq(raw, k, n_total)
        spec = spectrogram(iq)
        # 1. noise model: cells in quiet bins, normalised by their bin median / ln 2 -> Exp(1)
        q = spec[:, quiet]
        z = (q / (np.median(q, axis=0, keepdims=True) / np.log(2))).ravel()
        ks.append(kstest(z[::7], "expon").statistic)
        for n in DETS:
            m = dets[n].masks(spec, iq, [1e-3])[1e-3]
            boxes = segment(m)
            for f0, f1 in truth:
                hits[n].append(m[:, f0:f1].mean() >= 0.5)
                ious[n].append(max([iou(b, (0, 512, f0, f1)) for b in boxes], default=0.0))
            fa[n].append(m[:, quiet].mean())
    res = {"fs": fs, "fc": fc, "segments": N_SEG,
           "noise_model": {"ks_distance_to_exp_median": float(np.median(ks))},
           "detectors": {n: {"carrier_detected": float(np.mean(hits[n])), "mean_best_iou": float(np.mean(ious[n])),
                             "flag_rate_quiet_bins": float(np.mean(fa[n]))} for n in DETS}}
    print(f"noise model: median KS distance of quiet-bin cells to Exp(1) = {res['noise_model']['ks_distance_to_exp_median']:.3f} "
          "(0 = perfect; ~0.01 is sampling noise for this sample size)")
    print(f"{'detector':>10s} | carriers detected | mean box IoU | flag rate in quiet bins (target 1e-3)")
    for n in DETS:
        r = res["detectors"][n]
        print(f"{n:>10s} | {r['carrier_detected'] * 100:16.0f} % | {r['mean_best_iou']:12.2f} | {r['flag_rate_quiet_bins']:.1e}")

    # 3. characterization from a Welch PSD over the whole file (1 M-sample chunks, 8192-point FFT)
    from scipy.signal import welch
    psd = None
    for k in range(10):
        start = int(k * (n_total - 2**20) / 9)
        s = np.asarray(raw[2 * start: 2 * (start + 2**20)], dtype=np.float32)
        f, p = welch(s[0::2] + 1j * s[1::2], fs=fs, nperseg=8192, return_onesided=False)
        psd = p if psd is None else psd + p
    order = np.argsort(f); f, psd = f[order] + fc, psd[order]
    char = []
    for (c, b) in CARRIERS:
        reg = (f > c - 0.75 * b) & (f < c + 0.75 * b)
        side = ((f > c - 1.0 * b) & (f < c - 0.6 * b)) | ((f > c + 0.6 * b) & (f < c + 1.0 * b))
        n0 = np.median(psd[side])
        sig = np.clip(psd[reg] - n0, 0, None)
        fc_hat = np.sum(f[reg] * sig) / np.sum(sig)
        cdf = np.cumsum(sig) / np.sum(sig)
        obw = f[reg][np.searchsorted(cdf, 0.995)] - f[reg][np.searchsorted(cdf, 0.005)]
        char.append({"carrier_true_MHz": c / 1e6, "carrier_est_MHz": fc_hat / 1e6, "error_kHz": (fc_hat - c) / 1e3,
                     "occupied_bw_true_MHz": b / 1e6, "obw99_est_MHz": obw / 1e6,
                     "snr_inband_dB": float(10 * np.log10(np.mean(sig) / n0))})
        print(f"carrier {c / 1e6:7.2f} MHz: estimate {fc_hat / 1e6:8.3f} MHz (error {(fc_hat - c) / 1e3:+6.1f} kHz), "
              f"99 % OBW {obw / 1e6:5.2f} MHz (true {b / 1e6:.1f}), in-band SNR {char[-1]['snr_inband_dB']:.1f} dB")
    res["characterization"] = char
    out = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "atarfi_15-38-39.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iq = segment_iq(raw, N_SEG // 2, n_total); spec = spectrogram(iq)
    m = dets["energy"].masks(spec, iq, [1e-3])[1e-3]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ext = [(fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6, 0, SEG / fs * 1e3]
    ax[0].imshow(10 * np.log10(spec + 1e-9), aspect="auto", origin="lower", extent=ext); ax[0].set_title("Real recording, ~753 MHz (spectrogram)")
    ax[1].imshow(m, aspect="auto", origin="lower", extent=ext, cmap="gray"); ax[1].set_title("Energy detector mask (Pfa 1e-3)")
    for a in ax:
        for c, b in CARRIERS:
            a.axvspan((c - b / 2) / 1e6, (c + b / 2) / 1e6, color="lime", alpha=0.15)
        a.set_xlabel("Frequency (MHz)"); a.set_ylabel("Time (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "atarfi_15-38-39.png"), dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
