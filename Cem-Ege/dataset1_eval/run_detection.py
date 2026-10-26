"""Detection (WP4) on Dataset 1 v3: is there a signal in this 1,024-sample clip, or only noise?

Dataset 1 has one signal per clip, present for the whole clip, and every clip is scaled to power 1. So the
question per clip is H0 (NOISE class) vs H1 (any other class), and only detectors that do not need the
absolute noise power can work. The Cem-Ege detectors are applied UNCHANGED to the whole clip:

    block statistics (normally per 512-sample sub-band block; here per 1,024-sample clip)
        eigen_mme   wp4.detectors.EigenMME(L=8)                max/min eigenvalue ratio
        agm_L8      disc.eigen.EigenBlock("agm")               arithmetic/geometric mean of eigenvalues
        john_L8     disc.eigen.EigenBlock("john")              sphericity (EMR = John + 1, same detector)
        cav         ext.block_detectors.CAV(L=8)               covariance absolute value
        acorr       ext.block_detectors.AutoCorr(L=8)          Ljung-Box autocorrelation
        entropy_spec ext.block_detectors.SpectralEntropy(m=64) spectral entropy deficit
        entropy_hist ext.block_detectors.HistEntropy(16)       |FFT| histogram entropy
        cyclo       wp4.detectors.Cyclo()                      envelope / conjugate cyclic lines
        scd_fam     ext.block_detectors.SCDFam(Np=32)          spectral coherence (FAM)
        gid/grcr/pride/tid/aid  extra_methods/cov_index_detectors (L=4)
    spectrogram statistics (wp4 energy detector on a 16 x 64 spectrogram of the clip, fft 64)
        energy      wp4.detectors.Energy (kt=8, excision noise floor), clip statistic = max cell
        psd_occupancy wp4.detectors.PSDOccupancy, clip statistic = max over frequency

Thresholds: calibrated on fresh white Gaussian noise clips (1,024 samples, scaled to power 1, exactly how
make_synth_v3 makes NOISE clips), N_CAL clips, at Pfa 1e-2 and 1e-3, as in WP4 ("calibrated on noise").
The measured false-alarm rate is then checked on the 2,382 NOISE clips of the test split.
Pd is reported per class and per SNR (dataset Z label) on the test split.

    python run_detection.py            -> results/detection/
"""
from __future__ import annotations

import csv, json, os, time
import numpy as np
import ds1common as dc

OUT = os.path.join(dc.RESULTS, "detection")
os.makedirs(OUT, exist_ok=True)
N_CAL = 20000
PFAS = (1e-2, 1e-3)


def detectors():
    from wp4.detectors import EigenMME, Cyclo, Energy, PSDOccupancy
    from disc.eigen import EigenBlock
    from ext.block_detectors import CAV, AutoCorr, SpectralEntropy, HistEntropy, SCDFam
    import cov_index_detectors as cid
    from wp4.spectrum import spectrogram

    def block(det):
        return lambda x: det.block_stat(x.astype(np.complex128))

    def spec_det(det, reduce):
        def f(x):
            out = np.empty(len(x))
            for i, xi in enumerate(x):
                R = det.stat(spectrogram(xi, 64), xi)
                out[i] = reduce(R)
            return out
        return f

    d = {
        "eigen_mme": block(EigenMME(L=8)),
        "agm_L8": block(EigenBlock("agm", 8)),
        "john_L8": block(EigenBlock("john", 8)),
        "cav": block(CAV(L=8)),
        "acorr": block(AutoCorr(L=8)),
        "entropy_spec": block(SpectralEntropy(m=64)),
        "entropy_hist": block(HistEntropy(16)),
        "cyclo": block(Cyclo()),
        "scd_fam": block(SCDFam(Np=32)),
        "energy": spec_det(Energy(kt=8, kf=1, noise_method="excision"), lambda R: R.max()),
        "psd_occupancy": spec_det(PSDOccupancy(kt=8, kf=1, noise_method="excision"), lambda R: R.max()),
    }
    for k in cid.KINDS:
        d[f"{k}_L4"] = block(cid.CovIndex(k))
    return d


def stats_in_chunks(fn, x, chunk=2000):
    return np.concatenate([fn(x[s:s + chunk]) for s in range(0, len(x), chunk)])


def main():
    m = dc.meta()
    classes = m["classes"]
    te = np.flatnonzero(m["split"] == 2)
    x = dc.load_x(te)
    y, z = m["Y"][te], m["Z"][te]
    rng = np.random.default_rng(12345)
    h0 = (rng.standard_normal((N_CAL, 1024)) + 1j * rng.standard_normal((N_CAL, 1024))) / np.sqrt(2)
    h0 /= np.sqrt(np.mean(np.abs(h0) ** 2, axis=1, keepdims=True))
    noise_c = classes.index("NOISE")
    snrs = sorted(np.unique(z[~np.isnan(z)]).tolist())
    res, rows = {}, []
    for name, fn in detectors().items():
        t0 = time.perf_counter()
        s0 = stats_in_chunks(fn, h0)
        t_cal = time.perf_counter() - t0
        t0 = time.perf_counter()
        s1 = stats_in_chunks(fn, x)
        ms = (time.perf_counter() - t0) / len(x) * 1e3
        s0, s1 = np.nan_to_num(s0, nan=-np.inf), np.nan_to_num(s1, nan=-np.inf)
        r = {"ms_per_clip": ms, "calibration_clips": N_CAL, "pfa": {}}
        for pfa in PFAS:
            thr = float(np.quantile(s0, 1 - pfa))
            det = s1 > thr
            pfa_meas = float(det[y == noise_c].mean())
            pd_cls_snr = {classes[c]: {int(s): float(det[(y == c) & (z == s)].mean()) for s in snrs}
                          for c in range(len(classes)) if c != noise_c}
            # lowest SNR from which Pd stays >= 90 % (sustained), per class
            def sustained(d, target=0.9):
                best = None
                for s in sorted(d, reverse=True):
                    if d[s] < target:
                        break
                    best = s
                return best
            pd_all = {int(s): float(det[(y != noise_c) & (z == s)].mean()) for s in snrs}
            r["pfa"][str(pfa)] = {
                "threshold": thr, "measured_pfa_on_test_noise": pfa_meas,
                "n_test_noise": int((y == noise_c).sum()),
                "pd_all_signals": float(det[y != noise_c].mean()),
                "pd_vs_snr_all_classes": pd_all,
                "pd_class_snr": pd_cls_snr,
                "snr_pd90_sustained": {c: sustained(d) for c, d in pd_cls_snr.items()},
                "snr_pd90_sustained_all": sustained(pd_all),
            }
            for c, d in pd_cls_snr.items():
                for s, v in d.items():
                    rows.append([name, pfa, c, s, f"{v:.4f}"])
        res[name] = r
        p2 = r["pfa"]["0.01"]
        print(f"{name:>14s}  Pfa(1e-2) measured {p2['measured_pfa_on_test_noise']:.4f}  Pd all {p2['pd_all_signals'] * 100:5.1f} %  "
              f"90% from {p2['snr_pd90_sustained_all']} dB   cal {t_cal:.0f} s, {ms:.3f} ms/clip", flush=True)
        json.dump({"classes": classes, "detectors": res}, open(os.path.join(OUT, "detection.json"), "w"), indent=1)
    with open(os.path.join(OUT, "detection_class_snr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "pfa", "class", "snr_db", "pd"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
