"""Part A known-answer / theory checks. Run BEFORE any benchmark number is reported.
No classifier is trained here."""
import numpy as np
from scipy.signal import hilbert
import common
from common import save_json
from analog_gen import CLASSES, synth_clean, make_frames, B_NB
from analog_feats import analog_features, FI


def main():
    out = {}
    rng = np.random.default_rng(12345)
    N = 8192
    # ---- 1. construction checks against the defining identities --------------------------------
    chk = {}
    s, meta = synth_clean("SSB-USB", 50, N, rng)
    m = meta["m"]
    h = hilbert(m, axis=1)
    sc = np.sqrt(np.mean(np.abs(s) ** 2, 1, keepdims=True)); mm = np.sqrt(np.mean(m**2, 1, keepdims=True))
    chk["SSB-USB: |corr(Re s, m)|, |corr(Im s, hilbert(m))| (expect 1)"] = [
        float(np.mean([abs(np.corrcoef(s[i].real, m[i])[0, 1]) for i in range(50)])),
        float(np.mean([abs(np.corrcoef(s[i].imag, h[i].imag)[0, 1]) for i in range(50)]))]
    s, meta = synth_clean("VSB", 50, N, rng)
    chk["VSB: corr(Re s, m) (sum rule H(f)+H(-f)=1 -> Re s = m/2; expect 1)"] = float(np.mean([np.corrcoef(s[i].real, meta["m"][i])[0, 1] for i in range(50)]))
    s, meta = synth_clean("NBFM", 50, N, rng, force_tone=False)
    d = np.angle(s[:, 1:] * np.conj(s[:, :-1])) / (2 * np.pi)
    chk["NBFM: corr(discriminator, m) (expect 1)"] = float(np.mean([np.corrcoef(d[i], meta["m"][i, :-1])[0, 1] for i in range(50)]))
    s, meta = synth_clean("PM", 50, N, rng, force_tone=False)
    d = np.angle(s[:, 1:] * np.conj(s[:, :-1])) / (2 * np.pi)
    dm = np.diff(meta["m"], axis=1)
    chk["PM: corr(discriminator, dm/dt) (expect 1)"] = float(np.mean([np.corrcoef(d[i], dm[i])[0, 1] for i in range(50)]))
    out["construction"] = chk

    # ---- 2. theoretical spectral symmetry (noiseless, zero offset) ----------------------------
    sym = {}
    f = np.fft.fftfreq(N)
    for c in CLASSES:
        s, _ = synth_clean(c, 200, N, rng)
        P = np.abs(np.fft.fft(s, axis=1)) ** 2
        up = P[:, (f > 0)].sum(1); lo = P[:, (f < 0)].sum(1)
        r = 10 * np.log10(np.maximum(up, 1e-30) / np.maximum(lo, 1e-30))
        sym[c] = {"median_dB_up_over_low": float(np.median(r)), "median_abs_dB": float(np.median(np.abs(r))),
                  "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))}
    out["spectral_symmetry_noiseless_dB"] = sym

    # ---- 3. envelope theory: SSB envelope Rayleigh (std/mean 0.523), DSB-SC |Gaussian| (0.755) -------
    env = {}
    for c, th in (("AM-DSB-SC", np.sqrt(1 - 2 / np.pi) / np.sqrt(2 / np.pi)), ("SSB-USB", np.sqrt(4 / np.pi - 1))):
        s, _ = synth_clean(c, 200, 4096, rng, force_tone=False)
        a = np.abs(s)
        env[c] = {"measured_std_over_mean": float(np.median(a.std(1) / a.mean(1))), "theory": float(th)}
    out["envelope_noise_message"] = env

    # ---- 4. PM / FM slope feature on noiseless frames ---------------------------------------------
    sl = {}
    for tau_name, kw in (("flat message (tau=0)", dict(tau=np.zeros((400, 1)), force_tone=False)),
                         ("random tilt tau~U(0,1.2), noise message", dict(force_tone=False)),
                         ("random tilt, 30 % two-tone mix", dict())):
        v = {}
        for c in ("NBFM", "PM"):
            x, _ = make_frames(c, 400, 2048, 100.0, rng, **kw)
            v[c] = analog_features(x)[:, FI["fm_slope"]]
        thr = 0.5 * (np.median(v["NBFM"]) + np.median(v["PM"]))
        acc = 0.5 * (np.mean(v["NBFM"] < thr) + np.mean(v["PM"] > thr))
        sl[tau_name] = {"NBFM_slope_median": float(np.median(v["NBFM"])), "PM_slope_median": float(np.median(v["PM"])),
                        "NBFM_p10_p90": [float(np.percentile(v["NBFM"], 10)), float(np.percentile(v["NBFM"], 90))],
                        "PM_p10_p90": [float(np.percentile(v["PM"], 10)), float(np.percentile(v["PM"], 90))],
                        "midpoint_threshold": float(thr), "balanced_accuracy_noiseless": float(acc)}
    out["fm_slope_noiseless"] = sl

    # ---- 5. SNR estimator and x^2 line (DSB-SC vs SSB) --------------------------------------------
    est = {}
    for snr in (-5, 0, 10, 20):
        x, _ = make_frames("NBFM", 200, 2048, float(snr), rng)
        est[snr] = float(np.mean(analog_features(x)[:, FI["snr_est"]]))
    out["snr_estimate_mean_for_true_snr"] = est
    x1, _ = make_frames("AM-DSB-SC", 300, 2048, 30.0, rng, force_tone=False)
    x2, _ = make_frames("SSB-USB", 300, 2048, 30.0, rng, force_tone=False)
    out["x2_line_log10_median"] = {"AM-DSB-SC": float(np.median(analog_features(x1)[:, FI["line_x2"]])),
                                   "SSB-USB": float(np.median(analog_features(x2)[:, FI["line_x2"]]))}
    save_json(out, "partA_sanity.json")
    import json
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
