"""DSSS benchmark: noise calibration (Pfa 1e-2 / frame), validation of Pfa, Pd vs SNR, estimation, sanity checks.

  python dsss_bench.py cal        # thresholds on noise-only frames (seed block CAL) + Pfa validation (seed block VAL)
  python dsss_bench.py rep        # report run (seed block REP)
  python dsss_bench.py dev        # small tuning/exploration run (seed block DEV)
"""
import sys
import time
import numpy as np
from common import *
from dsss_gen import *
from dsss_methods import *

PFA = 1e-2
SNRS = [-25 + 2.5 * i for i in range(11)]
LS = (31, 127, 511)
DET = ["energy_ideal", "energy_unc", "energy_psd", "acf_fluct", "cyc_abs2", "cyc_x2", "acov_abs2"]


def unc_gain(rng):
    return 10 ** (rng.uniform(-0.5, 0.5) / 20)    # +-0.5 dB noise-power uncertainty (amplitude factor)


def frame_stats(x, x_unc, est=True):
    d = {}
    d["energy_ideal"] = energy_ideal(x)
    d["energy_unc"] = energy_ideal(x_unc)
    d["energy_psd"] = energy_psd(x)
    z, za_ = acf_fluct(x, True)
    d["acf_fluct"] = float(z.max())
    s, a = cyc_abs2(x)
    d["cyc_abs2"], d["alpha_abs2"] = s, a
    s2, rc, f0, _, _ = cyc_x2(x)
    d["cyc_x2"], d["rc_x2"], d["f0_x2"] = s2, rc, f0
    zb, rb = acov_abs2(x, True)
    d["acov_abs2"] = float(zb.max())
    if est:
        d["T_acf"] = pn_period(z, za_)
        d["T_acov"] = pn_period(zb, rb, rel=0.5)
        d["rc_model"], d["L_model"] = chip_rate_model(x, d["T_acf"])
    return d


def _noise_job(a):
    base, i = a
    rng = rng_for(base, 0, i)
    n = noise_frame(rng)
    g = unc_gain(rng)
    return frame_stats(n, g * n)


def _sig_job(a):
    base, L, snr, i, kind = a
    rng = rng_for(base, int(L), int(round((snr + 100) * 10)), i, hash_kind(kind))
    A = np.sqrt(10 ** (snr / 10))
    if kind == "oracle":
        x, info = dsss_frame(L, snr, rng, fixed_cfo=0.0)
        tpl = pulse_shape(np.tile(info["code"], 3))[L * SPS:2 * L * SPS]
        tpl = tpl / np.sqrt(np.mean(np.abs(tpl) ** 2))
        return {"oracle": oracle_mf(x, tpl)}
    s, info = dsss_frame(L, 0.0, rng, with_noise=False, random_chips=(kind == "plain"))
    n = cplx_noise(len(s), rng)
    x = A * s + n
    xu = A * s + unc_gain(rng) * n
    d = frame_stats(x, xu)
    d.update(L=L, snr=snr, T=info["T"], f0=info["f0"])
    return d


def hash_kind(k):
    return {"dsss": 0, "plain": 1, "oracle": 2}[k]


def _oracle_noise_job(a):
    base, L, i = a
    rng = rng_for(base, 99, int(L), i)
    n = noise_frame(rng)
    code = pn_code(L, rng)
    tpl = pulse_shape(np.tile(code, 3))[L * SPS:2 * L * SPS]
    tpl = tpl / np.sqrt(np.mean(np.abs(tpl) ** 2))
    return {"oracle": oracle_mf(n, tpl)}


def _tone_job(a):
    base, snr, i = a
    rng = rng_for(base, 77, int(snr + 100), i)
    n = noise_frame(rng)
    t = np.arange(len(n))
    x = n + np.sqrt(10 ** (snr / 10)) * np.exp(1j * (2 * np.pi * rng.uniform(-0.45, 0.45) * t + rng.uniform(0, 6.28)))
    return frame_stats(x, x, est=False)


def calibrate(n_cal=12000, n_val=6000, oracle_n=2000, n_tone=1000):
    t0 = time.time()
    cal = pool_map(_noise_job, [(SEED_CAL, i) for i in range(n_cal)])
    thr = {k: float(np.quantile([c[k] for c in cal], 1 - PFA)) for k in DET}
    val = pool_map(_noise_job, [(SEED_VAL, i) for i in range(n_val)])
    pfa = {}
    for k in DET:
        v = np.array([c[k] for c in val])
        hit = int((v > thr[k]).sum())
        lo, hi = wilson(hit, n_val)
        pfa[k] = {"pfa": hit / n_val, "ci95": [lo, hi], "n": n_val}
    # oracle thresholds per L
    othr, opfa = {}, {}
    for L in LS:
        c = [r["oracle"] for r in pool_map(_oracle_noise_job, [(SEED_CAL, L, i) for i in range(oracle_n)])]
        v = [r["oracle"] for r in pool_map(_oracle_noise_job, [(SEED_VAL, L, i) for i in range(oracle_n)])]
        othr[L] = float(np.quantile(c, 1 - PFA))
        opfa[L] = float(np.mean(np.array(v) > othr[L]))
    # stationary tone + noise (non-spread narrowband interferer): false alarms of the structure detectors
    tone = {}
    for snr in (0.0, 10.0):
        tv = pool_map(_tone_job, [(SEED_VAL, snr, i) for i in range(n_tone)])
        tone[snr] = {k: float(np.mean([c[k] > thr[k] for c in tv])) for k in DET}
    # no preferred chip rate on noise: distribution of alpha_hat
    al = np.array([c["alpha_abs2"] for c in val])
    near = float(np.mean(np.abs(al - 0.25) < 0.005))
    out = {"pfa_target": PFA, "n_cal": n_cal, "thr": thr, "pfa_val": pfa, "oracle_thr": othr, "oracle_pfa_val": opfa, "tone_plus_noise_false_alarm": tone, "n_tone": n_tone,
           "noise_alpha_hist": np.histogram(al, bins=np.linspace(0.02, 0.5, 25))[0].tolist(),
           "noise_frac_alpha_within_2pct_of_chiprate": near,
           "chance_expectation": 0.01 / 0.48, "noise_T_acf_samples_equal_any_PN_period": None}
    jdump(out, os.path.join(RES, "dsss_calibration.json"))
    np.savez(os.path.join(RES, "dsss_noise_stats.npz"), **{k: np.array([c[k] for c in val]) for k in DET}, alpha=al)
    print(json.dumps({k: out[k] for k in ("thr", "pfa_val", "oracle_thr", "oracle_pfa_val", "noise_frac_alpha_within_2pct_of_chiprate")}, indent=1))
    print("time", time.time() - t0)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cal":
        calibrate()
