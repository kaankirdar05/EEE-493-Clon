"""Characterization (WP6) on Dataset 1 v3: carrier, bandwidth, SNR and symbol rate, scored against the labels.

    python run_characterization.py wp6     # wp6.estimators.characterize on every signal clip of the test split
    python run_characterization.py sr      # symbol rate: inst_freq_s3 + the frozen 12-estimator consensus

WP6 (oracle detection, as in run_wp6.py): the truth box is [fo - bw99/2, fo + bw99/2], with fo = the clip's carrier
offset (recovered by gen_offsets.py) and bw99 = the dataset's occupied-bandwidth label. The estimators are called
unchanged with their default settings; the sample rate is given as 10 MHz (their design value; Dataset 1 is
normalised to 1) and every frequency is scaled by the same factor, so all errors are relative.
Truth:  carrier = fo;  bandwidth = bw99 label;  SNR = Z_inband label (signal / noise power inside the 99 % band).

Symbol rate: truth = 1 / sps label (classes OOK, 2FSK, 4FSK, GMSK, BPSK, QPSK, PI4DQPSK, 8PSK, 16QAM).
Success = within 2 %, as in BASELINE.md. Consensus parameters are the frozen ones in
../characterization_extended/results/consensus_params.json (tuned by Cem-Ege on TorchSig/RadioML dev data;
nothing is re-tuned on Dataset 1). Search range of every estimator: 2.5-25 samples/symbol.
"""
from __future__ import annotations

import csv, json, os, sys, time
import numpy as np
import ds1common as dc

OUT = os.path.join(dc.RESULTS, "characterization")
os.makedirs(OUT, exist_ok=True)
FS = 1e7
CARRIER = ["fc_peak", "fc_centroid", "fc_mid10", "fc_obw99", "fc_noisecross"]
BANDWIDTH = ["bw_3db", "bw_10db", "bw_26db", "bw_obw99", "bw_noisecross"]
SNR = ["snr_psd_obw", "snr_psd_10db", "snr_psd_trueband", "snr_peak512", "snr_m2m4"]


def wp6():
    from wp6.estimators import characterize
    m = dc.meta()
    cl = m["classes"]
    fo_all = np.load(os.path.join(dc.RESULTS, "ds1_v3_offsets.npz"))["fo"]
    te = np.flatnonzero((m["split"] == 2) & (m["Y"] != cl.index("NOISE")))
    x = dc.load_x(te)
    keys = CARRIER + BANDWIDTH + SNR
    E = np.full((len(te), len(keys)), np.nan)
    t0 = time.perf_counter()
    one = lambda f: np.ones_like(f)
    for k, (i, xi) in enumerate(zip(te, x)):
        bw, fo = float(m["bw99"][i]), float(fo_all[i])
        o = characterize(xi.astype(complex), FS, (fo - bw / 2) * FS, (fo + bw / 2) * FS, one)
        for j, key in enumerate(keys):
            v = o[key]
            if key in CARRIER:
                E[k, j] = (v / FS - fo) / bw                      # carrier error / true bandwidth
            elif key in BANDWIDTH:
                E[k, j] = v / FS / bw - 1                         # relative bandwidth error
            else:
                E[k, j] = v - m["Z_inband"][i]                    # SNR error, dB (inf -> nan below)
        if k % 5000 == 0:
            print(f"{k}/{len(te)}  {(time.perf_counter() - t0) / (k + 1) * 1e3:.1f} ms/clip", flush=True)
    E[~np.isfinite(E)] = np.nan
    np.savez(os.path.join(OUT, "wp6_errors.npz"), idx=te, E=E, keys=np.array(keys))
    y, z = m["Y"][te], m["Z"][te]
    snrs = sorted(np.unique(z).tolist())
    summary = {}
    rows = []
    for j, key in enumerate(keys):
        e = E[:, j]
        a = np.abs(e)
        s = {"median_abs": float(np.nanmedian(a)), "p90_abs": float(np.nanpercentile(a, 90)),
             "median_signed": float(np.nanmedian(e)), "invalid_fraction": float(np.mean(np.isnan(e))),
             "per_class_median_abs": {}, "per_class_median_signed": {}, "per_snr_median_abs": {},
             "per_class_snr_median_abs": {}}
        for c in sorted(np.unique(y)):
            mc = y == c
            s["per_class_median_abs"][cl[c]] = float(np.nanmedian(a[mc]))
            s["per_class_median_signed"][cl[c]] = float(np.nanmedian(e[mc]))
            s["per_class_snr_median_abs"][cl[c]] = {}
            for sn in snrs:
                v = float(np.nanmedian(a[mc & (z == sn)]))
                s["per_class_snr_median_abs"][cl[c]][int(sn)] = v
                rows.append([key, cl[c], int(sn), f"{v:.5f}", f"{float(np.nanmedian(e[mc & (z == sn)])):.5f}"])
        for sn in snrs:
            s["per_snr_median_abs"][int(sn)] = float(np.nanmedian(a[z == sn]))
        summary[key] = s
        print(f"{key:>18s}  median |err| {s['median_abs']:.4f}  p90 {s['p90_abs']:.4f}  bias {s['median_signed']:+.4f}  "
              f"invalid {s['invalid_fraction'] * 100:.1f} %", flush=True)
    json.dump({"units": {"carrier": "|fc_hat - fc| / true 99% bandwidth", "bandwidth": "bw_hat / bw99 - 1",
                         "snr": "dB, estimate - Z_inband"}, "n_clips": int(len(te)), "estimators": summary},
              open(os.path.join(OUT, "wp6.json"), "w"), indent=1)
    with open(os.path.join(OUT, "wp6_class_snr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "class", "snr_db", "median_abs_error", "median_signed_error"])
        w.writerows(rows)


def sr(per_cell=40):
    import sr_bank, eval_symbol_rate as ev
    import consensus as cs
    from symbol_rate import ESTIMATORS
    m = dc.meta()
    cl = m["classes"]
    sym = ["OOK", "2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "PI4DQPSK", "8PSK", "16QAM"]
    fsk = ["2FSK", "4FSK", "GMSK"]
    idx = dc.subsample(m, 2, per_cell, seed=1, classes=sym)
    x = dc.load_x(idx)
    rs = 1 / m["sps"][idx].astype(float)
    names, R, P = sr_bank.compute(x)
    names = [str(n) for n in names]
    params = json.load(open(os.path.join(dc.ROOT, "characterization_extended", "results", "consensus_params.json")))
    est = {"inst_freq_s3 (baseline, BASELINE.md sec. 3)": np.array([ESTIMATORS["inst_freq_s3"](xi.astype(complex)) for xi in x]),
           "consensus, 12 estimators (cons_sr_dev)": ev.apply(params["cons_sr_dev"], names, R, P),
           "consensus, 1,024-sample linear (cons_lin_dev)": ev.apply(params["cons_lin_dev"], names, R, P)}
    y = np.array([cl[c] for c in m["Y"][idx]])
    fam = np.isin(y, fsk)
    cf = est["consensus, 12 estimators (cons_sr_dev)"].copy()
    cf[fam] = ev.apply(params["cons_fsk_dev"], names, R[fam], P[fam])
    est["class-family aware (FSK consensus on 2FSK/4FSK/GMSK)"] = cf
    z = m["Z"][idx]
    in_range = (m["sps"][idx] >= 2.5) & (m["sps"][idx] <= 25)
    snrs = sorted(np.unique(z).tolist())
    res, rows = {}, []
    for name, rate in est.items():
        ok = cs.correct(rate, rs)
        r = {"all": float(ok.mean()), "in_search_range_only": float(ok[in_range].mean()),
             "per_class": {c: float(ok[y == c].mean()) for c in sym},
             "per_class_in_range": {c: float(ok[(y == c) & in_range].mean()) if ((y == c) & in_range).any() else None for c in sym},
             "per_snr": {int(s): float(ok[z == s].mean()) for s in snrs},
             "per_class_snr": {c: {int(s): float(ok[(y == c) & (z == s)].mean()) for s in snrs} for c in sym},
             "median_rel_error": float(np.median(np.abs(rate / rs - 1))),
             "failures_at_2x_rate": float(np.mean(np.abs(rate[~ok] / rs[~ok] - 2) < 0.04)) if (~ok).any() else 0.0,
             "failures_at_half_rate": float(np.mean(np.abs(rate[~ok] / rs[~ok] - 0.5) < 0.01)) if (~ok).any() else 0.0}
        res[name] = r
        for c in sym:
            for s in snrs:
                rows.append([name, c, int(s), f"{r['per_class_snr'][c][int(s)]:.4f}"])
        print(f"{name:>58s}  all {r['all'] * 100:5.1f} %  (in range {r['in_search_range_only'] * 100:5.1f} %)  " +
              " ".join(f"{c} {r['per_class'][c] * 100:.0f}" for c in sym), flush=True)
    out_range = {c: float(np.mean(~in_range[y == c])) for c in sym}
    json.dump({"n_clips": int(len(idx)), "per_cell": per_cell, "criterion": "|Rs_hat - Rs| / Rs < 2 %",
               "fraction_outside_search_range_2.5_25_sps": out_range, "results": res},
              open(os.path.join(OUT, "symbol_rate.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "symbol_rate_bank.npz"), idx=idx, names=np.array(names), R=R, P=P)
    with open(os.path.join(OUT, "symbol_rate_class_snr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "class", "snr_db", "success_within_2pct"])
        w.writerows(rows)


if __name__ == "__main__":
    {"wp6": wp6, "sr": sr}[sys.argv[1]]()
