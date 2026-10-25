"""WP6 benchmark: classical carrier / bandwidth / SNR estimators with oracle (truth) boxes.

    python run_wp6.py --dataset clean
    python run_wp6.py --dataset impaired
    python run_wp6.py --dataset impaired --dev-only --out <dir>    # tuning on scenes 0-99 only

Scenes 0-99 are the development split (tuning, choice of the "best" estimator per quantity);
all reported numbers are on scenes 100-299. Outputs in results/<dataset>/:
    metrics.json            test-split metrics (overall, per SNR bin, per family, per envelope group)
    per_emitter.csv         test split, one row per emitter, truth + every estimate
    per_emitter_dev.csv     development split, same columns
    error_vs_snr.png, snr_scatter.png, psd_*.png (sanity plots, dev scenes)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from wp6 import metrics as M
from wp6.data import FAMILIES, envelope_group, family, load, noise_map
from wp6.estimators import characterize
from wp6.plots import error_vs_snr, psd_example, snr_scatter

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = (100, 300)
DEV = (0, 100)


def run_scenes(dataset, scenes, cfg, sanity_dir=None, n_sanity=0):
    f, boxes, fs = load(dataset)
    iq_ds = f["iq"]
    n_iq = iq_ds.shape[1]
    rows = []
    sanity_left = n_sanity
    fft = 512
    centers = (np.arange(fft) - fft // 2) * fs / fft
    for sc in scenes:
        iq = iq_ds[sc]
        t0 = time.perf_counter()
        nmap = noise_map(iq, fft)
        t_map = time.perf_counter() - t0
        bs = boxes[boxes["scene"] == sc]
        for b in bs:
            s0 = int(np.floor(b["t_start"] * n_iq))
            s1 = max(int(np.ceil(b["t_stop"] * n_iq)), s0 + 64)
            seg = iq[s0:min(s1, n_iq)]
            tf0 = int(np.floor(b["t_start"] * nmap.shape[0]))
            tf1 = max(int(np.ceil(b["t_stop"] * nmap.shape[0])), tf0 + 1)
            nf = nmap[tf0:tf1].mean(0)

            def noise_fn(fr, nf=nf):
                return np.interp(fr, centers, nf)

            t1 = time.perf_counter()
            res = characterize(seg, fs, float(b["f_low"]), float(b["f_high"]), noise_fn, cfg)
            dt = time.perf_counter() - t1
            name = b["class_name"].decode()
            row = dict(scene=int(sc), class_name=name, family=family(name), env_group=envelope_group(name),
                       snr_true=float(b["snr_db"]), fc_true=0.5 * (b["f_low"] + b["f_high"]),
                       bw_true=float(b["f_high"] - b["f_low"]), f_low=float(b["f_low"]), f_high=float(b["f_high"]),
                       t_start=float(b["t_start"]), t_stop=float(b["t_stop"]),
                       band_edge=bool(b["f_low"] <= -fs / 2 + 1 or b["f_high"] >= fs / 2 - 2),
                       runtime_ms=1e3 * dt, noise_map_ms_per_emitter=1e3 * t_map / len(bs))
            row.update({k: v for k, v in res.items() if not k.startswith("_")})
            rows.append(row)
            if sanity_dir and sanity_left > 0 and sc % 7 == 0:
                psd_example(res, row, os.path.join(sanity_dir, f"psd_s{sc:03d}_{name}.png"))
                sanity_left -= 1
    f.close()
    return rows


def to_cols(rows):
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return cols


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["clean", "impaired"], required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--dev-only", action="store_true", help="run only scenes 0-99 (tuning)")
    p.add_argument("--noise", default="oob_lin", choices=["map", "oob", "oob_lin", "hybrid"])
    p.add_argument("--n-across", type=int, default=64)
    p.add_argument("--smooth-frac", type=float, default=1 / 16)
    p.add_argument("--sanity", type=int, default=8, help="number of PSD sanity plots (dev scenes)")
    a = p.parse_args()
    out = a.out or os.path.join(HERE, "results", a.dataset)
    os.makedirs(out, exist_ok=True)
    cfg = dict(noise=a.noise, n_across=a.n_across, smooth_frac=a.smooth_frac)

    dev_rows = run_scenes(a.dataset, range(*DEV), cfg, sanity_dir=out, n_sanity=a.sanity)
    write_csv(dev_rows, os.path.join(out, "per_emitter_dev.csv"))
    dev = to_cols(dev_rows)
    sdev = M.summarize(dev)
    # best estimator per quantity is chosen on the DEV split
    choice = {"carrier": M.best(sdev, M.CARRIER), "bandwidth": M.best(sdev, M.BANDWIDTH),
              "snr": M.best(sdev, M.SNR)}
    print("dev choice:", choice)
    if a.dev_only:
        print_table(sdev, "DEV")
        with open(os.path.join(out, "metrics_dev.json"), "w") as fh:
            json.dump({"config": cfg, "summary": sdev}, fh, indent=1)
        return

    test_rows = run_scenes(a.dataset, range(*TEST), cfg)
    write_csv(test_rows, os.path.join(out, "per_emitter.csv"))
    t = to_cols(test_rows)
    allsel = np.ones(len(test_rows), bool)
    s = M.summarize(t)
    res = {
        "dataset": a.dataset, "split": "scenes 100-299", "n_emitters": len(test_rows), "config": cfg,
        "best_estimator_chosen_on_dev": choice,
        "summary": s,
        "summary_excluding_band_edge_emitters": M.summarize(t, ~t["band_edge"]),
        "per_family": M.by_group(t, "family", FAMILIES, allsel),
        "family_counts": {g: int((t["family"] == g).sum()) for g in FAMILIES},
        "per_envelope_group": M.by_group(t, "env_group", ["ce_true", "ce_psk", "other"], allsel),
        "m2m4_invalid_fraction": float(1 - t["m2m4_valid"].mean()),
        "runtime_ms_per_emitter": {"median": float(np.median(t["runtime_ms"])),
                                   "p90": float(np.percentile(t["runtime_ms"], 90)),
                                   "noise_map_ms_per_emitter_median": float(np.median(t["noise_map_ms_per_emitter"]))},
        "n_tone_excluded_from_bw_metrics": int((t["class_name"] == "tone").sum()),
    }
    # constant-envelope (task definition: psk + fsk family + fm) vs others for M2M4
    ce = np.isin(t["env_group"], ["ce_true", "ce_psk"])
    res["m2m4_constant_envelope_vs_other"] = {
        "constant_envelope": M.summarize(t, ce)["snr_m2m4"]["overall"],
        "other": M.summarize(t, ~ce)["snr_m2m4"]["overall"],
    }
    # M2M4 vs the PSD estimator on the SAME band (99 % OBW): isolates the moment method's own error
    diff = t["snr_m2m4"] - t["snr_psd_obw"]
    res["m2m4_minus_psd_obw_median_db"] = {g: float(np.median(diff[t["env_group"] == g]))
                                           for g in ["ce_true", "ce_psk", "other"]}
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    error_vs_snr(s, os.path.join(out, "error_vs_snr.png"), f"WP6 {a.dataset}, scenes 100-299, oracle boxes")
    snr_scatter(t, os.path.join(out, "snr_scatter.png"), f"WP6 {a.dataset}: estimated vs TorchSig snr_db")
    print_table(s, f"TEST {a.dataset}")
    print("runtime", res["runtime_ms_per_emitter"], "m2m4 invalid", res["m2m4_invalid_fraction"])


def print_table(s, tag):
    print(f"--- {tag} ---")
    for k in M.CARRIER:
        o, ob = s[k]["overall"], s[k + "/bw"]["overall"]
        print(f"{k:18s} med {o['median_abs']:8.2f} kHz p90 {o['p90_abs']:8.2f} | frac bw med {ob['median_abs']:.3f} p90 {ob['p90_abs']:.3f}")
    for k in M.BANDWIDTH:
        o = s[k]["overall"]
        print(f"{k:18s} med|rel| {o['median_abs']:.3f} p90 {o['p90_abs']:.3f} bias {o['median_signed']:+.3f}")
    for k in M.SNR:
        o = s[k]["overall"]
        print(f"{k:18s} med|err| {o['median_abs']:.2f} dB p90 {o['p90_abs']:.2f} bias {o['median_signed']:+.2f}")


if __name__ == "__main__":
    main()
