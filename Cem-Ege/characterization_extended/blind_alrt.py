"""Task 4 - blind likelihood classifier (practical_alrt.classify) with a better symbol-rate step.

Same frames, loader arguments and test split as classical_amc/practical_alrt.py:
  RadioML 2018.01A, classes BPSK..64QAM, SNR -10..20 dB, per_group 120, order hdf5, seed 5;
  train_test_split(idx, test_size=0.5, stratify=y*100+(snr+50), random_state=0) -> 4,800 test frames.
  TorchSig: D:\\data\\alrt_torchsig.npz with the same split rule.
Only the samples-per-symbol estimate passed to classify() changes:
  inst_freq_s3     1 / inst_freq_s3(x)                  (practical_alrt baseline, re-run)
  cons_sr_dev      consensus tuned on the TorchSig symbol-rate DEV set (4,096-sample frames)
  cons_lin_dev     consensus tuned on 1,024-sample linear-modulation DEV data:
                   TorchSig alrt DEV set (seed 1) + the RadioML TRAIN half (true sps = 8)
  sps_known        8 (RadioML) / true sps (TorchSig)
Consensus member sets and normalisations come from results/consensus_params.json
(written by eval_symbol_rate.py; tuned on dev only).

    python blind_alrt.py radioml|torchsig [variant ...]   -> results/blind_alrt_parts/
    python blind_alrt.py merge                             -> results/blind_alrt.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import common
import consensus as cs
import sr_bank
from practical_alrt import classify, load, CLASSES
from symbol_rate import ESTIMATORS


def split(y, snr):
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=0.5, stratify=y * 100 + (snr + 50), random_state=0)


def consensus_rates(R, P, names, params):
    idx = {n: i for i, n in enumerate(names)}
    cols = [idx[n] for n in params["members"]]
    mu = np.zeros(len(names))
    for n, m in zip(params["members"], params["mu"]):
        mu[idx[n]] = m
    return cs.predict(R, P, mu, cols, params["rule"])


def summarise(pred, yt, st):
    snrs = sorted(set(st.tolist()))
    acc = {int(s): float(np.mean(pred[st == s] == yt[st == s])) for s in snrs}
    return {"accuracy_all": float(np.mean(pred == yt)), "accuracy_snr_ge_10": float(np.mean(pred[st >= 10] == yt[st >= 10])),
            "snr_first_85": next((s for s in snrs if acc[s] >= 0.85), None), "accuracy_vs_snr": acc}


VARIANTS = ("inst_freq_s3", "cons_sr_dev", "cons_lin_dev", "sps_known")


def main(source, variants):
    seed = 5
    x, y, snr, sps_true = load(source, seed)
    tr, te = split(y, snr)
    true_sps = np.full(len(x), 8.0) if source == "radioml" else sps_true.astype(float)
    yt, st = y[te], snr[te]
    need_bank = any(v.startswith("cons") for v in variants)
    if need_bank:
        names, R, P, meta = sr_bank.get_bank("radioml" if source == "radioml" else "alrt_heldout")
        names = [str(n) for n in names]
        assert len(R) == len(x), "bank / frame count mismatch"
        for i in te[:3]:   # order check: recompute bank members for a few test frames
            m = sr_bank.members(x[i])
            assert np.allclose([m[n][0] for n in names], R[i]), "bank rows are not in loader order"
        with open(os.path.join(common.RESULTS, "consensus_params.json")) as f:
            params = json.load(f)
    os.makedirs(os.path.join(common.RESULTS, "blind_alrt_parts"), exist_ok=True)
    for k in variants:
        t0 = time.perf_counter()
        if k == "inst_freq_s3":
            s = np.full(len(x), np.nan)
            s[te] = [1 / ESTIMATORS["inst_freq_s3"](x[i].astype(complex)) for i in te]
        elif k == "sps_known":
            s = true_sps
        else:
            s = 1 / consensus_rates(R, P, names, params[k])
        rel = np.abs(s[te] - true_sps[te]) / true_sps[te]
        sr = {"within_2pct_all": float(np.mean(rel < 0.02)), "within_0.5pct_all": float(np.mean(rel < 0.005)),
              "within_2pct_snr_ge_10": float(np.mean(rel[st >= 10] < 0.02)),
              "median_sps_hat_snr_ge_10": float(np.median(s[te][st >= 10])),
              "within_2pct_vs_snr": {int(v): float(np.mean(rel[st == v] < 0.02)) for v in sorted(set(st.tolist()))}}
        a, h = zip(*[classify(x[i], float(s[i])) for i in te])
        part = {"symbol_rate_accuracy": sr, "alrt": summarise(np.array(a), yt, st), "hlrt": summarise(np.array(h), yt, st),
                "runtime_s": time.perf_counter() - t0, "n_test": int(len(te))}
        with open(os.path.join(common.RESULTS, "blind_alrt_parts", f"{source}_{k}.json"), "w") as f:
            json.dump(part, f, indent=1)
        print(f"{source} {k:>13s}: sps within 2 % {sr['within_2pct_all'] * 100:5.1f} % (within 0.5 % {sr['within_0.5pct_all'] * 100:5.1f} %, "
              f"SNR>=10 {sr['within_2pct_snr_ge_10'] * 100:5.1f} %, median {sr['median_sps_hat_snr_ge_10']:.2f}) | "
              f"ALRT all {part['alrt']['accuracy_all'] * 100:5.1f} % SNR>=10 {part['alrt']['accuracy_snr_ge_10'] * 100:5.1f} % "
              f"85 % at {part['alrt']['snr_first_85']} | HLRT all {part['hlrt']['accuracy_all'] * 100:5.1f} % "
              f"SNR>=10 {part['hlrt']['accuracy_snr_ge_10'] * 100:5.1f} % 85 % at {part['hlrt']['snr_first_85']}", flush=True)


def merge():
    out = {}
    d = os.path.join(common.RESULTS, "blind_alrt_parts")
    for source in ("radioml", "torchsig"):
        out[source] = {}
        for k in VARIANTS:
            p = os.path.join(d, f"{source}_{k}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[source][k] = json.load(f)
    with open(os.path.join(common.RESULTS, "blind_alrt.json"), "w") as f:
        json.dump(out, f, indent=1)
    for source, v in out.items():
        for k, r in v.items():
            print(f"{source:8s} {k:>13s} sps2% {r['symbol_rate_accuracy']['within_2pct_all'] * 100:5.1f} "
                  f"sps0.5% {r['symbol_rate_accuracy']['within_0.5pct_all'] * 100:5.1f} | ALRT {r['alrt']['accuracy_all'] * 100:5.1f} "
                  f"{r['alrt']['accuracy_snr_ge_10'] * 100:5.1f} {r['alrt']['snr_first_85']} | HLRT {r['hlrt']['accuracy_all'] * 100:5.1f} "
                  f"{r['hlrt']['accuracy_snr_ge_10'] * 100:5.1f} {r['hlrt']['snr_first_85']}")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        main(sys.argv[1], sys.argv[2:] or list(VARIANTS))
