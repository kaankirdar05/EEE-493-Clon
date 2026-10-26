"""Practical likelihood classifier (ALRT / HLRT, classical_amc/practical_alrt.classify) on Dataset 1 v3.

Classes with a known constellation in Dataset 1: BPSK, QPSK, 8PSK, 16QAM (candidate set of practical_alrt:
BPSK, QPSK, 8PSK, 16QAM, 64QAM; a 64QAM decision counts as wrong). PI4DQPSK has no hypothesis in the
classifier; its test clips are run too and we report what they are called.

Clips: 20 per class and SNR, drawn from the test-split subsample of run_characterization.py sr (40 per
class and SNR), so the symbol-rate bank computed there is reused. Variants of the samples-per-symbol input, as in
characterization_extended/blind_alrt.py:
    sps_known     the dataset's sps label
    inst_freq_s3  1 / inst_freq_s3(x)                       (practical_alrt.py, fully blind)
    cons_sr_dev   1 / 12-estimator consensus (frozen params)
    cons_lin_dev  1 / 1,024-sample linear consensus (frozen params)
Comparator: 19 features + boosted trees trained on the Dataset-1 train split of the same 4 classes.

    python run_likelihood.py      (after run_characterization.py sr)   -> results/identification/likelihood.json
"""
from __future__ import annotations

import csv, json, os, time
import numpy as np
import ds1common as dc

OUT = os.path.join(dc.RESULTS, "identification")
PER_CELL = 20


def main():
    from practical_alrt import classify, CLASSES as CAND
    from symbol_rate import ESTIMATORS
    import eval_symbol_rate as ev
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = dc.meta()
    cl = m["classes"]
    four = ["BPSK", "QPSK", "8PSK", "16QAM"]
    b = np.load(os.path.join(dc.RESULTS, "characterization", "symbol_rate_bank.npz"))
    names = [str(n) for n in b["names"]]
    keep = np.isin(m["Y"][b["idx"]], [cl.index(c) for c in four + ["PI4DQPSK"]])
    # ~150 ms per clip and variant on Dataset 1 (up to 340 symbols per clip): keep PER_CELL of the 40 clips per cell
    rng = np.random.default_rng(0)
    cell = m["Y"][b["idx"]] * 100 + m["Z"][b["idx"]].astype(int)
    for c in np.unique(cell[keep]):
        ii = np.flatnonzero(keep & (cell == c))
        keep[rng.choice(ii, len(ii) - PER_CELL, replace=False)] = False
    idx, R, P = b["idx"][keep], b["R"][keep], b["P"][keep]
    x = dc.load_x(idx)
    params = json.load(open(os.path.join(dc.ROOT, "characterization_extended", "results", "consensus_params.json")))
    sps = {"sps_known": m["sps"][idx].astype(float),
           "inst_freq_s3": np.array([1 / ESTIMATORS["inst_freq_s3"](xi.astype(complex)) for xi in x]),
           "cons_sr_dev": 1 / ev.apply(params["cons_sr_dev"], names, R, P),
           "cons_lin_dev": 1 / ev.apply(params["cons_lin_dev"], names, R, P)}
    y = m["Y"][idx]
    z = m["Z"][idx]
    four_mask = np.isin(y, [cl.index(c) for c in four])
    pi4 = y == cl.index("PI4DQPSK")
    snrs = sorted(np.unique(z).tolist())

    def summ(pred_names):
        ok = (np.array([cl.index(p) if p in cl else -1 for p in pred_names]) == y) & four_mask
        acc = {int(s): float(ok[four_mask & (z == s)].mean()) for s in snrs}
        return {"accuracy_4_classes_all_snr": float(ok[four_mask].mean()),
                "accuracy_snr_ge_10": float(ok[four_mask & (z >= 10)].mean()),
                "snr_first_85": next((s for s in snrs if acc[s] >= 0.85), None),
                "sustained_snr_85": next((s for s in snrs if all(acc[t] >= 0.85 for t in snrs if t >= s)), None),
                "accuracy_vs_snr": acc,
                "per_class_vs_snr": {c: {int(s): float(ok[(y == cl.index(c)) & (z == s)].mean()) for s in snrs} for c in four},
                "per_class_snr_ge_10": {c: float(ok[(y == cl.index(c)) & (z >= 10)].mean()) for c in four},
                "pi4dqpsk_called": {k: float(np.mean(np.array(pred_names)[pi4 & (z >= 10)] == k)) for k in CAND}}

    res = {}
    for v, s in sps.items():
        rel = np.abs(s - m["sps"][idx]) / m["sps"][idx]
        t0 = time.perf_counter()
        a, h = zip(*[classify(xi, float(si)) for xi, si in zip(x, s)])
        ms = (time.perf_counter() - t0) / len(x) * 1e3
        res[f"ALRT, sps {v}"] = summ([CAND[k] for k in a]) | {"ms_per_clip": ms, "sps_within_2pct": float(np.mean(rel[four_mask] < 0.02))}
        res[f"HLRT, sps {v}"] = summ([CAND[k] for k in h]) | {"sps_within_2pct": float(np.mean(rel[four_mask] < 0.02))}
        for k in (f"ALRT, sps {v}", f"HLRT, sps {v}"):
            r = res[k]
            print(f"{k:>24s}: 4 classes {r['accuracy_4_classes_all_snr'] * 100:5.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:5.1f} %  "
                  f"85 % first at {r['snr_first_85']}  sps ok {r['sps_within_2pct'] * 100:.0f} %  PI4DQPSK->{max(r['pi4dqpsk_called'], key=r['pi4dqpsk_called'].get)}", flush=True)
    # comparator: features + boosted trees on the same 4 classes
    F = np.load(os.path.join(dc.RESULTS, "features_19.npz"))["F"]
    tr = np.flatnonzero((m["split"] == 0) & np.isin(m["Y"], [cl.index(c) for c in four]))
    clf = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0).fit(F[tr], m["Y"][tr])
    p = clf.predict(F[idx])
    res["19 features + boosted trees (4 classes, trained on ds1 train)"] = summ([cl[k] for k in p])
    r = res["19 features + boosted trees (4 classes, trained on ds1 train)"]
    print(f"{'features + HGB':>24s}: 4 classes {r['accuracy_4_classes_all_snr'] * 100:5.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:5.1f} %  85 % first at {r['snr_first_85']}")
    json.dump({"n_clips_4_classes": int(four_mask.sum()), "n_clips_pi4dqpsk": int(pi4.sum()), "results": res},
              open(os.path.join(OUT, "likelihood.json"), "w"), indent=1)
    with open(os.path.join(OUT, "likelihood_class_snr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "class", "snr_db", "accuracy"])
        for k, r in res.items():
            for c, d in r["per_class_vs_snr"].items():
                for s, a in d.items():
                    w.writerow([k, c, s, f"{a:.4f}"])


if __name__ == "__main__":
    main()
