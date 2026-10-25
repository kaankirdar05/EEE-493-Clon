"""Benchmark 3 + 4: KS goodness-of-fit and EM constellation classifiers on real frames,
5 digital classes (BPSK, QPSK, 8PSK, 16QAM, 64QAM), symbol rate KNOWN.

Frames and split are exactly those of practical_alrt.py (its load() and its 50/50 split
stratified by class x SNR), so the ALRT numbers in results/practical_alrt/*.json refer to the
same test frames and are quoted from there.

    python bench_symbol.py --source radioml     (RadioML 2018, per_group 120, SNR -10..20, seed 5)
    python bench_symbol.py --source torchsig    (D:\\data\\alrt_torchsig.npz)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import common
import numpy as np
from joblib import Parallel, delayed

import practical_alrt as pa
import symbol_classifiers as sc
from amc.features import extract_features
from cumulants_ext import extract_extended


def process(x, sps, do_em):
    t0 = time.perf_counter()
    z = pa.to_symbols(x, sps)
    t1 = time.perf_counter()
    D = sc.ks_distances(z)
    t2 = time.perf_counter()
    em_pred, lls = (sc.em_classify(z) if do_em else (-1, None))
    t3 = time.perf_counter()
    return D, em_pred, lls, t1 - t0, t2 - t1, t3 - t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["radioml", "torchsig"], required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-tune", type=int, default=1500, help="training frames used to pick the KS components")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y, snr, sps_true = pa.load(args.source, 5)
    tr, te = common.alrt_split(y, snr)
    sps = (lambda i: 8.0) if args.source == "radioml" else (lambda i: float(sps_true[i]))
    res, out = {}, {"source": args.source, "classes": sc.CLASSES, "n_train": int(len(tr)), "n_test": int(len(te))}

    # --- feature baselines on the same frames
    t0 = time.perf_counter(); Fb = extract_features(x); tb = (time.perf_counter() - t0) / len(x) * 1e3
    t0 = time.perf_counter(); Fe = extract_extended(x); te_ = (time.perf_counter() - t0) / len(x) * 1e3
    for name, F, fms in (("features_hgb (base19)", Fb, tb), ("features_hgb (base19+ext17)", np.hstack([Fb, Fe]), tb + te_)):
        clf = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0).fit(F[tr], y[tr])
        res[name] = common.summarize(clf.predict(F[te]), y[te], snr[te], fms)
        print(common.fmt_row(name, res[name]), flush=True)

    # --- KS: choose components on a training subsample
    rng = np.random.default_rng(0)
    tune_idx = rng.choice(tr, min(args.n_tune, len(tr)), replace=False)
    tune = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i), False) for i in tune_idx)
    tune_acc = {v: float(np.mean([sc.ks_decide(r[0], v) == y[i] for r, i in zip(tune, tune_idx)])) for v in sc.KS_VARIANTS}
    best = max(tune_acc, key=tune_acc.get)
    out["ks_tuning_train_accuracy"] = tune_acc
    out["ks_chosen_components"] = best
    print(f"KS components, train accuracy: {tune_acc} -> {best}", flush=True)

    # --- test frames: KS + EM
    t0 = time.perf_counter()
    rt = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i), True) for i in te)
    wall = time.perf_counter() - t0
    t_sym = float(np.mean([r[3] for r in rt])) * 1e3
    t_ks = float(np.mean([r[4] for r in rt])) * 1e3
    t_em = float(np.mean([r[5] for r in rt])) * 1e3
    for v in sc.KS_VARIANTS:
        p = np.array([sc.ks_decide(r[0], v) for r in rt])
        tag = " [chosen on train]" if v == best else " [not chosen; for reference]"
        res[f"ks_{v}{tag}"] = common.summarize(p, y[te], snr[te], t_sym + t_ks)
    p = np.array([r[1] for r in rt])
    res["em_gmm_ml"] = common.summarize(p, y[te], snr[te], t_sym + t_em)

    # --- existing ALRT / HLRT numbers on the same test frames
    pj = os.path.join(common.ROOT, "results", "practical_alrt", f"{args.source}.json")
    if os.path.exists(pj):
        prev = json.load(open(pj))
        for k in ("alrt_sps_known", "hlrt_sps_known"):
            if k in prev["results"]:
                r = dict(prev["results"][k])
                r["ms_per_frame"] = prev.get("ms_per_frame_likelihood")
                r["accuracy_vs_snr"] = {int(s): a for s, a in r["accuracy_vs_snr"].items()}
                res[f"{k} (practical_alrt.py)"] = r
        out["features_hgb_in_practical_alrt_json"] = prev["results"].get("features_hgb", {}).get("accuracy_all")
    for k, r in res.items():
        print(common.fmt_row(k, r))
    print(f"  symbols {t_sym:.1f} ms, KS {t_ks:.1f} ms, EM {t_em:.1f} ms per frame (worker time, {args.jobs} parallel jobs; "
          f"wall {wall:.0f} s)")
    out["timing_ms_per_frame"] = {"to_symbols": t_sym, "ks": t_ks, "em": t_em,
                                  "note": "per-frame CPU time inside a worker; the machine was shared with other jobs"}
    out["results"] = res
    common.save_json(out, f"bench_symbol_{args.source}.json")
    common.plot_overview(res, f"{args.source}: 5 digital classes, sps known (test split, {len(te)} frames)",
                         os.path.join(common.RESULTS, f"bench_symbol_{args.source}.png"),
                         highlight={"em_gmm_ml", f"ks_{best} [chosen on train]"})


if __name__ == "__main__":
    main()
