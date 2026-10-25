"""Cramer-von Mises and Anderson-Darling classifiers on real frames, 5 digital classes
(BPSK, QPSK, 8PSK, 16QAM, 64QAM), symbol rate KNOWN - same frames, split and KS tuning subsample
as extended/bench_symbol.py (practical_alrt.load(source, 5), 50/50 split stratified by class x SNR,
1500 training frames drawn with default_rng(0) to choose the component set).

KS is recomputed here with the same code path; it must reproduce extended/results/bench_symbol_*.json
(consistency check). EM, features+HGB and ALRT/HLRT are quoted from that JSON (same test frames).

    python bench_symbol.py --source radioml
    python bench_symbol.py --source torchsig
"""

from __future__ import annotations

import argparse
import json
import os
import time

import dcommon as dc
import numpy as np
from joblib import Parallel, delayed

import common
import dist_classifiers as dcl
import practical_alrt as pa
import symbol_classifiers as sc


def process(x, sps):
    import dcommon  # noqa: F401  (worker sys.path)
    import dist_classifiers as dcl_
    import practical_alrt as pa_
    t0 = time.perf_counter()
    z = pa_.to_symbols(x, sps)
    t = {"to_symbols": time.perf_counter() - t0}
    D = []
    for name in dcl_.TESTS:
        t0 = time.perf_counter()
        D.append(dcl_.gof_distances(z, (name,))[0])
        t[name] = time.perf_counter() - t0
    return np.stack(D), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["radioml", "torchsig"], required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-tune", type=int, default=1500)
    args = ap.parse_args()

    x, y, snr, sps_true = pa.load(args.source, 5)
    tr, te = common.alrt_split(y, snr)
    sps = (lambda i: 8.0) if args.source == "radioml" else (lambda i: float(sps_true[i]))
    prev = json.load(open(os.path.join(common.RESULTS, f"bench_symbol_{args.source}.json")))
    out = {"source": args.source, "classes": sc.CLASSES, "n_train": int(len(tr)), "n_test": int(len(te))}

    rng = np.random.default_rng(0)                        # same tuning subsample as extended/bench_symbol.py
    tune_idx = rng.choice(tr, min(args.n_tune, len(tr)), replace=False)
    tune = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i)) for i in tune_idx)
    chosen, tuning = {}, {}
    for t in dcl.TESTS:
        acc = {v: float(np.mean([dcl.decide(r[0], t, v) == y[i] for r, i in zip(tune, tune_idx)])) for v in sc.KS_VARIANTS}
        tuning[t] = acc
        chosen[t] = max(acc, key=acc.get)
    out["tuning_train_accuracy"] = tuning
    out["chosen_components"] = chosen
    print("train accuracy by component set:", json.dumps(tuning, indent=1), "\nchosen:", chosen, flush=True)

    t0 = time.perf_counter()
    rt = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i)) for i in te)
    wall = time.perf_counter() - t0
    tm = {k: float(np.mean([r[1][k] for r in rt])) * 1e3 for k in ["to_symbols"] + dcl.TESTS}
    res = {}
    for t in dcl.TESTS:
        for v in sc.KS_VARIANTS:
            p = np.array([dcl.decide(r[0], t, v) for r in rt])
            tag = "[chosen on train]" if v == chosen[t] else "[not chosen; reference]"
            res[f"{t}_{v} {tag}"] = common.summarize(p, y[te], snr[te], tm["to_symbols"] + tm[t])
    # paired comparison of the chosen CvM / AD against the chosen KS on the same test frames
    pk = np.array([dcl.decide(r[0], "ks", chosen["ks"]) for r in rt]) == y[te]
    paired = {}
    for t in ("cvm", "ad"):
        pt = np.array([dcl.decide(r[0], t, chosen[t]) for r in rt]) == y[te]
        b, c = int(np.sum(pt & ~pk)), int(np.sum(~pt & pk))
        paired[f"{t} vs ks"] = {"only_new_correct": b, "only_ks_correct": c,
                                "mcnemar_z": (b - c) / np.sqrt(b + c) if b + c else 0.0}
    out["paired_vs_ks"] = paired
    # consistency: KS here vs extended JSON
    prev_ks = {k.split(" ")[0]: v["accuracy_all"] for k, v in prev["results"].items() if k.startswith("ks_")}
    now_ks = {k.split(" ")[0]: v["accuracy_all"] for k, v in res.items() if k.startswith("ks_")}
    out["ks_reproduces_extended"] = {"extended": prev_ks, "here": now_ks,
                                     "max_abs_diff": max(abs(prev_ks[k] - now_ks[k]) for k in prev_ks)}
    print("KS consistency:", out["ks_reproduces_extended"], flush=True)
    for k in ("em_gmm_ml", "features_hgb (base19)", "features_hgb (base19+ext17)",
              "alrt_sps_known (practical_alrt.py)", "hlrt_sps_known (practical_alrt.py)"):
        if k in prev["results"]:
            r = dict(prev["results"][k])
            r["accuracy_vs_snr"] = {int(s): a for s, a in r["accuracy_vs_snr"].items()}
            res[f"{k} [quoted from extended]"] = r
    for k, r in res.items():
        print(common.fmt_row(k[:34], r))
    tm["note"] = "per-frame CPU time inside a worker; machine shared with the feature extraction"
    tm["wall_seconds"] = wall
    out["timing_ms_per_frame"] = tm
    out["results"] = res
    dc.save_json(out, f"bench_symbol_{args.source}.json")
    np.savez_compressed(os.path.join(dc.RESULTS, f"symbol_stats_{args.source}.npz"),
                        D_test=np.stack([r[0] for r in rt]), D_tune=np.stack([r[0] for r in tune]),
                        te=te, tune_idx=tune_idx, y=y, snr=snr, tests=np.array(dcl.TESTS))
    show = {k: v for k, v in res.items() if "[not chosen" not in k}
    common.plot_overview(show, f"{args.source}: 5 digital classes, sps known - CvM / AD vs KS, EM (test split, {len(te)} frames)",
                         os.path.join(dc.RESULTS, f"bench_symbol_{args.source}.png"),
                         highlight={f"cvm_{chosen['cvm']} [chosen on train]", f"ad_{chosen['ad']} [chosen on train]"})


if __name__ == "__main__":
    main()
