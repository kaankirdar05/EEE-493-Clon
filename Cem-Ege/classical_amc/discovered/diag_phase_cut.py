"""POST-HOC diagnostic (designed after seeing AD fail on RadioML BPSK in bench_symbol.py):
re-run KS / CvM / AD with the phase component measured from a class-specific branch cut placed in
the constellation's widest angular gap (dist_classifiers.gof_distances(..., cut=True)), so that the
cluster at +-pi (BPSK, 8PSK) is not split between the two ends of the phase ECDF - where AD's
1/(F(1-F)) weight is largest. Same frames, split, tuning subsample and component selection rule
as bench_symbol.py.

    python diag_phase_cut.py --source radioml
    python diag_phase_cut.py --source torchsig
"""

from __future__ import annotations

import argparse
import time

import dcommon as dc
import numpy as np
from joblib import Parallel, delayed

import common
import dist_classifiers as dcl
import practical_alrt as pa
import symbol_classifiers as sc


def process(x, sps):
    import dcommon  # noqa: F401
    import dist_classifiers as dcl_
    import practical_alrt as pa_
    return dcl_.gof_distances(pa_.to_symbols(x, sps), cut=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["radioml", "torchsig"], required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    x, y, snr, sps_true = pa.load(args.source, 5)
    tr, te = common.alrt_split(y, snr)
    sps = (lambda i: 8.0) if args.source == "radioml" else (lambda i: float(sps_true[i]))
    tune_idx = np.random.default_rng(0).choice(tr, min(1500, len(tr)), replace=False)
    t0 = time.perf_counter()
    Dt = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i)) for i in tune_idx)
    De = Parallel(n_jobs=args.jobs)(delayed(process)(x[i], sps(i)) for i in te)
    out = {"source": args.source, "posthoc": True, "cut_offsets_rad": dcl.CUT, "tuning": {}, "chosen": {}, "results": {}}
    hi = snr[te] >= 10
    for t in dcl.TESTS:
        acc = {v: float(np.mean([dcl.decide(D, t, v) == y[i] for D, i in zip(Dt, tune_idx)])) for v in sc.KS_VARIANTS}
        best = max(acc, key=acc.get)
        out["tuning"][t], out["chosen"][t] = acc, best
        for v in sc.KS_VARIANTS:
            p = np.array([dcl.decide(D, t, v) for D in De])
            tag = "[chosen on train]" if v == best else "[not chosen]"
            r = common.summarize(p, y[te], snr[te])
            r["per_class_acc_snr_ge_10"] = {c: float(np.mean(p[hi & (y[te] == k)] == k)) for k, c in enumerate(sc.CLASSES)}
            out["results"][f"{t}_{v}_cut {tag}"] = r
            if v == best:
                print(common.fmt_row(f"{t}_{v}_cut", r), r["per_class_acc_snr_ge_10"], flush=True)
    out["wall_seconds"] = time.perf_counter() - t0
    dc.save_json(out, f"diag_phase_cut_{args.source}.json")


if __name__ == "__main__":
    main()
