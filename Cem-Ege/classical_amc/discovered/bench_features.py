"""HGB on the new feature families, alone and appended to base19 and base19+wav24,
RadioML 2018.01A 7 and 24 classes - the same frames and the standard split as the cached baseline
(train_test_split(test_size=0.25, stratify=class x SNR, random_state=0)), same HGB settings as
extended/bench_features.py. No subsampling.

    python bench_features.py --set 7
    python bench_features.py --set 24

For every "X + family" run a paired McNemar comparison against "X" on the same test frames is
reported (z = (b - c) / sqrt(b + c), b = frames only the larger model gets right).
"""

from __future__ import annotations

import argparse
import os
import time

import dcommon as dc
import numpy as np

import common
from bench_features import make_hgb       # extended/bench_features.py (same HGB settings)

FAMS = {"bis30": "bis", "fd22": "fd", "info24": "info", "st23": "st"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["7", "24"], required=True)
    ap.add_argument("--shard", default=None, help="i/n: only fit runs with index %% n == i (results cached in "
                                                    "results/preds_{set}/); a final call without --shard assembles")
    args = ap.parse_args()
    old = np.load(os.path.join(common.RESULTS, f"features_{args.set}.npz"))
    new = np.load(os.path.join(dc.RESULTS, f"features_new_{args.set}.npz"))
    assert np.array_equal(old["y"], new["y"]) and np.array_equal(old["snr"], new["snr"]), "frames differ"
    assert bool(new["labels_identical"]) and float(new["base_max_abs_diff"]) == 0.0
    y, snr, classes = old["y"].astype(int), old["snr"].astype(int), [str(c) for c in old["classes"]]
    ms = {"base19": float(old["ms_base"]), "wav24": float(old["ms_wav"])}
    blocks = {"base19": old["base"], "wav24": old["wav"]}
    for name, key in FAMS.items():
        blocks[name] = new[key]
        ms[name] = float(new[f"ms_{key}"])
    allnew = list(FAMS)
    tr, te = common.standard_split(y, snr)
    print(f"RadioML {args.set} classes: {len(y)} frames, train {len(tr)}, test {len(te)}", flush=True)

    runs = [("base19", ["base19"], None), ("base19+wav24", ["base19", "wav24"], None)]
    for f in allnew:
        runs += [(f, [f], None), (f"base19+{f}", ["base19", f], "base19"),
                 (f"base19+wav24+{f}", ["base19", "wav24", f], "base19+wav24")]
    runs += [("all4new", allnew, None), ("base19+all4new", ["base19"] + allnew, "base19"),
             ("base19+wav24+all4new", ["base19", "wav24"] + allnew, "base19+wav24")]

    out = {"set": args.set, "classes": classes, "n_train": int(len(tr)), "n_test": int(len(te)),
           "feature_ms_per_frame": ms, "frames_identical_to_cache": True,
           "extraction_note": "ms/frame = single-threaded CPU time inside 16 parallel worker processes on a shared machine",
           "results": {}, "paired_vs_reference": {}}
    correct = {}
    # resumable: each finished run's test predictions and summary are cached
    cache_dir = os.path.join(dc.RESULTS, f"preds_{args.set}")
    os.makedirs(cache_dir, exist_ok=True)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        for j, (name, parts, ref) in enumerate(runs):
            cpath = os.path.join(cache_dir, name.replace("+", "_") + ".npz")
            if j % n != i or os.path.exists(cpath):
                continue
            X = np.hstack([blocks[p] for p in parts])
            m = make_hgb()
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = m.predict(X[te])
            pms = (time.perf_counter() - t0) / len(te) * 1e3
            np.savez(cpath, pred=p, pms=pms, fit_s=fit_s, n_iter=int(m.n_iter_))
            print(f"  shard {args.shard}: {name} acc {np.mean(p == y[te]) * 100:.2f} %, fit {fit_s:.0f} s", flush=True)
        return
    for name, parts, ref in runs:
        X = np.hstack([blocks[p] for p in parts])
        fms = sum(ms[p] for p in parts)
        hi = snr[te] >= 10
        cpath = os.path.join(cache_dir, name.replace("+", "_") + ".npz")
        if os.path.exists(cpath):
            c = np.load(cpath)
            p, pms, fit_s, n_iter = c["pred"], float(c["pms"]), float(c["fit_s"]), int(c["n_iter"])
        else:
            m = make_hgb()
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = m.predict(X[te])
            pms = (time.perf_counter() - t0) / len(te) * 1e3
            n_iter = int(m.n_iter_)
            np.savez(cpath, pred=p, pms=pms, fit_s=fit_s, n_iter=n_iter)
        per_cls = {c: float(np.mean(p[hi & (y[te] == k)] == k)) for k, c in enumerate(classes)}
        out["results"][f"hgb | {name}"] = common.summarize(
            p, y[te], snr[te], fms + pms, {"n_features": int(X.shape[1]), "fit_seconds": fit_s,
                                           "n_iter": n_iter, "per_class_acc_snr_ge_10": per_cls})
        correct[name] = p == y[te]
        line = common.fmt_row(f"hgb | {name}", out["results"][f"hgb | {name}"])
        if ref:
            b = int(np.sum(correct[name] & ~correct[ref]))
            c = int(np.sum(~correct[name] & correct[ref]))
            d_all = float(correct[name].mean() - correct[ref].mean())
            d_hi = float(correct[name][hi].mean() - correct[ref][hi].mean())
            out["paired_vs_reference"][name] = {"reference": ref, "delta_all": d_all, "delta_snr_ge_10": d_hi,
                                                "only_new_correct": b, "only_ref_correct": c,
                                                "mcnemar_z": (b - c) / np.sqrt(b + c) if b + c else 0.0}
            line += f"   delta vs {ref}: {d_all * 100:+.2f} pts (McNemar z {out['paired_vs_reference'][name]['mcnemar_z']:+.1f})"
        print(line, flush=True)
        dc.save_json(out, f"bench_features_{args.set}.json")

    sel = ["base19", "base19+wav24", "all4new", "base19+all4new", "base19+wav24+all4new"] + allnew
    common.plot_overview({f"hgb | {k}": out["results"][f"hgb | {k}"] for k in sel},
                         f"RadioML 2018 {args.set} classes - new feature families + HGB (test split)",
                         os.path.join(dc.RESULTS, f"bench_features_{args.set}.png"),
                         highlight={"hgb | base19", "hgb | base19+wav24", "hgb | base19+wav24+all4new"})
    print("saved", dc.save_json(out, f"bench_features_{args.set}.json"))


if __name__ == "__main__":
    main()
