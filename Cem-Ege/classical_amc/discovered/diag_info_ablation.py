"""Which part of info24 carries the 24-class gain? (diagnostic, test split as in bench_features.py)

Sub-groups appended to base19+wav24: histogram entropies (8), permutation entropies (6),
SampEn/ApEn (4), RQA (6), and the cheap part (histogram + permutation, 14) without the O(N^2) ones.
Also lists the per-class SNR>=10 accuracy change of base19+wav24+info24 vs base19+wav24.

    python diag_info_ablation.py --set 24
"""

from __future__ import annotations

import argparse
import os

import dcommon as dc
import numpy as np

import common
import infotheory as it
from bench_features import make_hgb

GROUPS = {"hist8": it.HIST_NAMES, "perm6": it.PE_NAMES, "sampen_apen4": it.SE_NAMES, "rqa6": it.RQA_NAMES,
          "hist8+perm6 (cheap)": it.HIST_NAMES + it.PE_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["7", "24"], default="24")
    args = ap.parse_args()
    old = np.load(os.path.join(common.RESULTS, f"features_{args.set}.npz"))
    new = np.load(os.path.join(dc.RESULTS, f"features_new_{args.set}.npz"))
    names = [str(n) for n in new["info_names"]]
    y, snr = old["y"].astype(int), old["snr"].astype(int)
    classes = [str(c) for c in old["classes"]]
    tr, te = common.standard_split(y, snr)
    base = np.hstack([old["base"], old["wav"]])
    hi = snr[te] >= 10
    ref = np.load(os.path.join(dc.RESULTS, f"preds_{args.set}", "base19_wav24.npz"))["pred"] == y[te]
    out = {"reference": "hgb | base19+wav24", "ref_all": float(ref.mean()), "ref_snr_ge_10": float(ref[hi].mean()),
           "groups": {}}
    for g, cols in GROUPS.items():
        idx = [names.index(c) for c in cols]
        X = np.hstack([base, new["info"][:, idx]])
        p = make_hgb().fit(X[tr], y[tr]).predict(X[te])
        ok = p == y[te]
        b, c = int(np.sum(ok & ~ref)), int(np.sum(~ok & ref))
        out["groups"][g] = {"n": len(cols), "acc_all": float(ok.mean()), "acc_snr_ge_10": float(ok[hi].mean()),
                            "delta_all": float(ok.mean() - ref.mean()), "delta_snr_ge_10": float(ok[hi].mean() - ref[hi].mean()),
                            "mcnemar_z": (b - c) / np.sqrt(b + c)}
        print(g, {k: round(v, 4) if isinstance(v, float) else v for k, v in out["groups"][g].items()}, flush=True)
    full = np.load(os.path.join(dc.RESULTS, f"preds_{args.set}", "base19_wav24_info24.npz"))["pred"]
    pref = np.load(os.path.join(dc.RESULTS, f"preds_{args.set}", "base19_wav24.npz"))["pred"]
    out["per_class_delta_snr_ge_10 (info24 full)"] = {
        c: float(np.mean(full[hi & (y[te] == k)] == k) - np.mean(pref[hi & (y[te] == k)] == k)) for k, c in enumerate(classes)}
    print(out["per_class_delta_snr_ge_10 (info24 full)"])
    dc.save_json(out, f"diag_info_ablation_{args.set}.json")


if __name__ == "__main__":
    main()
