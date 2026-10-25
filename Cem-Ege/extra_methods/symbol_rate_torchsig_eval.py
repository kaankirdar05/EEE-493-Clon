"""Run the symbol-rate estimator bank (classical_amc/symbol_rate.py) on TorchSig-generated
frames with known symbol rates (gen_symbol_rate_torchsig.py). Success = within 2 %.

    python symbol_rate_torchsig_eval.py   -> results/symbol_rate_torchsig.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical_amc"))
from symbol_rate import ESTIMATORS  # noqa: E402


def main():
    d = np.load(r"D:\data\symbol_rate_torchsig.npz")
    x, rs, cls, snr, pulse = d["x"], d["rs"], d["cls"], d["snr"], d["pulse"]
    est = {k: np.array([f(xi.astype(complex)) for xi in x]) for k, f in ESTIMATORS.items()}
    ok = {k: np.abs(v - rs) / rs < 0.02 for k, v in est.items()}
    classes = list(dict.fromkeys(cls.tolist()))
    snrs = sorted(set(snr.tolist()))
    summary = {}
    print(f"{len(x)} TorchSig frames; success = within 2 % of true symbol rate")
    print(f"{'estimator':>13s} all | " + " ".join(f"{c:>6s}" for c in classes) + " | " + " ".join(f"{s:>3d}dB" for s in snrs))
    for k in ESTIMATORS:
        pc = {c: float(ok[k][cls == c].mean()) for c in classes}
        ps = {int(s): float(ok[k][snr == s].mean()) for s in snrs}
        pp = {p: float(ok[k][pulse == p].mean()) for p in ("srrc", "rectangular")}
        summary[k] = {"success_all": float(ok[k].mean()), "per_class": pc, "per_snr": ps, "constellation_by_pulse": pp}
        print(f"{k:>13s} {ok[k].mean() * 100:3.0f} | " + " ".join(f"{pc[c] * 100:6.0f}" for c in classes) + " | " +
              " ".join(f"{ps[s] * 100:5.0f}" for s in snrs))
    best = {c: max(ESTIMATORS, key=lambda k: summary[k]["per_class"][c]) for c in classes}
    known = float(np.mean([ok[best[c]][i] for i, c in enumerate(cls)]))
    print("best per class:", best, f"-> {known * 100:.0f} % with the class known")
    for k in ("envelope", "inst_freq"):
        print(f"  {k}: SRRC-shaped {summary[k]['constellation_by_pulse']['srrc'] * 100:.0f} %, rectangular {summary[k]['constellation_by_pulse']['rectangular'] * 100:.0f} %")
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "symbol_rate_torchsig.json"), "w") as f:
        json.dump({"summary": summary, "best_per_class": best, "success_with_class_known": known, "n": int(len(x))}, f, indent=2)


if __name__ == "__main__":
    main()
