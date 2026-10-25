"""Task 3 - tune the symbol-rate consensus on DEV, report on the held-out TorchSig frames.

DEV      characterization_extended/data/symbol_rate_torchsig_dev.npz (TorchSig, seed 1)
HELD-OUT D:\\data\\symbol_rate_torchsig.npz (seed 0) - never used for any choice below.
Also fits the 1,024-sample linear-modulation consensus used by blind_alrt.py
(DEV = TorchSig alrt dev set, seed 1, + RadioML TRAIN half of practical_alrt's split).

Writes results/consensus_params.json, results/symbol_rate_extended.json, results/symbol_rate_extended.png

    python eval_symbol_rate.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import common
import consensus as cs
import sr_bank

FSK = ["2fsk", "4fsk", "2gfsk", "2msk", "2gmsk"]


def tune(names, R, P, rs, label):
    mu = cs.fit_mu(R, P, rs)
    best = None
    for rule in ("max", "vote"):
        print(f"[{label}] greedy selection, rule = {rule}")
        ch, acc = cs.greedy(R, P, rs, mu, list(range(len(names))), rule, verbose=True, names=names)
        if best is None or acc > best[2] + 1e-4:
            best = (rule, ch, acc)
    rule, ch, acc = best
    print(f"[{label}] chosen rule {rule}, {len(ch)} members, DEV accuracy {acc * 100:.1f} %")
    return {"rule": rule, "members": [str(names[c]) for c in ch], "mu": [float(mu[c]) for c in ch], "dev_accuracy": float(acc)}


def apply(params, names, R, P):
    idx = {n: i for i, n in enumerate(names)}
    cols = [idx[n] for n in params["members"]]
    mu = np.zeros(len(names))
    for n, m in zip(params["members"], params["mu"]):
        mu[idx[n]] = m
    return cs.predict(R, P, mu, cols, params["rule"])


def table(rate, rs, cls, snr):
    ok = cs.correct(rate, rs)
    classes = list(dict.fromkeys(cls.tolist()))
    lin = ~np.isin(cls, FSK)
    rel = rate / rs
    fail = ~ok
    return {"all": float(ok.mean()), "psk_qam": float(ok[lin].mean()), "fsk_family": float(ok[~lin].mean()),
            "per_class": {c: float(ok[cls == c].mean()) for c in classes},
            "per_snr": {int(s): float(ok[snr == s].mean()) for s in sorted(set(snr.tolist()))},
            "fsk_per_snr": {int(s): float(ok[(snr == s) & ~lin].mean()) for s in sorted(set(snr.tolist()))},
            "failures_harmonic_x2": float(np.mean(np.abs(rel[fail] - 2) < 0.04)) if fail.any() else 0.0,
            "failures_subharmonic_x0.5": float(np.mean(np.abs(rel[fail] - 0.5) < 0.01)) if fail.any() else 0.0,
            "median_rel_error": float(np.median(np.abs(rel - 1)))}


def main():
    names, R, P, meta = sr_bank.get_bank("dev")
    names = [str(n) for n in names]
    params = {"cons_sr_dev": tune(names, R, P, meta["rs"], "TorchSig symbol-rate DEV")}
    # best single member on DEV, for reference
    acc = [cs.correct(R[:, k], meta["rs"]).mean() for k in range(len(names))]
    best_single = names[int(np.argmax(acc))]
    fsk = np.isin(meta["cls"], FSK)
    acc_fsk = [cs.correct(R[fsk, k], meta["rs"][fsk]).mean() for k in range(len(names))]
    best_single_fsk = names[int(np.argmax(acc_fsk))]
    params["cons_fsk_dev"] = tune(names, R[fsk], P[fsk], meta["rs"][fsk], "FSK-only DEV (class family known)")

    # 1,024-sample linear modulation consensus for blind_alrt (dev = TorchSig alrt dev + RadioML train half)
    from blind_alrt import split
    n2, R2, P2, m2 = sr_bank.get_bank("alrt_dev")
    n3, R3, P3, m3 = sr_bank.get_bank("radioml")
    assert [str(a) for a in n2] == names and [str(a) for a in n3] == names
    tr, _ = split(m3["y"], m3["snr"])
    RL = np.vstack([R2, R3[tr]]); PL = np.vstack([P2, P3[tr]]); rsL = np.concatenate([m2["rs"], m3["rs"][tr]])
    params["cons_lin_dev"] = tune(names, RL, PL, rsL, "linear 1024-sample DEV (TorchSig alrt dev + RadioML train)")
    with open(os.path.join(common.RESULTS, "consensus_params.json"), "w") as f:
        json.dump(params, f, indent=1)

    # ---------------- held-out report
    hn, HR, HP, hm = sr_bank.get_bank("heldout")
    assert [str(a) for a in hn] == names
    rs, cls, snr = hm["rs"], hm["cls"], hm["snr"]
    col = {n: i for i, n in enumerate(names)}
    rows = {"inst_freq_s3 (baseline)": HR[:, col["ifr3.g"]],
            f"best single DEV member ({best_single})": HR[:, col[best_single]],
            f"best single DEV member on FSK ({best_single_fsk})": HR[:, col[best_single_fsk]],
            "consensus (tuned on DEV, all classes)": apply(params["cons_sr_dev"], names, HR, HP)}
    fskh = np.isin(cls, FSK)
    cf = np.full(len(rs), np.nan)
    cf[fskh] = apply(params["cons_fsk_dev"], names, HR[fskh], HP[fskh])
    cf[~fskh] = rows["consensus (tuned on DEV, all classes)"][~fskh]
    rows["class-family-aware: FSK consensus on FSK frames, general on the rest"] = cf
    res = {k: table(v, rs, cls, snr) for k, v in rows.items()}
    classes = list(dict.fromkeys(cls.tolist()))
    print(f"\nHELD-OUT ({len(rs)} frames, D:\\data\\symbol_rate_torchsig.npz): success = within 2 %")
    print(f"{'estimator':>62s}  all  PSK/QAM FSK | " + " ".join(f"{c:>5s}" for c in classes))
    for k, t in res.items():
        print(f"{k:>62s} {t['all'] * 100:4.1f} {t['psk_qam'] * 100:5.1f} {t['fsk_family'] * 100:5.1f} | " +
              " ".join(f"{t['per_class'][c] * 100:5.0f}" for c in classes))
    print("per SNR (consensus):", {s: round(v * 100, 1) for s, v in res["consensus (tuned on DEV, all classes)"]["per_snr"].items()})
    print("FSK per SNR (consensus):", {s: round(v * 100, 1) for s, v in res["consensus (tuned on DEV, all classes)"]["fsk_per_snr"].items()})
    # DEV numbers for the same rows (for the over-fitting check)
    dev_rows = {"inst_freq_s3 (baseline)": R[:, col["ifr3.g"]], "consensus (tuned on DEV, all classes)": apply(params["cons_sr_dev"], names, R, P)}
    dev_res = {k: table(v, meta["rs"], meta["cls"], meta["snr"]) for k, v in dev_rows.items()}
    with open(os.path.join(common.RESULTS, "symbol_rate_extended.json"), "w") as f:
        json.dump({"heldout": res, "dev_same_estimators": dev_res, "n_heldout": int(len(rs)), "criterion": "|Rs_hat-Rs|/Rs < 2 %",
                   "consensus_params": params}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    keys = ["inst_freq_s3 (baseline)", "consensus (tuned on DEV, all classes)",
            "class-family-aware: FSK consensus on FSK frames, general on the rest"]
    w = 0.27
    for j, k in enumerate(keys):
        ax[0].bar(np.arange(len(classes)) + (j - 1) * w, [res[k]["per_class"][c] * 100 for c in classes], w, label=k)
        s = sorted(res[k]["per_snr"])
        ax[1].plot(s, [res[k]["per_snr"][v] * 100 for v in s], marker="o", label=k)
    ax[0].set_xticks(np.arange(len(classes))); ax[0].set_xticklabels(classes)
    ax[0].set_ylabel("within 2 % (%)"); ax[0].set_title("held-out TorchSig frames, per class"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("within 2 % (%)"); ax[1].set_title("per SNR (all classes)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(common.RESULTS, "symbol_rate_extended.png"), dpi=110)


if __name__ == "__main__":
    main()
