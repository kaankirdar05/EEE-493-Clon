"""Part B step 2: select the Morlet configuration on DEV, report on the held-out TorchSig symbol-rate frames, refit the 12-estimator vote with the new member.

DEV      characterization_extended/data/symbol_rate_torchsig_dev.npz (3000 frames, seed 1)   -> every choice below
HELD-OUT D:\\data\\symbol_rate_torchsig.npz (3000 frames, seed 0)                                -> report only
Success = |Rs_hat - Rs| / Rs < 2 %.  Reference rows come from the cached banks of characterization_extended (sr_bank.get_bank).

    python b_eval.py tune      (DEV only -> results/b_tuned.json)
    python b_eval.py report    (held-out report + vote experiments -> results/b_symbol_rate.json/png)
"""
import json
import os
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import wavelet_sr as ws
import consensus as cs
import sr_bank

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FSK = ["2fsk", "4fsk", "2gfsk", "2msk", "2gmsk"]
CLASSES = ["bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "2gfsk", "2msk", "2gmsk"]


def load_feat(which):
    d = np.load(os.path.join(RES, f"b_features_{which}.npz"))
    return [str(n) for n in d["names"]], d["R"], d["P"], {k: d[k] for k in ("rs", "cls", "snr", "pulse")}


def correct(rate, rs):
    return cs.correct(rate, rs)


def tune():
    names, R, P, meta = load_feat("dev")
    ok = correct(R, meta["rs"][:, None])
    acc = ok.mean(0)
    order = np.argsort(-acc)
    fsk = np.isin(meta["cls"], FSK)
    print("DEV (3000 frames) leaderboard of the Morlet configurations (overall / PSK-QAM / FSK family), within 2 %:")
    board = []
    for i in order[:15]:
        print(f"  {names[i]:>26s} {acc[i] * 100:5.1f} {ok[~fsk, i].mean() * 100:5.1f} {ok[fsk, i].mean() * 100:5.1f}")
    for i in order:
        board.append((names[i], float(acc[i]), float(ok[~fsk, i].mean()), float(ok[fsk, i].mean())))
    best = names[order[0]]
    base, mode = best.rsplit(".", 1)
    parts = base.split("_")
    cfg = [float(parts[1][1:]), parts[2], int(parts[3][1:])]
    edge = cfg[0] in (ws.KAPPAS[0], ws.KAPPAS[-1])
    print("selected on DEV:", best, "(kappa at the edge of the grid)" if edge else "(kappa interior to the grid)")
    json.dump({"tuned_cfg": cfg, "mode": mode, "name": best, "dev_accuracy": float(acc[order[0]]), "kappa_grid": ws.KAPPAS, "centre_sets": ws.CENTRE_SETS,
               "dev_leaderboard": board}, open(os.path.join(RES, "b_tuned.json"), "w"), indent=1)


def table(rate, meta):
    ok = correct(rate, meta["rs"])
    cls, snr = meta["cls"], meta["snr"]
    fsk = np.isin(cls, FSK)
    rel = rate / meta["rs"]
    return {"all": float(ok.mean()), "psk_qam": float(ok[~fsk].mean()), "fsk_family": float(ok[fsk].mean()),
            "per_class": {c: float(ok[cls == c].mean()) for c in CLASSES}, "per_snr": {int(s): float(ok[snr == s].mean()) for s in sorted(set(snr.tolist()))},
            "fail_x2": float(np.mean(np.abs(rel[~ok] - 2) < 0.04)) if (~ok).any() else 0.0, "fail_half": float(np.mean(np.abs(rel[~ok] - 0.5) < 0.01)) if (~ok).any() else 0.0,
            "median_rel_error": float(np.median(np.abs(rel - 1)))}


def mcnemar(a_ok, b_ok):
    """exact two-sided McNemar: a vs b on paired frames."""
    from scipy.stats import binomtest
    n10 = int(np.sum(a_ok & ~b_ok)); n01 = int(np.sum(~a_ok & b_ok))
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
    return {"a_only": n10, "b_only": n01, "p_value": float(p)}


def boot_diff(a_ok, b_ok, n=4000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a_ok), (n, len(a_ok)))
    d = (b_ok.astype(float) - a_ok.astype(float))[idx].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def report():
    t0 = time.time()
    tj = json.load(open(os.path.join(RES, "b_tuned.json")))
    wn, WR_dev, WP_dev, mdev = load_feat("dev")
    _, WR_ho, WP_ho, mho = load_feat("heldout")
    bn, BR_dev, BP_dev, bmd = sr_bank.get_bank("dev")
    _, BR_ho, BP_ho, bmh = sr_bank.get_bank("heldout")
    bn = [str(n) for n in bn]
    assert np.allclose(bmd["rs"], mdev["rs"]) and np.allclose(bmh["rs"], mho["rs"]) and (bmh["cls"] == mho["cls"]).all()
    col = {n: i for i, n in enumerate(bn)}
    params = json.load(open(os.path.join(cs_dir(), "results", "consensus_params.json")))["cons_sr_dev"]
    # --- sanity: reproduce the reference numbers from the cached banks
    mu0 = np.zeros(len(bn))
    for n, m in zip(params["members"], params["mu"]):
        mu0[col[n]] = m
    cols0 = [col[n] for n in params["members"]]
    vote_ho = cs.predict(BR_ho, BP_ho, mu0, cols0, params["rule"])
    ifs3_ho = BR_ho[:, col["ifr3.g"]]
    base = {"inst_freq_s3": table(ifs3_ho, mho), "vote_12": table(vote_ho, mho)}
    print(f"[sanity] reproduce references on held-out: inst_freq_s3 {base['inst_freq_s3']['all'] * 100:.1f} % (task: 72.7), 12-estimator vote {base['vote_12']['all'] * 100:.1f} % "
          f"(89.9), PSK/QAM {base['vote_12']['psk_qam'] * 100:.1f} (96.9), FSK {base['vote_12']['fsk_family'] * 100:.1f} (82.9)")
    # mu convention check (fit_mu on DEV reproduces the stored values)
    mu_fit = cs.fit_mu(BR_dev, BP_dev, mdev["rs"])
    print("[sanity] fit_mu on DEV reproduces stored mu of the 12 members: max |diff| =", float(np.max(np.abs(mu_fit[cols0] - mu0[cols0]))))
    # --- the Morlet estimator (tuned on DEV)
    mname = tj["name"]
    wi = wn.index(mname)
    mor_ho = WR_ho[:, wi]
    mor_dev = WR_dev[:, wi]
    res = {"tuned": tj["name"], "dev_accuracy": tj["dev_accuracy"], "n_dev": 3000, "n_heldout": 3000, "criterion": "|Rs_hat-Rs|/Rs<2%",
           "heldout": {"inst_freq_s3": base["inst_freq_s3"], "vote_12": base["vote_12"], "morlet_cwt": table(mor_ho, mho)}}
    best_bank = max(range(len(bn)), key=lambda k: correct(BR_dev[:, k], mdev["rs"]).mean())
    res["heldout"][f"best_single_bank_member_on_DEV ({bn[best_bank]})"] = table(BR_ho[:, best_bank], mho)
    # --- E1: existing 12 members + the Morlet member (mu fitted on DEV by the same convention), same vote rule
    R_dev = np.concatenate([BR_dev, WR_dev], 1); P_dev = np.concatenate([BP_dev, WP_dev], 1)
    R_ho = np.concatenate([BR_ho, WR_ho], 1); P_ho = np.concatenate([BP_ho, WP_ho], 1)
    allnames = bn + wn
    mu_all = cs.fit_mu(R_dev, P_dev, mdev["rs"])
    cidx = {n: i for i, n in enumerate(allnames)}
    cols1 = [cidx[n] for n in params["members"]] + [cidx[mname]]
    e1_dev = cs.predict(R_dev, P_dev, mu_all, cols1, "vote")
    e1_ho = cs.predict(R_ho, P_ho, mu_all, cols1, "vote")
    res["heldout"]["vote_12_plus_morlet (E1)"] = table(e1_ho, mho)
    res["dev_vote"] = {"vote_12": float(correct(cs.predict(R_dev, P_dev, mu_all, [cidx[n] for n in params["members"]], "vote"), mdev["rs"]).mean()),
                       "vote_12_plus_morlet": float(correct(e1_dev, mdev["rs"]).mean())}
    print("DEV vote accuracy: 12 members", res["dev_vote"]["vote_12"], " + Morlet", res["dev_vote"]["vote_12_plus_morlet"])
    # --- E2: greedy re-selection on DEV from {bank} U {16 best Morlet members by DEV single accuracy}
    wacc = np.array([correct(WR_dev[:, k], mdev["rs"]).mean() for k in range(len(wn))])
    pool_w = [len(bn) + int(k) for k in np.argsort(-wacc)[:16]]
    cand = list(range(len(bn))) + pool_w
    print("greedy refit (vote) on DEV over", len(cand), "candidates ...", flush=True)
    chosen, dacc = cs.greedy(R_dev, P_dev, mdev["rs"], mu_all, cand, "vote", max_members=12, verbose=True, names=allnames)
    e2_ho = cs.predict(R_ho, P_ho, mu_all, chosen, "vote")
    res["heldout"]["vote_refit_greedy_with_morlet_pool (E2)"] = table(e2_ho, mho)
    res["E2"] = {"members": [allnames[c] for c in chosen], "dev_accuracy": float(dacc), "n_morlet_members": int(sum(c >= len(bn) for c in chosen)), "pool_morlet": [allnames[c] for c in pool_w]}
    # control: greedy with the bank only, same procedure (does the procedure itself reproduce ~89.9 % ?)
    print("control: greedy refit on DEV with the bank only ...", flush=True)
    ch0, dacc0 = cs.greedy(R_dev, P_dev, mdev["rs"], mu_all, list(range(len(bn))), "vote", max_members=12, verbose=False, names=allnames)
    c_ho = cs.predict(R_ho, P_ho, mu_all, ch0, "vote")
    res["heldout"]["vote_refit_greedy_bank_only (control)"] = table(c_ho, mho)
    res["control"] = {"members": [allnames[c] for c in ch0], "dev_accuracy": float(dacc0)}
    # --- significance (paired, held-out)
    ok = lambda r: correct(r, mho["rs"])
    res["paired_heldout"] = {
        "E1_vs_vote12": {"mcnemar": mcnemar(ok(vote_ho), ok(e1_ho)), "diff_ci95": boot_diff(ok(vote_ho), ok(e1_ho)), "diff": float(ok(e1_ho).mean() - ok(vote_ho).mean())},
        "E2_vs_vote12": {"mcnemar": mcnemar(ok(vote_ho), ok(e2_ho)), "diff_ci95": boot_diff(ok(vote_ho), ok(e2_ho)), "diff": float(ok(e2_ho).mean() - ok(vote_ho).mean())},
        "control_vs_vote12": {"mcnemar": mcnemar(ok(vote_ho), ok(c_ho)), "diff_ci95": boot_diff(ok(vote_ho), ok(c_ho)), "diff": float(ok(c_ho).mean() - ok(vote_ho).mean())},
        "morlet_vs_inst_freq_s3": {"mcnemar": mcnemar(ok(ifs3_ho), ok(mor_ho)), "diff_ci95": boot_diff(ok(ifs3_ho), ok(mor_ho)), "diff": float(ok(mor_ho).mean() - ok(ifs3_ho).mean())}}
    # --- how complementary is the Morlet member? oracle union / error overlap on held-out
    okm, okv = ok(mor_ho), ok(vote_ho)
    res["complementarity_heldout"] = {"morlet_right_vote_wrong": float(np.mean(okm & ~okv)), "vote_right_morlet_wrong": float(np.mean(~okm & okv)), "both_wrong": float(np.mean(~okm & ~okv))}
    # same, SNR-0 dB breakdown of the Morlet failures: harmonic / subharmonic
    json.dump(res, open(os.path.join(RES, "b_symbol_rate.json"), "w"), indent=1)
    print(f"\nHELD-OUT ({len(mho['rs'])} frames), within 2 %      all  PSK/QAM  FSK | " + " ".join(f"{c:>5s}" for c in CLASSES))
    for k, t in res["heldout"].items():
        print(f"{k[:58]:>58s} {t['all'] * 100:5.1f} {t['psk_qam'] * 100:6.1f} {t['fsk_family'] * 100:6.1f} | " + " ".join(f"{t['per_class'][c] * 100:5.0f}" for c in CLASSES))
    print("per SNR:", {k[:18]: {s: round(v * 100, 1) for s, v in t["per_snr"].items()} for k, t in res["heldout"].items() if k in ("inst_freq_s3", "morlet_cwt", "vote_12", "vote_12_plus_morlet (E1)")})
    print("paired:", json.dumps(res["paired_heldout"]))
    print("E2 members:", res["E2"]["members"], "DEV acc", res["E2"]["dev_accuracy"])
    print("control members:", res["control"]["members"], "DEV acc", res["control"]["dev_accuracy"])
    print(f"{time.time() - t0:.0f} s")
    plot(res)


def cs_dir():
    return ws.CE


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("inst_freq_s3", "tab:gray"), ("morlet_cwt", "tab:red"), ("vote_12", "tab:blue"), ("vote_12_plus_morlet (E1)", "tab:green")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.2
    for j, (k, c) in enumerate(keys):
        t = res["heldout"][k]
        ax[0].bar(np.arange(len(CLASSES)) + (j - 1.5) * w, [t["per_class"][cl] * 100 for cl in CLASSES], w, color=c, label=f"{k} ({t['all'] * 100:.1f} %)")
        s = sorted(t["per_snr"])
        ax[1].plot(s, [t["per_snr"][v] * 100 for v in s], marker="o", color=c, label=k)
    ax[0].set_xticks(range(len(CLASSES))); ax[0].set_xticklabels(CLASSES, fontsize=8); ax[0].set_ylabel("within 2 % of true Rs (%)")
    ax[0].set_title(f"B: held-out TorchSig frames (3000), tuned Morlet = {res['tuned']}", fontsize=9); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_title("per SNR (all classes)", fontsize=9); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "b_symbol_rate.png"), dpi=130)


if __name__ == "__main__":
    {"tune": tune, "report": report}[sys.argv[1]]()
