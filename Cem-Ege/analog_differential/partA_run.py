"""Part A FINAL run: train on TRAIN seeds (1e6+), report on fresh TEST seeds (3e6+). Nothing here was tuned on them."""
import sys, time, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from common import RES, save_json, wilson
import partA_analog as A
from partA_analog import CLASSES, K, SNRS, FI, build, fit_models, report, predict, HandTree

COL = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#9d755d", "#bab0ac"]


def stress_nbfm_pm(models_main):
    """NBFM vs PM at small, matched deviation: NBFM +-1.25 kHz (dev_scale 0.5), PM 0.5 rad. Binary problem."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cls = ["NBFM", "PM"]
    Ftr, ytr, _, _ = build(2000, 2048, (0.0, 25.0), 4_000_000, classes=cls, dev_scale=0.5)
    ytr = np.array([0 if c == 0 else 1 for c in (ytr * 0 + (ytr))])
    # build labels are CI indices (NBFM=5, PM=7); map to 0/1
    ytr = (ytr == A.CI["PM"]).astype(int)
    own = [FI[n] for n in A.OWN]
    gbt = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Ftr[:, own], ytr)
    thr, up, _ = A._best_thr(Ftr[:, FI["fm_slope"]], ytr == 1)
    res = {}
    for snr in (5, 10, 20):
        F, y, _, _ = build(500, 2048, float(snr), 5_000_000 + 100 * snr, classes=cls, dev_scale=0.5)
        y = (y == A.CI["PM"]).astype(int)
        ps = (F[:, FI["fm_slope"]] > thr) == up
        pg = gbt.predict(F[:, own])
        n = len(y)
        res[snr] = {"slope_only_acc": float(np.mean(ps == y)), "slope_only_ci": wilson(int((ps == y).sum()), n),
                    "gbt_all_features_acc": float(np.mean(pg == y)), "gbt_ci": wilson(int((pg == y).sum()), n), "n": n}
        print("  stress NBFM-vs-PM matched small deviation", snr, "dB", res[snr])
    return res


def stress_vsb(models):
    out = {}
    for fv in (0.15, 0.3, 0.5, 1.0):
        row = {}
        for snr in (10, 20):
            F, y, _, _ = build(300, 2048, float(snr), 6_000_000 + int(100 * fv) + snr, classes=["VSB"], fv_frac=fv)
            for name in ("hand_tree_snrbin", "gbt_own"):
                p = predict({name: models[name]}, F)[name]
                out[f"fv={fv}B|snr={snr}|{name}"] = {CLASSES[k]: float(np.mean(p == k)) for k in range(K) if np.mean(p == k) > 0}
    return out


def plot(res, data, models, path):
    Fte, yte, snrv, tone, preds = data
    sn = np.round(snrv).astype(int)
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    sty = {"hand_tree_global": ("--", "#999999"), "hand_tree_snrbin": ("-", "#4c78a8"), "cart_depth8": ("-", "#f58518"),
           "gbt_own": ("-", "#54a24b"), "gbt_own+19std": (":", "#222222")}
    for k, m in res["methods"].items():
        ax.plot(SNRS, [100 * m["acc_vs_snr"][s] for s in SNRS], ls=sty[k][0], color=sty[k][1], marker="o", ms=4, label=k)
    ax.set_xlabel("SNR (dB, per complex sample at fs)"); ax.set_ylabel("overall accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"8-class accuracy, fresh frames (n={res['n_per_class_snr']*K} per SNR)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    for j, (mname, ttl) in enumerate((("hand_tree_snrbin", "per class: hand tree (SNR-binned thresholds)"), ("gbt_own", "per class: gradient-boosted trees"))):
        ax = fig.add_subplot(gs[0, 1 + j])
        for i, c in enumerate(CLASSES):
            ax.plot(SNRS, [100 * res["methods"][mname]["per_class"][c][s] for s in SNRS], color=COL[i], marker="o", ms=3.5, label=c)
        ax.set_ylim(0, 102); ax.set_xlabel("SNR (dB)"); ax.set_title(ttl); ax.grid(alpha=.3)
        if j == 0: ax.legend(fontsize=7, ncol=2)
    for j, mname in enumerate(("hand_tree_snrbin", "gbt_own")):
        ax = fig.add_subplot(gs[1, j])
        C = np.array(res["methods"][mname]["confusion_10dB"], float); C = 100 * C / C.sum(1, keepdims=True)
        ax.imshow(C, cmap="Blues", vmin=0, vmax=100)
        for a_ in range(K):
            for b_ in range(K):
                ax.text(b_, a_, f"{C[a_, b_]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if C[a_, b_] > 55 else "black")
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(CLASSES, rotation=60, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_title(f"confusion at 10 dB (%): {mname}"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax = fig.add_subplot(gs[1, 2])
    m10 = (sn == 10) & ~tone
    for c, col in (("NBFM", "#4c78a8"), ("PM", "#e45756"), ("WBFM", "#72b7b2")):
        v = Fte[m10 & (yte == CLASSES.index(c)), FI["fm_slope"]]
        ax.hist(np.clip(v, -4, 4), bins=40, range=(-4, 4), alpha=.55, color=col, label=c, density=True)
    ax.set_xlabel("discriminator-output spectral slope (log-log)"); ax.set_title("PM vs FM slope feature, 10 dB, noise message"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle("Part A: analog modulation recognition (synthetic, fs = 8 x channel width)", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    t0 = time.time()
    summary = {}
    for N, n_test, seed_tr in ((2048, 500, 1_000_000), (1024, 200, 1_100_000), (4096, 200, 1_200_000)):
        Ftr, ytr, _, _ = build(3000, N, (-5.0, 25.0), seed_tr)
        models = fit_models(Ftr, ytr)
        res, data = report(models, N, n_test, 3_000_000 + N * 100, f"test N={N}")
        A.print_table(res)
        summary[N] = res
        if N == 2048:
            main_models, main_data = models, data
            res["hand_tree_thresholds"] = {"snrbin": main_models["hand_tree_snrbin"].log, "global": main_models["hand_tree_global"].log}
            plot(res, data, models, f"{RES}/partA_overview.png")
            with open(f"{RES}/partA_acc_vs_snr_N2048.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["method", "class"] + SNRS)
                for k, m in res["methods"].items():
                    w.writerow([k, "ALL"] + [f"{m['acc_vs_snr'][s]:.4f}" for s in SNRS])
                    for c in CLASSES:
                        w.writerow([k, c] + [f"{m['per_class'][c][s]:.4f}" for s in SNRS])
            # hard pairs at 10 dB for the two main methods
            hp = {}
            for mname in ("hand_tree_snrbin", "gbt_own"):
                C = np.array(res["methods"][mname]["confusion_10dB"], float); C = C / C.sum(1, keepdims=True)
                pairs = sorted([(C[i, j], CLASSES[i], CLASSES[j]) for i in range(K) for j in range(K) if i != j], reverse=True)[:6]
                hp[mname] = [{"true": a, "pred": b, "rate": float(r)} for r, a, b in pairs]
            summary["hard_pairs_10dB"] = hp
            print("hard pairs 10 dB:", json.dumps(hp, indent=0)[:900])
            summary["stress_nbfm_pm_matched"] = stress_nbfm_pm(models)
            summary["stress_vsb_vestige"] = stress_vsb(models)
            print(json.dumps(summary["stress_vsb_vestige"], indent=0))
    summary["elapsed_s"] = time.time() - t0
    save_json(summary, "partA_results.json")


if __name__ == "__main__":
    main()
