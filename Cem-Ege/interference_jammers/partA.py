"""Part A - classification of the 7 interference types (decision tree with learned thresholds + boosted trees).

stages:  python partA.py gen      -> generate frames + features (cached in results/cache_*.npz)
         python partA.py report   -> train on seeds A, tune on dev seeds, report on fresh seeds B
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import dataset
import jcommon as common
from jcommon import KINDS

OUT = os.path.join(common.HERE, "results")
INRS = [-10, -5, 0, 5, 10, 15, 20]
N_TRAIN, N_DEV, N_TEST, N_TEST_SIG, N_CAL, N_H0TEST = 350, 100, 50, 30, 3000, 1000
PFA = 0.01
GATE_STATS = ["pow_excess_db", "pk_med", "top5_med", "chirp_Gmax"]


def inr_draw(role, ki, idx):
    return float(-10 + 30 * np.random.default_rng([99, role, ki, idx]).random())


def tasks_train(role, n):
    t = []
    for ki, k in enumerate(KINDS):
        for i in range(n):
            t.append(dict(role=role, kind=k, inr=round(inr_draw(role, ki, i), 3), sig=bool(i % 2), idx=i))
    return t


def tasks_test():
    t = []
    for sig in (False, True):
        for k in KINDS:
            for inr in INRS:
                for i in range(N_TEST_SIG if sig else N_TEST):
                    t.append(dict(role=dataset.ROLE_TEST, kind=k, inr=float(inr), sig=sig, idx=i))
    return t


def cache(name, tasks, **kw):
    path = os.path.join(OUT, f"cache_{name}.npz")
    if os.path.exists(path):
        return
    X, nm, metas = dataset.run(tasks, label=name, **kw)
    kind = np.array([t["kind"] for t in tasks])
    inr = np.array([t["inr"] for t in tasks])
    sig = np.array([t["sig"] for t in tasks])
    np.savez(path, X=X, names=np.array(nm), kind=kind, inr=inr, sig=sig)


def gen():
    cache("train", tasks_train(dataset.ROLE_TRAIN, N_TRAIN))
    cache("dev", tasks_train(dataset.ROLE_DEV, N_DEV))
    cache("test", tasks_test())
    cal = [dict(role=dataset.ROLE_CAL, kind="none", inr=0.0, sig=False, idx=i, bispec=False) for i in range(N_CAL)]
    cache("h0cal", cal)
    h0 = [dict(role=dataset.ROLE_TEST, kind="none", inr=0.0, sig=False, idx=i, bispec=False) for i in range(N_H0TEST)]
    cache("h0test", h0)


# ------------------------------------------------------------------ report

def load(name):
    d = np.load(os.path.join(OUT, f"cache_{name}.npz"))
    return d["X"], list(d["names"]), d["kind"], d["inr"], d["sig"]


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeClassifier, export_text

    Xtr, nm, ktr, itr, str_ = load("train")
    Xdv, _, kdv, idv, sdv = load("dev")
    Xte, _, kte, ite, ste = load("test")
    use = [i for i, n in enumerate(nm) if not n.startswith("aux_") and n != "hough_npts"]
    fn = [nm[i] for i in use]
    ytr = np.array([KINDS.index(k) for k in ktr]); ydv = np.array([KINDS.index(k) for k in kdv]); yte = np.array([KINDS.index(k) for k in kte])
    R = {"n_train": int(len(ytr)), "n_dev": int(len(ydv)), "n_test": int(len(yte)), "n_per_test_cell_nosig": N_TEST, "n_per_test_cell_sig": N_TEST_SIG,
         "n_features": len(fn), "features": fn}

    # ---- hyper-parameters chosen on dev seeds only
    best, grid = None, []
    for d in (3, 4, 5, 6, 8, 10, 14):
        for leaf in (1, 3, 5, 10, 20):
            m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=0).fit(Xtr[:, use], ytr)
            a = float(np.mean(m.predict(Xdv[:, use]) == ydv))
            grid.append((d, leaf, a))
            if best is None or a > best[2] + 1e-9 or (abs(a - best[2]) < 1e-9 and d < best[0]):
                best = (d, leaf, a)
    R["dt_dev_grid"] = grid
    R["dt_params"] = dict(max_depth=best[0], min_samples_leaf=best[1], dev_acc=best[2])
    dt = DecisionTreeClassifier(max_depth=best[0], min_samples_leaf=best[1], random_state=0).fit(Xtr[:, use], ytr)
    bestb, gridb = None, []
    for d in (3, 6):
        for it in (100, 300):
            m = HistGradientBoostingClassifier(max_depth=d, max_iter=it, learning_rate=0.1, early_stopping=False, random_state=0).fit(Xtr[:, use], ytr)
            a = float(np.mean(m.predict(Xdv[:, use]) == ydv))
            gridb.append((d, it, a))
            if bestb is None or a > bestb[2]:
                bestb = (d, it, a)
    R["gb_dev_grid"] = gridb
    R["gb_params"] = dict(max_depth=bestb[0], max_iter=bestb[1], dev_acc=bestb[2])
    gb = HistGradientBoostingClassifier(max_depth=bestb[0], max_iter=bestb[1], learning_rate=0.1, early_stopping=False, random_state=0).fit(Xtr[:, use], ytr)
    with open(os.path.join(OUT, "partA_tree.txt"), "w") as f:
        f.write(export_text(dt, feature_names=fn, class_names=KINDS, decimals=3))
    imp = sorted(zip(fn, dt.feature_importances_), key=lambda z: -z[1])[:15]
    R["dt_top_features"] = [(a, float(b)) for a, b in imp]
    R["dt_n_leaves"] = int(dt.get_n_leaves())

    # ---- gate calibrated on noise-only frames (seed role CAL), verified on fresh noise-only frames
    Xc, nmc, *_ = load("h0cal")
    Xh, nmh, *_ = load("h0test")
    gi = {s: nmc.index(s) for s in GATE_STATS}
    m_ = len(GATE_STATS)
    thr = {s: float(np.quantile(Xc[:, gi[s]], 1 - PFA / m_)) for s in GATE_STATS}

    def gate(X, names):
        g = np.zeros(len(X), bool)
        for s in GATE_STATS:
            g |= X[:, names.index(s)] > thr[s]
        return g
    fa_cal, fa_test = gate(Xc, nmc), gate(Xh, nmh)
    R["gate"] = dict(pfa_target=PFA, thresholds=thr, stats=GATE_STATS, n_cal=int(len(Xc)), n_h0_test=int(len(Xh)),
                     pfa_cal=float(fa_cal.mean()), pfa_fresh=float(fa_test.mean()), fa_fresh_count=int(fa_test.sum()),
                     per_stat_fresh={s: float(np.mean(Xh[:, nmh.index(s)] > thr[s])) for s in GATE_STATS},
                     noise_only_classified_as_jammer=float(fa_test.mean()))
    g_te = gate(Xte, nm)

    preds = {"dt": dt.predict(Xte[:, use]), "gb": gb.predict(Xte[:, use])}
    wilson = lambda p, n: 1.96 * np.sqrt(max(p * (1 - p), 1e-4) / n)
    for sig in (False, True):
        tag = "sig" if sig else "nosig"
        for mname, pr in preds.items():
            acc = {k: {} for k in KINDS}
            e2e = {k: {} for k in KINDS}
            for ki, k in enumerate(KINDS):
                for inr in INRS:
                    s = (kte == k) & (ite == inr) & (ste == sig)
                    acc[k][inr] = float(np.mean(pr[s] == ki))
                    e2e[k][inr] = float(np.mean((pr[s] == ki) & g_te[s]))
            R[f"{mname}_{tag}_acc_vs_inr"] = acc
            R[f"{mname}_{tag}_acc_endtoend_vs_inr"] = e2e
            R[f"{mname}_{tag}_overall_by_inr"] = {inr: float(np.mean([acc[k][inr] for k in KINDS])) for inr in INRS}
            R[f"{mname}_{tag}_overall_by_inr_endtoend"] = {inr: float(np.mean([e2e[k][inr] for k in KINDS])) for inr in INRS}
            for inr in (-5, 0, 10):
                s = (ite == inr) & (ste == sig)
                R[f"{mname}_{tag}_confusion_inr{inr}"] = confusion_matrix(yte[s], pr[s], labels=range(7)).tolist()
        # gate Pd vs INR
        R[f"gate_pd_{tag}"] = {k: {inr: float(np.mean(g_te[(kte == k) & (ite == inr) & (ste == sig)])) for inr in INRS} for k in KINDS}
    R["pd_note"] = "gate_pd_sig: the gate is calibrated WITHOUT the legitimate signal, so with a QPSK present it also fires on the signal; use the nosig table for Pd"

    # ---- feature-group ablation (decision tree, same hyper-parameters, no-signal test)
    groups = {
        "occupancy+spectrum": ["tf_occ", "bw_frac", "bw_rel", "n_comp", "comp_width", "n_peaks", "pk_med", "pk_mean", "flat_occ", "flat_all", "kurt_f", "pow_excess_db"],
        "+kurtosis": ["sk_mean", "sk_med", "env_kurt"],
        "+time/periodicity": ["duty", "on_db", "q_cv", "top5_med", "per_h", "per_lag", "bin_acf_h", "bin_acf_lag"],
        "+sweep (Hough, dechirp)": [n for n in fn if n.startswith(("hough_", "chirp_"))],
        "+bispectrum": [n for n in fn if n.startswith("bis_")],
    }
    abl, cum = {}, []
    for gname, cols in groups.items():
        cum += [c for c in cols if c in fn]
        idx = [fn.index(c) for c in cum]
        m = DecisionTreeClassifier(max_depth=best[0], min_samples_leaf=best[1], random_state=0).fit(Xtr[:, use][:, idx], ytr)
        s = ~ste
        abl[gname] = dict(n_features=len(idx),
                          acc_all_inr_nosig=float(np.mean(m.predict(Xte[:, use][s][:, idx]) == yte[s])),
                          acc_inr10_nosig=float(np.mean(m.predict(Xte[:, use][s & (ite == 10)][:, idx]) == yte[s & (ite == 10)])),
                          acc_inr0_nosig=float(np.mean(m.predict(Xte[:, use][s & (ite == 0)][:, idx]) == yte[s & (ite == 0)])))
    R["ablation_cumulative_tree"] = abl
    for mname in ("dt", "gb"):
        s = ~ste
        R[f"{mname}_nosig_overall_all_inr"] = float(np.mean(preds[mname][s] == yte[s]))
        R[f"{mname}_sig_overall_all_inr"] = float(np.mean(preds[mname][ste] == yte[ste]))
    with open(os.path.join(OUT, "partA_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- CSV
    with open(os.path.join(OUT, "partA_acc_vs_inr.csv"), "w") as f:
        f.write("model,signal,class,inr_db,accuracy,endtoend_accuracy,n\n")
        for mname in ("dt", "gb"):
            for tag in ("nosig", "sig"):
                for k in KINDS:
                    for inr in INRS:
                        f.write(f"{mname},{tag},{k},{inr},{R[f'{mname}_{tag}_acc_vs_inr'][k][inr]:.4f},{R[f'{mname}_{tag}_acc_endtoend_vs_inr'][k][inr]:.4f},{N_TEST_SIG if tag == 'sig' else N_TEST}\n")

    # ---- overview figure
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    mk = ["o", "s", "^", "D", "v", "P", "X"]
    ink, mut, grid_c, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": mut, "xtick.color": mut, "ytick.color": mut,
                         "axes.facecolor": surf, "figure.facecolor": surf, "axes.titlecolor": ink})
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    def curves(ax, key, title):
        for ki, k in enumerate(KINDS):
            ax.plot(INRS, [100 * R[key][k][i] for i in INRS], color=cols[ki], marker=mk[ki], lw=1.6, ms=5, label=common.KIND_LABEL[k])
        ax.plot(INRS, [100 * np.mean([R[key][k][i] for k in KINDS]) for i in INRS], color=ink, lw=2.2, ls="--", label="mean")
        ax.set_ylim(0, 102); ax.set_xlabel("INR per sample (dB)"); ax.set_ylabel("accuracy (%)")
        ax.set_title(title, fontsize=10, loc="left"); ax.grid(color=grid_c, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 0]); curves(ax, "dt_nosig_acc_vs_inr", f"Decision tree (depth {best[0]}), no legit. signal")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = fig.add_subplot(gs[0, 1]); curves(ax, "gb_nosig_acc_vs_inr", "Boosted trees (reference), no legit. signal")
    ax = fig.add_subplot(gs[0, 2]); curves(ax, "dt_sig_acc_vs_inr", "Decision tree, QPSK also present (SNR 5-20 dB)")

    def cm(ax, key, title):
        M = np.array(R[key], float)
        M = 100 * M / M.sum(1, keepdims=True)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
        for i in range(7):
            for j in range(7):
                if M[i, j] >= 1:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 55 else ink)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(KINDS, rotation=45, ha="right"); ax.set_yticklabels(KINDS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10, loc="left")

    cm(fig.add_subplot(gs[1, 0]), "dt_nosig_confusion_inr10", f"Decision tree, confusion at +10 dB (n={N_TEST}/class, %)")
    cm(fig.add_subplot(gs[1, 1]), "dt_nosig_confusion_inr0", f"Decision tree, confusion at 0 dB (n={N_TEST}/class, %)")
    ax = fig.add_subplot(gs[1, 2])
    for ki, k in enumerate(KINDS):
        ax.plot(INRS, [100 * R["gate_pd_nosig"][k][i] for i in INRS], color=cols[ki], marker=mk[ki], lw=1.6, ms=5, label=common.KIND_LABEL[k])
    ax.axhline(100 * PFA, color=mut, lw=1, ls=":")
    ax.text(-9.8, 4, f"gate Pfa target {PFA:.0%} (measured on {len(Xh)} fresh noise frames: {R['gate']['pfa_fresh']:.2%})", fontsize=7.5, color=mut)
    ax.set_ylim(0, 102); ax.set_xlabel("INR per sample (dB)"); ax.set_ylabel("jammer-present gate Pd (%)")
    ax.set_title("Presence gate (4 noise-calibrated statistics, Bonferroni)", fontsize=10, loc="left")
    ax.grid(color=grid_c, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Part A - classification of 7 interference types; train seeds A ({len(ytr)} frames), report on fresh seeds B ({N_TEST} frames per class x INR cell, {N_TEST_SIG} with QPSK)", fontsize=12, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "partA_overview.png"), dpi=130, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        report()
