"""Part A driver: analog-modulation recognition (hand tree with learned thresholds, CART, GBT).

    python partA_analog.py dev             # design-time look on DEV seeds only
    python partA_analog.py final           # train on TRAIN seeds, report on fresh TEST seeds
Seeds: train 1xxx, dev 2xxx, test 3xxx (never used for tuning).
"""
import sys, time, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import common
from common import RES, NCPU, save_json, wilson
from analog_gen import CLASSES, make_frames
from analog_feats import analog_features, NAMES, OWN, FI

K = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
SNRS = [-5, 0, 5, 10, 15, 20, 25]


# ---------------------------------------------------------------- data generation (parallel)
def _task(args):
    cls, n, N, snr, seed, kw = args
    rng = np.random.default_rng(seed)
    if isinstance(snr, tuple):                       # uniform random SNR in [lo, hi]
        snr_v = rng.uniform(snr[0], snr[1], n)
    else:
        snr_v = np.full(n, float(snr))
    x, meta = make_frames(cls, n, N, snr_v, rng, **kw)
    return analog_features(x), snr_v, meta["is_tone"]


def build(n_per, N, snr, seed0, chunk=500, classes=CLASSES, **kw):
    """class-balanced feature set. snr: scalar, list of scalars (n_per each) or (lo,hi) tuple."""
    snr_list = snr if isinstance(snr, list) else [snr]
    tasks, ys = [], []
    s = seed0
    for cls in classes:
        for sv in snr_list:
            left = n_per
            while left > 0:
                m = min(chunk, left); left -= m
                tasks.append((cls, m, N, sv, s, kw)); ys.append(CI[cls]); s += 1
    with ProcessPoolExecutor(NCPU) as ex:
        res = list(ex.map(_task, tasks))
    F = np.concatenate([r[0] for r in res]); snrv = np.concatenate([r[1] for r in res])
    tone = np.concatenate([r[2] for r in res])
    y = np.concatenate([np.full(len(r[0]), c) for r, c in zip(res, ys)])
    return F, y, snrv, tone


# ---------------------------------------------------------------- hand-structured tree, learned thresholds
class Node:
    def __init__(self, name, A, B, cands, childA=None, childB=None):
        self.name, self.A, self.B, self.cands = name, set(A), set(B), cands
        self.child = {"A": childA, "B": childB}
        self.feat = None; self.thr = {}; self.dirA = {}   # per SNR-bin: threshold, 'A is above threshold'


def make_tree():
    c = lambda *a: list(a)
    n_ud = Node("N7 USB/LSB", ["SSB-USB"], ["SSB-LSB"], c("asym_nom_db", "asym_pk_db"))
    n_vs = Node("N6 SSB/VSB", ["SSB-USB", "SSB-LSB"], ["VSB"], c("asym_nom_abs", "asym_pk_abs", "line_x2", "occ", "kurt_f"), childA=n_ud)
    n_sc = Node("N5 DSB-SC", ["AM-DSB-SC"], ["SSB-USB", "SSB-LSB", "VSB"], c("line_x2", "sig_aa_c", "kurt_f", "asym_nom_abs", "bw90"), childB=n_vs)
    n_car = Node("N4 carrier", ["AM-DSB-WC"], ["AM-DSB-SC", "SSB-USB", "SSB-LSB", "VSB"], c("carrier_frac", "line_x1", "sig_aa_c", "bw90"), childB=n_sc)
    n_fm = Node("N3 PM/NBFM", ["PM"], ["NBFM"], c("fm_slope"))
    n_wb = Node("N2 WBFM", ["WBFM"], ["NBFM", "PM"], c("bw90", "sig_f", "kurt_f", "carrier_frac", "disc_bw"), childB=n_fm)
    n_env = Node("N1 envelope", ["AM-DSB-WC", "AM-DSB-SC", "SSB-USB", "SSB-LSB", "VSB"], ["NBFM", "WBFM", "PM"],
                 c("sig_aa_c", "sig_aa", "kurt_amp", "skew_amp"), childA=n_car, childB=n_wb)
    return n_env


SNR_EDGES = list(np.arange(-3.75, 25, 2.5))


def _bin(snr_est, edges):
    return np.digitize(snr_est, edges) if edges else np.zeros(len(snr_est), int)


def _best_thr(v, isA):
    """threshold and orientation minimising class-balanced error -> (thr, A_above, err)."""
    cand = np.unique(np.quantile(v, np.linspace(0.01, 0.99, 300)))
    nA, nB = max(isA.sum(), 1), max((~isA).sum(), 1)
    vA, vB = np.sort(v[isA]), np.sort(v[~isA])
    below_A = np.searchsorted(vA, cand, side="right") / nA            # fraction of A <= t
    below_B = np.searchsorted(vB, cand, side="right") / nB
    fa = (below_A + (1 - below_B)) / 2                                # error if A is above threshold
    k = int(np.argmin(fa)); k2 = int(np.argmax(fa))
    if fa[k] <= 1 - fa[k2]:
        return float(cand[k]), True, float(fa[k])
    return float(cand[k2]), False, float(1 - fa[k2])


def fit_node(node, F, ylab, snr_est, edges, log):
    names = np.array(CLASSES)[ylab]
    m = np.isin(names, list(node.A | node.B))
    isA = np.isin(names[m], list(node.A))
    b = _bin(snr_est[m], edges)
    Fm = F[m]
    node.feat = {}
    for bi in np.unique(_bin(np.arange(-10, 40, 1.0), edges)):
        s = b == bi
        if s.sum() < 40 or isA[s].all() or (~isA[s]).all():
            s = np.ones_like(s)            # fall back to pooled data
        best = None
        for fn in node.cands:              # feature AND threshold chosen per SNR bin (training frames only)
            t, up, e = _best_thr(Fm[s, FI[fn]], isA[s])
            if best is None or e < best[0]:
                best = (e, fn, t, up)
        e, fn, t, up = best
        node.feat[int(bi)] = fn; node.thr[int(bi)] = t; node.dirA[int(bi)] = up
        log.append({"node": node.name, "feature": fn, "snr_bin": int(bi), "thr": t, "A_above": up, "train_err": e})
    for k in ("A", "B"):
        ch = node.child[k]
        if isinstance(ch, Node):
            fit_node(ch, F, ylab, snr_est, edges, log)


def predict_node(node, F, snr_est, edges, idx=None, out=None):
    if out is None:
        out = np.full(len(F), -1); idx = np.arange(len(F))
    if len(idx) == 0:
        return out
    b = _bin(snr_est[idx], edges)
    thr = np.array([node.thr[int(i)] for i in b]); up = np.array([node.dirA[int(i)] for i in b])
    val = np.array([F[j, FI[node.feat[int(i)]]] for j, i in zip(idx, b)])
    isA = (val > thr) == up
    for k, sel in (("A", isA), ("B", ~isA)):
        sub = idx[sel]
        ch = node.child[k]
        if isinstance(ch, Node):
            predict_node(ch, F, snr_est, edges, sub, out)
        else:
            grp = node.A if k == "A" else node.B
            out[sub] = CI[next(iter(grp))]
    return out


class HandTree:
    def __init__(self, snr_bins=True):
        self.edges = SNR_EDGES if snr_bins else []
    def fit(self, F, y):
        self.root = make_tree(); self.log = []
        fit_node(self.root, F, y, F[:, FI["snr_est"]], self.edges, self.log)
        return self
    def predict(self, F):
        return predict_node(self.root, F, F[:, FI["snr_est"]], self.edges)


# ---------------------------------------------------------------- classifiers
def fit_models(Ftr, ytr):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    own = [FI[n] for n in OWN]
    models = {}
    models["hand_tree_global"] = HandTree(False).fit(Ftr, ytr)
    models["hand_tree_snrbin"] = HandTree(True).fit(Ftr, ytr)
    cart = DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, random_state=0).fit(Ftr[:, own], ytr)
    models["cart_depth8"] = (cart, own)
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Ftr[:, own], ytr)
    models["gbt_own"] = (g, own)
    g2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Ftr, ytr)
    models["gbt_own+19std"] = (g2, list(range(len(NAMES))))
    return models


def predict(models, F):
    return {k: (m.predict(F) if isinstance(m, HandTree) else m[0].predict(F[:, m[1]])) for k, m in models.items()}


def confusion(y, p):
    C = np.zeros((K, K), int)
    np.add.at(C, (y, p), 1)
    return C


def report(models, N, n_per, seed0, tag):
    t = time.time()
    Fte, yte, snrv, tone = build(n_per, N, [float(s) for s in SNRS], seed0)
    sn = np.round(snrv).astype(int)
    preds = predict(models, Fte)
    res = {"N": N, "n_per_class_snr": n_per, "snrs": SNRS, "methods": {}}
    for name, p in preds.items():
        res["methods"][name] = {
            "acc_vs_snr": {int(s): float(np.mean(p[sn == s] == yte[sn == s])) for s in SNRS},
            "per_class": {c: {int(s): float(np.mean(p[(sn == s) & (yte == i)] == i)) for s in SNRS} for i, c in enumerate(CLASSES)},
            "acc_all": float(np.mean(p == yte)),
            "acc_noise_msg": {int(s): float(np.mean(p[(sn == s) & ~tone] == yte[(sn == s) & ~tone])) for s in SNRS},
            "acc_tone_msg": {int(s): float(np.mean(p[(sn == s) & tone] == yte[(sn == s) & tone])) for s in SNRS},
            "confusion_10dB": confusion(yte[sn == 10], p[sn == 10]).tolist()}
    res["n_test_frames"] = int(len(yte))
    print(f"[{tag}] N={N}: {len(yte)} frames ({time.time()-t:.0f}s)")
    return res, (Fte, yte, snrv, tone, preds)


def print_table(res):
    print("overall accuracy vs SNR (%%), n = %d frames per SNR point" % (res["n_per_class_snr"] * K))
    print(f"{'method':18s} " + " ".join(f"{s:>5d}" for s in SNRS))
    for k, m in res["methods"].items():
        print(f"{k:18s} " + " ".join(f"{100*m['acc_vs_snr'][s]:5.1f}" for s in SNRS))


def print_pc(res, method, snr=10):
    for i, c in enumerate(CLASSES):
        print(f"  {c:10s} " + " ".join(f"{100*res['methods'][method]['per_class'][c][s]:5.1f}" for s in SNRS))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "dev":
        Ftr, ytr, str_, _ = build(3000, 2048, (-5.0, 25.0), 1_000)
        models = fit_models(Ftr, ytr)
        res, _ = report(models, 2048, 100, 2_000, "dev")
        print_table(res)
        for m in ("hand_tree_snrbin", "gbt_own"):
            print(m); print_pc(res, m)
        C = np.array(res["methods"]["hand_tree_snrbin"]["confusion_10dB"]); print(C)
