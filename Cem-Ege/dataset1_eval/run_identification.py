"""Identification (job 5.1) on Dataset 1 v3: the Cem-Ege classifiers, trained on the train split, scored on the test split.

    python run_identification.py classifiers   # 8 models of compare_classifiers.py on the 19 features
    python run_identification.py extended      # boosted trees on 19 + Haar wavelet / histogram-entropy features
    python run_identification.py rules         # Azzouz-Nandi tree (7 classes) + Swami-Sadler hierarchy (PSK/QAM)

Scoring (same split for every method): train = split 0 (156,800 clips), test = split 2 (33,600 clips).
SNR axis = the dataset's Z label (Es/N0 for single-carrier digital classes, in-band SNR for AM/NBFM/WBFM/OFDM).
NOISE clips have no SNR: they count in "all clips" and get their own accuracy, but not in the per-SNR curves.
Outputs in results/identification/.
"""
from __future__ import annotations

import csv, json, os, sys, time
import numpy as np
import ds1common as dc

OUT = os.path.join(dc.RESULTS, "identification")
os.makedirs(OUT, exist_ok=True)
RF_MAX_TRAIN = 60000


def load_features(sets=("19",)):
    Fs, names = [], []
    for s in sets:
        d = np.load(os.path.join(dc.RESULTS, f"features_{s}.npz"))
        Fs.append(d["F"]); names += [str(n) for n in d["names"]]
    return np.concatenate(Fs, axis=1), names


def score(pred, idx, m, label, extra=None):
    """Accuracy summary + per-class x SNR matrix for test indices idx."""
    from compare_classifiers import first_snr_reaching, sustained_snr_reaching
    classes = m["classes"]
    y, z = m["Y"][idx], m["Z"][idx]
    ok = pred == y
    sig = ~np.isnan(z)
    snrs = sorted(np.unique(z[sig]).tolist())
    acc_snr = [float(ok[sig & (z == s)].mean()) for s in snrs]
    present = sorted(np.unique(y).tolist())
    per_cls_snr = {classes[c]: {int(s): float(ok[(y == c) & (z == s)].mean()) for s in snrs if ((y == c) & (z == s)).any()}
                   for c in present if classes[c] != "NOISE"}
    hi = sig & (z >= 10)
    res = {
        "method": label,
        "n_test": int(len(idx)),
        "accuracy_all_clips": float(ok.mean()),
        "accuracy_signals_all_snr": float(ok[sig].mean()),
        "accuracy_snr_ge_10": float(ok[hi].mean()),
        "accuracy_snr_le_0": float(ok[sig & (z <= 0)].mean()),
        "noise_clips_correct": float(ok[~sig].mean()) if (~sig).any() else None,
        "first_snr_ge_85": first_snr_reaching(snrs, acc_snr),
        "sustained_snr_ge_85": sustained_snr_reaching(snrs, acc_snr),
        "accuracy_vs_snr": {int(s): a for s, a in zip(snrs, acc_snr)},
        "per_class_snr_ge_10": {classes[c]: float(ok[(y == c) & (hi if classes[c] != "NOISE" else ~sig)].mean())
                                for c in present},
        "per_class_all_snr": {classes[c]: float(ok[y == c].mean()) for c in present},
        "per_class_vs_snr": per_cls_snr,
    }
    if extra:
        res.update(extra)
    return res


def confusion(pred, idx, m, mask=None):
    from sklearn.metrics import confusion_matrix
    y = m["Y"][idx]
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return confusion_matrix(y, pred, labels=list(range(len(m["classes"])))).tolist()


def top_confusions(cm, classes, k=10):
    cm = np.array(cm)
    off = []
    for i in range(len(classes)):
        n = cm[i].sum()
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0 and n:
                off.append((cm[i, j] / n, classes[i], classes[j]))
    off.sort(reverse=True)
    return [{"true": t, "pred": p, "fraction_of_true": round(float(f), 4)} for f, t, p in off[:k]]


def write_long_csv(path, results):
    """method,class,snr_db,accuracy (one row per cell), for plotting."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "class", "snr_db", "accuracy"])
        for r in results:
            for c, d in r["per_class_vs_snr"].items():
                for s, a in d.items():
                    w.writerow([r["method"], c, s, f"{a:.4f}"])


def classifiers():
    from compare_classifiers import make_model, CumulantTheoryClassifier, MODELS
    from sklearn.model_selection import train_test_split
    m = dc.meta()
    F, fnames = load_features(("19",))
    tr, te = np.flatnonzero(m["split"] == 0), np.flatnonzero(m["split"] == 2)
    y = m["Y"]
    strata = y * 1000 + np.nan_to_num(m["Z"], nan=-99).astype(int) + 100
    results, cms = [], {}
    for name in MODELS:
        tr_m, te_m, note = tr, te, ""
        if name == "svm":
            tr_m = train_test_split(tr, train_size=40000, stratify=strata[tr], random_state=0)[0]
            note = f"training subsampled {len(tr)} -> {len(tr_m)} (stratified), as in compare_classifiers.py"
        if name == "rf":   # 300 full-depth trees on 156,800 clips need > 4.5 GB RAM (killed); cap like the SVM
            tr_m = train_test_split(tr, train_size=RF_MAX_TRAIN, stratify=strata[tr], random_state=0)[0]
            note = f"training subsampled {len(tr)} -> {len(tr_m)} (stratified) to fit in memory"
        if name == "cumulant_theory":
            model = CumulantTheoryClassifier(m["classes"])
            te_m = te[np.isin(y[te], model.idx)]
            note = "training-free; scored only on " + ",".join(model.names)
        else:
            model = make_model(name, 0)
        cache = os.path.join(OUT, f"pred_{name}.npy")
        if os.path.exists(cache) and os.path.exists(cache + ".json"):     # resume after an interrupted run
            te_m, pred = np.load(cache)
            fit_s, ms = json.load(open(cache + ".json"))
        else:
            t0 = time.perf_counter()
            model.fit(F[tr_m], y[tr_m])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = model.predict(F[te_m])
            ms = (time.perf_counter() - t0) / len(te_m) * 1e3
            np.save(cache, np.stack([te_m, pred]))
            json.dump([fit_s, ms], open(cache + ".json", "w"))
        r = score(pred, te_m, m, name, {"n_train": 0 if name == "cumulant_theory" else int(len(tr_m)),
                                        "fit_seconds": fit_s, "predict_ms_per_clip": ms, "note": note})
        hi = (m["Z"][te_m] >= 10) | np.isnan(m["Z"][te_m])
        cms[name] = {"all": confusion(pred, te_m, m), "snr_ge_10": confusion(pred, te_m, m, hi)}
        r["top_confusions_snr_ge_10"] = top_confusions(cms[name]["snr_ge_10"], m["classes"])
        results.append(r)
        print(f"[{name}] all {r['accuracy_all_clips'] * 100:.1f} %  signals {r['accuracy_signals_all_snr'] * 100:.1f} %  "
              f"SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  85% sustained from {r['sustained_snr_ge_85']}  "
              f"fit {fit_s:.0f} s  {ms:.3f} ms/clip", flush=True)
        print("   top confusions:", ", ".join(f"{c['true']}->{c['pred']} {c['fraction_of_true'] * 100:.0f}%"
                                         for c in r["top_confusions_snr_ge_10"][:5]), flush=True)
    json.dump({"classes": m["classes"], "features": fnames, "results": results, "confusion": cms},
              open(os.path.join(OUT, "classifiers.json"), "w"), indent=1)
    write_long_csv(os.path.join(OUT, "classifiers_class_snr.csv"), results)


def extended():
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = dc.meta()
    tr, te = np.flatnonzero(m["split"] == 0), np.flatnonzero(m["split"] == 2)
    results, cms = [], {}
    for sets in (("19",), ("19", "hist"), ("19", "wav"), ("19", "wav", "hist")):
        if not all(os.path.exists(os.path.join(dc.RESULTS, f"features_{s}.npz")) for s in sets):
            print("skip", sets, "(features not extracted)")
            continue
        F, names = load_features(sets)
        label = " + ".join({"19": "19 features", "wav": "24 Haar wavelet", "hist": "8 histogram entropies"}[s] for s in sets)
        cache = os.path.join(OUT, "pred_hgb.npy")
        if sets == ("19",) and os.path.exists(cache + ".json"):   # identical model to compare_classifiers "hgb"
            te_c, pred = np.load(cache)
            assert np.array_equal(te_c, te)
            fit_s = json.load(open(cache + ".json"))[0]
        else:
            clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True,
                                                 validation_fraction=0.1, random_state=0)
            t0 = time.perf_counter(); clf.fit(F[tr], m["Y"][tr]); fit_s = time.perf_counter() - t0
            pred = clf.predict(F[te])
        r = score(pred, te, m, label, {"n_features": F.shape[1], "fit_seconds": fit_s})
        hi = (m["Z"][te] >= 10) | np.isnan(m["Z"][te])
        cms[label] = {"snr_ge_10": confusion(pred, te, m, hi)}
        r["top_confusions_snr_ge_10"] = top_confusions(cms[label]["snr_ge_10"], m["classes"])
        results.append(r)
        print(f"[{label}] all {r['accuracy_all_clips'] * 100:.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  "
              f"85% sustained from {r['sustained_snr_ge_85']}", flush=True)
    json.dump({"classes": m["classes"], "results": results, "confusion": cms},
              open(os.path.join(OUT, "extended.json"), "w"), indent=1)
    write_long_csv(os.path.join(OUT, "extended_class_snr.csv"), results)


def rules():
    import requirements_methods as rm
    from practical_alrt import to_symbols
    m = dc.meta()
    F, _ = load_features(("19",))
    classes = m["classes"]
    out = {}
    # Azzouz-Nandi hand tree: the 7 RadioML classes mapped onto their Dataset-1 equivalents
    mapping = {"AM": "AM-DSB-WC", "NBFM": "FM", "BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK", "16QAM": "16QAM", "GMSK": "GMSK"}
    y7 = np.full(len(m["Y"]), -1)
    for ds1_name, rml_name in mapping.items():
        y7[m["Y"] == classes.index(ds1_name)] = rm.CLASSES7.index(rml_name)
    tr = np.flatnonzero((m["split"] == 0) & (y7 >= 0))
    te = np.flatnonzero((m["split"] == 2) & (y7 >= 0))
    thr, info = rm.fit_tree(F[tr], y7[tr], np.nan_to_num(m["Z"][tr], nan=-99))
    p7 = rm.predict_tree(F[te], thr)
    back = {rm.CLASSES7.index(v): classes.index(k) for k, v in mapping.items()}
    pred = np.array([back[p] for p in p7])
    out["azzouz_nandi_tree"] = score(pred, te, m, "Azzouz-Nandi hand tree (7 classes)",
                                     {"nodes": info, "mapping": mapping,
                                      "note": "thresholds learned on train clips at Z >= 10 dB, as in requirements_methods.fit_tree"})
    r = out["azzouz_nandi_tree"]
    print(f"Azzouz-Nandi: all {r['accuracy_all_clips'] * 100:.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  85% from {r['sustained_snr_ge_85']}")
    print("  node train errors:", {k: round(v["train_error"], 3) for k, v in info.items()})

    # Swami-Sadler hierarchical cumulants, training-free, symbol rate known (from the sps label)
    dig = ["BPSK", "QPSK", "8PSK", "16QAM"]
    te = np.flatnonzero((m["split"] == 2) & np.isin(m["Y"], [classes.index(c) for c in dig]))
    x = dc.load_x(te)
    t0 = time.perf_counter()
    p5 = np.array([rm.hier_classify(to_symbols(x[i], float(m["sps"][te[i]]))) for i in range(len(te))])
    ms = (time.perf_counter() - t0) / len(te) * 1e3
    pred = np.array([classes.index(rm.DIGITAL[p]) if rm.DIGITAL[p] in classes else -1 for p in p5])
    out["swami_sadler"] = score(pred, te, m, "Swami-Sadler hierarchy (BPSK/QPSK/8PSK/16QAM, symbol rate known)",
                                {"ms_per_clip": ms, "predicted_64QAM_fraction": float(np.mean(p5 == 4)),
                                 "note": "training-free; outputs one of BPSK, QPSK, 8PSK, 16QAM, 64QAM (64QAM counts as wrong)"})
    r = out["swami_sadler"]
    print(f"Swami-Sadler: all {r['accuracy_all_clips'] * 100:.1f} %  SNR>=10 {r['accuracy_snr_ge_10'] * 100:.1f} %  85% from {r['sustained_snr_ge_85']}")
    json.dump(out, open(os.path.join(OUT, "rules.json"), "w"), indent=1)
    write_long_csv(os.path.join(OUT, "rules_class_snr.csv"), list(out.values()))


if __name__ == "__main__":
    {"classifiers": classifiers, "extended": extended, "rules": rules}[sys.argv[1]]()
