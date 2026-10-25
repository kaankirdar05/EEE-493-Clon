"""Benchmark 1 + 2 + 5: feature sets x classifiers on RadioML 2018.01A.

Feature sets (extract_radioml_features.py; identical frames to the cached baseline):
    base19      the 19 features of amc/features.py
    ext17       extended cumulants (C61..C84, ratios) + cyclic moments / cumulants
    wav24       Haar wavelet transform statistics
    and their concatenations.
Classifiers:
    hgb         HistGradientBoosting, same settings as train_feature_classifier.py
    gnb         Gaussian naive Bayes
    qda         quadratic discriminant analysis (regularised)
    gmm         per-class Gaussian mixture (full covariance, K chosen by BIC on the class's
                training data), Bayes rule with class priors
Generative models get a monotone per-feature transform (Yeo-Johnson or plain standardisation);
the transform, QDA's reg_param are chosen by 3-fold CV on (a subsample of) the TRAINING split only.

Split: train_test_split(..., test_size=0.25, stratify=class x SNR, random_state=0) - the standard one.

    python bench_features.py --set 7
    python bench_features.py --set 24
"""

from __future__ import annotations

import argparse
import os
import time

import common
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


class GMMBayes(BaseEstimator, ClassifierMixin):
    """Per-class GaussianMixture; number of components chosen by BIC on the class's training data."""

    def __init__(self, ks=(1, 2, 4, 8, 16), reg_covar=1e-4, seed=0):
        self.ks, self.reg_covar, self.seed = ks, reg_covar, seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_, self.k_, self.logprior_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            best = None
            for k in self.ks:
                if k * (X.shape[1] + 1) * 3 > len(Xc):           # too many parameters for the data
                    continue
                g = GaussianMixture(k, covariance_type="full", reg_covar=self.reg_covar, n_init=1,
                                    max_iter=300, random_state=self.seed).fit(Xc)
                bic = g.bic(Xc)
                if best is None or bic < best[0]:
                    best = (bic, g, k)
            self.models_.append(best[1])
            self.k_.append(best[2])
            self.logprior_.append(np.log(len(Xc) / len(X)))
        return self

    def predict(self, X):
        L = np.stack([m.score_samples(X) + lp for m, lp in zip(self.models_, self.logprior_)], axis=1)
        return self.classes_[np.argmax(L, axis=1)]


def tune(estimators: dict, X, y, seed=0, n_max=40000):
    """Pick the best of several candidate pipelines by 3-fold CV accuracy on training data."""
    if len(y) > n_max:
        idx = np.random.default_rng(seed).choice(len(y), n_max, replace=False)
        X, y = X[idx], y[idx]
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    scores = {name: float(np.mean(cross_val_score(est, X, y, cv=cv, n_jobs=3))) for name, est in estimators.items()}
    best = max(scores, key=scores.get)
    return best, scores


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def transforms():
    return {"std": StandardScaler(), "yeoj": PowerTransformer(method="yeo-johnson", standardize=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["7", "24"], required=True)
    ap.add_argument("--skip-generative", action="store_true")
    args = ap.parse_args()
    d = np.load(os.path.join(common.RESULTS, f"features_{args.set}.npz"))
    y, snr, classes = d["y"].astype(int), d["snr"].astype(int), [str(c) for c in d["classes"]]
    B, E, W = d["base"], d["ext"], d["wav"]
    ms = {"base19": float(d["ms_base"]), "ext17": float(d["ms_ext"]), "wav24": float(d["ms_wav"])}
    sets = {"base19": (B, ms["base19"]), "ext17": (E, ms["ext17"]), "wav24": (W, ms["wav24"]),
            "base19+ext17": (np.hstack([B, E]), ms["base19"] + ms["ext17"]),
            "base19+wav24": (np.hstack([B, W]), ms["base19"] + ms["wav24"]),
            "base19+ext17+wav24": (np.hstack([B, E, W]), sum(ms.values()))}
    tr, te = common.standard_split(y, snr)
    print(f"RadioML {args.set} classes: {len(y)} frames, train {len(tr)}, test {len(te)}; "
          f"cache labels identical: {bool(d['cache_labels_identical'])}, max |base - cache| {float(d['cache_max_abs_diff'])}")
    out = {"set": args.set, "classes": classes, "n_train": len(tr), "n_test": len(te),
           "feature_ms_per_frame": ms, "frames_identical_to_cache": bool(d["cache_labels_identical"]),
           "max_abs_diff_base_vs_cache": float(d["cache_max_abs_diff"]), "results": {}, "tuning": {}}
    res = out["results"]

    def run(name, model, X, fms, extra=None):
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = model.predict(X[te])
        pms = (time.perf_counter() - t0) / len(te) * 1e3
        hi = snr[te] >= 10
        per_cls = {c: float(np.mean(p[hi & (y[te] == k)] == k)) for k, c in enumerate(classes)}
        res[name] = common.summarize(p, y[te], snr[te], fms + pms,
                                     {"fit_seconds": fit_s, "predict_ms_per_frame": pms,
                                      "per_class_acc_snr_ge_10": per_cls, **(extra or {})})
        print(common.fmt_row(name, res[name]), flush=True)
        common.save_json(out, f"bench_features_{args.set}.json")

    for fs, (X, fms) in sets.items():
        run(f"hgb | {fs}", make_hgb(), X, fms)

    if not args.skip_generative:
        for fs in ("base19", "base19+ext17", "base19+ext17+wav24"):
            X, fms = sets[fs]
            Xtr = X[tr]
            # Gaussian naive Bayes: choose transform on train CV
            cand = {t: make_pipeline(tf, GaussianNB()) for t, tf in transforms().items()}
            b, sc = tune(cand, Xtr, y[tr])
            out["tuning"][f"gnb | {fs}"] = sc
            run(f"gnb | {fs}", cand[b], X, fms, {"chosen": b})
            # QDA: transform x reg_param
            cand = {f"{t}|reg={r}": make_pipeline(clone(tf), QuadraticDiscriminantAnalysis(reg_param=r))
                    for t, tf in transforms().items() for r in (1e-4, 1e-3, 1e-2, 0.05, 0.2)}
            b, sc = tune(cand, Xtr, y[tr])
            out["tuning"][f"qda | {fs}"] = sc
            run(f"qda | {fs}", cand[b], X, fms, {"chosen": b})
            # GMM Bayes: transform chosen by QDA's CV; K per class by BIC on training data
            tf = transforms()[b.split("|")[0]]
            gm = make_pipeline(tf, GMMBayes())
            run(f"gmm | {fs}", gm, X, fms, {"transform": b.split("|")[0]})
            res[f"gmm | {fs}"]["components_per_class"] = dict(zip(classes, map(int, gm[-1].k_)))
            common.save_json(out, f"bench_features_{args.set}.json")

    common.plot_overview({k: v for k, v in res.items()},
                         f"RadioML 2018 {args.set} classes - extended features and generative classifiers (test split)",
                         os.path.join(common.RESULTS, f"bench_features_{args.set}.png"),
                         highlight={"hgb | base19", "hgb | base19+ext17", "hgb | base19+ext17+wav24"})
    print("saved", common.save_json(out, f"bench_features_{args.set}.json"))


if __name__ == "__main__":
    main()
