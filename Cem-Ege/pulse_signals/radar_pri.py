"""Radar PRI estimation / deinterleaving from times of arrival (TOA), classical methods.

Times are in microseconds.  PRI search band [TMIN, TMAX].

Estimators (all take sorted TOAs and return a dict with coarse PRI, a detection statistic, a 'validated'
flag and the pulses of the validated sequence):
  cdif   : cumulative difference histogram (Mardia 1989; Milojevic & Popovic 1992) on log-spaced bins, level by level,
           Poisson-null threshold, smallest-lag peak group first, verified by sequence search
  sdif   : sequential difference histogram - the histogram of the k-th order differences ONLY at each level
  fft    : periodogram of the TOA impulse train (binned train + zero-padded FFT), Rayleigh/Schuster null,
           sub-harmonic resolution
  acorr  : autocorrelation of the binned (uniform bins) pulse train, Poisson null
  lsgrid : least-squares fit of a regular grid t_k = t0 + n_k PRI (candidates from the periodogram, robust
           inlier iteration) + binomial goodness-of-fit p-value

The coarse PRI of every method is refined with the same sequence search (`refine_chain`): pulses are chained
by predicted-time nearest-neighbour search allowing skipped pulses; PRI = (t_last - t_first) / (number of slots).
So methods differ in how they find the candidate PRI and in how they decide that one exists.

Null (false declaration) thresholds are calibrated on Poisson TOA sequences (dev seeds) by `calibrate`.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binom, poisson

TMIN, TMAX = 100.0, 3000.0
LMIN = 6                  # minimum pulses in a validated sequence
TOL_REL = 0.06            # sequence-search tolerance (fraction of the PRI) - covers +-5 % jitter
MAX_SKIP = 4              # consecutive missing pulses tolerated by the sequence search
RHO = 0.02                # relative width of the log-spaced difference bins
HERE = os.path.dirname(os.path.abspath(__file__))
NULL_FILE = os.path.join(HERE, "results", "radar_null.json")


# ------------------------------------------------------------------ generators
def gen_train(rng, kind="const", n_true=200, T=None, miss=0.0, spur=0.0, jitter=0.0, nlev=2, toa_sigma_rel=0.0):
    """Returns (toas sorted, labels (0 true, -1 spurious), info).  kind: const | jitter (interval jitter +-jitter,
    TOA is the cumulative sum) | toajit (regular grid + uniform TOA error +-jitter*T) | stagger (nlev levels)."""
    if T is None:
        T = float(np.exp(rng.uniform(np.log(300.0), np.log(1500.0))))
    if kind == "stagger":
        spread = rng.uniform(0.10, 0.25)
        u = np.linspace(-1, 1, nlev)
        levels = T * (1 + spread * u)
        levels = levels[rng.permutation(nlev)]
        iv = np.tile(levels, n_true // nlev + 2)[:n_true - 1]
        answers = list(levels) + [float(levels.mean()), float(levels.sum())]
    elif kind == "jitter":
        iv = T * (1 + rng.uniform(-jitter, jitter, n_true - 1))
        answers = [T]
    else:
        iv = np.full(n_true - 1, T)
        answers = [T]
    t = np.concatenate([[0.0], np.cumsum(iv)])
    if kind == "toajit":
        t = t + rng.uniform(-jitter, jitter, n_true) * T
    t = t + rng.uniform(0, T)
    keep = rng.random(n_true) >= miss
    if keep.sum() < 3:
        keep[:3] = True
    tt = t[keep]
    lab = np.zeros(len(tt), int)
    nsp = int(round(spur * n_true))
    if nsp:
        sp = rng.uniform(t[0], t[-1], nsp)
        tt = np.concatenate([tt, sp])
        lab = np.concatenate([lab, -np.ones(nsp, int)])
    o = np.argsort(tt)
    return tt[o], lab[o], dict(T=T, answers=answers, kind=kind, n_true=n_true)


def poisson_train(rng, n, T=None):
    if T is None:
        T = float(np.exp(rng.uniform(np.log(150.0), np.log(1500.0))))
    return np.sort(rng.uniform(0, n * T, n))


def correct(est, answers, tol=0.01):
    return est is not None and any(abs(est / a - 1) <= tol for a in answers)


# ------------------------------------------------------------------ sequence search
def _next_ptr(t, T, tol, max_skip):
    """For each pulse, index of the pulse nearest to t + m*T (smallest m in 1..max_skip+1 within tol) or -1."""
    n = len(t)
    nxt = np.full(n, -1)
    slots = np.zeros(n, int)
    todo = np.ones(n, bool)
    for m in range(1, max_skip + 2):
        pred = t + m * T
        j = np.searchsorted(t, pred)
        jl = np.clip(j - 1, 0, n - 1)
        jr = np.clip(j, 0, n - 1)
        pick = np.where(np.abs(t[jl] - pred) <= np.abs(t[jr] - pred), jl, jr)
        good = todo & (np.abs(t[pick] - pred) <= tol) & (pick > np.arange(n))
        nxt[good] = pick[good]
        slots[good] = m
        todo &= ~good
    return nxt, slots


CH_THR = 6.0              # -log10 p-value a sequence must reach (random-hit binomial null) to be validated


def best_chain(t, T, tol, max_skip=MAX_SKIP):
    """Most significant predicted-time chain.  Significance: the chain has n pulses over S slots; if pulses were
    uniformly random with the observed density, a slot is hit with probability h = 1 - exp(-2 tol lambda), so
    p = P(Bin(S, h) >= n - 1).  Returns (indices, total slots, -log10 p)."""
    n = len(t)
    nxt, slots = _next_ptr(t, T, tol, max_skip)
    ln = np.ones(n, int)
    sl = np.zeros(n, int)
    for i in range(n - 1, -1, -1):
        j = nxt[i]
        if j >= 0:
            ln[i] = 1 + ln[j]
            sl[i] = slots[i] + sl[j]
    lam = n / max(t[-1] - t[0], 1e-9)
    h = min(1 - np.exp(-2 * tol * lam), 0.999)
    ok = ln >= LMIN
    score = np.full(n, -1.0)
    if ok.any():
        score[ok] = -binom.logsf(ln[ok] - 2, np.maximum(sl[ok], 1), h) / np.log(10.0)
    s = int(np.argmax(score))
    idx = [s]
    while nxt[idx[-1]] >= 0:
        idx.append(int(nxt[idx[-1]]))
    return np.array(idx), int(sl[s]), float(score[s])


def refine_chain(t, T0, binw=0.0):
    """Two-pass sequence search around a coarse PRI.  Returns (pri, member indices) or (None, None)."""
    tol = max(TOL_REL * T0, 1.2 * binw)
    idx, sl, sc = best_chain(t, T0, tol)
    if sc < CH_THR or sl == 0:
        return None, None
    pri = (t[idx[-1]] - t[idx[0]]) / sl
    idx2, sl2, sc2 = best_chain(t, pri, TOL_REL * pri)
    if sc2 >= CH_THR and len(idx2) >= len(idx):
        idx, pri = idx2, (t[idx2[-1]] - t[idx2[0]]) / sl2
    return float(pri), idx


# ------------------------------------------------------------------ null thresholds
_NULL = None


def null_thr(method, n_obs):
    global _NULL
    if _NULL is None:
        _NULL = json.load(open(NULL_FILE)) if os.path.exists(NULL_FILE) else {}
    d = _NULL.get(method)
    if d is None:
        return np.inf
    return float(np.interp(np.log(max(n_obs, 5)), np.log(d["n"]), d["thr"]))


def reset_null():
    global _NULL
    _NULL = None


# ------------------------------------------------------------------ difference histograms (CDIF / SDIF)
def _edges():
    nb = int(np.ceil(np.log(TMAX / TMIN) / np.log1p(RHO)))
    return TMIN * (1 + RHO) ** np.arange(nb + 1)


EDGES = _edges()
CENT = np.sqrt(EDGES[:-1] * EDGES[1:])
WID = np.diff(EDGES)


def level_counts(t, kmax=25):
    """Histogram (log bins) of the k-th order differences for k = 1..: list of count arrays."""
    out = []
    for k in range(1, min(kmax, len(t) - 1) + 1):
        d = t[k:] - t[:-k]
        d = d[(d >= EDGES[0]) & (d < EDGES[-1])]
        if len(d) == 0 and k > 1 and (t[k:] - t[:-k]).min() > EDGES[-1]:
            break
        out.append(np.bincount(np.searchsorted(EDGES, d, side="right") - 1, minlength=len(CENT)))
        if (t[k:] - t[:-k]).min() > EDGES[-1]:
            break
    return out


def _null_means(n, span, kmax, cumulative):
    lam = n / span
    base = n * lam * WID * np.clip(1 - CENT / span, 0, None)
    mu = []
    for k in range(1, kmax + 1):
        pk = poisson.cdf(k - 1, lam * CENT) if cumulative else poisson.pmf(k - 1, lam * CENT)
        mu.append(np.maximum(base * pk, 1e-3))
    return mu


def _nlogp(c, mu):
    """-log10 P(Poisson(mu) >= c) (0 for c = 0)."""
    s = -poisson.logsf(c - 1, mu) / np.log(10.0)
    return np.where(c > 0, s, 0.0)


def hist_estimator(t, method):
    """CDIF (cumulative) / SDIF (sequential) estimator."""
    n = len(t)
    if n < 8:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    span = t[-1] - t[0]
    lv = level_counts(t)
    cum = method == "cdif"
    mus = _null_means(n, span, len(lv), cum)
    acc = np.zeros(len(CENT))
    stats = []
    for k, c in enumerate(lv):
        acc = acc + c if cum else c
        stats.append(_nlogp(acc, mus[k]))
    stat = float(max(s.max() for s in stats)) if stats else 0.0
    thr = null_thr(method, n)
    res = dict(pri=None, stat=stat, ok=False, members=None, coarse=None)
    done = False
    acc = np.zeros(len(CENT))
    for k, c in enumerate(lv):
        acc = acc + c if cum else c
        s = stats[k]
        hit = (s >= thr) & (acc >= 3)
        if not hit.any():
            continue
        # group adjacent bins above threshold, candidates ascending in lag
        idx = np.nonzero(hit)[0]
        groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for g in groups[:4]:
            w = acc[g].astype(float)
            T0 = float(np.exp(np.sum(w * np.log(CENT[g])) / w.sum()))
            if res["coarse"] is None:
                res["coarse"] = T0
            pri, mem = refine_chain(t, T0, binw=0.5 * WID[g].sum())
            if pri is not None and TMIN <= pri <= TMAX:
                res.update(pri=pri, ok=True, members=mem)
                done = True
                break
        if done:
            break
    return res


# ------------------------------------------------------------------ binned-train spectral methods
BIN_FFT = 4.0           # us, impulse-train bin for the periodogram
BIN_AC = 20.0           # us, bin of the autocorrelation (chosen on dev seeds)


def _binned(t, dt):
    nb = int(np.ceil((t[-1] - t[0]) / dt)) + 2
    x = np.bincount(((t - t[0]) / dt).astype(int), minlength=nb).astype(float)
    return x


def periodogram(t, dt=BIN_FFT, over=4):
    x = _binned(t, dt)
    x = x - x.mean()                      # remove DC (its leakage would dominate the lowest frequencies)
    nfft = 1 << int(np.ceil(np.log2(len(x) * over)))
    X = np.fft.rfft(x, nfft)
    P = np.abs(X) ** 2 / len(t)
    f = np.arange(len(P)) / (nfft * dt)
    return f, P


def _band(f, span):
    """PRI band; also at least 4 periods must fit in the record."""
    return (f >= max(1.0 / TMAX, 4.0 / span)) & (f <= 1.0 / TMIN)


def fft_estimator(t):
    n = len(t)
    if n < 8:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    f, P = periodogram(t)
    span = t[-1] - t[0]
    band = _band(f, span)
    if band.sum() < 3:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    fb, Pb = f[band], P[band]
    keff = max((fb[-1] - fb[0]) * span, 1.0)
    jmax = int(np.argmax(Pb))
    stat = float(Pb[jmax] - np.log(keff))
    pk = Pb[jmax]
    df = 1.0 / (4 * span)
    # sub-harmonic resolution: smallest f*/m (m = 1..6) with a local peak >= 0.5 of the maximum
    fbest = fb[jmax]
    for m in range(6, 1, -1):
        fm = fb[jmax] / m
        if fm < 1.0 / TMAX:
            continue
        sel = (f >= fm - 2.5 * df) & (f <= fm + 2.5 * df)
        if sel.any() and P[sel].max() >= 0.5 * pk:
            fbest = float(f[sel][np.argmax(P[sel])])
            break
    T0 = 1.0 / fbest
    res = dict(pri=None, stat=stat, ok=False, members=None, coarse=T0)
    pri, mem = refine_chain(t, T0, binw=0.0)
    if pri is not None and TMIN <= pri <= TMAX:
        res.update(pri=pri, members=mem, ok=stat >= null_thr("fft", n))
    return res


def acorr_estimator(t, dt=None):
    dt = dt or BIN_AC
    n = len(t)
    if n < 8:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    x = _binned(t, dt)
    nfft = 1 << int(np.ceil(np.log2(2 * len(x))))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)
    lag = np.arange(len(r)) * dt
    sel = (lag >= TMIN) & (lag <= min(TMAX, (len(x) - 1) * dt))
    L, R = lag[sel], np.rint(r[sel])
    if len(L) == 0:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    span = t[-1] - t[0]
    lam = n / span
    mu = np.maximum(n * lam * dt * np.clip(1 - L / span, 0, None), 1e-3)
    s = _nlogp(R, mu)
    stat = float(s.max())
    thr = null_thr("acorr", n)
    res = dict(pri=None, stat=stat, ok=False, members=None, coarse=None)
    hit = (s >= thr) & (R >= 3)
    if not hit.any():
        return res
    idx = np.nonzero(hit)[0]
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    for g in groups[:4]:
        w = R[g]
        T0 = float(np.sum(w * L[g]) / w.sum())
        if res["coarse"] is None:
            res["coarse"] = T0
        pri, mem = refine_chain(t, T0, binw=dt)
        if pri is not None and TMIN <= pri <= TMAX:
            res.update(pri=pri, ok=True, members=mem)
            break
    return res


LS_TOL = 0.03


def _ls_fit(t, T0, tol_abs):
    """Fine coherence search around T0, then robust LS grid fit with inlier iteration (absolute tolerance)."""
    n = len(t)
    span = t[-1] - t[0]
    nslot = max(span / T0, 2.0)
    fr = (1.0 / T0) * (1 + np.linspace(-1.0, 1.0, 161) / nslot)
    R = np.abs(np.exp(2j * np.pi * np.outer(fr, t - t[0])).sum(1))
    f0 = fr[int(np.argmax(R))]
    T = 1.0 / f0
    T_start = T
    ph = np.angle(np.sum(np.exp(-2j * np.pi * f0 * (t - t[0]))))
    t0 = t[0] + ph / (2 * np.pi) * T
    tol = tol_abs
    for _ in range(4):
        k = np.rint((t - t0) / T)
        inl = np.abs(t - t0 - k * T) <= tol
        if inl.sum() < 3:
            return None
        A = np.vstack([np.ones(inl.sum()), k[inl]]).T
        (t0, T), *_ = np.linalg.lstsq(A, t[inl], rcond=None)
    if not (np.isfinite(T) and abs(T / T_start - 1) <= 2.0 / nslot):
        return None                       # the LS fit wandered away from the periodogram candidate
    k = np.rint((t - t0) / T)
    inl = np.abs(t - t0 - k * T) <= tol
    nin = int(inl.sum())
    p0 = min(2 * tol / T, 0.5)
    pval = binom.logsf(nin - 1, n, p0) / -np.log(10.0)
    return float(T), float(pval), np.nonzero(inl)[0]


def lsgrid_estimator(t, tol_rel=None):
    """Candidates: the 6 strongest periodogram peaks and the sub-harmonics f*/2..f*/4 of the strongest.  One common
    ABSOLUTE tolerance tol_rel / f* (f* = strongest peak) is used for all hypotheses, so that a harmonic
    (shorter grid period, more slots) is less significant than the fundamental."""
    tol_rel = tol_rel or LS_TOL
    n = len(t)
    if n < 8:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    f, P = periodogram(t)
    band = _band(f, t[-1] - t[0])
    if band.sum() < 3:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    fb, Pb = f[band], P[band]
    loc = np.nonzero((Pb[1:-1] > Pb[:-2]) & (Pb[1:-1] >= Pb[2:]))[0] + 1
    if len(loc) == 0:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    loc = loc[np.argsort(-Pb[loc])][:6]
    fstar = fb[loc[0]]
    cands = [1.0 / fb[i] for i in loc] + [m / fstar for m in (2, 3, 4)]
    tol_abs = tol_rel / fstar
    best = None
    for T0 in cands:
        if T0 > TMAX:
            continue
        r = _ls_fit(t, T0, tol_abs)
        if r is None:
            continue
        if TMIN <= r[0] <= TMAX and (best is None or r[1] > best[1]):
            best = r
    if best is None:
        return dict(pri=None, stat=0.0, ok=False, members=None, coarse=None)
    T, st, mem = best
    return dict(pri=T, stat=st, ok=st >= null_thr("lsgrid", n), members=mem, coarse=T)


METHODS = {
    "cdif": lambda t: hist_estimator(t, "cdif"),
    "sdif": lambda t: hist_estimator(t, "sdif"),
    "fft": fft_estimator,
    "acorr": acorr_estimator,
    "lsgrid": lsgrid_estimator,
}


def estimate_pri(t, method):
    return METHODS[method](np.sort(np.asarray(t, float)))


# ------------------------------------------------------------------ deinterleaving by peeling
def deinterleave(t, method, max_emitters=4):
    """Iteratively estimate a PRI, extract its sequence, remove it, repeat.  Returns list of
    (pri, indices into t).  Sequences with a PRI within 2 % of an earlier one are merged into it."""
    t = np.asarray(t, float)
    remaining = np.arange(len(t))
    out = []
    for _ in range(max_emitters + 2):
        if len(remaining) < 8:
            break
        r = METHODS[method](t[remaining])
        if not r["ok"] or r["members"] is None:
            break
        mem = remaining[r["members"]]
        # extend over the whole record: all pulses of the full-train chain within tolerance of the PRI grid
        merged = False
        for e in out:
            if abs(r["pri"] / e[0] - 1) < 0.02:
                e[1] = np.union1d(e[1], mem)
                merged = True
        if not merged:
            out.append([r["pri"], mem])
            if len(out) >= max_emitters:
                remaining = np.setdiff1d(remaining, mem)
                break
        remaining = np.setdiff1d(remaining, mem)
    return [(p, np.asarray(m)) for p, m in out]


# ------------------------------------------------------------------ null calibration
def calibrate(seed=1000, n_grid=(25, 50, 100, 200, 400, 800, 1600), reps=300, alpha=0.01, methods=None, ex=None):
    """99 % quantile of each method's detection statistic on Poisson TOA sequences (pooled over mean spacing)."""
    methods = methods or ["cdif", "sdif", "fft", "acorr", "lsgrid"]
    jobs = [(m, int(n), reps, seed) for m in methods for n in n_grid]
    res = list(ex.map(_cal_job, jobs)) if ex is not None else [_cal_job(j) for j in jobs]
    out = {m: {"n": list(n_grid), "thr": [], "alpha": alpha} for m in methods}
    for (m, n, _, _), v in zip(jobs, res):
        out[m]["thr"].append(float(np.quantile(v, 1 - alpha)))
    return out


def _cal_job(job):
    m, n, reps, seed = job
    rng = np.random.default_rng([seed, n, sum(map(ord, m))])
    reset_null()
    global _NULL
    _NULL = {k: {"n": [5, 1e5], "thr": [-1e9, -1e9]} for k in METHODS}      # no thresholding while calibrating
    return [METHODS[m](poisson_train(rng, n))["stat"] for _ in range(reps)]
