"""Radar PRI benchmark.  Usage: python radar_bench.py <stage> [dev|test] [trials]
stages: cal | sanity | pri | deint | iq | tune
Seeds: namespace 1000 = dev (calibration of null thresholds, tuning), 2000 = test (reported)."""
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import radar_pri as R

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
NW = 20
METH = ["cdif", "sdif", "fft", "acorr", "lsgrid"]
N_CHOICES = [50, 100, 200, 500]


def _rng(ns, *a):
    return np.random.default_rng([ns, *[int(x) for x in a]])


def cid(name):
    return zlib.crc32(name.encode()) % 100003


# ------------------------------------------------------------------ PRI cells
def run_cell(job):
    ns, name, kind, miss, spur, jit, nlev, trials = job
    rng = _rng(ns, cid(name))
    cnt = {m: dict(decl_ok=0, est_ok=0, decl_bad=0, decl=0) for m in METH}
    byn = {m: {n: [0, 0] for n in N_CHOICES} for m in METH}
    for i in range(trials):
        n_true = N_CHOICES[i % 4]
        t, lab, info = R.gen_train(rng, kind, n_true=n_true, miss=miss, spur=spur, jitter=jit, nlev=nlev)
        for m in METH:
            r = R.METHODS[m](t)
            ok_est = R.correct(r["pri"], info["answers"]) if r["pri"] is not None else False
            decl = bool(r["ok"]) and r["pri"] is not None
            c = cnt[m]
            c["decl"] += decl
            c["est_ok"] += ok_est
            c["decl_ok"] += decl and ok_est
            c["decl_bad"] += decl and not ok_est
            byn[m][n_true][0] += decl and ok_est
            byn[m][n_true][1] += 1
    return dict(name=name, kind=kind, miss=miss, spur=spur, jit=jit, nlev=nlev, trials=trials, cnt=cnt, byn=byn)


def pri_cells():
    cells = []
    for miss in (0.0, 0.1, 0.2, 0.3, 0.4):
        for spur in (0.0, 0.25, 0.5, 1.0):
            cells.append((f"const_m{miss}_s{spur}", "const", miss, spur, 0.0, 2))
    for j in (0.01, 0.03, 0.05):
        for miss in (0.0, 0.2, 0.4):
            for spur in (0.0, 0.5, 1.0):
                cells.append((f"jit{j}_m{miss}_s{spur}", "jitter", miss, spur, j, 2))
    for miss in (0.0, 0.2, 0.4):
        for spur in (0.0, 0.5):
            cells.append((f"toajit0.03_m{miss}_s{spur}", "toajit", miss, spur, 0.03, 2))
    for nl in (2, 3, 4):
        for miss in (0.0, 0.2, 0.4):
            for spur in (0.0, 0.5):
                cells.append((f"stag{nl}_m{miss}_s{spur}", "stagger", miss, spur, 0.0, nl))
    return cells


# ------------------------------------------------------------------ stages
def stage_cal(ex, ns=1000, reps=300):
    t = time.time()
    out = R.calibrate(seed=ns, reps=reps, ex=ex)
    json.dump(out, open(R.NULL_FILE, "w"), indent=1)
    print({m: [round(x, 2) for x in v["thr"]] for m, v in out.items()}, f"{time.time() - t:.0f}s")
    R.reset_null()


def _san_job(job):
    ns, method, n, reps = job
    rng = _rng(ns, 77, n, zlib.crc32(method.encode()) % 997)
    stat_exceed = decl = 0
    for _ in range(reps):
        r = R.estimate_pri(R.poisson_train(rng, n), method)
        stat_exceed += r["stat"] >= R.null_thr(method, n)
        decl += bool(r["ok"]) and r["pri"] is not None
    return method, n, reps, stat_exceed, decl


def stage_sanity(ex, ns=2000):
    out = {}
    # 1. noiseless regular trains: exact PRI with every method (coarse = before sequence-search refinement)
    rng = _rng(ns, 5)
    rows = {m: dict(max_rel_err_final=0.0, max_rel_err_coarse=0.0, n=0) for m in METH}
    for _ in range(100):
        T = float(np.exp(rng.uniform(np.log(150), np.log(2500))))
        n = int(rng.choice(N_CHOICES))
        t, _, _ = R.gen_train(rng, "const", n_true=n, T=T)
        for m in METH:
            r = R.estimate_pri(t, m)
            rows[m]["n"] += 1
            rows[m]["max_rel_err_final"] = max(rows[m]["max_rel_err_final"], abs(r["pri"] / T - 1) if r["pri"] else 1.0)
            rows[m]["max_rel_err_coarse"] = max(rows[m]["max_rel_err_coarse"], abs(r["coarse"] / T - 1) if r["coarse"] else 1.0)
    out["noiseless_regular"] = rows
    # 2. Poisson TOA: fraction of sequences whose statistic exceeds the null threshold / that are declared
    jobs = [(ns, m, n, 400) for m in METH for n in (50, 100, 200, 500)]
    res = list(ex.map(_san_job, jobs))
    po = {}
    for m, n, reps, se, dc in res:
        po.setdefault(m, {})[n] = dict(seqs=reps, stat_above_thr=se / reps, declared=dc / reps)
    out["poisson_null"] = po
    json.dump(out, open(os.path.join(OUT, "radar_sanity.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def stage_pri(ex, ns, trials):
    cells = pri_cells()
    jobs = [(ns, c[0], c[1], c[2], c[3], c[4], c[5], trials) for c in cells]
    t = time.time()
    res = list(ex.map(run_cell, jobs))
    print(f"pri grid: {len(res)} cells x {trials} trials, {time.time() - t:.0f}s")
    tag = "dev" if ns == 1000 else "test"
    json.dump(res, open(os.path.join(OUT, f"radar_pri_{tag}.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ two interleaved emitters
def _two_emitters(rng, miss, spur, jit):
    while True:
        Ta, Tb = np.exp(rng.uniform(np.log(300), np.log(1500), 2))
        r = max(Ta, Tb) / min(Ta, Tb)
        if r < 1.07 or r > 4.5:
            continue
        # keep away from simple harmonic ratios p/q (q <= 4): their pulse trains share many TOAs
        if any(abs(r - p / q) < 0.03 * p / q for q in range(1, 5) for p in range(q, 5 * q + 1)):
            continue
        break
    W = 100 * max(Ta, Tb)
    ts, labs = [], []
    for e, T in enumerate((Ta, Tb)):
        n = int(W / T) + 1
        t, _, _ = R.gen_train(rng, "jitter" if jit else "const", n_true=n, T=float(T), miss=miss, spur=0.0, jitter=jit)
        ts.append(t)
        labs.append(np.full(len(t), e))
    t = np.concatenate(ts)
    lab = np.concatenate(labs)
    nsp = int(round(spur * (len(t))))
    if nsp:
        t = np.concatenate([t, rng.uniform(t.min(), t.max(), nsp)])
        lab = np.concatenate([lab, -np.ones(nsp, int)])
    o = np.argsort(t)
    return t[o], lab[o], (float(Ta), float(Tb))


def run_deint(job):
    ns, name, miss, spur, jit, trials = job
    rng = _rng(ns, cid(name), 31)
    out = {m: dict(both=0, one=0, acc=0.0, spur_assigned=0.0, n_found=0.0, trials=0) for m in METH}
    for _ in range(trials):
        t, lab, T2 = _two_emitters(rng, miss, spur, jit)
        for m in METH:
            em = R.deinterleave(t, m, max_emitters=3)
            o = out[m]
            o["trials"] += 1
            o["n_found"] += len(em)
            # match found emitters to true ones by PRI (1 %)
            match = {}
            for e, T in enumerate(T2):
                for k, (p, idx) in enumerate(em):
                    if abs(p / T - 1) <= 0.01 and k not in match.values():
                        match[e] = k
                        break
            o["both"] += len(match) == 2
            o["one"] += len(match) >= 1
            ntrue = int((lab >= 0).sum())
            good = 0
            for e, k in match.items():
                good += int((lab[em[k][1]] == e).sum())
            o["acc"] += good / ntrue
            nsp = int((lab < 0).sum())
            if nsp:
                assigned = sum(int((lab[idx] < 0).sum()) for _, idx in em)
                o["spur_assigned"] += assigned / nsp
    for m in METH:
        n = out[m]["trials"]
        for k in ("both", "one", "acc", "spur_assigned", "n_found"):
            out[m][k] /= n
    return dict(name=name, miss=miss, spur=spur, jit=jit, trials=trials, res=out)


def stage_deint(ex, ns, trials):
    jobs = []
    for jit in (0.0, 0.02):
        for miss in (0.0, 0.2, 0.3):
            for spur in (0.0, 0.25, 0.5):
                jobs.append((ns, f"deint_j{jit}_m{miss}_s{spur}", miss, spur, jit, trials))
    t = time.time()
    res = list(ex.map(run_deint, jobs))
    print(f"deinterleave: {len(res)} cells x {trials}, {time.time() - t:.0f}s")
    json.dump(res, open(os.path.join(OUT, f"radar_deint_{'dev' if ns == 1000 else 'test'}.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ TOAs from an I/Q envelope detector
FS_IQ = 5.0            # MS/s
PW = 2.0               # pulse width, us


def iq_train(rng, snr_db, T, n_pulses, jit):
    """Complex baseband record with n rectangular pulses (random carrier phase per pulse), unit noise."""
    w = int(round(PW * FS_IQ))
    iv = T * (1 + rng.uniform(-jit, jit, n_pulses - 1)) if jit else np.full(n_pulses - 1, T)
    tp = np.concatenate([[0.0], np.cumsum(iv)]) + rng.uniform(5.0, T)
    L = int((tp[-1] + T) * FS_IQ) + 8
    x = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    A = 10 ** (snr_db / 20)
    s0 = np.round(tp * FS_IQ).astype(int)          # pulses start on the sample grid (timing refs are at starts)
    for s in s0:
        x[s:s + w] += A * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return x, s0 / FS_IQ + PW / 2                   # true centres (us)


def detect_toas(x, fa_per_record=1.0):
    from scipy.signal import find_peaks
    from scipy.special import gammaincinv
    w = int(round(PW * FS_IQ))
    p = x.real ** 2 + x.imag ** 2
    y = np.convolve(p, np.ones(w) / w, "valid")          # matched (moving-average) energy detector, sigma^2 = 1
    thr = gammaincinv(w, 1 - fa_per_record / len(y)) / w
    pk, _ = find_peaks(y, height=thr, distance=w)
    # parabolic interpolation of the peak position; window k covers samples k..k+w-1 -> centre k + (w-1)/2 + 0.5
    pos = pk.astype(float)
    ok = (pk > 0) & (pk < len(y) - 1)
    a, b, c = y[pk[ok] - 1], y[pk[ok]], y[pk[ok] + 1]
    den = a - 2 * b + c
    d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    pos[ok] += np.clip(d, -0.5, 0.5)
    return (pos + w / 2.0) / FS_IQ, thr


def run_iq(job):
    ns, snr, jit, trials = job
    rng = _rng(ns, 91, int(snr * 10) + 500, int(jit * 1000))
    from scipy.stats import ncx2
    w = int(round(PW * FS_IQ))
    n_p = 100
    stats = dict(pd=0, pn=0, errs=[], nfalse=0, ndet=0)
    cnt = {m: dict(decl_ok=0, est_ok=0) for m in METH}
    for _ in range(trials):
        T = float(np.exp(rng.uniform(np.log(300), np.log(1500))))
        x, tc = iq_train(rng, snr, T, n_p, jit)
        toa, thr = detect_toas(x)
        j = np.searchsorted(toa, tc)
        jl, jr = np.clip(j - 1, 0, max(len(toa) - 1, 0)), np.clip(j, 0, max(len(toa) - 1, 0))
        if len(toa):
            near = np.where(np.abs(toa[jl] - tc) <= np.abs(toa[jr] - tc), jl, jr)
            e = toa[near] - tc
            hit = np.abs(e) <= PW / 2
            stats["pd"] += int(hit.sum())
            stats["errs"] += list(e[hit])
            stats["nfalse"] += len(toa) - int(hit.sum())
            stats["ndet"] += len(toa)
        stats["pn"] += n_p
        for m in METH:
            if len(toa) < 8:
                continue
            r = R.METHODS[m](toa)
            ok = r["pri"] is not None and R.correct(r["pri"], [T])
            cnt[m]["est_ok"] += ok
            cnt[m]["decl_ok"] += ok and bool(r["ok"])
    s = 10 ** (snr / 10)
    pd_aligned = float(ncx2.sf(2 * w * thr, 2 * w, 2 * w * s))      # P(y > thr) with the window exactly on the pulse
    errs = np.array(stats["errs"])
    return dict(snr=snr, jit=jit, trials=trials, pd=stats["pd"] / stats["pn"], pd_aligned_theory=pd_aligned,
                rms_toa_err_us=float(np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
                false_per_record=stats["nfalse"] / trials, pri_ok={m: cnt[m]["decl_ok"] / trials for m in METH},
                pri_est_ok={m: cnt[m]["est_ok"] / trials for m in METH})


def stage_iq(ex, ns, trials):
    snrs = list(np.arange(-8, 9, 2.0))
    jobs = [(ns, s, j, trials) for j in (0.0, 0.02) for s in snrs]
    t = time.time()
    res = list(ex.map(run_iq, jobs))
    print(f"iq: {len(res)} cells x {trials}, {time.time() - t:.0f}s")
    json.dump(res, open(os.path.join(OUT, f"radar_iq_{'dev' if ns == 1000 else 'test'}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    stage = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "dev"
    ns = 1000 if which == "dev" else 2000
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else (40 if which == "dev" else 100)
    with ProcessPoolExecutor(NW) as ex:
        if stage == "cal":
            stage_cal(ex)
        if stage == "sanity":
            stage_sanity(ex)
        if stage == "pri":
            stage_pri(ex, ns, trials)
        if stage == "deint":
            stage_deint(ex, ns, trials)
        if stage == "iq":
            stage_iq(ex, ns, trials)
