"""Part C - sequential change-point detection of burst onset on a frame-energy sequence.

    python partC_changepoint.py sanity   # CUSUM vs Siegmund / Lorden for a Gaussian mean shift
    python partC_changepoint.py bench    # common ARL0 = 1000 frames, delay & miss vs SNR

Observation model: frames of M = 64 complex samples, unit-power complex Gaussian
noise; after the onset nu a complex Gaussian signal of per-sample SNR rho is
added. Normalised frame energy E_t = sum |x|^2 / sigma^2 is then exactly
Gamma(M, 1) before and Gamma(M, 1 + rho) after the change (sigma^2 known).
Per-frame log-likelihood ratio for a design rho_d:
    l_t = E_t rho_d / (1 + rho_d) - M log(1 + rho_d)

Procedures (all run without restart; alarm time tau = first t with stat_t > h)
    cusum_oracle  Page CUSUM, S_t = max(0, S_{t-1} + l_t), rho_d = true rho
    cusum_fixed   same, rho_d fixed at -6 dB (mismatched design)
    sr_oracle     Shiryaev-Roberts, R_t = (1 + R_{t-1}) e^{l_t}, rho_d = true rho
    page_hinkley  Page-Hinkley on y_t = E_t / M with running mean (pre-change level
                  NOT assumed known), drift delta = rho_d/2 for rho_d = -6 dB
    glr           window-limited GLR, unknown post-change power (MLE, theta >= 1),
                  window W = 400 frames (Willsky & Jones / Lai)
    bocpd         Bayesian online change-point detection (Adams & MacKay 2007):
                  run-length posterior, initial segment = known noise model,
                  new segments = Gamma-scale with conjugate Gamma(1, 1) prior on 1/theta,
                  hazard 1/1000, run lengths truncated at 600; stat = posterior log-odds
                  that a change has occurred.
Thresholds: each procedure's h is set by bisection so that the mean first-alarm
time on noise-only sequences is ARL0 = 1000 frames (1000 sequences x 8000
frames), then checked on an independent noise set.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "partC")
M = 64
ARL0 = 1000
RHO_FIXED_DB = -6.0
SNRS_DB = [-15, -12, -9, -6, -3, 0]


def llr(E, rho):
    return E * (rho / (1 + rho)) - M * np.log1p(rho)


# ------------------------------------------------------------------ statistic paths (R, T)
def cusum_path(l):
    S = np.zeros(l.shape)
    s = np.zeros(l.shape[0])
    for t in range(l.shape[1]):
        s = np.maximum(0.0, s + l[:, t])
        S[:, t] = s
    return S


def sr_path(l):
    """log R_t, R_t = (1 + R_{t-1}) exp(l_t)."""
    out = np.zeros(l.shape)
    lr = np.full(l.shape[0], -np.inf)
    for t in range(l.shape[1]):
        lr = np.logaddexp(0.0, lr) + l[:, t]
        out[:, t] = lr
    return out


def ph_path(E, delta):
    y = E / M
    out = np.zeros(E.shape)
    mean = np.zeros(E.shape[0])
    m = np.zeros(E.shape[0])
    mmin = np.zeros(E.shape[0])
    for t in range(E.shape[1]):
        mean += (y[:, t] - mean) / (t + 1)
        m += y[:, t] - mean - delta
        mmin = np.minimum(mmin, m)
        out[:, t] = m - mmin
    return out


def glr_path(E, W=400):
    R, T = E.shape
    C = np.concatenate([np.zeros((R, 1)), np.cumsum(E, 1)], 1)
    G = np.zeros((R, T))
    for n in range(1, min(W, T) + 1):
        th = (C[:, n:] - C[:, :-n]) / (n * M)
        g = np.where(th > 1, n * M * (th - 1 - np.log(np.maximum(th, 1e-300))), 0.0)
        np.maximum(G[:, n - 1:], g, out=G[:, n - 1:])
    return G


def bocpd_path(E, hazard=1e-3, K=600, a0=1.0, b0=1.0):
    R, T = E.shape
    lh, l1h = np.log(hazard), np.log1p(-hazard)
    logp0 = np.zeros(R)                    # initial (no-change) hypothesis
    logw = np.full((R, K), -np.inf)        # new-segment run lengths r = 0..K-1
    b = np.full((R, K), b0)
    a = a0 + M * np.arange(K)              # rate posterior shape after r observations
    ga = gammaln(a + M) - gammaln(a)
    out = np.zeros((R, T))
    for t in range(T):
        e = E[:, t]
        tot = np.logaddexp(logp0, np.logaddexp.reduce(logw, axis=1))
        logw = np.concatenate([(lh + tot)[:, None], l1h + logw[:, :-1]], 1)
        b = np.concatenate([np.full((R, 1), b0), b[:, :-1]], 1)
        logp0 = l1h + logp0
        # predictive log densities (common E^(M-1)/Gamma(M) factor dropped from all)
        logp0 = logp0 - e
        logw = logw + a * np.log(b) - (a + M) * np.log(b + e[:, None]) + ga
        z = np.logaddexp(logp0, np.logaddexp.reduce(logw, axis=1))
        logp0 -= z
        logw -= z[:, None]
        b = b + e[:, None]
        out[:, t] = np.logaddexp.reduce(logw, axis=1) - logp0
    return out


def path(method, E, rho):
    rf = 10 ** (RHO_FIXED_DB / 10)
    if method == "cusum_oracle":
        return cusum_path(llr(E, rho))
    if method == "cusum_fixed":
        return cusum_path(llr(E, rf))
    if method == "sr_oracle":
        return sr_path(llr(E, rho))
    if method == "page_hinkley":
        return ph_path(E, rf / 2)
    if method == "glr":
        return glr_path(E)
    if method == "bocpd":
        return bocpd_path(E)
    raise ValueError(method)


METHODS = ["cusum_oracle", "cusum_fixed", "sr_oracle", "page_hinkley", "glr", "bocpd"]
RHO_DEPENDENT = {"cusum_oracle", "sr_oracle"}


def first_passage(runmax, h, start=0):
    """runmax (R, T) running max of stat; first index >= start with stat > h (T if never)."""
    hit = runmax[:, start:] > h
    fp = np.where(hit.any(1), hit.argmax(1), runmax.shape[1] - start)
    return fp + start


def calibrate_arl(stat, target=ARL0):
    rm = np.maximum.accumulate(stat, axis=1)
    hi = float(np.max(rm[:, -1]) + 1)
    lo = float(np.quantile(stat[:, :50], 0.5))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(first_passage(rm, mid) + 1) < target:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    fp = first_passage(rm, h) + 1
    return h, float(np.mean(fp)), float(np.mean(fp >= stat.shape[1]))


# ------------------------------------------------------------------ sanity
def sanity():
    """CUSUM for N(0,1) -> N(mu,1): Monte-Carlo ARL0 and ARL1 (change at t=0, S_0=0, which is
    Lorden's worst case) vs Siegmund's approximation and Lorden's asymptote log(ARL0)/I."""
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []

    def siegmund(theta, mu, h):
        d = theta - mu / 2
        b = h / mu + 1.166
        return (np.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)

    for mu in (0.5, 1.0):
        for h in (2.0, 4.0, 6.0):
            T0 = int(20 * siegmund(0.0, mu, h)) + 100
            R0 = 2000
            res = {}
            for name, theta, T, R in (("arl0", 0.0, T0, R0), ("arl1", mu, 400, 4000)):
                fps = []
                for _ in range(max(1, (R * T) // 20_000_000 + 1)):
                    r = R // max(1, (R * T) // 20_000_000 + 1)
                    x = rng.standard_normal((r, T)) + theta
                    S = cusum_path(mu * x - mu * mu / 2)
                    fps.append(first_passage(np.maximum.accumulate(S, 1), h) + 1)
                fp = np.concatenate(fps)
                res[name] = (float(fp.mean()), float(fp.std() / np.sqrt(len(fp))), float(np.mean(fp >= T)))
            rows.append({"mu": mu, "h": h,
                         "arl0_sim": res["arl0"][0], "arl0_se": res["arl0"][1], "arl0_censored_frac": res["arl0"][2],
                         "arl0_siegmund": float(siegmund(0.0, mu, h)),
                         "arl1_sim": res["arl1"][0], "arl1_se": res["arl1"][1],
                         "arl1_siegmund": float(siegmund(mu, mu, h)),
                         "delay_lorden_asymptote": float(np.log(res["arl0"][0]) / (mu * mu / 2))})
            r = rows[-1]
            print(f"mu={mu} h={h}: ARL0 sim {r['arl0_sim']:.1f}+-{r['arl0_se']:.1f} vs Siegmund {r['arl0_siegmund']:.1f};"
                  f"  ARL1 sim {r['arl1_sim']:.2f}+-{r['arl1_se']:.2f} vs Siegmund {r['arl1_siegmund']:.2f}"
                  f" (Lorden asymptote {r['delay_lorden_asymptote']:.2f})", flush=True)
    rel0 = max(abs(r["arl0_sim"] / r["arl0_siegmund"] - 1) for r in rows)
    rel1 = max(abs(r["arl1_sim"] / r["arl1_siegmund"] - 1) for r in rows)
    out = {"rows": rows, "max_rel_err_arl0": rel0, "max_rel_err_arl1": rel1, "pass": bool(rel0 < 0.1 and rel1 < 0.1)}
    with open(os.path.join(OUT, "sanity_cusum.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("max rel err ARL0", round(rel0, 3), "ARL1", round(rel1, 3), "PASS" if out["pass"] else "FAIL")


# ------------------------------------------------------------------ benchmark
def bench(R0=1000, T0=8000, R1=1000, d_max=500, nu_lo=100, nu_hi=300):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    E0 = rng.gamma(M, 1.0, (R0, T0))
    E0v = rng.gamma(M, 1.0, (R0 // 2, T0))          # independent noise set for checking ARL0
    thr, arl_check = {}, {}
    out = {"M": M, "ARL0_target": ARL0, "rho_fixed_db": RHO_FIXED_DB, "snrs_db": SNRS_DB, "d_max": d_max,
           "onset": [nu_lo, nu_hi], "rows": []}
    for m in METHODS:
        for s in (SNRS_DB if m in RHO_DEPENDENT else [None]):
            rho = 10 ** (s / 10) if s is not None else None
            key = f"{m}@{s}" if s is not None else m
            h, arl, cens = calibrate_arl(path(m, E0, rho))
            fpv = first_passage(np.maximum.accumulate(path(m, E0v, rho), 1), h) + 1
            thr[key] = h
            arl_check[key] = {"arl0_cal": arl, "censored_frac": cens, "arl0_independent": float(fpv.mean()),
                              "arl0_independent_se": float(fpv.std() / np.sqrt(len(fpv)))}
            print(f"{key}: h={h:.3f} ARL0 cal {arl:.0f}, independent {fpv.mean():.0f}+-{fpv.std() / np.sqrt(len(fpv)):.0f}",
                  flush=True)
    out["thresholds"], out["arl0_check"] = thr, arl_check
    for s in SNRS_DB:
        rho = 10 ** (s / 10)
        nu = rng.integers(nu_lo, nu_hi + 1, R1)
        T1 = nu_hi + d_max + 1
        E1 = rng.gamma(M, 1.0, (R1, T1))
        post = np.arange(T1)[None, :] >= nu[:, None]
        E1 = np.where(post, rng.gamma(M, 1.0 + rho, (R1, T1)), E1)
        kl = M * (rho - np.log1p(rho))     # KL(post || pre) per frame
        row = {"snr_db": s, "kl_per_frame": float(kl), "lorden_delay_approx": float(np.log(ARL0) / kl)}
        for m in METHODS:
            key = f"{m}@{s}" if m in RHO_DEPENDENT else m
            st = path(m, E1, rho)
            rm = np.maximum.accumulate(st, 1)
            tau = first_passage(rm, thr[key])                # 0-based alarm index (T1 if none)
            early = tau < nu
            delay = tau - nu
            det = (~early) & (delay < d_max)
            row[m] = {"early_alarm_rate": float(early.mean()),
                      "miss_rate": float(np.mean((~early) & (delay >= d_max)) / max(1e-12, 1 - early.mean())),
                      "mean_delay": float(delay[det].mean()) if det.any() else None,
                      "median_delay": float(np.median(delay[det])) if det.any() else None}
        out["rows"].append(row)
        print(s, {m: (row[m]["mean_delay"], round(row[m]["miss_rate"], 3), round(row[m]["early_alarm_rate"], 3))
                  for m in METHODS}, flush=True)
    with open(os.path.join(OUT, "changepoint_bench.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    plot(out)


def siegmund_delay(h, rho):
    """Siegmund-corrected CUSUM ARL1 (S_0 = 0, change at t = 0) for the energy LLR, treating the
    LLR as Gaussian: (exp(-b) + b - 1) / KL1, b = h + 1.166 sd(l). Minus 1 to match the
    'frames after the first post-change frame' delay convention used in the bench."""
    kl1 = M * (rho - np.log1p(rho))
    sd = rho * np.sqrt(M)          # sd of l_t = E rho/(1+rho) - M log(1+rho) under Gamma(M, 1+rho)
    b = h + 1.166 * sd
    return float((np.exp(-b) + b - 1) / kl1 - 1)


def report():
    p = os.path.join(OUT, "changepoint_bench.json")
    out = json.load(open(p))
    for r in out["rows"]:
        r["siegmund_cusum_oracle_delay"] = siegmund_delay(out["thresholds"][f"cusum_oracle@{r['snr_db']}"],
                                                          10 ** (r["snr_db"] / 10))
    json.dump(out, open(p, "w"), indent=2)
    lines = ["| SNR dB | " + " | ".join(METHODS) + " | Siegmund (CUSUM oracle) |", "|---" * (len(METHODS) + 2) + "|"]
    for r in out["rows"]:
        cells = []
        for m in METHODS:
            d = r[m]
            cells.append(f"{d['mean_delay']:.1f} / {d['miss_rate'] * 100:.1f}%" if d["mean_delay"] is not None else "-")
        lines.append(f"| {r['snr_db']} | " + " | ".join(cells) + f" | {r['siegmund_cusum_oracle_delay']:.1f} |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "changepoint_table.md"), "w").write(text)
    print(text)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [r["snr_db"] for r in out["rows"]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for m in METHODS:
        axs[0].semilogy(s, [r[m]["mean_delay"] or np.nan for r in out["rows"]], marker="o", ms=3, label=m)
        axs[1].plot(s, [r[m]["miss_rate"] * 100 for r in out["rows"]], marker="o", ms=3, label=m)
    if "siegmund_cusum_oracle_delay" in out["rows"][0]:
        axs[0].semilogy(s, [max(r["siegmund_cusum_oracle_delay"], 1e-3) for r in out["rows"]], "k--", lw=1,
                        label="Siegmund approx. (CUSUM oracle)")
    axs[0].set_xlabel("per-sample SNR (dB), M = 64 samples/frame")
    axs[0].set_ylabel("mean detection delay (frames)")
    axs[0].grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=7)
    axs[0].set_title(f"Burst-onset delay at ARL0 = {out['ARL0_target']} frames")
    axs[1].set_xlabel("per-sample SNR (dB)")
    axs[1].set_ylabel(f"miss rate (%), no alarm within {out['d_max']} frames")
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "changepoint_bench.png"), dpi=130)


if __name__ == "__main__":
    {"sanity": sanity, "bench": bench, "report": report}[sys.argv[1]]()
