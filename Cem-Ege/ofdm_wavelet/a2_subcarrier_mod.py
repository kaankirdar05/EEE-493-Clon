"""Part A2 - subcarrier modulation classification inside OFDM (QPSK / 16QAM / 64QAM).

Pipeline per frame (8192 samples, N in {64..1024}, cp = N/4..N/32, 75 % used subcarriers, random timing / carrier phase / CFO up to +-0.3 spacings):
  1. (N, cp): blind = ofdm_params.estimate(x) (20-candidate grid)  | oracle = truth
  2. timing theta (CP start) = argmax of the CP-correlation window of the chosen (N, cp)  | oracle = truth
  3. fractional CFO = angle of the CP correlation, derotate; remove CP (FFT window starts `backoff = max(1, cp//4)` samples inside the CP when equalising);
     FFT per OFDM symbol; used bins from the average power profile (smoothed, two-level threshold, DC excluded, 2-bin erosion); pilot comb removed
  4. channel: AWGN, or static 2-3 tap multipath (all delays <= min(cp/2, 4) samples < cp, exponentially decaying Rayleigh taps, new draw per frame)
  5. equalisation (variant "eq"): ASSUMED PILOTS = comb on every 4th FFT bin inside the used band (k % 4 == 0), fixed known QPSK sequence in every OFDM
     symbol (25 % of the used bins, excluded from classification). Per-symbol common phase from the pilots, bulk-delay slope removal, MMSE-ridge LS fit of a short
     FIR (taps -3..D+3) on the pilot bins, one-tap zero-forcing on the data bins, bins with |H|^2 < 0.3 mean dropped (0.3 chosen on dev seed 100 among {0, 0.1, 0.3}).
     Variant "noeq": plain FFT at the estimated end of the CP (no backoff), only the CFO derotation, no phase tracking / equalisation.
  6. classify the unit-power subcarrier symbols (4000-5000 per frame):
     hier  = Swami-Sadler hierarchy restricted to {QPSK, 16QAM, 64QAM}, thresholds = midpoints of the theoretical cumulants from requirements_methods.TH
     em    = EM constellation fit (symbol_classifiers.em_fit), max-likelihood over the 3 classes (<= 1000 randomly chosen symbols, 2 phase restarts, <= 60 iterations, for speed)

    python a2_subcarrier_mod.py --check
    python a2_subcarrier_mod.py --dev        (tuning probes on seed 100; not reported)
    python a2_subcarrier_mod.py              (test seed 200)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from joblib import Parallel, delayed

from ofdmlib import *  # noqa
import requirements_methods as rm   # noqa  existing (Swami-Sadler thresholds)
import symbol_classifiers as sc     # noqa  existing (EM)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODS3 = ["QPSK", "16QAM", "64QAM"]
SNRS = [0, 5, 10, 15, 20, 25, 30]
NS = (64, 128, 256, 512, 1024)
CPD = (4, 8, 16, 32)
TH = rm.TH


def cum_stats(z, noise=None):
    """(|C40|, C42) normalised by the signal power c21 - noise. noise=None -> M2M4 estimate (assumes constant modulus; BIASED for QAM, see README)."""
    z = z - z.mean()
    n = rm.m2m4_noise(z) if noise is None else noise * np.mean(np.abs(z) ** 2)
    c21 = np.mean(np.abs(z) ** 2)
    ps = max(c21 - n, 1e-3 * c21)
    m20, m40, m42 = np.mean(z ** 2), np.mean(z ** 4), np.mean(np.abs(z) ** 4)
    return abs(m40 - 3 * m20 ** 2) / ps ** 2, (m42 - abs(m20) ** 2 - 2 * c21 ** 2) / ps ** 2


def hier3(z, noise=None):
    c40, c42 = cum_stats(z, noise)
    if c40 > (TH["QPSK"][0] + TH["16QAM"][0]) / 2:
        return 0
    return 2 if c42 > (TH["16QAM"][1] + TH["64QAM"][1]) / 2 else 1


def hier_default(z):
    """existing rm.hier_classify EXACTLY as the baseline calls it (noise=None -> M2M4; 5 outputs BPSK, QPSK, 8PSK, 16QAM, 64QAM), mapped to 0/1/2, -1 otherwise."""
    k = rm.hier_classify(z)
    return {1: 0, 3: 1, 4: 2}.get(k, -1)


def em3(z, rng):
    if len(z) > 1000:
        z = z[rng.choice(len(z), 1000, replace=False)]
    return int(np.argmax([sc.em_fit(z, c, n_restarts=2, max_iter=60)[0] for c in MODS3]))


def run_frame(seed, i, snr, chan, drop_fade=0.3, with_em=True):
    rng = np.random.default_rng([seed, i])
    n = int(rng.choice(NS)); d = int(rng.choice(CPD)); cp = n // d
    lab = int(rng.integers(0, 3))
    h = make_channel(cp, rng) if chan == "multipath" else None
    x, tr = make_ofdm(n, cp, MODS3[lab], snr, rng, channel=h)
    (ne, ce), _ = op.estimate(x)
    ok_est = (ne, ce) == (n, cp)
    out = {"lab": lab, "snr": snr, "chan": chan, "n": n, "cp": cp, "est_ok": ok_est, "est_n_ok": ne == n}
    erng = np.random.default_rng([seed, i, 1])
    for tag, (nn, cc, th) in {"oracle": (n, cp, tr["theta"]), "blind": (ne, ce, None)}.items():
        for eqn, eq in (("eq", True), ("noeq", False)):
            try:
                z, info = subcarrier_symbols(x, nn, cc, th, equalize=eq, backoff=None if eq else 0, drop_fade=drop_fade, return_info=True)
                if len(z) < 200:
                    raise ValueError
                out[f"{tag}_{eqn}_hier"] = hier3(z, info["nvar"])
                out[f"{tag}_{eqn}_hier_default"] = hier_default(z)
                out[f"{tag}_{eqn}_em"] = em3(z, erng) if with_em else -2
                if tag == "oracle" and eq:
                    tu, tp = tr["used"], tr["pil"]
                    mk = info["used"]
                    out["mask_recall_interior"] = float(mk[tu & np.roll(tu, 3) & np.roll(tu, -3)].mean())
                    out["mask_false_pos"] = float((mk & ~tu).sum() / max(mk.sum(), 1))
                    out["keep_frac"] = info.get("keep", 1.0)
            except Exception:
                out[f"{tag}_{eqn}_hier"] = out[f"{tag}_{eqn}_hier_default"] = out[f"{tag}_{eqn}_em"] = -1
    return out


def check():
    """Sanity: (1) FFT of a noiseless frame = tx symbols (common phase) [see check_ofdmlib.py]; (2) 30 dB AWGN, oracle timing, hier/EM ~ 100 %;
    (3) the unmodified rm.hier_classify check on ideal symbols still passes; (4) blind-estimate agrees with truth at 30 dB."""
    rng = np.random.default_rng(300)
    # (1)
    for (n, cp) in [(64, 16), (256, 32), (1024, 128)]:
        x, tr = make_ofdm(n, cp, "16QAM", 100, rng, cfo=False, noise=False)
        j = tr["first_sym"]; s0 = j * tr["period"] - tr["off"] + cp
        Y = np.fft.fft(x[s0:s0 + n]); u = tr["used"]
        r = Y[u] / tr["sym"][j][u]
        print(f"FFT(noiseless frame, N={n}) / transmitted symbols: |std/mean| of the ratio over used bins = {np.std(r) / abs(np.mean(r)):.1e}; null-bin power / used power = {np.mean(np.abs(Y[~u]) ** 2) / np.mean(np.abs(Y[u]) ** 2):.0e}")
    # (2)
    res = {}
    rows = Parallel(n_jobs=20)(delayed(run_frame)(300, i, 30, "awgn") for i in range(300))
    for key in ("oracle_noeq_hier", "oracle_noeq_hier_default", "oracle_noeq_em", "oracle_eq_hier", "oracle_eq_em", "blind_eq_hier", "blind_eq_em"):
        a = np.mean([r[key] == r["lab"] for r in rows])
        res[key] = float(a)
        print(f"30 dB AWGN, 300 frames, {key}: {a * 100:.1f} %")
    print("blind (N, cp) exact at 30 dB:", np.mean([r["est_ok"] for r in rows]))
    print("mask recall (interior) / false-positive share:", np.mean([r["mask_recall_interior"] for r in rows]), np.mean([r["mask_false_pos"] for r in rows]))
    print("rm.check() (ideal symbols, hierarchy):", rm.check())
    json.dump(res, open(os.path.join(RES, "a2_sanity.json"), "w"), indent=1)
    return all(res[k] > 0.97 for k in ("oracle_noeq_hier", "oracle_noeq_em"))


def summarize(rows, key, by):
    a = np.array([r[key] == r["lab"] for r in rows])
    grp = np.array([r[by] for r in rows])
    return {str(g): float(a[grp == g].mean()) for g in sorted(set(grp.tolist()))}, float(a.mean())


def dev():
    for chan in ("awgn", "multipath"):
        for df in (0.0, 0.1, 0.3):
            rows = []
            for snr in (10, 20, 30):
                rows += Parallel(n_jobs=20)(delayed(run_frame)(100, i + 1000 * snr, snr, chan, df, False) for i in range(300))
            a = {k: summarize(rows, k, "snr")[0] for k in ("oracle_eq_hier", "blind_eq_hier")}
            print(chan, "drop_fade", df, {k: {s: round(v, 3) for s, v in d.items()} for k, d in a.items()}, flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    per_snr = 360
    out = {"seed": 200, "per_snr_per_channel": per_snr, "snrs": SNRS, "tables": {}}
    allrows = {}
    for chan in ("awgn", "multipath"):
        rows = []
        for snr in SNRS:
            rows += Parallel(n_jobs=20, batch_size=2)(delayed(run_frame)(200, i + 100000 * (snr + 1) + (0 if chan == "awgn" else 50000), snr, chan) for i in range(per_snr))
            print(chan, snr, f"{time.time() - t0:.0f} s", flush=True)
        allrows[chan] = rows
        T = {}
        for key in sorted(k for k in rows[0] if k.startswith(("oracle_", "blind_")) and k[-3:] != "_ok"):
            v, a = summarize(rows, key, "snr")
            T[key] = {"acc_all": a, "acc_vs_snr": v}
        T["blind_Ncp_exact_vs_snr"] = {str(s): float(np.mean([r["est_ok"] for r in rows if r["snr"] == s])) for s in SNRS}
        T["blind_N_only_vs_snr"] = {str(s): float(np.mean([r["est_n_ok"] for r in rows if r["snr"] == s])) for s in SNRS}
        T["mask_recall_interior_vs_snr"] = {str(s): float(np.mean([r["mask_recall_interior"] for r in rows if r["snr"] == s and "mask_recall_interior" in r])) for s in SNRS}
        T["mask_false_pos_vs_snr"] = {str(s): float(np.mean([r["mask_false_pos"] for r in rows if r["snr"] == s and "mask_false_pos" in r])) for s in SNRS}
        # per-(N, cp ratio) at 20 dB and per-modulation confusion at 20 dB for blind_eq
        for key in ("blind_eq_hier", "blind_eq_em"):
            sel = [r for r in rows if r["snr"] == 20]
            cm = np.zeros((3, 4), int)
            for r in sel:
                cm[r["lab"], r[key] if r[key] >= 0 else 3] += 1
            T[f"confusion_20dB_{key}"] = {"rows_true": MODS3, "cols_pred": MODS3 + ["fail"], "counts": cm.tolist()}
            T[f"acc_by_N_20dB_{key}"] = {str(n): float(np.mean([r[key] == r["lab"] for r in sel if r["n"] == n])) for n in NS}
            T[f"acc_by_cp_ratio_20dB_{key}"] = {f"N/{d}": float(np.mean([r[key] == r["lab"] for r in sel if r["n"] // r["cp"] == d])) for d in CPD}
            T[f"acc_by_mod_vs_snr_{key}"] = {m: {str(s): float(np.mean([r[key] == r["lab"] for r in rows if r["snr"] == s and r["lab"] == k])) for s in SNRS} for k, m in enumerate(MODS3)}
        out["tables"][chan] = T
        print(chan, "done")
        for key in ("oracle_noeq_hier", "oracle_eq_hier", "blind_eq_hier", "blind_noeq_hier", "oracle_noeq_em", "oracle_eq_em", "blind_eq_em", "blind_noeq_em"):
            print(f"  {key:>18s}: " + " ".join(f"{s}:{T[key]['acc_vs_snr'][str(s)] * 100:5.1f}" for s in SNRS))
        print("  blind (N,cp) exact:", " ".join(f"{s}:{T['blind_Ncp_exact_vs_snr'][str(s)] * 100:5.1f}" for s in SNRS))
        print(f"  confusion 20 dB blind_eq_hier: {T['confusion_20dB_blind_eq_hier']['counts']}  blind_eq_em: {T['confusion_20dB_blind_eq_em']['counts']}")
        print("  by N @20dB blind_eq_hier", T["acc_by_N_20dB_blind_eq_hier"], "by cp", T["acc_by_cp_ratio_20dB_blind_eq_hier"])
    json.dump(out, open(os.path.join(RES, "a2_subcarrier_mod.json"), "w"), indent=1)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    sty = {"oracle_eq": ("tab:green", "-"), "blind_eq": ("tab:blue", "-"), "oracle_noeq": ("tab:red", "--"), "blind_noeq": ("tab:orange", "--")}
    for r, chan in enumerate(("awgn", "multipath")):
        T = out["tables"][chan]
        for c, cl in enumerate(("hier", "em")):
            a = ax[r, c]
            for k, (col, ls) in sty.items():
                v = T[f"{k}_{cl}"]
                a.plot(SNRS, [v["acc_vs_snr"][str(s)] * 100 for s in SNRS], color=col, ls=ls, marker="o", ms=4, label=f"{k} ({v['acc_all'] * 100:.1f} %)")
            a.axhline(100 / 3, color="gray", ls=":", lw=1)
            a.set_title(f"{chan}, classifier = {cl}", fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right")
            if r == 1: a.set_xlabel("per-sample SNR (dB)")
            if c == 0: a.set_ylabel("QPSK/16QAM/64QAM accuracy (%)")
    fig.suptitle(f"A2: subcarrier modulation inside OFDM ({out['per_snr_per_channel']} test frames per SNR per channel, seed {out['seed']}); dotted = chance", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "a2_subcarrier_mod.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--dev", action="store_true"); a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.check:
        sys.exit(0 if check() else 1)
    if a.dev:
        dev()
    else:
        main()
