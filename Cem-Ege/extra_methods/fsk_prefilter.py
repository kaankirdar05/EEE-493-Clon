"""FSK-family symbol rate at low SNR: band-limiting, and why GFSK/GMSK still fail below 5 dB.

1. Band-limit before the FM discriminator. The discriminator sees all the noise in the sampled band, but
   an FSK signal occupies only a fraction of it. Masking the spectrum to the occupied band first removes
   the out-of-band noise (input SNR gain of about 10*log10(fs / B_occupied)).
   NOTE: the 12-estimator vote of characterization_extended ALREADY contains band-limited discriminator
   members (sr_bank.members filters with fr.bandlimit and a narrower factor 0.4), so band-limiting the
   frame again before the vote is double filtering, not a new method. It is measured here anyway.
2. Diagnosis of the remaining failures: accuracy vs IN-BAND SNR = per-sample SNR + 10*log10(fs / B_kept).
   If the failures are the FM discriminator's threshold effect, accuracy collapses below roughly 8-10 dB
   in-band SNR for all FSK-type classes alike.

Variants per held-out frame (D:\\data\\symbol_rate_torchsig.npz, FSK family, all SNRs 0-20 dB):
    plain_if / bl_if      inst_freq_s3 on the frame / on the band-limited frame
    plain_vote / bl_vote  12-estimator vote (parameters tuned on the DEV set only) on the frame / on the
                          band-limited frame
Nothing is tuned on the held-out frames.

    python fsk_prefilter.py --check
    python fsk_prefilter.py               (parallel; writes results/fsk_prefilter.json and _frames.csv)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "classical_amc"))
sys.path.insert(0, os.path.join(HERE, "..", "characterization_extended"))
FSK = ("2fsk", "4fsk", "2gfsk", "2msk", "2gmsk")


def bandlimit(x, widen=0.25, smooth=33):
    n = len(x)
    X = np.fft.fft(x)
    P = uniform_filter1d(np.abs(np.fft.fftshift(X)) ** 2, smooth, mode="wrap")
    noise = np.median(P)
    k0 = int(np.argmax(P))
    above = P > 2.0 * noise                       # +3 dB over the noise median
    lo = k0
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = k0
    while hi < n - 1 and above[hi + 1]:
        hi += 1
    w = hi - lo + 1
    if w < 8:                                     # no clear occupied band: leave the frame unfiltered
        return x, 1.0
    lo, hi = max(0, lo - int(widen * w)), min(n - 1, hi + int(widen * w))
    mask = np.zeros(n)
    mask[lo:hi + 1] = 1
    return np.fft.ifft(X * np.fft.ifftshift(mask)), (hi - lo + 1) / n


_G = {}


def _init(path):
    from symbol_rate import ESTIMATORS
    import sr_bank
    from blind_alrt import consensus_rates
    _G["est"] = ESTIMATORS["inst_freq_s3"]
    _G["bank"] = sr_bank
    _G["cons"] = consensus_rates
    _G["params"] = json.load(open(os.path.join(HERE, "..", "characterization_extended", "results", "consensus_params.json")))["cons_sr_dev"]
    d = np.load(path)
    _G["d"] = {k: d[k] for k in ("x", "rs", "cls", "snr")}


def _vote(x):
    m = _G["bank"].members(x)
    names = list(m)
    R = np.array([[m[k][0] for k in names]]); P = np.array([[m[k][1] for k in names]])
    return float(_G["cons"](R, P, names, _G["params"])[0])


def _one(i):
    d = _G["d"]
    x, rs = d["x"][i].astype(complex), float(d["rs"][i])
    xb, frac = bandlimit(x)
    ok = lambda e: abs(e - rs) / rs < 0.02
    est = _G["est"]
    return (str(d["cls"][i]), int(d["snr"][i]), rs, frac, ok(est(x)), ok(est(xb)), ok(_vote(x)), ok(_vote(xb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--set", choices=["heldout", "dev"], default="heldout")
    ap.add_argument("--all-classes", action="store_true", help="include PSK/QAM frames (default: FSK family only)")
    args = ap.parse_args()
    path = (r"D:\data\symbol_rate_torchsig.npz" if args.set == "heldout"
            else os.path.join(HERE, "..", "characterization_extended", "data", "symbol_rate_torchsig_dev.npz"))
    tag = args.set + ("_all" if args.all_classes else "_fsk")
    if args.check:
        _init(path)
        from msk_line_pair import msk
        rng = np.random.default_rng(3)
        x = msk(4096, 8.0, rng)
        xb, frac = bandlimit(x)
        e = _G["est"]
        print(f"clean MSK sps 8: plain {1 / e(x):.3f}, band-limited {1 / e(xb):.3f} sps (true 8), kept {frac * 100:.0f} % of band")
        xn = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
        print(f"pure noise: kept {bandlimit(xn)[1] * 100:.1f} % of band")
        return
    d = np.load(path)
    idx = [i for i, c in enumerate(d["cls"]) if args.all_classes or str(c) in FSK]
    with Pool(args.workers, initializer=_init, initargs=(path,)) as p:
        rows = p.map(_one, idx, chunksize=8)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"fsk_prefilter_{tag}_frames.csv"), "w") as f:
        f.write("cls,snr,rs,kept_frac,plain_if,bl_if,plain_vote,bl_vote\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)
    cls = np.array([r[0] for r in rows]); snr = np.array([r[1] for r in rows]); frac = np.array([r[3] for r in rows])
    ok = {k: np.array([r[j] for r in rows], bool) for k, j in (("plain_if", 4), ("bl_if", 5), ("plain_vote", 6), ("bl_vote", 7))}
    res = {"by_class_snr": {}, "by_inband_snr": {}}
    print(f"{len(rows)} frames from the {args.set} set")
    fam = np.isin(cls, FSK)
    res["overall"] = {k: {"all": float(v.mean()), "fsk": float(v[fam].mean()), "psk_qam": float(v[~fam].mean()) if (~fam).any() else None} for k, v in ok.items()}
    for k, q in res["overall"].items():
        print(f"overall {k:>10s}: all {q['all'] * 100:5.1f} %   FSK family {q['fsk'] * 100:5.1f} %   PSK/QAM {('%5.1f %%' % (q['psk_qam'] * 100)) if q['psk_qam'] is not None else 'n/a'}")
    for c in FSK:
        for s in (0, 5, 10):
            m = (cls == c) & (snr == s)
            res["by_class_snr"][f"{c}@{s}"] = {k: float(v[m].mean()) for k, v in ok.items()}
            q = res["by_class_snr"][f"{c}@{s}"]
            print(f"{c:>6s} {s:2d} dB: inst-freq {q['plain_if'] * 100:4.0f} -> {q['bl_if'] * 100:4.0f} %   vote {q['plain_vote'] * 100:4.0f} -> {q['bl_vote'] * 100:4.0f} % (double band-limited)")
    inband = snr - 10 * np.log10(frac)
    edges = [-5, 0, 3, 6, 9, 12, 15, 40]
    print("\nplain vote accuracy vs IN-BAND SNR (per-sample SNR + filter gain), by class group:")
    groups = {"GFSK/GMSK": ("2gfsk", "2gmsk"), "2FSK/4FSK/MSK": ("2fsk", "4fsk", "2msk")}
    print("   in-band SNR [dB)  " + "  ".join(f"{g:>14s}" for g in groups) + "      n")
    for lo, hi in zip(edges[:-1], edges[1:]):
        row, nn = [], 0
        for g, cs in groups.items():
            m = np.isin(cls, cs) & (inband >= lo) & (inband < hi)
            row.append(f"{ok['plain_vote'][m].mean() * 100:13.0f} %" if m.sum() >= 10 else f"{'n<10':>14s}"); nn += int(m.sum())
            res["by_inband_snr"][f"{g}@[{lo},{hi})"] = {"acc": float(ok["plain_vote"][m].mean()) if m.sum() >= 10 else None, "n": int(m.sum())}
        print(f"   [{lo:3d},{hi:3d})        " + "  ".join(row) + f"   {nn:5d}")
    json.dump(res, open(os.path.join(HERE, "results", f"fsk_prefilter_{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
