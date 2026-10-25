"""ADS-B / Mode S benchmark.  Usage:  python adsb_bench.py <stage> <dev|test> [trials]
stages: sanity | detect | decode | fa | interf | all
Seeds: namespace 1000 = dev (used only to choose design constants), 2000 = test (reported).
"""
import json
import zlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import modes as M
import adsb_theory as T

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
NW = 20                       # worker processes
FS_LIST = [10, 4, 2]
DET_RATE = 10.0               # target false alarms / s for the detector comparison
DETS = ["energy", "peak", "mf", "mf_norm", "parhalf", "parhalf_norm"]
# decode configurations: (name, detector, candidate rate /s, decoder, fix_bits)
DEC_CFG = [
    ("mf_energy_r10", "mf", 10.0, "energy", 0),
    ("parhalf_energy_r10", "parhalf", 10.0, "energy", 0),
    ("mf_energy", "mf", 2000.0, "energy", 0),
    ("mfnorm_energy", "mf_norm", 2000.0, "energy", 0),
    ("mf_energy_fix1", "mf", 2000.0, "energy", 1),
    ("mf_mlse", "mf", 2000.0, "mlse", 0),
    ("mf_auto_fix1", "mf", 2000.0, "auto", 1),
    ("parhalf_energy", "parhalf", 2000.0, "energy", 0),
]


def _seed(ns, *a):
    return np.random.default_rng([ns, *a])


# ------------------------------------------------------------------ worker
def run_frames(job):
    """Run n frames at (fs, snr) with optional interference; return counters."""
    ns, fs, snr, n, chunk, scen, do_det, cfgs = job
    rng = _seed(ns, fs, int(round(snr * 10)) + 1000, chunk, zlib.crc32(scen.get("name", "").encode()) % 9973)
    rxs = {d: M.get_rx(fs, d, DET_RATE)[0] for d in DETS} if do_det else {}
    cnt = {k: 0 for k in DETS} if do_det else {}
    dec = {c[0]: 0 for c in cfgs}
    wrong = {c[0]: 0 for c in cfgs}
    kw = {k: scen[k] for k in ("garble_sir_db", "dme_mean", "dme_rel_db") if k in scen}
    for _ in range(n):
        bits = M.make_df17(rng)
        iq, s = M.synth_frame(fs, snr, bits, rng, **kw)
        t0 = s / M.FINE
        if do_det:
            p = np.abs(iq) ** 2
            sig2 = float(np.median(p[:int(300 * fs)]) / np.log(2))   # noise floor from the signal-free first 300 us
            for d, rx in rxs.items():
                sc = rx.scores(p, sig2)
                tt = np.arange(sc.shape[1])[None, :] / fs + np.array([ph.phi for ph in rx.phases])[:, None]
                m = np.abs(tt - t0) <= 1.0
                if sc[m].max() > 1.0:
                    cnt[d] += 1
        hexs = M.bits2hex(bits)
        sig2 = float(np.median(np.abs(iq[:int(300 * fs)]) ** 2) / np.log(2))
        for name, det, rate, decd, fix in cfgs:
            r = M.decode_modes_ex(iq, fs * 1e6, fa_rate_hz=rate, detector=det, decoder=decd, fix_bits=fix, sig2=sig2)
            oks = [x for x in r if x["crc_ok"]]
            if any(x["hex"] == hexs for x in oks):
                dec[name] += 1
            elif oks:
                wrong[name] += 1
    return dict(n=n, det=cnt, dec=dec, wrong=wrong)


def sweep(ns, fs, snrs, trials, scen, do_det, cfgs, ex):
    per = 100
    jobs = []
    for snr in snrs:
        for c in range(trials // per):
            jobs.append((ns, fs, snr, per, c, scen, do_det, cfgs))
    res = list(ex.map(run_frames, jobs))
    out = {}
    for (ns_, fs_, snr, per_, c, *_), r in zip(jobs, res):
        o = out.setdefault(snr, dict(n=0, det={}, dec={}, wrong={}))
        o["n"] += r["n"]
        for key in ("det", "dec", "wrong"):
            for k, v in r[key].items():
                o[key][k] = o[key].get(k, 0) + v
    return out


def interp_snr(snrs, pd, target=0.9):
    snrs = np.asarray(snrs, float)
    pd = np.asarray(pd, float)
    for i in range(1, len(snrs)):
        if pd[i] >= target > pd[i - 1]:
            return float(snrs[i - 1] + (target - pd[i - 1]) / (pd[i] - pd[i - 1]) * (snrs[i] - snrs[i - 1]))
    return None if pd[-1] < target else float(snrs[0])


# ------------------------------------------------------------------ stages
def stage_sanity(ns, ex):
    out = {}
    # 1. published example messages
    ex_msgs = ["8D4840D6202CC371C32CE0576098", "8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6"]
    out["crc_examples"] = {h: int(M.remainder(M.hex2bits(h))) for h in ex_msgs}
    out["crc_examples_pass"] = all(v == 0 for v in out["crc_examples"].values())
    # bit-serial parity of the first 88 bits equals the published PI field
    out["crc_parity_matches_PI"] = all(
        f"{M.remainder(np.concatenate([M.hex2bits(h)[:88], np.zeros(24, np.uint8)])):06X}" == h[-6:] for h in ex_msgs)
    # 2. random words: pass probability 2^-24, single-bit-repair acceptance 112 * 2^-24; byte-table == bitwise CRC
    rng = _seed(ns, 1)
    chk = rng.integers(0, 256, (2000, 14), dtype=np.uint8)
    sy = M.syndrome_bytes(chk)
    out["table_equals_bitwise"] = all(int(sy[i]) == M.remainder(np.unpackbits(chk[i])) for i in range(2000))
    tot, npass, nrep = 0, 0, 0
    keys = np.array(sorted(M.SYNDROME_TO_BIT), dtype=np.int64)
    for _ in range(100):
        w = rng.integers(0, 256, (2_000_000, 14), dtype=np.uint8)
        s = M.syndrome_bytes(w)
        npass += int((s == 0).sum())
        nrep += int(np.isin(s, keys).sum())
        tot += len(s)
    out["random_words"] = dict(n=tot, crc_pass=npass, crc_pass_expected=tot / 2 ** 24, single_bit_fixable=nrep,
                               single_bit_fixable_expected=tot * 112 / 2 ** 24)
    # 3. noiseless (80 dB) decode, three rates, energy decoder and auto
    out["noiseless"] = {}
    for fs in FS_LIST:
        res = {}
        for decd in ("energy", "auto"):
            ok = 0
            n = 500
            r2 = _seed(ns, 2, fs)
            for _ in range(n):
                b = M.make_df17(r2)
                iq, s = M.synth_frame(fs, 80, b, r2)
                iq = iq * r2.uniform(0.1, 5)
                r = M.decode_modes_ex(iq, fs * 1e6, decoder=decd)
                ok += any(x["crc_ok"] and x["hex"] == M.bits2hex(b) for x in r)
            res[decd] = ok / n
        out["noiseless"][str(fs)] = res
    # 4. genie-timing BER vs theory (phase 0, known start)
    out["ber"] = {}
    for fs in (10, 2):
        N = int(round(0.5 * fs))
        D = int(round(M.FINE / fs))
        rx, dm = M.get_rx(fs, "mf", 10.0)
        d = dm[0]
        rows = []
        for snr in np.arange(0, 11, 2.0):
            rng = _seed(ns, 3, fs, int(snr))
            err = bits_n = ok = nm = 0
            for _ in range(400):
                b = M.make_df17(rng)
                st = int(rng.integers(100 * 20 // D, 120 * 20 // D)) * D
                iq, s = M.synth_frame(fs, snr, b, rng, start_fine=st, freq_off_khz=0.0)
                p = np.abs(iq) ** 2
                got, _ = d.bits(p[s // D:s // D + d.L])
                e = int((got != b).sum())
                err += e
                bits_n += 112
                ok += e == 0
                nm += 1
            pb = T.pb_theory(snr, N)
            rows.append(dict(snr=snr, ber_meas=err / bits_n, ber_theory=pb, msg_ok_meas=ok / nm,
                             msg_ok_theory=(1 - pb) ** 112))
        out["ber"][str(fs)] = rows
    json.dump(out, open(os.path.join(OUT, "adsb_sanity.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def _theory_detection(fs):
    """Analytic Pd curves at phase 0, exact thresholds of the FIRST phase (genie alignment)."""
    out = {}
    for d in DETS:
        rx = M.Receiver(fs, d, DET_RATE)
        ph = rx.phases[0]
        thr = rx.thr[0]
        if d == "energy":
            f = lambda x, ph=ph, thr=thr: T.pd_energy(x, ph.Np, ph.Nw, thr)
        elif d == "peak":
            f = lambda x, ph=ph, thr=thr: T.pd_peak(x, ph.Np, ph.Nw, thr)
        elif d == "mf":
            f = lambda x, ph=ph, thr=thr: T.pd_sum(x, ph.Np, thr)
        elif d == "mf_norm":
            f = lambda x, ph=ph, thr=thr: T.pd_ratio(x, ph.Np, ph.Nw, thr)
        elif d == "parhalf":
            f = lambda x, ph=ph, thr=thr: T.pd_sum(x, ph.Npa, thr[0]) * T.pd_sum(x, ph.Npb, thr[1])
        else:
            f = lambda x, ph=ph, thr=thr: T.pd_ratio(x, ph.Npa, ph.Nwa, thr[0]) * T.pd_ratio(x, ph.Npb, ph.Nwb, thr[1])
        out[d] = f
    return out


def run_aligned(job):
    """Detection Pd with phase-0 aligned start, no frequency offset, at the analytic 90 % SNR of each detector."""
    ns, fs, det, snr, n, chunk = job
    rng = _seed(ns, 55, fs, zlib.crc32(det.encode()) % 997, chunk)
    rx = M.get_rx(fs, det, DET_RATE)[0]
    D = int(round(M.FINE / fs))
    hit = 0
    for _ in range(n):
        bits = M.make_df17(rng)
        st = int(rng.integers(400 * 20 // D, 420 * 20 // D)) * D
        iq, s = M.synth_frame(fs, snr, bits, rng, start_fine=st, freq_off_khz=0.0)
        p = np.abs(iq) ** 2
        sc = rx.scores(p, 1.0)
        tt = np.arange(sc.shape[1])[None, :] / fs + np.array([ph.phi for ph in rx.phases])[:, None]
        m = np.abs(tt - s / M.FINE) <= 1.0
        hit += sc[m].max() > 1.0
    return hit, n


def stage_aligned(ns, ex, n=1000):
    out = {}
    for fs in FS_LIST:
        th = _theory_detection(fs)
        jobs = []
        for det in DETS:
            snr = T.snr_at(th[det])
            for c in range(n // 100):
                jobs.append((ns, fs, det, snr, 100, c))
        res = list(ex.map(run_aligned, jobs))
        for i, det in enumerate(DETS):
            h = sum(r[0] for r in res[i * (n // 100):(i + 1) * (n // 100)])
            out[f"{det}@{fs}MS/s"] = dict(snr_db=round(T.snr_at(th[det]), 2), pd_measured=h / n, pd_theory=0.9, trials=n)
            print(det, fs, out[f"{det}@{fs}MS/s"], flush=True)
    json.dump(out, open(os.path.join(OUT, "adsb_aligned_check.json"), "w"), indent=1)
    return out


def stage_detect(ns, trials, ex, snrs=None):
    snrs = snrs or list(np.arange(-12, 13, 2.0))
    res = {}
    for fs in FS_LIST:
        t = time.time()
        o = sweep(ns, fs, snrs, trials, {"name": "clean"}, True, [], ex)
        th = _theory_detection(fs)
        curves = {d: [o[s]["det"][d] / o[s]["n"] for s in snrs] for d in DETS}
        tcurves = {d: [th[d](s) for s in snrs] for d in DETS}
        res[str(fs)] = dict(snrs=snrs, trials=trials, pd=curves, pd_theory=tcurves,
                            snr90={d: interp_snr(snrs, curves[d]) for d in DETS},
                            snr90_theory={d: T.snr_at(th[d]) for d in DETS})
        print(fs, "snr90 meas", res[str(fs)]["snr90"], "theory", {k: round(v, 2) for k, v in res[str(fs)]["snr90_theory"].items()},
              f"{time.time() - t:.0f}s", flush=True)
    json.dump(res, open(os.path.join(OUT, f"adsb_detect_{'dev' if ns == 1000 else 'test'}.json"), "w"), indent=1)
    return res


def stage_decode(ns, trials, ex, snrs=None):
    snrs = snrs or list(np.arange(-2, 21, 2.0))
    res = {}
    for fs in FS_LIST:
        t = time.time()
        o = sweep(ns, fs, snrs, trials, {"name": "clean"}, False, DEC_CFG, ex)
        curves = {c[0]: [o[s]["dec"][c[0]] / o[s]["n"] for s in snrs] for c in DEC_CFG}
        wrong = {c[0]: sum(o[s]["wrong"][c[0]] for s in snrs) for c in DEC_CFG}
        res[str(fs)] = dict(snrs=snrs, trials=trials, pdec=curves, wrong_crc_accepts=wrong,
                            snr90={k: interp_snr(snrs, v) for k, v in curves.items()},
                            snr50={k: interp_snr(snrs, v, 0.5) for k, v in curves.items()})
        print(fs, "decode snr90", res[str(fs)]["snr90"], f"{time.time() - t:.0f}s", flush=True)
    json.dump(res, open(os.path.join(OUT, f"adsb_decode_{'dev' if ns == 1000 else 'test'}.json"), "w"), indent=1)
    return res


def _fa_chunk(job):
    fs, det, rate, n_samp, seed = job
    rng = np.random.default_rng(seed)
    rx = M.get_rx(fs, det, rate)[0]
    exceed = 0
    clusters = 0
    last = -10 ** 12
    gap = int(8 * fs)
    B = 1_000_000
    ov = 400
    carry = None
    done = 0
    pos0 = 0
    while done < n_samp:
        n = min(B, n_samp - done)
        x = rng.standard_normal(n + ov) + 1j * rng.standard_normal(n + ov)
        p = (x.real ** 2 + x.imag ** 2) * 0.5
        sc = rx.scores(p, 1.0).max(0)[:n]                # positions fully inside this block (overlap supplies the tail)
        idx = np.nonzero(sc > 1.0)[0] + done
        exceed += len(idx)
        for i in idx:
            if i - last > gap:
                clusters += 1
            last = i
        done += n
    return exceed, clusters


def stage_fa(ns, ex, seconds=20.0):
    res = {}
    for fs in (10, 2):
        n_samp = int(seconds * fs * 1e6)
        nj = 10
        for det in DETS:
            jobs = [(fs, det, DET_RATE, n_samp // nj, [ns, 7, fs, i, zlib.crc32(det.encode()) % 997]) for i in range(nj)]
            r = list(ex.map(_fa_chunk, jobs))
            e = sum(a for a, _ in r)
            c = sum(b for _, b in r)
            res[f"{det}@{fs}MS/s"] = dict(seconds=seconds, target_per_s=DET_RATE, exceedances=e,
                                          exceed_per_s=e / seconds, clusters=c, clusters_per_s=c / seconds)
            print(det, fs, res[f"{det}@{fs}MS/s"], flush=True)
    # end-to-end noise decode: candidates and CRC passes at the decoding operating point (rate 2000/s)
    for fs in (10, 2):
        rng = _seed(ns, 9, fs)
        n = int(5 * fs * 1e6)
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        t = time.time()
        r = M.decode_modes_ex(x, fs * 1e6, fa_rate_hz=2000.0, detector="mf", decoder="auto", fix_bits=1)
        res[f"noise_decode@{fs}MS/s"] = dict(seconds=5.0, candidates=len(r), crc_ok=sum(1 for q in r if q["crc_ok"]),
                                             cands_per_s=len(r) / 5.0, runtime_s=time.time() - t)
        print(res[f"noise_decode@{fs}MS/s"], flush=True)
    json.dump(res, open(os.path.join(OUT, "adsb_fa.json"), "w"), indent=1)
    return res


def stage_interf(ns, trials, ex):
    cfgs = [c for c in DEC_CFG if c[0] in ("mf_energy", "mf_auto_fix1", "parhalf_energy")]
    res = {}
    for fs in (10, 2):
        for snr in (10.0, 20.0):
            rows = {}
            scens = [dict(name="clean")]
            scens += [dict(name=f"garble_sir{s}", garble_sir_db=float(s)) for s in (0, 3, 6, 10, 15)]
            scens += [dict(name=f"dme1_rel{r}", dme_mean=1.0, dme_rel_db=float(r)) for r in (-10, 0, 10)]
            for sc in scens:
                o = sweep(ns, fs, [snr], trials, sc, False, cfgs, ex)[snr]
                rows[sc["name"]] = {k: v / o["n"] for k, v in o["dec"].items()}
                rows[sc["name"]]["wrong"] = {k: v for k, v in o["wrong"].items()}
            res[f"{fs}MS/s_snr{snr:g}"] = dict(trials=trials, rows=rows)
            print(fs, snr, {k: {a: round(b, 3) for a, b in v.items() if a != "wrong"} for k, v in rows.items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "adsb_interf.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    stage = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "dev"
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else (300 if which == "dev" else 1000)
    ns = 1000 if which == "dev" else 2000
    with ProcessPoolExecutor(NW) as ex:
        if stage in ("sanity", "all"):
            stage_sanity(ns, ex)
        if stage in ("aligned", "all"):
            stage_aligned(ns, ex)
        if stage in ("detect", "all"):
            stage_detect(ns, trials, ex)
        if stage in ("decode", "all"):
            stage_decode(ns, trials, ex)
        if stage in ("fa", "all"):
            stage_fa(ns, ex)
        if stage in ("interf", "all"):
            stage_interf(ns, trials, ex)
