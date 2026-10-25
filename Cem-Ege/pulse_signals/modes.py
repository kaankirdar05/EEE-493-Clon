"""Mode S / ADS-B (DF17) toolbox: CRC-24, synthetic frame generator, preamble detectors, PPM decoder.

Time unit: microseconds.  fs is given in Hz in the public API `decode_modes(iq, fs)` and in MS/s inside
the helper classes (suffix _mhz).

Sample convention (shared by synthesiser and receiver): sample n of the receiver is the average of the
continuous-time complex baseband signal over [n/fs, (n+1)/fs) (boxcar integrate-and-dump, white AWGN
of unit variance added afterwards at the sample rate).  A message that starts at time t0 = n/fs + phi
therefore has sub-sample phase phi in [0, 1/fs).  The receiver tries K = round(20/fs_mhz) evenly spaced
phases (0.05 us spacing at 10/4/2 MS/s) -> the synthetic timing grid is matched exactly.

References: Mode S parity: ICAO Annex 10 Vol IV 3.1.2.3.3.1 (generator 0x1FFF409);
Gertz, "Fundamentals of Mode S parity coding", MIT-LL ATC-117 (1984);
J. Sun, "The 1090 MHz Riddle" (example messages used in the CRC check).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d
from scipy.special import betaincinv, gammaincinv, i0e

POLY = 0xFFF409
CAPTURE = 0.90                   # a pulse sample set must hold >= 90 % of the pulse energy
FINE = 20.0                      # synthesiser fine grid, ticks per us (0.05 us)
PREAMBLE_PULSES = [(0.0, 0.5), (1.0, 1.5), (3.5, 4.0), (4.5, 5.0)]
PRE_US, MSG_US, NBITS = 8.0, 120.0, 112
HALF_A = [(0.0, 0.5), (1.0, 1.5)]          # pulse pair A and its normalisation window (0..2 us)
HALF_B = [(3.5, 4.0), (4.5, 5.0)]          # pulse pair B (3.5..5.5 us)
WIN_A, WIN_B = (0.0, 2.0), (3.5, 5.5)


# ------------------------------------------------------------------ CRC-24
def remainder(bits) -> int:
    """Bitwise long division of the bit string (MSB first) by the generator; valid DF17 frame -> 0."""
    reg = 0
    for b in bits:
        top = (reg >> 23) & 1
        reg = ((reg << 1) & 0xFFFFFF) ^ int(b)
        if top:
            reg ^= POLY
    return reg


def hex2bits(h: str) -> np.ndarray:
    return np.array([int(c) for c in bin(int(h, 16))[2:].zfill(len(h) * 4)], dtype=np.uint8)


def bits2hex(bits) -> str:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:0{len(bits) // 4}X}"


# syndrome of every single-bit error pattern in a 112-bit word (linear code: syndrome(x) = XOR of these)
_UNIT = np.array([remainder(np.eye(1, NBITS, i, dtype=np.uint8)[0]) for i in range(NBITS)], dtype=np.int64)
assert len(set(_UNIT.tolist())) == NBITS and 0 not in _UNIT
SYNDROME_TO_BIT = {int(s): i for i, s in enumerate(_UNIT)}
# byte-wise lookup tables for fast vectorised syndromes: syndrome = XOR_j T[j][byte_j]
_BYTE_T = np.zeros((14, 256), dtype=np.int64)
for _j in range(14):
    for _v in range(256):
        _s = 0
        for _k in range(8):
            if (_v >> (7 - _k)) & 1:
                _s ^= int(_UNIT[8 * _j + _k])
        _BYTE_T[_j, _v] = _s


def syndrome_bytes(arr: np.ndarray) -> np.ndarray:
    """arr: (N,14) uint8 -> (N,) syndromes."""
    s = np.zeros(arr.shape[0], dtype=np.int64)
    for j in range(14):
        s ^= _BYTE_T[j][arr[:, j]]
    return s


def make_df17(rng) -> np.ndarray:
    """Random valid DF17 frame (112 bits): DF=17, CA=5, random ICAO + 56-bit ME, correct CRC parity."""
    body = np.concatenate([[1, 0, 0, 0, 1], [1, 0, 1], rng.integers(0, 2, 80)]).astype(np.uint8)  # 88 bits
    par = remainder(np.concatenate([body, np.zeros(24, np.uint8)]))
    pb = np.array([(par >> (23 - i)) & 1 for i in range(24)], np.uint8)
    out = np.concatenate([body, pb])
    assert remainder(out) == 0
    return out


# ------------------------------------------------------------------ synthesiser
def _msg_env(bits) -> np.ndarray:
    """Fine-grid (20 ticks/us) 0/1 envelope of a Mode S message (preamble + PPM data), length 2400."""
    b = np.asarray(bits, np.uint8)
    slots = np.zeros(16 + 2 * len(b), np.uint8)
    slots[[0, 2, 7, 9]] = 1
    slots[16 + 2 * np.arange(len(b))] = b            # bit 1: pulse in first half
    slots[16 + 2 * np.arange(len(b)) + 1] = 1 - b    # bit 0: pulse in second half
    return np.repeat(slots, 10).astype(float)


def synth_frame(fs_mhz, snr_db, bits, rng, frame_us=560.0, start_fine=None, freq_off_khz=200.0,
                garble_sir_db=None, dme_mean=0.0, dme_rel_db=0.0, noise=True):
    """Complex baseband frame at fs_mhz.  Returns (iq, start_fine) where start_fine is the message start in
    fine ticks (0.05 us).  snr_db = (power while a pulse is on) / (noise power per complex sample)."""
    D = int(round(FINE / fs_mhz))
    assert abs(D * fs_mhz - FINE) < 1e-9, "fs must divide 20 MS/s for the synthesiser"
    Lf = int(round(frame_us * FINE))
    Lf -= Lf % D
    t = np.arange(Lf) / FINE
    sig = np.zeros(Lf, complex)

    def add(env, s, amp, f_khz):
        seg = np.zeros(Lf)
        e = min(len(env), Lf - s)
        seg[s:s + e] = env[:e]
        ph = rng.uniform(0, 2 * np.pi)
        sig[:] += amp * seg * np.exp(1j * (2 * np.pi * f_khz * 1e-3 * t + ph))

    if start_fine is None:
        start_fine = int(rng.integers(int(400 * FINE), int(420 * FINE)))
    A = 10 ** (snr_db / 20)
    add(_msg_env(bits), start_fine, A, rng.uniform(-freq_off_khz, freq_off_khz))
    if garble_sir_db is not None:
        b2 = make_df17(rng)
        d = rng.uniform(-119, 119)
        s2 = max(0, min(int(round(start_fine + d * FINE)), Lf - 2400))
        add(_msg_env(b2), s2, A * 10 ** (-garble_sir_db / 20), rng.uniform(-freq_off_khz, freq_off_khz))
    if dme_mean > 0:
        # DME interrogation: Gaussian pulse pair, 3.5 us width at half amplitude (c = 1.75/sqrt(ln 2)), 12 us spacing
        c = 1.75 / np.sqrt(np.log(2))
        for _ in range(rng.poisson(dme_mean)):
            t0 = rng.uniform(0, frame_us)
            env = np.exp(-(((t - t0) / c) ** 2)) + np.exp(-(((t - t0 - 12.0) / c) ** 2))
            ph = rng.uniform(0, 2 * np.pi)
            sig += A * 10 ** (dme_rel_db / 20) * env * np.exp(1j * (2 * np.pi * rng.uniform(-500, 500) * 1e-3 * t + ph))
    iq = sig.reshape(-1, D).mean(1)
    if noise:
        n = len(iq)
        iq = iq + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return iq, start_fine


# ------------------------------------------------------------------ templates / detectors
def _ov(fs_mhz, phi, a, b, nmax):
    """Overlap (as a fraction of the sample length) of sample k = [k/fs,(k+1)/fs) with [phi+a, phi+b)."""
    k = np.arange(nmax)
    lo = np.maximum(k / fs_mhz, phi + a)
    hi = np.minimum((k + 1) / fs_mhz, phi + b)
    return np.clip(hi - lo, 0, None) * fs_mhz


def _range(fs_mhz, phi, a, b):
    """Contiguous sample range (start, end_exclusive) holding >= CAPTURE of the pulse energy of [phi+a, phi+b)
    (energy of a sample of the boxcar-averaged rectangular pulse is proportional to overlap^2)."""
    n = int(np.ceil((phi + b) * fs_mhz)) + 2
    o2 = _ov(fs_mhz, phi, a, b, n) ** 2
    order = np.argsort(-o2, kind="stable")
    cum = np.cumsum(o2[order])
    m = int(np.searchsorted(cum, CAPTURE * cum[-1] - 1e-12)) + 1
    sel = order[:m]
    return int(sel.min()), int(sel.max()) + 1


def _cover(win, pulses):
    """Normalisation window must contain every pulse sample (so that stat = pulse energy / window energy <= 1
    and the Beta null is exact)."""
    return (min([win[0]] + [p[0] for p in pulses]), max([win[1]] + [p[1] for p in pulses]))


class Phase:
    """Sample sets of the preamble for one sub-sample phase."""

    def __init__(self, fs_mhz, phi):
        self.phi = phi
        self.pulses = [_range(fs_mhz, phi, a, b) for a, b in PREAMBLE_PULSES]
        self.win = _cover(_range(fs_mhz, phi, 0.0, PRE_US), self.pulses)
        self.Np = sum(e - s for s, e in self.pulses)
        self.Nw = self.win[1] - self.win[0]
        self.pa = [_range(fs_mhz, phi, a, b) for a, b in HALF_A]
        self.pb = [_range(fs_mhz, phi, a, b) for a, b in HALF_B]
        self.wa = _cover(_range(fs_mhz, phi, *WIN_A), self.pa)
        self.wb = _cover(_range(fs_mhz, phi, *WIN_B), self.pb)
        self.Npa = sum(e - s for s, e in self.pa)
        self.Npb = sum(e - s for s, e in self.pb)
        self.Nwa = self.wa[1] - self.wa[0]
        self.Nwb = self.wb[1] - self.wb[0]


def _rsum(C, rngs, npos):
    tot = 0.0
    for s, e in rngs:
        tot = tot + (C[e:e + npos] - C[s:s + npos])
    return tot


def _beta_thr(a, b, p):
    """Upper-tail threshold of Beta(a,b): P(X > thr) = p (exact null of pulse-energy / window-energy)."""
    return float(betaincinv(a, b, 1.0 - p))


SUM_TYPE = ("energy", "peak", "mf", "parhalf")       # need the noise power; the *_norm variants do not
NORM_TYPE = ("mf_norm", "parhalf_norm")


def noise_floor(p, q=0.10):
    """Robust noise power per complex sample from a block of power samples: low quantile of an exponential
    distribution (valid while the block is not dominated by signal; bias from signal duty < 1 - q is small)."""
    return float(np.quantile(p, q) / -np.log(1.0 - q))


class Receiver:
    """Envelope-power detectors with analytic thresholds for a false-alarm rate (exceedances / s on noise).

    detector (statistic / threshold > 1 means detection):
      'energy'       : energy of the 8 us preamble span / sigma^2, Gamma(Nw) null
      'peak'         : max power over the 8 us span / sigma^2 (single-sample threshold)
      'mf'           : non-coherent preamble matched filter: energy collected in the 4 pulse sample sets / sigma^2,
                       Gamma(Np) null.  sigma^2 = long-term noise floor (CFAR reference), not the off-pulse samples
      'mf_norm'      : the same but normalised by the energy of the 8 us window (needs no noise power):
                       Beta(Np, Nw-Np) null.  Costs SNR because the 'noise estimate' has only Nw-Np samples
      'parhalf'      : two pulse-pair filters (pulses 0,1.0 us and 3.5,4.5 us) on the energy / sigma^2; both must
                       trigger at the same candidate start; each at probability sqrt(p) (independent under H0)
      'parhalf_norm' : the same with each pair normalised by its own 2 us window
    """

    def __init__(self, fs, detector="mf", rate_hz=10.0):
        self.fs_mhz = fs / 1e6 if fs > 1e3 else float(fs)
        self.detector = detector
        self.K = max(1, int(round(FINE / self.fs_mhz))) if detector in ("mf", "parhalf", "mf_norm", "parhalf_norm") else 1
        self.phases = [Phase(self.fs_mhz, i / (self.K * self.fs_mhz)) for i in range(self.K)]
        self.set_rate(rate_hz)

    def set_rate(self, rate_hz):
        self.rate_hz = rate_hz
        pfa = rate_hz / (self.fs_mhz * 1e6)       # per candidate position
        self.pfa_pos = pfa
        a = pfa / self.K                          # Bonferroni over the sub-sample phases
        self.thr = []
        for ph in self.phases:
            d = self.detector
            if d == "energy":
                self.thr.append(float(gammaincinv(ph.Nw, 1.0 - pfa)))
            elif d == "peak":
                self.thr.append(float(-np.log(1.0 - (1.0 - pfa) ** (1.0 / ph.Nw))))
            elif d == "mf":
                self.thr.append(float(gammaincinv(ph.Np, 1.0 - a)))
            elif d == "mf_norm":
                self.thr.append(_beta_thr(ph.Np, ph.Nw - ph.Np, a))
            elif d == "parhalf":
                q = np.sqrt(a)
                self.thr.append((float(gammaincinv(ph.Npa, 1.0 - q)), float(gammaincinv(ph.Npb, 1.0 - q))))
            elif d == "parhalf_norm":
                q = np.sqrt(a)
                self.thr.append((_beta_thr(ph.Npa, ph.Nwa - ph.Npa, q), _beta_thr(ph.Npb, ph.Nwb - ph.Npb, q)))
            else:
                raise ValueError(d)
        self.span = max(max(e for _, e in (ph.pulses + [ph.win, ph.wa, ph.wb])) for ph in self.phases)

    def scores(self, p, sig2=None):
        """(K, npos) array of statistic / threshold for candidate start positions (sample index of t0 - phi).
        sig2: noise power per sample for the sum-type detectors (default: noise_floor(p))."""
        npos = len(p) - self.span + 1
        if npos <= 0:
            return np.zeros((self.K, 0))
        if self.detector in SUM_TYPE:
            p = p / (sig2 if sig2 is not None else noise_floor(p))
        C = np.concatenate([[0.0], np.cumsum(p)])
        out = np.empty((self.K, npos))
        eps = 1e-20
        d = self.detector
        for i, ph in enumerate(self.phases):
            if d == "energy":
                out[i] = _rsum(C, [ph.win], npos) / self.thr[i]
            elif d == "peak":
                m = sliding_window_view(p, ph.Nw).max(1)
                out[i] = m[:npos] / self.thr[i]
            elif d == "mf":
                out[i] = _rsum(C, ph.pulses, npos) / self.thr[i]
            elif d == "mf_norm":
                out[i] = (_rsum(C, ph.pulses, npos) / np.maximum(_rsum(C, [ph.win], npos), eps)) / self.thr[i]
            elif d == "parhalf":
                out[i] = np.minimum(_rsum(C, ph.pa, npos) / self.thr[i][0], _rsum(C, ph.pb, npos) / self.thr[i][1])
            else:
                sa = _rsum(C, ph.pa, npos) / np.maximum(_rsum(C, [ph.wa], npos), eps) / self.thr[i][0]
                sb = _rsum(C, ph.pb, npos) / np.maximum(_rsum(C, [ph.wb], npos), eps) / self.thr[i][1]
                out[i] = np.minimum(sa, sb)
        return out


# ------------------------------------------------------------------ decoder
class Demod:
    """Half-bit energy weights (overlap^2: locally most powerful for a weak pulse) for one sub-sample phase."""

    def __init__(self, fs_mhz, phi):
        self.L = int(np.ceil((phi + MSG_US) * fs_mhz)) + 1
        W = np.zeros((2 * NBITS, self.L))
        for b in range(NBITS):
            for h in range(2):
                a = PRE_US + b + 0.5 * h
                W[2 * b + h] = _ov(fs_mhz, phi, a, a + 0.5, self.L) ** 2
        self.W = W
        self.ws = W.sum(1)
        # zero-mean, unit-norm power-domain template of the 4 preamble pulses (for sub-sample phase refinement)
        Lp = int(np.ceil((phi + PRE_US) * fs_mhz)) + 2
        t = sum(_ov(fs_mhz, phi, a, b, Lp) ** 2 for a, b in PREAMBLE_PULSES)
        t = t - t.mean()
        self.Lp = Lp
        self.tpl = t / np.linalg.norm(t)

        self._init_mlse(fs_mhz, phi)

    def _init_mlse(self, fs_mhz, phi):
        """Tables for the 2-state Viterbi detector.  A sample (<= 0.5 us long) overlaps at most two adjacent
        half-slots, hence at most two adjacent bits; amplitude = A * sum(overlap * pulse-present)."""
        L = self.L
        ovs = np.zeros((L, 2 * NBITS))
        for b in range(NBITS):
            for h in range(2):
                a = PRE_US + b + 0.5 * h
                ovs[:, 2 * b + h] = _ov(fs_mhz, phi, a, a + 0.5, L)
        self.step = np.full(L, -1)
        self.AK = np.zeros((L, 4))                  # amplitude per combination index 2*bit_prev + bit_cur
        for k in range(L):
            hs = np.nonzero(ovs[k] > 1e-12)[0]
            if len(hs) == 0:
                continue
            bs = sorted(set((hs // 2).tolist()))
            assert len(bs) <= 2 and (len(bs) == 1 or bs[1] == bs[0] + 1), "sample spans >2 bits (fs too low)"
            cur = bs[-1]
            self.step[k] = cur
            for combo in range(4):
                bp, bc = combo >> 1, combo & 1
                amp = 0.0
                for h in hs:
                    b, hh = h // 2, h % 2
                    bit = bc if b == cur else bp
                    amp += ovs[k, h] * (bit if hh == 0 else 1 - bit)
                self.AK[k, combo] = amp
        Lp = self.Lp
        self.a_pre = sum(_ov(fs_mhz, phi, a, b, Lp) for a, b in PREAMBLE_PULSES)
        self.use = self.step >= 0
        self.step_u = self.step[self.use]
        self.AK_u = self.AK[self.use]
        self.ov_pre = None

    def mlse(self, seg, a2, sig2):
        """Viterbi over the 112 bits with the noncentral-exponential likelihood of the sample power."""
        x = seg[self.use] / sig2
        m = a2 * self.AK_u ** 2                                   # (Ls, 4)
        z = 2.0 * np.sqrt(x[:, None] * m)
        ll = -m + np.log(i0e(z)) + z
        G = np.zeros((NBITS, 4))
        for c in range(4):
            G[:, c] = np.bincount(self.step_u, weights=ll[:, c], minlength=NBITS)
        # bit 0 has no previous bit: its combos with bp=0 and bp=1 are identical (amplitude does not use bp)
        d = np.array([G[0, 0], G[0, 1]])                           # state = bit_0
        back = np.zeros((NBITS, 2), np.int8)
        for b in range(1, NBITS):
            c00, c01, c10, c11 = G[b]
            n0a, n0b = d[0] + c00, d[1] + c10                      # to bit 0 from prev 0 / prev 1
            n1a, n1b = d[0] + c01, d[1] + c11
            back[b, 0] = 0 if n0a >= n0b else 1
            back[b, 1] = 0 if n1a >= n1b else 1
            d = np.array([max(n0a, n0b), max(n1a, n1b)])
        st = int(d[1] > d[0])
        bits = np.zeros(NBITS, np.uint8)
        for b in range(NBITS - 1, -1, -1):
            bits[b] = st
            if b > 0:
                st = back[b, st]
        return bits

    def amp2(self, seg, sig2):
        """Pulse power A^2 (same units as p) from the preamble: weighted LS fit of p - sigma^2 on a_pre^2."""
        a = self.a_pre
        num = float(np.sum(a ** 2 * (seg[:len(a)] - sig2)))
        return min(max(num / float(np.sum(a ** 4)), 0.1 * sig2), 1e4 * sig2)   # cap: huge-SNR model mismatch

    def bits(self, seg):
        e = (self.W @ seg) / self.ws
        return (e[0::2] > e[1::2]).astype(np.uint8), np.abs(e[0::2] - e[1::2]) / (e[0::2] + e[1::2] + 1e-30)


_CACHE = {}


def get_rx(fs_mhz, detector, rate_hz):
    key = (round(fs_mhz, 6), detector, rate_hz)
    if key not in _CACHE:
        rx = Receiver(fs_mhz, detector, rate_hz)
        dm = [Demod(fs_mhz, ph.phi) for ph in rx.phases]
        _CACHE[key] = (rx, dm)
    return _CACHE[key]


def candidates(rx, p, sig2=None):
    """Local maxima (+-1 us) of the best-phase score above 1 -> list of (n, phase_index, score)."""
    sc = rx.scores(p, sig2)
    if sc.shape[1] == 0:
        return []
    best = sc.max(0)
    ph = sc.argmax(0)
    size = 2 * int(np.ceil(1.0 * rx.fs_mhz)) + 1
    loc = (best == maximum_filter1d(best, size=size, mode="constant", cval=0.0)) & (best > 1.0)
    idx = np.nonzero(loc)[0]
    return [(int(n), int(ph[n]), float(best[n])) for n in idx]


def refine(dm, p, n, top=3):
    """Rank (position, phase) hypotheses around coarse candidate n by the normalised power-domain correlation
    with the ov^2 preamble template (the binary-set detector cannot separate phases whose pulse sets are
    identical).  Returns the best `top` (n, k) pairs."""
    res = []
    for nn in (n - 1, n, n + 1):
        if nn < 0:
            continue
        for k, d in enumerate(dm):
            seg = p[nn:nn + d.Lp]
            if len(seg) < d.Lp:
                continue
            res.append((float(d.tpl @ seg), nn, k))
    res.sort(reverse=True)
    return [(nn, k) for _, nn, k in res[:top]]


def local_sig2(d, blk, n, fs_mhz):
    """Noise power (per complex sample) near candidate n: median of the 10 us before the preamble plus the
    preamble samples untouched by pulses; robust to a partly contaminated pool (mean = median / ln 2)."""
    pre = blk[max(0, n - int(np.ceil(10 * fs_mhz))):n]
    gap = blk[n:n + d.Lp][d.a_pre == 0]
    pool = np.concatenate([pre, gap])
    return float(np.median(pool) / np.log(2.0)) + 1e-30


def _try(d, blk, n, fix_bits, df_filter, sig2, mode="energy"):
    seg = blk[n:n + d.L]
    if len(seg) < d.L:
        return None
    if mode == "energy":
        bits, _ = d.bits(seg)
    else:
        bits = d.mlse(seg, d.amp2(blk[n:n + d.Lp], sig2), sig2)
    s = int(remainder(bits))
    repaired = False
    if s != 0 and fix_bits >= 1 and s in SYNDROME_TO_BIT:
        bits = bits.copy()
        bits[SYNDROME_TO_BIT[s]] ^= 1
        repaired = True
        s = 0
    df = int("".join(map(str, bits[:5])), 2)
    ok = (s == 0) and (df_filter is None or df in df_filter)
    return bits, ok, repaired and ok, df


def decode_modes_ex(iq, fs, fa_rate_hz=2000.0, detector="mf", fix_bits=1, df_filter=(17, 18), retry=3, decoder="auto", sig2=None, chunk=1 << 21):
    """Detect and decode Mode S 112-bit extended squitters in a complex baseband recording.

    iq: complex array (any scale; the detectors normalise by the local energy), fs: sample rate in Hz
    (>= ~2 MS/s).  fa_rate_hz: candidate (preamble) false-alarm rate on noise; chosen high on purpose because
    the CRC-24 is the real filter.  fix_bits=1: dump1090-style single-bit repair with the CRC syndrome.
    df_filter: only these downlink formats count as crc_ok (None = all); it protects the repaired path
    (112 syndromes -> 6.7e-6 false-accept probability per noise candidate without it).
    retry: number of (position, phase) hypotheses, ranked by preamble template correlation, tried per candidate
    (needed at 2 MS/s where a half-sample timing offset makes adjacent half-bits share a sample).
    sig2: noise power per complex sample (default: robust low-quantile estimate per block).
    decoder: 'energy' (compare the two half-bit energies), 'mlse' (2-state Viterbi, handles inter-bit sharing of
    samples at low sample rates) or 'auto' (energy first, MLSE if the CRC fails).
    Returns a list of dicts: index (sample of the preamble start), phase_us (sub-sample offset), hex, crc_ok,
    repaired, df, score.  Short (56-bit) squitters are NOT decoded.
    """
    fs_mhz = fs / 1e6
    rx, dm = get_rx(fs_mhz, detector, fa_rate_hz)
    p = np.abs(np.asarray(iq)) ** 2
    ov = int(np.ceil((MSG_US + 10) * fs_mhz)) + 4
    out = []
    N = len(p)
    i0 = 0
    while i0 < N:
        i1 = min(N, i0 + chunk + ov)
        blk = p[i0:i1]
        sig2_blk = noise_floor(blk) if sig2 is None else sig2
        for n, k, sc in candidates(rx, blk, sig2_blk):
            if i1 < N and n >= chunk:
                continue                                     # belongs to the next block
            first = None
            modes_ = {"energy": ("energy",), "mlse": ("mlse",), "auto": ("energy", "mlse")}[decoder]
            for nn, kk in refine(dm, blk, n, top=retry):
                for md in modes_:
                    r = _try(dm[kk], blk, nn, fix_bits, df_filter, sig2_blk, md)
                    if r is None:
                        continue
                    if first is None:
                        first = (nn, kk, r)
                    if r[1]:
                        first = (nn, kk, r)
                        break
                if first is not None and first[2][1]:
                    break
            if first is None:
                continue
            nn, kk, (bits, ok, rep, df) = first
            out.append(dict(index=i0 + nn, phase_us=rx.phases[kk].phi, hex=bits2hex(bits), crc_ok=bool(ok),
                            repaired=bool(rep), df=df, score=sc))
        i0 += chunk
    return out


def decode_modes(iq, fs, **kw):
    """Required interface: list of (sample index, 14-byte message hex, crc_ok)."""
    return [(r["index"], r["hex"], r["crc_ok"]) for r in decode_modes_ex(iq, fs, **kw)]
