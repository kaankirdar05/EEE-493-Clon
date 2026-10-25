"""Synthetic DSSS frames: BPSK data x PN code (symbol period = one code period), RRC chips."""
import numpy as np
from scipy.signal import fftconvolve
from common import cplx_noise  # sets sys.path for the repo imports below
from amc.synthetic import _rrc

SPS = 4
BETA = 0.35
N_FRAME = 65536
CFO_MAX = 0.005          # cycles/sample (2 % of the chip rate)
_H = _rrc(BETA, 8, SPS)  # unit-energy RRC, 65 taps


def lfsr(n, taps):
    """m-sequence of period 2^n-1 for x^n = sum(x^t for t in taps) + 1  (bits 0/1)."""
    s = [1] * n
    out = []
    for _ in range(2**n - 1):
        out.append(s[0])
        new = s[0]
        for t in taps:
            new ^= s[t]
        s = s[1:] + [new]
    return np.array(out, dtype=np.int8)


def _primitive_taps(n):
    """All tap sets t (subset of 1..n-1) for which x^n = sum x^t + 1 gives period 2^n-1 (found by brute force)."""
    import itertools
    out = []
    for r in range(1, n):
        for taps in itertools.combinations(range(1, n), r):
            s = [1] * n
            ok = True
            for k in range(1, 2**n):
                new = s[0]
                for t in taps:
                    new ^= s[t]
                s = s[1:] + [new]
                if all(v == 1 for v in s):
                    ok = (k == 2**n - 1)
                    break
            if ok and k == 2**n - 1:
                out.append(list(taps))
    return out


_PRIM = {}


def pn_code(L, rng=None, fixed=False):
    """+-1 chips of length L in {31, 511: m-sequence with a random primitive polynomial,
    127: random member of the Gold family from the preferred pair x^7+x^3+1, x^7+x^3+x^2+x+1}.
    fixed=True uses the first polynomial / shift 0 (deterministic)."""
    if L in (31, 511):
        n = 5 if L == 31 else 9
        if n not in _PRIM:
            _PRIM[n] = _primitive_taps(n)
        k = 0 if (rng is None or fixed) else int(rng.integers(0, len(_PRIM[n])))
        b = lfsr(n, _PRIM[n][k])
    elif L == 127:
        u = lfsr(7, [3])
        v = lfsr(7, [3, 2, 1])
        k = 0 if (rng is None or fixed) else int(rng.integers(0, 127))
        b = u ^ np.roll(v, k)
    else:
        raise ValueError(L)
    return (1 - 2 * b.astype(float))


def pulse_shape(chips, sps=SPS):
    up = np.zeros(len(chips) * sps)
    up[::sps] = chips
    return fftconvolve(up, _H, "same")


def dsss_frame(L, snr_db, rng, N=N_FRAME, code=None, random_chips=False, cfo_max=CFO_MAX,
               with_noise=True, fixed_cfo=None):
    """One frame (complex64-ish array) at per-sample SNR snr_db. random_chips=True gives a plain
    BPSK-at-chip-rate signal (i.i.d. chips, no PN period) as the 'no processing gain' reference."""
    T = L * SPS
    nsym = (N + 2 * T) // T + 2
    offset = int(rng.integers(0, T)) + T
    if random_chips:
        chips = rng.choice([-1.0, 1.0], nsym * L)
    else:
        c = pn_code(L, rng) if code is None else code
        data = rng.choice([-1.0, 1.0], nsym)
        chips = (data[:, None] * c[None, :]).ravel()
    s = pulse_shape(chips)[offset:offset + N]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    f0 = rng.uniform(-cfo_max, cfo_max) if fixed_cfo is None else fixed_cfo
    s = s * np.exp(1j * (2 * np.pi * f0 * np.arange(N) + rng.uniform(0, 2 * np.pi)))
    x = np.sqrt(10 ** (snr_db / 10)) * s
    if with_noise:
        x = x + cplx_noise(N, rng)
    return x, {"L": L, "f0": f0, "chip_rate": 1.0 / SPS, "T": T, "code": None if random_chips else c}


def noise_frame(rng, N=N_FRAME):
    return cplx_noise(N, rng)
