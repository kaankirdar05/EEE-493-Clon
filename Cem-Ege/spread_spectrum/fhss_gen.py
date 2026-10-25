"""Synthetic FHSS frames (fixed dwell, random pattern over K channels), plus non-hopping look-alikes."""
import numpy as np
from common import cplx_noise

N_FRAME = 262144
RATE_RANGE = (50.0, 2000.0)      # hops per 1e6 samples
K_CHOICES = (16, 24, 32, 48, 64)
BH_RANGE = (0.2, 0.8)            # hop band as a fraction of the sampled band


def _hann_smooth(v, n):
    if n < 2:
        return v
    k = np.hanning(n + 2)[1:-1]
    return np.convolve(v, k / k.sum(), "same")


def fhss_frame(rng, snr_db, N=N_FRAME, rate=None, K=None, bh=None, mod=None, center=None,
               freqs=None, pattern=None, with_noise=True, ramp=16, rs_frac=1 / 3):
    """Return (x, truth). Per-sample SNR = signal power / complex noise power (unit) over the full band.
    freqs/pattern override the random channel plan (known-answer tests)."""
    if rate is None:
        rate = float(np.exp(rng.uniform(np.log(RATE_RANGE[0]), np.log(RATE_RANGE[1]))))
    D = int(round(1e6 / rate))
    if freqs is None:
        K = int(rng.choice(K_CHOICES)) if K is None else K
        bh = float(rng.uniform(*BH_RANGE)) if bh is None else bh
        lim = 0.5 - bh / 2 - 0.02
        c0 = float(rng.uniform(-lim, lim)) if center is None else center
        delta = bh / K
        freqs = c0 + (np.arange(K) - (K - 1) / 2) * delta
    else:
        freqs = np.asarray(freqs, float)
        K = len(freqs)
        delta = float(np.min(np.diff(np.sort(freqs)))) if K > 1 else 0.01
        bh = float(freqs.max() - freqs.min() + delta)
    mod = str(rng.choice(["2fsk", "bpsk"])) if mod is None else mod
    u = int(rng.integers(0, D))
    nh = (N + u) // D + 2
    if pattern is None:
        pat = np.empty(nh, int)
        pat[0] = rng.integers(0, K)
        for k in range(1, nh):
            j = int(rng.integers(0, K - 1))
            pat[k] = j if j < pat[k - 1] else j + 1       # no immediate repeat
    else:
        pat = np.resize(np.asarray(pattern, int), nh)
    t = np.arange(N)
    hop_idx = (t + u) // D
    fh = freqs[pat[hop_idx]]
    rs = delta * rs_frac
    Ts = max(2, int(round(1.0 / rs)))
    nsym = N // Ts + 2
    bits = rng.choice([-1.0, 1.0], nsym)
    b = np.repeat(bits, Ts)[:N]
    ph_hop = rng.uniform(0, 2 * np.pi, nh)[hop_idx]
    if mod == "2fsk":
        finst = fh + 0.5 * rs * b
        phase = 2 * np.pi * np.cumsum(finst) + ph_hop
        s = np.exp(1j * phase)
    else:
        bb = _hann_smooth(b, max(2, Ts // 3))
        s = bb * np.exp(1j * (2 * np.pi * np.cumsum(fh) + ph_hop))
    # amplitude ramps at the hop boundaries
    R = max(1, min(ramp, D // 8))
    pos_in = (t + u) % D
    dist = np.minimum(pos_in, D - 1 - pos_in)
    env = 0.5 * (1 - np.cos(np.pi * np.clip(dist / R, 0, 1)))
    s = s * env
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    x = np.sqrt(10 ** (snr_db / 10)) * s
    if with_noise:
        x = x + cplx_noise(N, rng)
    bounds = np.arange(-u + D, N, D)                    # hop instants inside the frame (start of hop k>=1)
    # hop segments (start, stop clipped to frame, channel freq)
    starts = -u + D * np.arange(nh)
    seg = [(max(0, st), min(N, st + D), float(freqs[pat[k]])) for k, st in enumerate(starts) if st < N and st + D > 0]
    vis = [f for (a, b_, f) in seg if (b_ - a) >= 0.5 * D]
    truth = dict(rate=1e6 / D, D=D, K=K, bh=bh, delta=delta, mod=mod, bounds=bounds, segs=seg, freqs=freqs,
                 n_distinct=len(set(np.round(vis, 9))), hop_freqs_visible=sorted(set(np.round(vis, 9))))
    return x, truth


def tone_frame(rng, snr_db, N=N_FRAME, f=None):
    f = float(rng.uniform(-0.4, 0.4)) if f is None else f
    s = np.exp(1j * (2 * np.pi * f * np.arange(N) + rng.uniform(0, 2 * np.pi)))
    return np.sqrt(10 ** (snr_db / 10)) * s + cplx_noise(N, rng)


def cw_fsk_frame(rng, snr_db, N=N_FRAME):
    """Stationary (non-hopping) narrowband 2-FSK, same symbol-rate range as the hoppers."""
    f0 = float(rng.uniform(-0.4, 0.4))
    rs = float(rng.uniform(0.0005, 0.01))
    Ts = int(round(1 / rs))
    b = np.repeat(rng.choice([-1.0, 1.0], N // Ts + 2), Ts)[:N]
    s = np.exp(1j * (2 * np.pi * np.cumsum(f0 + 0.5 * rs * b) + rng.uniform(0, 6.28)))
    return np.sqrt(10 ** (snr_db / 10)) * s + cplx_noise(N, rng)


def noise_frame(rng, N=N_FRAME):
    return cplx_noise(N, rng)
