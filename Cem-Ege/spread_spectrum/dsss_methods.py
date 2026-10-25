"""Blind DSSS detectors / estimators (no knowledge of chip rate, code or timing).

All statistics are scalars per frame; thresholds are set by simulation on noise-only frames.
  energy_ideal : mean |x|^2 (noise power assumed known = 1)               [reference]
  energy_unc   : same, with the true noise power uncertain by +-0.5 dB     [reference, shows the SNR wall]
  energy_psd   : max band energy of a Welch PSD / median PSD (noise power NOT needed)
  acf_fluct    : Burel/Deng-style fluctuation of the normalised autocorrelation across blocks,
                 max over lags 96..2400 (lag-dependent block length W ~ lag/4..lag/2)
  cyc_abs2     : strongest cyclic line of |x|^2 for alpha in [0.02, 0.5)  (chip-rate line)
  cyc_x2       : strongest cyclic line of x^2 for |alpha| in [0.05, 0.5)  (2*f0 +- chip rate)
  acov_abs2    : peak of the normalised autocovariance of |x|^2 at lags 96..2400 (period estimator 'B')
"""
import numpy as np
from scipy import fft as sfft

LAG_LO, LAG_HI = 96, 2400
# (block length W, first lag, last lag): W ~ lag/4 .. lag/2 so a block lies inside one data symbol
GROUPS = [(32, 96, 127), (64, 128, 255), (128, 256, 511), (256, 512, 1023), (512, 1024, 2047), (1024, 2048, 2400)]
LAGS = np.arange(LAG_LO, LAG_HI + 1)


def energy_ideal(x):
    return float(np.mean(np.abs(x) ** 2))


def energy_psd(x, nfft=256, band=64):
    n = (len(x) // nfft) * nfft
    P = (np.abs(sfft.fft(x[:n].reshape(-1, nfft), axis=1)) ** 2).mean(0) / nfft
    cs = np.concatenate([P, P[:band]]).cumsum()
    B = cs[band:band + nfft] - cs[:nfft]
    return float(B.max() / (band * np.median(P)))


def acf_profile(x):
    """Per-lag fluctuation profile. Returns (z, a):
    T(lag) = mean_b |s_b - mean_b s_b|^2 / (W sigma^4)  (=1 on noise, s_b = sum over block b of x[t] conj x[t+lag]),
    z = sqrt(nb) (T - 1)  (noise z-score),  a = (T - 1) / W  (signal level, comparable across block lengths)."""
    N = len(x)
    sig2 = np.mean(np.abs(x) ** 2)
    zs, as_ = [], []
    for W, lo, hi in GROUPS:
        nb = (N - hi - 1) // W
        u = x[:nb * W].reshape(nb, W)
        v = np.lib.stride_tricks.as_strided(x, shape=(nb, W + hi), strides=(x.strides[0] * W, x.strides[0]))
        M = sfft.next_fast_len(W + hi)
        s = sfft.ifft(np.conj(sfft.fft(u, M, axis=1)) * sfft.fft(v, M, axis=1), axis=1)[:, lo:hi + 1]
        s = s - s.mean(0, keepdims=True)
        T = (np.abs(s) ** 2).sum(0) / ((nb - 1) * W * sig2**2)
        zs.append(np.sqrt(nb) * (T - 1.0))
        as_.append((T - 1.0) / W)
    return np.concatenate(zs), np.concatenate(as_)


def acf_fluct(x, return_all=False):
    z, a = acf_profile(x)
    return (z, a) if return_all else float(z.max())


def fundamental(lags, z, thr, rel=0.7):
    """Smallest local-maximum lag whose z >= max(thr, rel*zmax)."""
    zm = z.max()
    lim = max(thr, rel * zm)
    ok = np.flatnonzero(z >= lim)
    if len(ok) == 0:
        return int(lags[np.argmax(z)])
    # group consecutive indices into clusters; take the first cluster's argmax
    br = np.flatnonzero(np.diff(ok) > 3)
    first = ok[:br[0] + 1] if len(br) else ok
    return int(lags[first[np.argmax(z[first])]])


_NF = {}


def _nfft(N):
    return 2 * N + 2   # deliberately not a multiple of 4: chip-rate line (0.25) falls half a bin off-grid


def _line_spectrum(y, N):
    P = np.abs(sfft.fft(y, _nfft(N))) ** 2
    return P


def _peak_refine(P, k):
    a, b, c = np.log(P[k - 1] + 1e-300), np.log(P[k] + 1e-300), np.log(P[k + 1] + 1e-300)
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den < 0 else 0.0
    return k + float(np.clip(d, -0.5, 0.5))


def cyc_abs2(x, amin=0.02):
    """Return (stat, alpha_hat). stat = peak / noise-line level (median/ln2)."""
    N = len(x)
    y = np.abs(x) ** 2
    y = y - y.mean()
    P = _line_spectrum(y, N)
    M = len(P)
    k0, k1 = int(np.ceil(amin * M)), M // 2 - 1
    seg = P[k0:k1]
    floor = np.median(seg) / np.log(2)
    k = k0 + int(np.argmax(seg))
    return float(P[k] / floor), _peak_refine(P, k) / M


def cyc_x2(x, amin=0.05):
    """x^2 lines at 2 f0 +- Rc. Return (stat, Rc_hat, f0_hat); Rc from the peak pair (+side, -side)."""
    N = len(x)
    y = x * x
    y = y - y.mean()
    P = _line_spectrum(y, N)
    M = len(P)
    a = int(np.ceil(amin * M))
    pos = P[a:M // 2 - 1]
    neg = P[M // 2 + 2:M - a + 1]            # negative frequencies -0.5..-amin
    allv = np.concatenate([pos, neg])
    floor = np.median(allv) / np.log(2)
    kp = a + int(np.argmax(pos))
    kn = M // 2 + 2 + int(np.argmax(neg))
    sp, sn = P[kp], P[kn]
    stat = max(sp, sn) / floor
    ap = _peak_refine(P, kp) / M
    an = (_peak_refine(P, kn) - M) / M
    return float(stat), float((ap - an) / 2), float((ap + an) / 4), float(sp / floor), float(sn / floor)


def acov_abs2(x, return_all=False):
    """Normalised autocovariance of |x|^2 (which is exactly T-periodic for DSSS-BPSK)."""
    N = len(x)
    y = np.abs(x) ** 2
    y = y - y.mean()
    M = sfft.next_fast_len(2 * N)
    r = sfft.ifft(np.abs(sfft.fft(y, M)) ** 2).real[:LAG_HI + 1]
    # unbiased, normalised so that noise-only values have std ~ 1/sqrt(N)
    r = r / (N - np.arange(len(r))) / (y.var())
    z = np.sqrt(N) * r[LAG_LO:LAG_HI + 1]
    if return_all:
        return z, r[LAG_LO:LAG_HI + 1]
    return float(z.max())


def L_from(T_hat, rc_hat):
    """PN length in chips from the period in samples and the chip rate (cycles/sample)."""
    return int(round(T_hat * rc_hat))


def pn_period(z, a, zsub=4.5, rel=0.5, tol=4, mmax=25, refine=3):
    """Fundamental period (samples) from the lag profile: global peak of z, then its sub-multiples cand/m
    (m = mmax..2): accept the smallest one with a local peak z >= zsub + 3 * (median z of the +-50-lag neighbourhood)
    and level a >= rel * a(cand).
    The accepted peak is refined by a z^2-weighted centroid over +-refine lags."""
    ic = int(np.argmax(z))
    pick = ic
    for m in range(mmax, 1, -1):
        c = int(round(LAGS[ic] / m))
        lo, hi = c - tol, c + tol
        if lo < LAG_LO:
            continue
        i0, i1 = lo - LAG_LO, hi - LAG_LO + 1
        j = i0 + int(np.argmax(z[i0:i1]))
        nb = np.concatenate([z[max(0, i0 - 50):i0], z[i1:i1 + 50]])
        bg = max(float(np.median(nb)), 0.0) if len(nb) else 0.0
        if z[j] >= zsub + 3.0 * bg and a[j] >= rel * a[ic]:
            pick = j
            break
    if refine:
        i0, i1 = max(0, pick - refine), min(len(z), pick + refine + 1)
        w = np.maximum(z[i0:i1], 0.0) ** 2
        if w.sum() > 0:
            return int(round(float((LAGS[i0:i1] * w).sum() / w.sum())))
    return int(LAGS[pick])


def oracle_mf(x, tpl, sig2=None):
    """Known-code (and CFO = 0) matched correlator, folded non-coherently over data symbols.
    c(t) = sum_n x[t+n] conj(tpl[n]); S(t0) = sum_k |c(t0 + kT)|^2 / (sigma^2 ||tpl||^2); stat = max_t0 (S - nk)/sqrt(nk)."""
    N, T = len(x), len(tpl)
    if sig2 is None:
        sig2 = np.mean(np.abs(x) ** 2)
    M = sfft.next_fast_len(N + T)
    c = sfft.ifft(sfft.fft(x, M) * np.conj(sfft.fft(tpl, M)))[:N - T + 1]
    p = np.abs(c) ** 2 / (sig2 * np.sum(np.abs(tpl) ** 2))
    nk = (len(p)) // T
    P = p[:nk * T].reshape(nk, T)
    S = P.sum(0)
    return float(((S - nk) / np.sqrt(nk)).max())


def occupied_bw(x, nfft=256, k=2.5, close=3):
    """Occupied bandwidth (cycles/sample) of the largest significant PSD region (Welch, floor = median PSD)."""
    from scipy import ndimage
    n = (len(x) // nfft) * nfft
    nseg = n // nfft
    P = np.fft.fftshift((np.abs(sfft.fft(x[:n].reshape(-1, nfft), axis=1)) ** 2).mean(0))
    floor = np.median(P)
    m = P > floor * (1 + k / np.sqrt(nseg))
    m = ndimage.binary_closing(np.pad(m, close + 1), np.ones(2 * close + 1, bool))[close + 1:-close - 1]
    lab, nl = ndimage.label(m)
    if nl == 0:
        return 0.0
    sizes = ndimage.sum(m, lab, np.arange(1, nl + 1))
    return float(sizes.max() / nfft)


def chip_rate_model(x, T_hat, div=1.1):
    """Extension (needs the PN period and the prior that the code is maximal-length, L = 2^n-1):
    coarse chip rate Rc ~ B_occ/div (div = 1.1: effective, because the measured occupied bandwidth
    is biased low versus the true (1+0.35) Rc at low SNR; chosen on dev seeds); L_hat = 2^n-1 nearest (log scale) to T_hat*Rc; Rc_hat = L_hat/T_hat."""
    B = occupied_bw(x)
    if B <= 0:
        return float("nan"), 0
    target = T_hat * B / div
    cands = 2 ** np.arange(2, 13) - 1
    L = int(cands[np.argmin(np.abs(np.log(cands / max(target, 1e-9))))])
    return L / T_hat, L
