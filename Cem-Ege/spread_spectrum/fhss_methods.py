"""Classical FHSS detectors / estimators on a single frame (no knowledge of the hop plan).

  stft_norm       Hann STFT power, normalised so that noise cells ~ Exp(1) (floor = median / ln 2)
  method1 (ridge) STFT ridge (power centroid around the peak bin) + optimal-partition change-point detection
                  (offline multi-change generalisation of CUSUM mean-shift detection) -> hop instants,
                  dwell, hop frequencies; hop rate from a comb fit to the hop instants
  method2 (2DFT)  2-D Fourier view of the (thresholded) spectrogram: temporal modulation spectrum summed
                  over frequency, fitted with the hop on/off model sinc^2(nu D) (Whittle likelihood) -> dwell D
  method3 (MHAC)  multiple-hop autocorrelation: autocorrelation of the ridge-jump indicator, peaks at m*D,
                  regression of the peak lags on m -> D
"""
import numpy as np
from scipy import fft as sfft

N_FRAME = 262144


# ------------------------------------------------------------------ STFT
def stft_norm(x, nfft, step):
    nfr = (len(x) - nfft) // step + 1
    w = np.hanning(nfft)
    idx = np.arange(nfft)[None, :] + step * np.arange(nfr)[:, None]
    F = sfft.fft(x[idx] * w, axis=1)
    S = np.fft.fftshift(np.abs(F) ** 2 / np.sum(w**2), axes=1)
    floor = np.median(S) / np.log(2)
    return S / floor


def ridge_centroid(Sn, half=12, base=4.0):
    """Per frame: peak bin and power centroid of max(S - base, 0) within +-half bins. Returns (f_bins, peak)."""
    T, F = Sn.shape
    pk = np.argmax(Sn, axis=1)
    val = Sn[np.arange(T), pk]
    offs = np.arange(-half, half + 1)
    idx = np.clip(pk[:, None] + offs[None, :], 0, F - 1)
    w = np.maximum(np.take_along_axis(Sn, idx, axis=1) - base, 0.0)
    pos = pk[:, None] + offs[None, :]
    f = (w * pos).sum(1) / np.maximum(w.sum(1), 1e-12)
    f = np.where(w.sum(1) > 0, f, pk.astype(float))
    return f, val


# ------------------------------------------------------------------ change points
def opt_partition(y, beta, minlen=2, maxlen=900):
    """Optimal partitioning with L2 (mean-shift) cost + beta per segment. Returns segment start indices."""
    n = len(y)
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, int)
    for t in range(minlen, n + 1):
        s0 = max(0, t - maxlen)
        s = np.arange(s0, t - minlen + 1)
        ln = t - s
        cost = (c2[t] - c2[s]) - (c1[t] - c1[s]) ** 2 / ln
        tot = F[s] + cost + beta
        j = int(np.argmin(tot))
        F[t] = tot[j]
        last[t] = s[j]
    starts = []
    t = n
    while t > 0:
        starts.append(last[t])
        t = last[t]
    return starts[::-1]


def cluster_count(vals, gap):
    v = np.sort(np.asarray(vals, float))
    if len(v) == 0:
        return 0, []
    cl = [[v[0]]]
    for a in v[1:]:
        if a - cl[-1][-1] <= gap:
            cl[-1].append(a)
        else:
            cl.append([a])
    return len(cl), [float(np.mean(c)) for c in cl]


def comb_period(times, dmin=300.0, dmax=65000.0, tol=0.9):
    """Hop period from hop instants: D = largest value (>= tol * best score) maximising |sum exp(-j 2 pi t / D)|."""
    t = np.asarray(times, float)
    if len(t) < 4:
        return float("nan")
    grid = np.exp(np.arange(np.log(dmin), np.log(dmax), 0.0008))
    sc = np.abs(np.exp(-2j * np.pi * t[None, :] / grid[:, None]).sum(1)) / len(t)
    sm = sc.max()
    cand = np.flatnonzero(sc >= tol * sm)
    i = cand[-1]
    lo, hi = grid[i] * 0.99, grid[i] * 1.01
    fine = np.linspace(lo, hi, 401)
    sf = np.abs(np.exp(-2j * np.pi * t[None, :] / fine[:, None]).sum(1))
    return float(fine[int(np.argmax(sf))])


# ------------------------------------------------------------------ method 1
def method1(x, nfft=256, step=64, gate=14.0, c=8.0, jmin=1.2, mseg=3, Sn=None):
    if Sn is None:
        Sn = stft_norm(x, nfft, step)
    f, pk = ridge_centroid(Sn)
    ok = pk >= gate
    idx = np.flatnonzero(ok)
    out = dict(bounds=np.array([]), freqs=np.array([]), n_hops=0, n_gated=int(len(idx)), D_comb=float("nan"),
               D_raw=float("nan"), n_distinct=0)
    if len(idx) < 2 * mseg:
        return out
    y = f[idx]
    from scipy.ndimage import median_filter
    res = y - median_filter(y, 9, mode="nearest")
    sig = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.3)
    beta = c * sig**2 * np.log(len(y))
    st = opt_partition(y, beta)
    ends = st[1:] + [len(y)]
    segs = [(a, b, float(np.median(y[a:b]))) for a, b in zip(st, ends) if b - a >= mseg]
    merged = []
    for a, b, fm in segs:
        if merged and abs(merged[-1][2] - fm) < jmin:
            a0, b0, f0 = merged[-1]
            merged[-1] = (a0, b, float(np.median(y[a0:b])))
        else:
            merged.append((a, b, fm))
    if len(merged) < 2:
        out["freqs"] = np.array([(m[2] - Sn.shape[1] / 2) / Sn.shape[1] for m in merged])
        out["n_distinct"] = len(merged)
        return out
    cen = lambda i: idx[i] * step + nfft / 2.0
    bounds = np.array([0.5 * (cen(merged[k][1] - 1) + cen(merged[k + 1][0])) for k in range(len(merged) - 1)])
    freqs = np.array([(m[2] - Sn.shape[1] / 2) / Sn.shape[1] for m in merged])
    out["bounds"], out["freqs"], out["n_hops"] = bounds, freqs, len(bounds)
    out["D_raw"] = float(np.median(np.diff(bounds))) if len(bounds) >= 2 else float("nan")
    out["D_comb"] = comb_period(bounds)
    out["n_distinct"] = cluster_count(freqs, 0.8 / Sn.shape[1])[0]
    return out


# ------------------------------------------------------------------ method 2
_M2 = {}


def _m2_model(nk, nfft, T):
    key = (nk, nfft, T)
    if key not in _M2:
        nu = np.arange(1, nk + 1) / (T * nfft)
        grid = np.exp(np.linspace(np.log(420.0), np.log(60000.0), 1400))
        m = np.zeros((len(grid), nk))
        for a in range(-3, 4):
            m += np.sinc(grid[:, None] * (nu[None, :] - a / nfft)) ** 2 * np.sinc(nfft * nu[None, :] - a) ** 2
        _M2[key] = (nu, grid, m)
    return _M2[key]


R_GRID = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def method2(x, nfft=256, gam=8.0):
    """Returns dict(D, llr). Binary map B = (cell > gam) (noise floor 1), per-bin mean removed over time, temporal
    periodogram summed over frequency bins (= integral over eta of |2-D FFT|^2)."""
    Sn = stft_norm(x, nfft, nfft)
    B = (Sn > gam).astype(np.float64)
    B -= B.mean(0, keepdims=True)
    T = B.shape[0]
    P = (np.abs(sfft.fft(B, axis=0)) ** 2).sum(1)[1:T // 2 + 1]
    nk = len(P)
    if P.sum() <= 1e-9:
        return dict(D=float("nan"), llr=0.0)
    P = P + 1e-9 * P.mean()
    nu, grid, m = _m2_model(nk, nfft, T)
    nll0 = nk * np.log(P.mean())
    best = (np.inf, None, None)
    for r in R_GRID:
        den = 1.0 + r * m
        b = (P[None, :] / den).mean(1)
        nll = nk * np.log(b) + np.log(den).sum(1)
        g = int(np.argmin(nll))
        if nll[g] < best[0]:
            best = (nll[g], g, r)
    nllb, g, r = best
    fine = np.exp(np.linspace(np.log(grid[max(g - 2, 0)]), np.log(grid[min(g + 2, len(grid) - 1)]), 81))
    mf = np.zeros((len(fine), nk))
    for a in range(-3, 4):
        mf += np.sinc(fine[:, None] * (nu[None, :] - a / nfft)) ** 2 * np.sinc(nfft * nu[None, :] - a) ** 2
    den = 1.0 + r * mf
    nll = nk * np.log((P[None, :] / den).mean(1)) + np.log(den).sum(1)
    j = int(np.argmin(nll))
    return dict(D=float(fine[j]), llr=float(nll0 - nll[j]))


# ------------------------------------------------------------------ method 3
def method3(x, nfft=256, step=64, kappa=4, lmin=6, Sn=None):
    """Multiple-hop autocorrelation. c_t = 1 if the peak bin of frame t differs from that of frame t-kappa by >= 2 bins.
    rho(l) = normalised autocorrelation of c. stat = max rho over l in [lmin, n/2]; D from the regression of the
    peak lags near m*l1 on m (m = 1, 2, ...)."""
    if Sn is None:
        Sn = stft_norm(x, nfft, step)
    g = np.argmax(Sn, axis=1)
    c = (np.abs(g[kappa:] - g[:-kappa]) >= 2).astype(float)
    n = len(c)
    if c.std() < 1e-9:
        return dict(D=float("nan"), stat=0.0)
    cc = c - c.mean()
    M = sfft.next_fast_len(2 * n)
    r = sfft.ifft(np.abs(sfft.fft(cc, M)) ** 2).real[: n // 2 + 1]
    rho = r / r[0] * n / (n - np.arange(len(r)))
    lo, hi = lmin, n // 2
    seg = rho[lo:hi + 1]
    stat = float(seg.max())
    loc = np.flatnonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if len(loc) == 0:
        return dict(D=float("nan"), stat=stat)
    pk_l = loc + lo
    pk_v = seg[loc]
    cand = pk_l[pk_v >= 0.5 * pk_v.max()]
    l1 = None
    for l in cand:
        sup = 0
        for mm in (2, 3):
            L = int(round(mm * l))
            if L > hi:
                sup += 1
                continue
            w = max(3, int(0.03 * L))
            win = rho[max(L - w, lo):min(L + w, hi) + 1]
            sup += int(win.max() >= 0.3 * rho[l])
        if sup >= 2:
            l1 = int(l)
            break
    if l1 is None:
        l1 = int(cand[0])
    ms, ls = [], []
    cur = float(l1)
    for mm in range(1, 40):
        Lc = mm * cur
        if Lc > hi:
            break
        w = max(3, int(0.04 * Lc))
        a, b = int(Lc) - w, int(Lc) + w
        a, b = max(a, lo), min(b, hi)
        if b <= a + 1:
            continue
        j = a + int(np.argmax(rho[a:b + 1]))
        if rho[j] < 0.3 * rho[l1] and mm > 1:
            continue
        if 0 < j < len(rho) - 1:
            y0, y1, y2 = rho[j - 1], rho[j], rho[j + 1]
            den = y0 - 2 * y1 + y2
            dj = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        else:
            dj = 0.0
        ms.append(mm)
        ls.append(j + float(np.clip(dj, -1, 1)))
        if mm == 1:
            cur = ls[-1]
    ms, ls = np.array(ms, float), np.array(ls, float)
    D = float((ms * ls).sum() / (ms * ms).sum() * step) if len(ms) else float(l1 * step)
    return dict(D=D, stat=stat)
