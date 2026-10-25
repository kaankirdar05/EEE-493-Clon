"""Classical carrier / bandwidth / SNR estimators for one emitter, given an oracle box.

PSD scaling convention (same as wp4.spectrum): |X|^2 / sum(w^2), so complex white
noise of power s^2 per sample gives a mean of s^2 in every bin, whatever the FFT
size. A signal of total power P then satisfies P = sum_bins(PSD_signal) / nfft.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

DEFAULTS = dict(
    n_across=64,        # target number of Welch bins across the truth box width
    nfft_min=256,
    nfft_max=16384,
    min_segments=8,     # Welch segments (50 % overlap) required; caps nfft for short bursts
    smooth_frac=1 / 16, # moving-average length, as a fraction of the bins across the box
    widen=0.5,          # search region = box widened by this fraction of its width per side
    min_box_hz=2 * 10e6 / 512,  # floor on the box width used for the region / nfft (tones are 1 Hz)
    # noise floor: 'oob_lin' = robust mean of the region left/right of the box (+10 % guard),
    # linearly interpolated across the box (chosen on dev scenes 0-99); 'oob' = flat;
    # 'map' = WP4 excision noise map; 'hybrid' = map shape rescaled to the out-of-box level
    noise="oob_lin",
)

BW_LEVELS = (3, 10, 26)
LOG_AVG_NOISE_DB = 10 * np.log10(np.exp(np.euler_gamma))  # 2.507 dB: E[10log10 Exp(1)] below 10log10 mean


def welch(x: np.ndarray, nfft: int, window: str = "hann", overlap: float = 0.5):
    """Averaged periodogram, fftshifted, noise-normalized. Returns (psd, n_segments)."""
    w = np.hanning(nfft) if window == "hann" else np.blackman(nfft)
    step = max(1, int(nfft * (1 - overlap)))
    if len(x) < nfft:
        x = np.concatenate([x, np.zeros(nfft - len(x), x.dtype)])
    n_seg = 1 + (len(x) - nfft) // step
    idx = np.arange(nfft)[None, :] + step * np.arange(n_seg)[:, None]
    frames = x[idx].astype(np.complex128) * w
    p = np.abs(np.fft.fft(frames, axis=1)) ** 2 / np.sum(w**2)
    return np.fft.fftshift(p.mean(0)), n_seg


def _crossing(f, y, i_from, i_to, thr):
    """Walk from i_from towards i_to while y >= thr; return interpolated crossing freq and
    whether the walk hit the region boundary (saturated)."""
    step = 1 if i_to > i_from else -1
    i = i_from
    while i != i_to and y[i + step] >= thr:
        i += step
    if i == i_to:
        return f[i], True
    j = i + step  # y[i] >= thr > y[j]
    t = (y[i] - thr) / (y[i] - y[j]) if y[i] != y[j] else 0.0
    return f[i] + t * (f[j] - f[i]), False


def _band_power_snr(f, d, nz, a, b):
    """Noise-subtracted power over noise power in [a, b] (on bins inside)."""
    m = (f >= a) & (f <= b)
    if m.sum() == 0:
        k = int(np.argmin(np.abs(f - 0.5 * (a + b))))
        m = np.zeros_like(m)
        m[k] = True
    s, n = d[m].sum(), nz[m].sum()
    return s / n


def m2m4_snr(y: np.ndarray) -> tuple[float, bool]:
    """M2M4 SNR (linear) for a constant-modulus signal in complex Gaussian noise.

    SNR = sqrt(2 M2^2 - M4) / (M2 - sqrt(2 M2^2 - M4)). Returns (snr, valid);
    invalid when 2 M2^2 - M4 <= 0 (S = 0) or M2 <= sqrt(...) (N <= 0).
    """
    p = np.abs(y) ** 2
    m2, m4 = p.mean(), (p**2).mean()
    disc = 2 * m2**2 - m4
    if disc <= 0:
        return 0.0, False
    s = np.sqrt(disc)
    n = m2 - s
    if n <= 0:
        return np.inf, False
    return s / n, True


def characterize(seg: np.ndarray, fs: float, f_low: float, f_high: float, noise_fn, cfg=None) -> dict:
    """Estimate carrier, bandwidth and SNR of one emitter.

    seg      : complex I/Q samples of the box's time span
    f_low/high : truth box edges (Hz) - used only to place the search region and the nfft,
                 and by the '*_trueband' SNR estimator
    noise_fn : callable(freqs) -> noise power per sample at those baseband freqs over the
               box's time span (from the WP4 excision map); used when cfg['noise'] == 'map'
    """
    c = dict(DEFAULTS, **(cfg or {}))
    out: dict = {}
    bw_box = max(f_high - f_low, c["min_box_hz"])
    fc_box = 0.5 * (f_low + f_high)
    r_lo = max(-fs / 2, fc_box - bw_box / 2 - c["widen"] * bw_box)
    r_hi = min(fs / 2, fc_box + bw_box / 2 + c["widen"] * bw_box)

    # ---- Welch PSD with ~n_across bins over the box, enough segments for the span ----
    nfft = 2 ** int(np.round(np.log2(c["n_across"] * fs / bw_box)))
    cap = 2 ** int(np.floor(np.log2(max(2 * len(seg) / (c["min_segments"] + 1), 16))))
    nfft = int(np.clip(min(nfft, cap), c["nfft_min"], c["nfft_max"]))
    nfft = min(nfft, max(16, 2 ** int(np.floor(np.log2(max(len(seg), 16))))))
    psd, n_seg = welch(seg, nfft)
    fgrid = (np.arange(nfft) - nfft // 2) * fs / nfft
    reg = (fgrid >= r_lo) & (fgrid <= r_hi)
    if reg.sum() < 5:  # very narrow region: take the 5 nearest bins
        k = int(np.argmin(np.abs(fgrid - fc_box)))
        reg = np.zeros(nfft, bool)
        reg[max(k - 2, 0):k + 3] = True
    f = fgrid[reg]
    p = psd[reg]
    out.update(nfft=nfft, n_seg=n_seg)

    # ---- noise floor ----
    nz_map = np.asarray(noise_fn(f), float)
    nz = nz_map
    if c["noise"] != "map":
        g = 0.1 * bw_box  # guard beyond the box edges
        sides = [(f < f_low - g), (f > f_high + g)]

        def robust_mean(v):  # drop bins > 3x median (neighbour emitters, spurs)
            return np.mean(v[v <= 3 * np.median(v)])

        lv = [robust_mean(p[s]) if s.sum() >= 4 else np.nan for s in sides]
        if c["noise"] == "oob":  # flat floor from both out-of-box sides
            oob = sides[0] | sides[1]
            if oob.sum() >= 8:
                nz = np.full_like(p, robust_mean(p[oob]))
        elif c["noise"] == "oob_lin":  # linear across the region between the two sides
            if np.all(np.isfinite(lv)):
                fl, fr = np.mean(f[sides[0]]), np.mean(f[sides[1]])
                nz = lv[0] + (lv[1] - lv[0]) * (f - fl) / (fr - fl)
                nz = np.maximum(nz, 0.25 * min(lv))
            elif np.any(np.isfinite(lv)):
                nz = np.full_like(p, np.nanmean(lv))
        elif c["noise"] == "hybrid":  # map shape, level rescaled to the out-of-box sides
            oob = sides[0] | sides[1]
            if oob.sum() >= 8:
                nz = nz_map * robust_mean(p[oob] / nz_map[oob])
    d = p - nz
    m = max(1, int(round(c["smooth_frac"] * bw_box / (fs / nfft))))
    ds = uniform_filter1d(d, m, mode="nearest") if m > 1 else d

    # ---- carrier (a) peak ----
    ipk = int(np.argmax(ds))
    dpk = ds[ipk]
    out["fc_peak"] = f[ipk]

    # ---- x-dB bandwidths from the peak outward ----
    edges = {}
    for x in BW_LEVELS:
        thr = dpk * 10 ** (-x / 10)
        lo, sl = _crossing(f, ds, ipk, 0, thr)
        hi, sh = _crossing(f, ds, ipk, len(f) - 1, thr)
        if hi - lo <= 0:
            hi, lo = f[ipk] + 0.5 * fs / nfft, f[ipk] - 0.5 * fs / nfft
        edges[x] = (lo, hi)
        out[f"bw_{x}db"] = hi - lo
        out[f"sat_{x}db"] = bool(sl or sh)

    # ---- 99 % occupied bandwidth (ITU-R SM.443), cumulative of noise-subtracted PSD ----
    cum = np.cumsum(d)
    tot = cum[-1]
    if tot > 0:
        lo99 = float(np.interp(0.005 * tot, np.maximum.accumulate(cum), f))
        hi99 = float(np.interp(0.995 * tot, np.maximum.accumulate(cum), f))
        if hi99 <= lo99:
            lo99, hi99 = edges[10]
    else:
        lo99, hi99 = edges[10]
    out["bw_obw99"] = hi99 - lo99

    # ---- carrier (b) centroid over the -26 dB band, (c) -10 dB midpoint, (c') OBW midpoint ----
    lo26, hi26 = edges[26]
    sup = (f >= lo26) & (f <= hi26)
    w = np.clip(d[sup], 0, None)
    out["fc_centroid"] = float(np.sum(w * f[sup]) / np.sum(w)) if w.sum() > 0 else f[ipk]
    out["fc_mid10"] = 0.5 * sum(edges[10])
    out["fc_obw99"] = 0.5 * (lo99 + hi99)

    # ---- SNR ----
    def todb(v):
        return 10 * np.log10(v) if v > 0 else -np.inf

    out["snr_psd_obw"] = todb(_band_power_snr(f, d, nz, lo99, hi99))
    out["snr_psd_10db"] = todb(_band_power_snr(f, d, nz, *edges[10]))
    out["snr_psd_trueband"] = todb(_band_power_snr(f, d, nz, f_low, f_high) if f_high - f_low > fs / nfft
                                   else _band_power_snr(f, d, nz, fc_box - 0.5 * fs / nfft, fc_box + 0.5 * fs / nfft))

    # (d) TorchSig-matched peak-PSD SNR: 512-pt Blackman, non-overlapping frames,
    # peak of the time-averaged signal PSD over the noise PSD, within the truth box.
    pb, _ = welch(seg, 512, window="blackman", overlap=0.0)
    fb = (np.arange(512) - 256) * fs / 512
    inb = (fb >= f_low - fs / 512) & (fb <= f_high + fs / 512)
    if not inb.any():
        inb[int(np.argmin(np.abs(fb - fc_box)))] = True
    # noise on the 512-bin grid: the same floor as above (interpolated, flat outside the region)
    nfb = np.asarray(noise_fn(fb), float) if c["noise"] == "map" else np.interp(fb, f, nz)
    nb = np.maximum(nfb[inb], 1e-30)
    out["snr_peak512"] = todb(np.max((pb[inb] - nb) / nb))

    # (e) M2M4 on the band-limited time signal: FFT mask = estimated 99 % band
    X = np.fft.fft(seg.astype(np.complex128))
    fx = np.fft.fftfreq(len(seg), 1 / fs)
    mask = (fx >= lo99) & (fx <= hi99)
    if mask.sum() < 4:
        mask = (fx >= f_low) & (fx <= f_high)
    y = np.fft.ifft(X * mask)
    s, ok = m2m4_snr(y)
    out["snr_m2m4"] = float(np.clip(todb(s), -30, 60))
    out["m2m4_valid"] = bool(ok)

    # (f) noise-crossing bandwidth (TorchSig-matched definition): TorchSig takes the bins
    # where the signal-only MAX-HOLD spectrogram exceeds (log-avg noise + 3 dB) = (mean noise
    # + 0.49 dB). For a noise-like signal max-hold ~ mean * H_K over K frames, so the
    # equivalent threshold on the time-averaged PSD is d > N * 10^(0.049) / H_K.
    k_frames = max(1, len(seg) // 512)
    h_k = np.sum(1.0 / np.arange(1, k_frames + 1))
    thr_nc = 10 ** ((3 - LOG_AVG_NOISE_DB) / 10) / h_k
    pbs = uniform_filter1d(pb - nfb, 3, mode="nearest")
    rb = pbs / np.maximum(nfb, 1e-30)
    rin = (fb >= r_lo) & (fb <= r_hi)
    ii = np.flatnonzero(rin)
    if len(ii) >= 2:
        k0 = ii[np.argmax(rb[ii])]
        lo_i, hi_i = k0, k0
        while lo_i - 1 >= ii[0] and rb[lo_i - 1] > thr_nc:
            lo_i -= 1
        while hi_i + 1 <= ii[-1] and rb[hi_i + 1] > thr_nc:
            hi_i += 1
        lo_nc, hi_nc = fb[lo_i] - fs / 1024, fb[hi_i] + fs / 1024
    else:
        lo_nc, hi_nc = edges[10]
    out["bw_noisecross"] = hi_nc - lo_nc
    out["fc_noisecross"] = 0.5 * (lo_nc + hi_nc)

    # keep what the sanity plots need
    out["_plot"] = dict(f=f, p=p, nz=nz, ds=ds, edges=edges, obw=(lo99, hi99), nc=(lo_nc, hi_nc))
    return out
