"""Classical feature extraction for interference / jammer frames (no learning inside).

Everything is computed from one frame x (N = 262144 complex samples) after dividing by the ASSUMED
receiver noise power sigma_hat^2 (known to a calibration error, see `cal_db`).  After this scaling
the noise floor of a 512 x 512 WP4 spectrogram cell is Exp(1) (mean 1) -- the same convention as
wp4.spectrum.spectrogram -- so every threshold below is a noise-referenced number with a known
distribution under H0.

Feature groups (names in FEATURES):
  level        pow_excess_db          10 log10(mean |x|^2)   (0 on noise; needs the noise floor)
  occupancy    tf_occ                 fraction of spectrogram cells > 4.6 (1 % cell false-alarm level)
               bw_frac                fraction of frequency bins whose time-averaged spectrum Pbar > 1.3
               bw_rel                 same but also > peak/100 (INR-insensitive width)
               n_comp, comp_width     connected occupied components and their mean width (bins)
               n_peaks                prominent spectral peaks (prominence > 6 dB, >1.3 floor)
  spectrum     pk_med, pk_mean        max Pbar / median Pbar, / mean Pbar
               flat_occ, flat_all     Wiener spectral flatness inside the occupied set / over all bins
               kurt_f                 excess kurtosis of Pbar across frequency
  kurtosis     sk_mean, sk_med        spectral kurtosis  E[S^2]/E[S]^2  per occupied bin (2 = Gaussian noise,
                                      1 = constant envelope, 2/fill for intermittent signals)
               env_kurt               time-domain envelope kurtosis E|x|^4 / (E|x|^2)^2 (2 for Gaussian, 1 for CW/chirp)
  time         duty                   fraction of fine time blocks (32-128 samples) where the occupied-band power is above
                                      its noise-only 99.9 % quantile (Gamma law) -> time duty cycle
               top5_med               mean of top-5 % block powers / median block power (noise-floor free)
               on_db                  mean on-block power relative to its noise floor, dB
               q_cv                   coefficient of variation of block powers
  periodicity  per_h, per_lag         height and lag (log10 samples) of the first strong peak of the
                                      autocorrelation of the occupied-band block-power profile
                                      (cyclic line of the time-occupancy profile)
               bin_acf_h, bin_acf_lag same for the per-bin spectrogram time series (a sweep revisits a bin once per period)
  sweep        hough_*                Hough line (wp4.hough.hough_lines) on the thresholded spectrogram: votes fraction and |slope| (bins/frame)
               chirp_*                dechirp / FrFT (ext.chirp: |FFT(x exp(-j pi k n^2))|^2) on 256- and 1024-sample segments:
                                      best normalised peak gain G in [0,1] over chirp rates k != 0, its rate, and the k = 0 (tone) gain
  third order  bis_*                  bispectrum/bicoherence statistics (classical_amc/discovered/bispectrum.py) on the first 32768 samples
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal, stats
from scipy.special import gammaincinv

import jcommon as common  # noqa: F401  (sets sys.path)
from jcommon import N
from wp4.spectrum import spectrogram
from wp4.hough import hough_lines

T1 = 1.3          # Pbar occupancy threshold (noise-referenced; > 4 sigma of Pbar noise + 0.25 dB calibration error)
CELL_THR = 4.6    # cell level: exp(-4.6) = 1 %
BIS_LEN = 32768

_GQ = {}


def _gamma_thr(dof, p=1e-3):
    key = (round(dof, 3), p)
    if key not in _GQ:
        _GQ[key] = gammaincinv(dof, 1 - p) / dof
    return _GQ[key]


# ------------------------------------------------------------------ dechirp (FrFT)

def dechirp_gains(x, L, ks, nseg, nfft_mult=2, chunk=40):
    """Per-segment normalised dechirp peak gain on a k grid.
    G(seg, k) = max_f |sum_n x e^{-j pi k n'^2} e^{-j 2 pi f n}|^2 / (L * sum |x|^2)   in [0, 1],
    n' = n - L/2 (same dechirp as ext.chirp.dechirp_search, normalised by the segment energy).
    Returns G (nseg, len(ks))."""
    starts = np.linspace(0, len(x) - L, nseg).astype(int)
    segs = np.stack([x[s:s + L] for s in starts]).astype(np.complex64)
    E = np.sum(np.abs(segs.astype(np.complex128)) ** 2, axis=1) + 1e-30
    n = np.arange(L) - L / 2
    G = np.empty((nseg, len(ks)))
    for a in range(0, len(ks), chunk):
        kk = ks[a:a + chunk]
        ph = np.exp(-1j * np.pi * kk[:, None] * n[None, :] ** 2).astype(np.complex64)            # (nk, L)
        Y = np.fft.fft(segs[:, None, :] * ph[None], n=nfft_mult * L, axis=2)
        G[:, a:a + chunk] = np.max(np.abs(Y) ** 2, axis=2) / (L * E[:, None])
    return G


def k_grid(L, kmax, step_mult=1.0):
    """Chirp-rate grid with step ~ 1/L^2 (residual sweep over the segment < 1 FFT bin)."""
    d = step_mult / L ** 2
    m = int(np.ceil(kmax / d))
    return np.arange(-m, m + 1) * d


KG = {256: (k_grid(256, 1.5e-3), 16), 1024: (k_grid(1024, 3.0e-4), 8)}


def chirp_features(x):
    out = {}
    for L, (ks, nseg) in KG.items():
        G = dechirp_gains(x, L, ks, nseg)
        kmin = 4.0 / L ** 2                                   # inside this |k| a chirp still looks like a tone at this L (residual sweep < 4 bins)
        zero = np.abs(ks) < 1e-12
        far = np.abs(ks) >= kmin
        Gc = G[:, far]
        per_seg = Gc.max(1)
        i = np.argmax(per_seg)
        j = np.argmax(Gc[i])
        out[f"chirp_G{L}"] = per_seg.max()
        out[f"chirp_k{L}"] = abs(ks[far][j])
        out[f"chirp_top{L}"] = np.mean(np.sort(per_seg)[-max(1, nseg // 4):])
        out[f"chirp_G0_{L}"] = G[:, zero].max()
    return out


# ------------------------------------------------------------------ small helpers

def _components(occ, gap=2):
    """Connected runs of True after bridging gaps <= `gap` bins. Returns list of (start, stop)."""
    if not occ.any():
        return []
    m = ndimage.binary_closing(np.pad(occ, gap + 1), structure=np.ones(gap + 1, bool))[gap + 1:-(gap + 1)]
    lab, n = ndimage.label(m)
    return [(sl[0].start, sl[0].stop) for sl in ndimage.find_objects(lab)]


def _acf_peak(z, min_lag, max_lag):
    """Normalised autocorrelation of z (mean removed); first strong local maximum AFTER the first zero crossing
    of the autocorrelation (a pulse train of any duty crosses zero within P/4; this skips the short-lag wiggles
    of noise inside a pulse, which are otherwise picked up as 'peaks').  The peak lag is refined by a parabola.
    Returns (height, lag in samples of z) or (0, 0)."""
    z = z - z.mean()
    M = len(z)
    f = np.fft.rfft(z, 2 * M)
    r = np.fft.irfft(np.abs(f) ** 2)[:M]
    r = r / max(r[0], 1e-30)
    seg = r[:max_lag]
    neg = np.flatnonzero(seg < 0)
    if len(neg) == 0:
        return 0.0, 0.0
    start = max(min_lag, int(neg[0]))
    pk, _ = signal.find_peaks(seg, height=0.0)
    pk = pk[pk >= start]
    pk = pk[(pk > 0) & (pk < len(seg) - 1)]
    if len(pk) == 0:
        return 0.0, 0.0
    h = seg[pk]
    ok = np.flatnonzero(h >= 0.5 * h.max())
    i = pk[ok[0]]
    d = seg[i - 1] - 2 * seg[i] + seg[i + 1]
    off = 0.5 * (seg[i - 1] - seg[i + 1]) / d if d < 0 else 0.0
    return float(seg[i]), float(i + off)


def _mask_fine(occ_dil, n=N):
    """512-bin occupancy (fftshifted) -> boolean mask over the N FFT bins (FFT order)."""
    fine = np.roll(np.repeat(occ_dil, n // 512), -(n // 1024))
    return np.fft.ifftshift(fine)


def _flat(v):
    v = np.maximum(v, 1e-12)
    return float(np.exp(np.mean(np.log(v))) / np.mean(v))


# ------------------------------------------------------------------ main extractor

BIS_ON = True


def extract(x, cal_db=0.0, bispec=True):
    """Feature dict for one frame. cal_db: assumed noise power = 10^(cal_db/10) (true value is 1)."""
    xn = (x / np.sqrt(10 ** (cal_db / 10))).astype(np.complex64)
    S = spectrogram(xn, 512)                                   # (512 time, 512 freq), noise mean 1
    T, F = S.shape
    f = {}
    p_all = np.abs(xn.astype(np.complex128)) ** 2
    f["pow_excess_db"] = 10 * np.log10(p_all.mean())
    f["env_kurt"] = float(np.mean(p_all ** 2) / np.mean(p_all) ** 2)

    Pbar = S.mean(0)
    med = np.median(Pbar)
    occ = Pbar > T1
    f["tf_occ"] = float(np.mean(S > CELL_THR))
    f["bw_frac"] = float(occ.mean())
    f["bw_rel"] = float(np.mean(occ & (Pbar > Pbar.max() / 100)))
    if occ.sum() >= 2:
        plateau = np.percentile(Pbar[occ], 90)
        f["bw_half"] = float(np.mean(occ & (Pbar > 0.5 * plateau)))     # half-power width of the occupied spectrum (skirt-insensitive)
    else:
        f["bw_half"] = 0.0
    comps = _components(occ)
    f["n_comp"] = len(comps)
    f["comp_width"] = float(np.mean([b - a for a, b in comps])) if comps else 0.0
    LdB = 10 * np.log10(np.maximum(Pbar, 1e-3))
    pk, _ = signal.find_peaks(LdB, height=10 * np.log10(T1), prominence=6.0, distance=2)
    f["n_peaks"] = len(pk)
    # auxiliary estimates for the sanity checks only (NOT classifier features: position-dependent)
    fgrid = (np.arange(F) - F // 2) / F
    f["aux_f_peak"] = float(fgrid[np.argmax(Pbar)])
    f["aux_peak_spacing"] = float(np.median(np.diff(fgrid[pk]))) if len(pk) > 1 else 0.0
    f["aux_f_center"] = float(fgrid[occ].mean()) if occ.any() else 0.0
    f["pk_med"] = float(Pbar.max() / med)
    f["pk_mean"] = float(Pbar.max() / Pbar.mean())
    f["flat_occ"] = _flat(Pbar[occ]) if occ.sum() >= 2 else 1.0
    f["flat_all"] = _flat(Pbar)
    f["kurt_f"] = float(stats.kurtosis(Pbar))

    occ_dil = ndimage.binary_dilation(occ, structure=np.ones(5, bool)) if occ.any() else np.ones(F, bool)
    Sk = S[:, occ_dil]
    sk = np.mean(Sk ** 2, axis=0) / np.maximum(np.mean(Sk, axis=0), 1e-12) ** 2
    f["sk_mean"] = float(sk.mean())
    f["sk_med"] = float(np.median(sk))

    # ---- fine time profile of the occupied band
    mask = _mask_fine(occ_dil)
    f_o = float(mask.mean())
    y = np.fft.ifft(np.fft.fft(xn.astype(np.complex128)) * mask)
    Lb = int(np.clip(2 ** np.ceil(np.log2(4 / f_o)), 32, 128))
    pb = np.mean(np.abs(y[: (N // Lb) * Lb]).reshape(-1, Lb) ** 2, axis=1)         # block powers
    thr = f_o * _gamma_thr(max(Lb * f_o, 1.0))
    on = pb > thr
    f["duty"] = float(on.mean())
    f["on_db"] = float(10 * np.log10(max(pb[on].mean() / f_o, 1e-3))) if on.any() else 0.0
    f["q_cv"] = float(pb.std() / pb.mean())
    # noise-floor-free time anomaly on the whole band (64-sample blocks)
    pw = np.mean(p_all[: (N // 64) * 64].reshape(-1, 64), axis=1)
    top = np.sort(pw)[-len(pw) // 20:]
    f["top5_med"] = float(top.mean() / max(np.median(pw), 1e-12))
    h, lag = _acf_peak(pb, 2, len(pb) // 2)
    f["per_h"] = h
    f["per_lag"] = float(np.log10(lag * Lb)) if lag > 0 else 0.0
    # per-bin periodicity of the spectrogram time series (sweep revisits a bin once per period)
    Z = Sk - Sk.mean(0)
    Fz = np.fft.rfft(Z, 2 * T, axis=0)
    R = np.fft.irfft(np.abs(Fz) ** 2, axis=0)[:T].mean(1)
    R = R / max(R[0], 1e-30)
    neg = np.flatnonzero(R[: T // 2] < 0)
    pk2 = np.array([], int)
    if len(neg):
        pk2, _ = signal.find_peaks(R[: T // 2], height=0.0)
        pk2 = pk2[(pk2 >= max(2, neg[0])) & (pk2 > 0) & (pk2 < T // 2 - 1)]
    if len(pk2):
        hh = R[pk2]
        i = pk2[np.flatnonzero(hh >= 0.5 * hh.max())[0]]
        d = R[i - 1] - 2 * R[i] + R[i + 1]
        off = 0.5 * (R[i - 1] - R[i + 1]) / d if d < 0 else 0.0
        f["bin_acf_h"], f["bin_acf_lag"] = float(R[i]), float(np.log10((i + off) * 512))
    else:
        f["bin_acf_h"], f["bin_acf_lag"] = 0.0, 0.0

    # ---- Hough on the thresholded spectrogram (wp4.hough)
    tt, ff = np.nonzero(S > np.maximum(CELL_THR, 0.05 * S.max(1, keepdims=True)))   # row-relative threshold: thin lines at any INR
    npts = len(tt)
    f["hough_npts"] = npts
    if npts > 20:
        if npts > 8000:
            sel = np.random.default_rng(0).choice(npts, 8000, replace=False)
            tt, ff = tt[sel], ff[sel]
        m = np.zeros(S.shape, bool)
        m[tt, ff] = True
        lines = hough_lines(m, n_theta=360, min_votes=1, exclude_deg=1.5, nms=(8, 6), max_lines=1)
        if lines:
            rho, th = lines[0]
            d = np.abs(ff * np.cos(th) + tt * np.sin(th) - rho)
            v = int(np.sum(d <= 1.0))
            f["hough_frac"] = v / len(tt)
            f["hough_votes"] = v
            f["hough_slope"] = float(np.log10(max(abs(np.tan(th)), 1e-3)))
        else:
            f["hough_frac"], f["hough_votes"], f["hough_slope"] = 0.0, 0, 0.0
    else:
        f["hough_frac"], f["hough_votes"], f["hough_slope"] = 0.0, 0, 0.0

    f.update(chirp_features(xn))
    f["chirp_Gmax"] = max(f["chirp_G256"], f["chirp_G1024"])
    f["aux_k_hat"] = f["chirp_k256"] if f["chirp_G256"] >= f["chirp_G1024"] else f["chirp_k1024"]
    f["chirp_ratio"] = f["chirp_Gmax"] / max(f["chirp_G0_256"], f["chirp_G0_1024"], 1e-6)

    if bispec:
        import bispectrum as bs
        xb = bs.dc.normalize(xn[None, :BIS_LEN].astype(np.complex128))
        for sig_, sname in ((xb, "x"), (np.abs(xb), "amp")):          # inst-frequency signal dropped (noise-like, cost)
            v = bs._stats(*bs.bispectrum(sig_))
            for st, val in zip(bs.STATS, v):
                f[f"bis_{sname}_{st}"] = float(np.nan_to_num(val[0], nan=0.0, posinf=1e6, neginf=-1e6))
    return f


def feature_names(bispec=True):
    rng = np.random.default_rng(0)
    x = common.cnoise(rng).astype(np.complex64)
    return list(extract(x, bispec=bispec).keys())
