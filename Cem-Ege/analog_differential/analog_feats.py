"""Azzouz-Nandi style + spectral-symmetry features for analog classes (Part A).

Front end (receiver model): channel filter = zero-phase FFT brick-wall |f| < 0.075 cycles/sample
(channel is +-0.0625, +-0.0125 margin for the residual offset). The out-of-channel region (|f|>0.1) of the
UNFILTERED frame is pure noise and gives N0 (noise power per sample at fs); this is used (i) to estimate the
SNR and (ii) to noise-compensate envelope variance and spectral sums.
"""
import numpy as np
import common  # noqa  (sets sys.path)
from amc.features import extract_features, FEATURE_NAMES

CH = 0.075          # channel-filter half width
NB_LO = (0.004, 0.012)    # discriminator spectrum sub-bands for the PM/FM slope (voice band 0.13B..0.92B, B = 0.03)
NB_HI = (0.012, 0.0275)

OWN = ["snr_est", "sig_aa", "sig_aa_c", "kurt_amp", "skew_amp", "gamma_max", "sig_f", "kurt_f",
       "carrier_frac", "line_x1", "line_x2", "asym_pk_db", "asym_pk_abs", "asym_nom_db", "asym_nom_abs", "bw90", "bw99", "occ", "fm_slope", "disc_bw"]
BL = ["bl_" + n for n in FEATURE_NAMES]
NAMES = OWN + BL
FI = {n: i for i, n in enumerate(NAMES)}


def _box(a, w):
    """circular moving average along axis 1 (width w bins)."""
    k = np.zeros(a.shape[1]); k[:w] = 1.0 / w
    k = np.roll(k, -(w // 2))
    return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(k)[None, :], axis=1).real


def _feats_batch(x):
    n, N = x.shape
    f = np.fft.fftfreq(N)
    X = np.fft.fft(x, axis=1)
    oob = np.abs(f) > 0.1
    w = np.hanning(N)
    Xw = np.fft.fft(x * w[None, :], axis=1)                            # windowed: no edge leakage into the noise floor
    N0 = np.mean(np.abs(Xw[:, oob]) ** 2, axis=1) / (N * np.mean(w**2))   # noise power / sample (full band)
    inb = np.abs(f) <= CH
    y = np.fft.ifft(X * inb[None, :], axis=1)
    frac = inb.mean()
    Pin = np.mean(np.abs(y) ** 2, axis=1)
    Ps = np.maximum(Pin - N0 * frac, 1e-3 * Pin)                       # signal power
    snr_full = Ps / N0
    snr_in = Ps / (N0 * frac)

    a = np.abs(y)
    ac = a / a.mean(1, keepdims=True) - 1
    sig_aa = ac.std(1)
    sig_aa_c = np.sqrt(np.maximum(sig_aa**2 - 1.0 / (2 * snr_in), 0))
    d = a - a.mean(1, keepdims=True)
    v = np.maximum((d**2).mean(1), 1e-20)
    kurt_amp = (d**4).mean(1) / v**2
    skew_amp = (d**3).mean(1) / v**1.5
    gamma_max = (np.abs(np.fft.fft(ac, axis=1)) ** 2).max(1) / N

    # discriminator output (cycles / sample)
    dsc = np.angle(y[:, 1:] * np.conj(y[:, :-1])) / (2 * np.pi)
    dsc = dsc - np.median(dsc, axis=1, keepdims=True)
    sig_f = np.sqrt(np.mean(dsc**2, axis=1))
    kurt_f = np.mean(dsc**4, axis=1) / np.maximum(np.mean(dsc**2, axis=1) ** 2, 1e-30)

    # windowed PSD of the filtered frame; noise floor per bin = N0
    Pw = np.abs(np.fft.fft(y * w[None, :], axis=1)) ** 2 / (N * np.mean(w**2))
    Pw = Pw * inb[None, :]
    S = np.where(inb[None, :], np.maximum(Pw - N0[:, None], 0), 0.0)
    Stot = np.maximum(S.sum(1), 1e-20)
    k_pk = np.argmax(_box(Pw, max(3, int(round(0.003 * N)))), axis=1)
    carrier_frac = np.array([S[i, np.arange(k_pk[i] - 2, k_pk[i] + 3) % N].sum() for i in range(n)]) / Stot
    # asymmetry about the smoothed spectral peak (carrier line for AM/PM, centre for DSB-SC, band edge for SSB)
    kk = np.arange(N)[None, :]
    rel = ((kk - k_pk[:, None] + N // 2) % N) - N // 2                 # signed bin offset to the peak
    up = (S * (rel > 2)).sum(1); lo = (S * (rel < -2)).sum(1)
    dl = 1e-3 * Stot
    asym_db = 10 * np.log10(np.maximum(up, dl) / np.maximum(lo, dl))
    # asymmetry about the NOMINAL channel centre (f = 0), peak lobe (+-3 bins) excluded
    fk = f[None, :]
    keep = (np.abs(rel) > 3)
    upn = (S * keep * (fk > 0.0005)).sum(1); lon = (S * keep * (fk < -0.0005)).sum(1)
    dn = 1e-3 * np.maximum((S * keep).sum(1), 1e-20)
    asym_nom_db = 10 * np.log10(np.maximum(upn, dn) / np.maximum(lon, dn))

    # carrier-line prominence (x^1): peak of the filtered spectrum vs mean in-band level
    line_x1 = np.log10(Pw.max(1) / np.maximum(Pw[:, inb].mean(1), 1e-20))
    P2 = np.abs(np.fft.fft((y**2) * w[None, :], axis=1)) ** 2
    in2 = np.abs(f) <= 2 * CH
    line_x2 = np.log10(P2.max(1) / np.maximum(P2[:, in2].mean(1), 1e-20))

    # occupancy
    cs = np.cumsum(np.fft.fftshift(S, axes=1), axis=1) / Stot[:, None]
    def width(q):
        lo_i = np.argmax(cs >= q, axis=1); hi_i = np.argmax(cs >= 1 - q, axis=1)
        return (hi_i - lo_i + 1) / N
    bw90, bw99 = width(0.05), width(0.005)
    Ps_s = _box(Pw, max(3, int(round(0.003 * N))))
    occ = ((Ps_s > 2 * N0[:, None]) & inb[None, :]).sum(1) / inb.sum()

    # PM vs FM: spectral slope of the discriminator output, two equal-width sub-bands of the voice band
    Pd = np.abs(np.fft.rfft(dsc * np.hanning(N - 1)[None, :], axis=1)) ** 2
    fd = np.arange(Pd.shape[1]) / (N - 1)
    lo_m = (fd >= NB_LO[0]) & (fd < NB_LO[1]); hi_m = (fd >= NB_HI[0]) & (fd < NB_HI[1])
    Plo, Phi = Pd[:, lo_m].mean(1), Pd[:, hi_m].mean(1)
    fl, fh_ = np.mean(NB_LO), np.mean(NB_HI)   # band centres
    fm_slope = np.log10(np.maximum(Phi, 1e-30) / np.maximum(Plo, 1e-30)) / np.log10(fh_ / fl)

    cd = np.cumsum(Pd[:, fd < 0.075], axis=1); cd = cd / np.maximum(cd[:, -1:], 1e-30)
    disc_bw = np.argmax(cd >= 0.9, axis=1) / (N - 1)          # 90 % point of the discriminator-output power
    own = np.stack([10 * np.log10(snr_full), sig_aa, sig_aa_c, kurt_amp, skew_amp, gamma_max, sig_f, kurt_f,
                    carrier_frac, line_x1, line_x2, asym_db, np.abs(asym_db), asym_nom_db, np.abs(asym_nom_db), bw90, bw99, occ, fm_slope, disc_bw], axis=1)
    bl = extract_features(y)
    return np.concatenate([own, bl], axis=1)


def analog_features(x, batch=256):
    out = [_feats_batch(x[i:i + batch]) for i in range(0, len(x), batch)]
    return np.nan_to_num(np.concatenate(out, 0), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
