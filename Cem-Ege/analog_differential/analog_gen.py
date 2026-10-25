"""Synthetic analog-modulation frames (Part A).

Sampling convention: every frame is a channelised complex-baseband stream at fs = 8 x CHANNEL WIDTH
  narrow classes  (AM-DSB-WC, AM-DSB-SC, SSB-USB, SSB-LSB, VSB, NBFM, PM): 12.5 kHz channel -> fs = 100 kHz
  WBFM                                                                     : 200  kHz channel -> fs = 1.6 MHz
so all frames are expressed in NORMALISED frequency (cycles/sample, channel = +-0.0625):
  narrow message bandwidth  B = 3 kHz / 100 kHz   = 0.03
  WBFM message bandwidth    B = 15 kHz / 1.6 MHz  = 0.009375
  NBFM peak deviation       2.5 kHz / 100 kHz     = 0.025   (jittered x U(0.8,1.2))
  WBFM peak deviation       75 kHz / 1.6 MHz      = 0.046875 (jittered x U(0.8,1.2))
  PM peak phase deviation   U(0.8,1.2) rad
Message: 70 % low-pass Gaussian noise with a voice-like random spectral tilt (PSD ~ f^-tau, tau~U(0,1.2)
above 0.1 B, 1st-order high-pass at 0.08 B, 4th-order Butterworth low-pass at B), 30 % two-tone
(f1~U(.15,.35)B, f2~U(.6,.9)B, a2/a1~U(.5,1)). Message is peak-normalised over the frame.
AM index mu~U(0.5,1). VSB: suppressed carrier, ideal H(f)=0.5(1+clip(s f/fv,-1,1)), fv~U(.15,.4)B, s=+-1.
SSB: ideal analytic signal (carrier suppressed). Residual carrier offset U(-0.004,0.004) cyc/sample,
random phase. SNR = signal power / white-noise power per complex sample at fs (as in the baseline synthetic.py).
"""
import numpy as np

CLASSES = ["AM-DSB-WC", "AM-DSB-SC", "SSB-USB", "SSB-LSB", "VSB", "NBFM", "WBFM", "PM"]
B_NB, B_WB = 0.03, 0.009375
DEV_NB, DEV_WB = 0.025, 0.046875
OFFSET_MAX = 0.004


def _message(n, M, B, rng, force_tone=None, tau=None):
    """(n, M) real messages, plus is_tone flags."""
    f = np.fft.fftfreq(M)
    af = np.abs(f)
    fh = 0.08 * B
    hp = af / np.sqrt(af**2 + fh**2)
    lp = 1 / np.sqrt(1 + (af / B) ** 8)
    if tau is None:
        tau = rng.uniform(0, 1.2, size=(n, 1))
    tilt = (1 + af[None, :] / (0.1 * B)) ** (-tau / 2)
    H = (hp * lp)[None, :] * tilt
    H[:, 0] = 0
    m_noise = np.fft.ifft(np.fft.fft(rng.standard_normal((n, M)), axis=1) * H, axis=1).real
    t = np.arange(M)[None, :]
    f1 = rng.uniform(0.15, 0.35, (n, 1)) * B
    f2 = rng.uniform(0.6, 0.9, (n, 1)) * B
    a2 = rng.uniform(0.5, 1.0, (n, 1))
    m_tone = (np.cos(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi, (n, 1)))
              + a2 * np.cos(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi, (n, 1))))
    is_tone = rng.random(n) < 0.3 if force_tone is None else np.full(n, bool(force_tone))
    m = np.where(is_tone[:, None], m_tone, m_noise)
    return m, is_tone


def synth_clean(cls, n, N, rng, force_tone=None, tau=None, dev_scale=None, fv_frac=None):
    """Noise-free unit-power complex frames (n, N) and per-frame metadata dict."""
    B = B_WB if cls == "WBFM" else B_NB
    M = 2 * N
    m, is_tone = _message(n, M, B, rng, force_tone, tau)
    off = rng.integers(0, M - N + 1, size=n)
    idx = off[:, None] + np.arange(N)[None, :]
    crop = lambda a: np.take_along_axis(a, idx, axis=1)
    m = m / np.max(np.abs(crop(m)), axis=1, keepdims=True)
    f = np.fft.fftfreq(M)[None, :]
    meta = {"is_tone": is_tone, "m": crop(m), "fs_B": B}
    if cls == "AM-DSB-WC":
        mu = rng.uniform(0.5, 1.0, (n, 1)); meta["mu"] = mu[:, 0]
        s = (1 + mu * m).astype(complex)
    elif cls == "AM-DSB-SC":
        s = m.astype(complex)
    elif cls in ("SSB-USB", "SSB-LSB"):
        M_ = np.fft.fft(m, axis=1)
        sel = (f > 0) if cls == "SSB-USB" else (f < 0)
        s = np.fft.ifft(2 * M_ * sel, axis=1)
    elif cls == "VSB":
        fv = (rng.uniform(0.15, 0.4, (n, 1)) if fv_frac is None else fv_frac) * B
        sgn = rng.choice([-1.0, 1.0], (n, 1)); meta["vsb_sign"] = sgn[:, 0]
        Hv = 0.5 * (1 + np.clip(sgn * f / fv, -1, 1))
        s = np.fft.ifft(np.fft.fft(m, axis=1) * Hv, axis=1)
    elif cls in ("NBFM", "WBFM"):
        dev = (DEV_WB if cls == "WBFM" else DEV_NB) * (rng.uniform(0.8, 1.2, (n, 1)) if dev_scale is None else dev_scale)
        meta["dev"] = np.broadcast_to(dev, (n, 1))[:, 0]
        s = np.exp(1j * 2 * np.pi * dev * np.cumsum(m, axis=1))
    elif cls == "PM":
        dphi = rng.uniform(0.8, 1.2, (n, 1)) if dev_scale is None else dev_scale
        meta["dphi"] = np.broadcast_to(dphi, (n, 1))[:, 0]
        s = np.exp(1j * dphi * m)
    else:
        raise ValueError(cls)
    s = crop(s)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2, axis=1, keepdims=True))
    return s, meta


def impair(s, snr_db, rng, offset_max=OFFSET_MAX):
    """Random carrier phase / offset and AWGN at `snr_db` (per complex sample at fs)."""
    n, N = s.shape
    f0 = rng.uniform(-offset_max, offset_max, (n, 1))
    ph = rng.uniform(0, 2 * np.pi, (n, 1))
    x = s * np.exp(1j * (ph + 2 * np.pi * f0 * np.arange(N)[None, :]))
    snr = np.broadcast_to(np.asarray(snr_db, float).reshape(-1, 1) if np.ndim(snr_db) else np.float64(snr_db), (n, 1))
    sig = 10 ** (-snr / 20) / np.sqrt(2)
    w = sig * (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N)))
    return x + w, f0[:, 0]


def make_frames(cls, n, N, snr_db, rng, **kw):
    s, meta = synth_clean(cls, n, N, rng, **kw)
    x, f0 = impair(s, snr_db, rng)
    meta["f0"] = f0
    return x, meta
