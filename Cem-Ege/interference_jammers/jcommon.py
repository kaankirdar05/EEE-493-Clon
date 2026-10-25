"""Frame generators for interference / jammer detection and classification.

Conventions (all frequencies in cycles/sample, i.e. fs = 1; the physical rate is fs = 10 MS/s):
  * a frame is N = 262144 complex samples (same size as the wideband scenes, 512 x 512 spectrogram)
  * receiver noise is complex white Gaussian with power sigma^2 = 1 per sample
  * INR = (average interference power over the frame) / sigma^2, per sample.  For gated types this
    is the AVERAGE power, so the on-state power is INR/duty.  Frame power is normalised to the
    requested value exactly (measured power, not expected power).
  * JSR (Part B) = jammer power / legitimate-signal power, both total per-sample powers.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "wp4_detection"), os.path.join(ROOT, "detection_extended"),
          os.path.join(ROOT, "classical_amc", "discovered")):
    if p not in sys.path:
        sys.path.append(p)

N = 262144
FS_HZ = 10e6
KINDS = ["cw", "comb", "spot", "barrage", "sweep", "pnoise", "pcw"]
KIND_LABEL = {"cw": "CW tone", "comb": "multi-tone comb", "spot": "narrowband (spot) noise",
              "barrage": "barrage noise", "sweep": "sweep (sawtooth chirp)",
              "pnoise": "pulsed noise", "pcw": "pulsed CW"}


def cnoise(rng, n=N):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def _scale_to(x, power):
    p = np.mean(np.abs(x) ** 2)
    return x * np.sqrt(power / max(p, 1e-30))


def band_noise(rng, fc, bw, n=N):
    """Gaussian noise with a brick-wall spectrum on bins whose frequency is in [fc-bw/2, fc+bw/2]."""
    X = cnoise(rng, n)
    F = np.fft.fft(X)
    f = np.fft.fftfreq(n)
    d = (f - fc + 0.5) % 1.0 - 0.5            # circular distance to the centre
    F[np.abs(d) > bw / 2] = 0
    return np.fft.ifft(F)


def tone(rng, f, n=N):
    return np.exp(1j * (2 * np.pi * f * np.arange(n) + rng.uniform(0, 2 * np.pi)))


def gate(rng, period, duty, n=N):
    """Periodic rectangular gate, random phase. period in samples (float ok), duty in (0,1)."""
    ph = rng.uniform(0, period)
    t = (np.arange(n) + ph) % period
    return (t < duty * period).astype(float)


def sawtooth_chirp(f_lo, bw, period, up=True, n=N, phase0=0):
    """Periodic sawtooth sweep: IF goes f_lo -> f_lo+bw (or reverse) linearly over `period` samples
    (integer), then jumps back.  x[n+period] = x[n] exactly.  Returns x, k (rate, cycles/sample^2)."""
    P = int(period)
    m = np.arange(P)
    k = bw / P
    if up:
        ph = 2 * np.pi * (f_lo * m + 0.5 * k * m ** 2)
    else:
        ph = 2 * np.pi * ((f_lo + bw) * m - 0.5 * k * m ** 2)
    one = np.exp(1j * ph)
    reps = int(np.ceil(n / P)) + 2
    off = int(phase0) % P
    x = np.tile(one, reps)[off: off + n]
    return x, (k if up else -k)


def rrc(beta, sps, span=10):
    n = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(t) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


_RRC = {}


def qpsk(rng, bw_frac=None, fc=None, n=N, beta=0.35):
    """Legitimate QPSK, RRC 0.35, occupied bandwidth bw_frac*fs (default U(1,5)%), centre fc.
    Integer samples/symbol sps = 1.35/bw (rounded), so the realised bandwidth is 1.35/sps.
    Returns unit-power x, dict(fc, bw)."""
    if bw_frac is None:
        bw_frac = rng.uniform(0.01, 0.05)
    sps = int(round((1 + beta) / bw_frac))
    bw = (1 + beta) / sps
    if fc is None:
        fc = rng.uniform(-0.5 + bw / 2 + 0.01, 0.5 - bw / 2 - 0.01)
    nsym = n // sps + 40
    s = (rng.choice([-1, 1], nsym) + 1j * rng.choice([-1, 1], nsym)) / np.sqrt(2)
    up = np.zeros(nsym * sps, complex)
    up[::sps] = s
    if sps not in _RRC:
        _RRC[sps] = rrc(beta, sps)
    y = fftconvolve(up, _RRC[sps], mode="same")
    y = y[20 * sps: 20 * sps + n]
    y = y * np.exp(2j * np.pi * fc * np.arange(n))
    return _scale_to(y, 1.0), dict(fc=fc, bw=bw)


def make_jammer(kind, rng, inr_db, n=N, jam_band=None):
    """Jammer of the given kind with average power 10^(inr_db/10) (sigma^2 = 1).
    Returns (j, params); params holds the TRUE parameters used by the sanity checks.
    jam_band: optional (fc, bw) override for spot noise (co-channel spot jammer)."""
    P = 10 ** (inr_db / 10)
    if kind == "cw":
        f = rng.uniform(-0.5, 0.5)
        j = tone(rng, f, n)
        par = dict(f=f, n_lines=1)
    elif kind == "comb":
        m = int(rng.integers(3, 11))
        dmax = min(0.10, 0.9 / (m - 1))
        d = rng.uniform(0.02, dmax)
        f0 = rng.uniform(-0.5, 0.5 - (m - 1) * d)
        fs_ = f0 + d * np.arange(m)
        t = np.arange(n)
        j = sum(np.exp(1j * (2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))) for f in fs_)
        par = dict(f0=f0, spacing=d, n_lines=m, bw=d * (m - 1))
    elif kind == "spot":
        bw = rng.uniform(0.01, 0.05)
        fc = rng.uniform(-0.5 + bw / 2, 0.5 - bw / 2)
        if jam_band is not None:
            fc, bw = jam_band
        j = band_noise(rng, fc, bw, n)
        par = dict(fc=fc, bw=bw)
    elif kind == "barrage":
        bw = rng.uniform(0.30, 1.0)
        fc = rng.uniform(-0.5 + bw / 2, 0.5 - bw / 2) if bw < 1 else 0.0
        j = band_noise(rng, fc, bw, n)
        par = dict(fc=fc, bw=bw)
    elif kind == "sweep":
        B = rng.uniform(0.10, 0.80)
        Pper = int(round(rng.uniform(0.002, 0.05) * n))
        f_lo = rng.uniform(-0.5, 0.5 - B)
        up = bool(rng.integers(0, 2))
        j, k = sawtooth_chirp(f_lo, B, Pper, up, n, phase0=rng.integers(0, Pper))
        j = j * np.exp(1j * rng.uniform(0, 2 * np.pi))
        par = dict(bw=B, period=Pper, k=k, f_lo=f_lo, up=up,
                   slope_bins_per_frame=k * 512 * 512)
    elif kind == "pnoise":
        bw = rng.uniform(0.20, 1.0)
        fc = rng.uniform(-0.5 + bw / 2, 0.5 - bw / 2) if bw < 1 else 0.0
        period = rng.uniform(0.01, 0.10) * n
        duty = rng.uniform(0.05, 0.50)
        g = gate(rng, period, duty, n)
        j = band_noise(rng, fc, bw, n) * g
        par = dict(fc=fc, bw=bw, period=period, duty=float(g.mean()), duty_nominal=duty)
    elif kind == "pcw":
        f = rng.uniform(-0.5, 0.5)
        period = rng.uniform(0.01, 0.10) * n
        duty = rng.uniform(0.05, 0.50)
        g = gate(rng, period, duty, n)
        j = tone(rng, f, n) * g
        par = dict(f=f, period=period, duty=float(g.mean()), duty_nominal=duty)
    else:
        raise ValueError(kind)
    return _scale_to(j, P), par


def make_frame(kind, rng, inr_db=None, with_sig=False, sig_snr_db=None, noise=True, n=N,
               sig_gain_db=0.0, jsr_db=None):
    """One frame. kind in KINDS, 'spot_cochannel', or 'none' (noise and/or signal only).
    inr_db: jammer power / sigma^2.  If with_sig the legitimate QPSK has power 10^(sig_snr_db/10)
    (random U(5,20) dB if None); sig_gain_db perturbs the actual signal power (power-uncertainty test).
    jsr_db (needs sig_snr_db): jammer power = NOMINAL signal power * 10^(jsr/10).
    noise=False builds a noiseless frame (sanity checks).  Returns x (complex64), meta."""
    x = cnoise(rng, n) if noise else np.zeros(n, complex)
    meta = dict(kind=kind)
    if with_sig:
        s, sp = qpsk(rng, n=n)
        snr = rng.uniform(5, 20) if sig_snr_db is None else sig_snr_db
        x = x + s * np.sqrt(10 ** ((snr + sig_gain_db) / 10))
        meta.update(sig=sp, sig_snr_db=snr)
    if kind != "none":
        if jsr_db is not None:
            inr_db = sig_snr_db + jsr_db
        if kind == "spot_cochannel":
            fcs, bws = meta["sig"]["fc"], meta["sig"]["bw"]
            j, par = make_jammer("spot", rng, inr_db, n, jam_band=(fcs, min(0.2, 1.5 * bws)))
        else:
            j, par = make_jammer(kind, rng, inr_db, n)
        x = x + j
        meta.update(par=par, inr_db=inr_db)
    return x.astype(np.complex64), meta
