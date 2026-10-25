"""Bell-202 AFSK / APRS on NBFM (Part B) and negative (voice-like NBFM) generators.

fs = 96 kHz complex baseband around the channel (12.5 kHz channel -> 7.7x oversampled; AFSK on +-3 kHz NBFM:
Carson ~ 2(3+2.2) = 10.4 kHz). 1200 baud = exactly 80 samples/baud. Frame = 0.2 s = 19200 samples (240 bit).
AFSK audio: phase-continuous marks/spaces 1200/2200 Hz, NRZI (0 = toggle), HDLC: preamble of 5..45 flags (0x7E)
then random payload with bit stuffing; the frame is a random window (random start bit and random sub-bit offset).
FM peak deviation 3 kHz x U(0.8,1.2) (audio amplitude 1). Carrier offset U(-500, 500) Hz, random phase,
AWGN at SNR = signal power / noise power per complex sample at fs.
Negatives: 'voice' = NBFM with LPF-noise message (band 300-3000 Hz, random spectral tilt), same peak deviation;
'twotone' = NBFM with two tones in 500..2800 Hz; 'tone' = steady 1200 or 2200 Hz tone on NBFM.
"""
import numpy as np

FS = 96000.0
BAUD = 1200.0
SPB = int(FS / BAUD)           # 80
NFR = 19200
FM_, FS_ = 1200.0, 2200.0
DEV = 3000.0
OFF_MAX = 500.0


def _flag_bits():
    return np.array([0, 1, 1, 1, 1, 1, 1, 0])


def _packet_bits(rng, nbits_needed):
    n_pre = int(rng.integers(5, 46))
    pre = np.tile(_flag_bits(), n_pre)
    pay = rng.integers(0, 2, nbits_needed + 200)
    out, ones = [], 0
    for b in pay:                      # HDLC bit stuffing
        out.append(b)
        ones = ones + 1 if b == 1 else 0
        if ones == 5:
            out.append(0); ones = 0
    return np.concatenate([pre, np.array(out)]), n_pre


def afsk_audio(rng, nfr=NFR):
    nbit = nfr // SPB + 3
    bits, n_pre = _packet_bits(rng, nbit)
    start = int(rng.integers(0, 8 * n_pre + 120))
    b = bits[start:start + nbit]
    # NRZI: bit 0 -> toggle tone, bit 1 -> keep
    tone = np.cumsum(1 - b) % 2            # 0 -> mark(1200), 1 -> space(2200)
    f = np.where(tone == 0, FM_, FS_)
    f = np.repeat(f, SPB)
    sub = int(rng.integers(0, SPB))
    f = f[sub:sub + nfr]
    ph = 2 * np.pi * np.cumsum(f) / FS + rng.uniform(0, 2 * np.pi)
    return np.cos(ph)


def voice_audio(rng, nfr=NFR, kind="voice"):
    M = 2 * nfr
    f = np.fft.fftfreq(M, 1 / FS)
    af = np.abs(f)
    if kind == "voice":
        tau = rng.uniform(0, 1.2)
        H = (af / np.sqrt(af**2 + 300.0**2)) / np.sqrt(1 + (af / 3000.0) ** 8) * (1 + af / 300.0) ** (-tau / 2)
        H[0] = 0
        m = np.fft.ifft(np.fft.fft(rng.standard_normal(M)) * H).real
        o = int(rng.integers(0, M - nfr + 1))
        m = m[o:o + nfr]
        return m / np.max(np.abs(m))
    t = np.arange(nfr) / FS
    if kind == "twotone":
        f1, f2 = rng.uniform(500, 1500), rng.uniform(1600, 2800)
        m = np.cos(2 * np.pi * f1 * t + rng.uniform(0, 6.28)) + rng.uniform(0.5, 1) * np.cos(2 * np.pi * f2 * t + rng.uniform(0, 6.28))
        return m / np.max(np.abs(m))
    if kind == "tone":
        f1 = rng.choice([1200.0, 2200.0])
        return np.cos(2 * np.pi * f1 * t + rng.uniform(0, 6.28))
    raise ValueError(kind)


def make_frames(kind, n, snr_db, rng, nfr=NFR):
    """kind in {'afsk','voice','twotone','tone'} -> complex frames (n, nfr) at fs, SNR per complex sample."""
    X = np.empty((n, nfr), complex)
    snr = np.broadcast_to(np.asarray(snr_db, float), (n,))
    for i in range(n):
        a = afsk_audio(rng, nfr) if kind == "afsk" else voice_audio(rng, nfr, kind)
        dev = DEV * rng.uniform(0.8, 1.2)
        s = np.exp(1j * 2 * np.pi * dev * np.cumsum(a) / FS)
        f0 = rng.uniform(-OFF_MAX, OFF_MAX)
        s = s * np.exp(1j * (rng.uniform(0, 2 * np.pi) + 2 * np.pi * f0 * np.arange(nfr) / FS))
        sig = 10 ** (-snr[i] / 20) / np.sqrt(2)
        X[i] = s + sig * (rng.standard_normal(nfr) + 1j * rng.standard_normal(nfr))
    return X
