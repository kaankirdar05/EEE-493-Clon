"""Dataset 1 (synthetic), version 3: labelled 1,024-sample I/Q clips in our agreed HDF5 format. No SDR needed.

  python make_synth_v3.py --per-snr 1000 --out ds1_v3.h5
  python make_synth_v3.py --classes NBFM 4FSK PI4DQPSK GMSK --per-snr 1000 --out ds1_v3_small.h5
  python make_synth_v3.py --per-snr 1000 --fading --out ds1_v3_fading.h5

14 classes chosen from what is actually transmitted in the V/UHF band (see README.md):
  analog:  AM (aircraft voice), NBFM (walkie-talkies, land mobile, military VHF), WBFM (broadcast, with stereo pilot)
  digital: OOK (ISM remotes and sensors), 2FSK (pagers, telemetry), 4FSK (DMR, dPMR, NXDN), GMSK (AIS, GSM),
           BPSK, QPSK, PI4DQPSK (TETRA), 8PSK (VDL Mode 2), 16QAM, OFDM (DAB, DVB-T, LTE), NOISE

Every signal fills 10-35 % of the clip's band (random), like a real signal after our preprocessing (4 x its bandwidth),
so the classifier has to learn the modulation itself and cannot rely on signal width alone.

SNR labels (Z): Es/N0 for single-carrier digital classes; in-band SNR for AM, NBFM, WBFM and OFDM; NaN for NOISE.
Z_inband: in-band SNR for every clip (signal power / noise power inside the 99 % occupied bandwidth; AM: 2 x message band).
"""
import argparse, datetime, json, pathlib
import numpy as np, h5py

ALL_CLASSES = ["AM", "NBFM", "WBFM", "OOK", "2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "PI4DQPSK", "8PSK", "16QAM", "OFDM", "NOISE"]
KNOWN = set(ALL_CLASSES)
_txt = pathlib.Path(__file__).resolve().parents[1] / "classes.txt"          # the team's shared class list
DEFAULT = _txt.read_text().split() if _txt.exists() else ALL_CLASSES
ES_N0 = {"OOK", "2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "PI4DQPSK", "8PSK", "16QAM"}   # labelled with Es/N0
N = 1024                                                                             # samples per clip

p = argparse.ArgumentParser()
p.add_argument("--classes", nargs="+", default=DEFAULT)
p.add_argument("--snr", nargs=3, type=float, default=[-10, 20, 2], metavar=("MIN", "MAX", "STEP"))
p.add_argument("--per-snr", type=int, default=1000)                 # clips per class per SNR level
p.add_argument("--bw", nargs=2, type=float, default=[0.10, 0.35])   # occupied bandwidth range, fraction of sample rate
p.add_argument("--fading", action="store_true")                     # add light two-path multipath
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--out", default="ds1_v3.h5")
a = p.parse_args()
unknown = [c for c in a.classes if c not in KNOWN]
if unknown:
    raise SystemExit(f"No generator for {unknown}. Known classes: {ALL_CLASSES}")
rng = np.random.default_rng(a.seed)
SNRS = np.arange(a.snr[0], a.snr[1] + 1e-9, a.snr[2])

# ---------- building blocks ----------
def rrc(sps, beta, span=8):
    """Root-raised-cosine pulse: how real transmitters shape digital symbols."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.empty(len(t))
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))

def gaussian(sps, bt, span=4):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    g = np.exp(-t ** 2 / (2 * (np.sqrt(np.log(2)) / (2 * np.pi * bt)) ** 2))
    return g / g.sum()

def message(n, cut):
    """A smooth random 'audio' message between -1 and +1, band-limited to `cut` (fraction of the sample rate)."""
    spec = np.fft.rfft(rng.standard_normal(n)); spec[max(2, int(cut * n)):] = 0
    m = np.fft.irfft(spec, n)
    return m / (np.max(np.abs(m)) + 1e-12)

def sps_for(k, b):
    """Samples per symbol so that a signal k symbol-rates wide fills a fraction b of the band."""
    return int(np.clip(round(k / b), 2, 64))

def shaped(symbols, sps, beta):
    up = np.zeros(len(symbols) * sps, complex); up[::sps] = symbols
    return np.convolve(up, rrc(sps, beta), "same")[20 * sps:20 * sps + N]

def fm(freq):
    """Frequency (cycles per sample) -> constant-envelope signal."""
    return np.exp(2j * np.pi * np.cumsum(freq))

QAM16 = ((np.arange(16) % 4) * 2 - 3) + 1j * ((np.arange(16) // 4) * 2 - 3)

def clean(name):
    """One clean clip of N samples, average power 1. Returns (x, samples per symbol or 0, bandwidth or None)."""
    b = rng.uniform(*a.bw)                                   # target occupied bandwidth for this clip
    sps = 0
    if name == "AM":                                         # double sideband with carrier; band = 2 x message band
        x = (1 + rng.uniform(0.3, 0.9) * message(N, b / 2)).astype(complex)
        return x / np.sqrt(np.mean(np.abs(x) ** 2)), 0, b
    elif name == "NBFM":                                     # voice FM: modulation index 0.5-2, Carson band = b
        beta = rng.uniform(0.5, 2.0); fmax = b / (2 * (beta + 1))
        x = fm(beta * fmax * message(N, fmax))
    elif name == "WBFM":                                     # broadcast FM: 15 kHz audio, 75 kHz deviation, 19 kHz pilot
        if rng.random() < 0.7:                               # stereo multiplex: L+R, pilot, L-R around 38 kHz
            u = b / 256; n = np.arange(N); L, R = message(N, 15 * u), message(N, 15 * u)
            mpx = 0.9 * (0.5 * (L + R) / 2 + 0.5 * (L - R) / 2 * np.cos(2 * np.pi * 38 * u * n)) + 0.1 * np.cos(2 * np.pi * 19 * u * n)
        else:                                                # mono
            u = b / 180; mpx = message(N, 15 * u)
        x = fm(75 * u * mpx)
    elif name == "OOK":                                      # on-off keying (ISM remotes, sensors)
        beta = rng.uniform(0.2, 0.5); sps = sps_for(1 + beta, b)
        x = shaped(rng.integers(0, 2, N // sps + 40).astype(complex), sps, beta)
    elif name == "2FSK":                                     # pagers, telemetry: index 0.5-2.5
        h = rng.uniform(0.5, 2.5); sps = sps_for(h + 2, b)
        x = fm(np.repeat(2 * rng.integers(0, 2, N // sps + 40) - 1, sps) * h / (2 * sps))[20 * sps:20 * sps + N]
    elif name == "4FSK":                                     # DMR / dPMR / NXDN style: small index, smoothed tones
        h = rng.uniform(0.25, 0.6); sps = sps_for(3 * h + 2, b)
        lv = np.repeat(2 * rng.integers(0, 4, N // sps + 40) - 3, sps).astype(float)
        x = fm(np.convolve(lv, gaussian(sps, 0.5), "same") * h / (2 * sps))[20 * sps:20 * sps + N]
    elif name == "GMSK":                                     # AIS, GSM: index 0.5, BT 0.3-0.5
        sps = sps_for(1.0, b)
        nrz = np.repeat(2 * rng.integers(0, 2, N // sps + 40) - 1, sps).astype(float)
        x = fm(np.convolve(nrz, gaussian(sps, rng.uniform(0.3, 0.5)), "same") * 0.5 / (2 * sps))[20 * sps:20 * sps + N]
    elif name in ("BPSK", "QPSK", "PI4DQPSK", "8PSK", "16QAM"):
        beta = rng.uniform(0.2, 0.5); sps = sps_for(1 + beta, b)
        k = rng.integers(0, 16, N // sps + 40)
        if name == "PI4DQPSK":                               # TETRA: each symbol turns the phase by +-45 or +-135 degrees
            s = np.exp(1j * np.cumsum(np.array([1, 3, -1, -3])[k % 4] * np.pi / 4))
        else:
            s = {"BPSK": 2.0 * (k % 2) - 1, "QPSK": np.exp(1j * (np.pi / 4 + np.pi / 2 * (k % 4))),
                 "8PSK": np.exp(2j * np.pi * (k % 8) / 8), "16QAM": QAM16[k] / np.sqrt(10)}[name]
        x = shaped(s, sps, beta)
    elif name == "OFDM":                                     # DAB, DVB-T, LTE: many subcarriers, band = b
        nfft = int(rng.choice([64, 128, 256])); used = max(2, int(nfft * b / 2))
        bins = np.r_[1:used + 1, nfft - used:nfft]           # subcarriers around 0 Hz, DC empty
        cp = nfft // int(rng.choice([4, 8])); qam = rng.random() < 0.5
        out = []
        while sum(len(o) for o in out) < 2 * N:
            X = np.zeros(nfft, complex); k = rng.integers(0, 16, len(bins))
            X[bins] = QAM16[k] / np.sqrt(10) if qam else np.exp(1j * (np.pi / 4 + np.pi / 2 * (k % 4)))
            sym = np.fft.ifft(X); out.append(np.r_[sym[-cp:], sym])
        x = np.concatenate(out); st = int(rng.integers(0, len(x) - N)); x = x[st:st + N]
    else:                                                    # NOISE: no signal at all
        return np.zeros(N, complex), 0, 1.0
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), sps, None

def occupied_bw(x):
    """99 % occupied bandwidth as a fraction of the sample rate (ITU style: 0.5 % of power cut each side)."""
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
    c = np.cumsum(P) / P.sum()
    return (np.searchsorted(c, 0.995) - np.searchsorted(c, 0.005) + 1) / len(x)

def channel(x):
    """Light real-world damage: two-path echo (optional), random phase, small frequency offset."""
    if a.fading:
        d = int(rng.integers(1, 5)); h = np.zeros(d + 1, complex); h[0] = 1
        h[d] = rng.uniform(0, 0.5) * np.exp(2j * np.pi * rng.random())
        x = np.convolve(x, h)[:N]; x /= np.sqrt(np.mean(np.abs(x) ** 2))
    n = np.arange(N)
    return x * np.exp(1j * (2 * np.pi * rng.random() + 2 * np.pi * rng.uniform(-0.002, 0.002) * n))

# ---------- generate ----------
C, S, R = len(a.classes), len(SNRS), a.per_snr
total = C * S * R
with h5py.File(a.out, "w") as f:
    X = f.create_dataset("X", (total, N, 2), np.float32)
    Y = np.empty(total, np.int64); Z = np.empty(total, np.float32); Zi = np.empty(total, np.float32)
    SPS = np.empty(total, np.float32); BW = np.empty(total, np.float32)
    i = 0
    for c, name in enumerate(a.classes):
        block = np.empty((S * R, N, 2), np.float32)
        for j, snr in enumerate(SNRS):
            for r in range(R):
                x, sps, bw = clean(name)
                bw = bw if bw is not None else occupied_bw(x)
                x = channel(x)
                if name == "NOISE":
                    var, z, zi = 1.0, np.nan, np.nan
                elif name in ES_N0:                          # Es/N0 = sps / noise variance (signal power 1)
                    var = sps / 10 ** (snr / 10); z = snr; zi = 10 * np.log10(1 / (var * bw))
                else:                                        # in-band: noise counted inside the occupied band
                    var = 1 / (10 ** (snr / 10) * bw); z = zi = snr
                x = x + np.sqrt(var / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
                x /= np.sqrt(np.mean(np.abs(x) ** 2))        # agreement: every clip has average power 1
                k = j * R + r
                block[k, :, 0], block[k, :, 1] = x.real, x.imag
                Y[i + k], Z[i + k], Zi[i + k], SPS[i + k], BW[i + k] = c, z, zi, sps, bw
        X[i:i + S * R] = block; i += S * R
        print(f"{name:8s} done ({i:,} / {total:,} clips)")
    split = np.zeros(total, np.int8); perm = rng.permutation(total)
    split[perm[int(0.7 * total):int(0.85 * total)]] = 1; split[perm[int(0.85 * total):]] = 2
    for key, val in dict(Y=Y, Z=Z, Z_inband=Zi, sps=SPS, bw99=BW, split=split).items():
        f[key] = val
    f.attrs.update(classes=a.classes, sample_rate=1.0, source="synthetic", version="v3",
                   snr_definition="Z: Es/N0 (dB) for " + ", ".join(sorted(ES_N0)) + "; in-band SNR (dB) for AM, NBFM, WBFM, OFDM; "
                                  "NaN for NOISE. Z_inband: signal power / noise power inside the 99% occupied bandwidth (AM: 2 x message bandwidth).",
                   split_codes="0 train (70%), 1 validation (15%), 2 test (15%)",
                   settings=json.dumps(vars(a)), created=datetime.date.today().isoformat())
print(f"saved {total:,} clips to {a.out}")
