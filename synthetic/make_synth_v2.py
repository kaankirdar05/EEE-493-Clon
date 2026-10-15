"""Dataset 1 (synthetic), version 2: labelled 1,024-sample I/Q clips in our agreed HDF5 format. No SDR needed.

  python make_synth_v2.py --per-snr 1000 --out ds1_v2.h5
  python make_synth_v2.py --classes BPSK QPSK 8PSK 16QAM FM --per-snr 1000 --out ds1_v2_5cls.h5
  python make_synth_v2.py --per-snr 1000 --fading --out ds1_v2_fading.h5

SNR labels (Z): Es/N0 for single-carrier digital classes; in-band SNR for AM, FM and OFDM.
Z_inband: in-band SNR for every clip (signal power / noise power inside the 99 % occupied bandwidth;
for AM inside 2 x the message bandwidth, because the carrier alone holds 99 % of the power).
"""
import argparse, datetime, json, pathlib
import numpy as np, h5py

ALL_CLASSES = ["AM", "FM", "2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "8PSK", "16QAM", "OFDM", "NOISE"]
KNOWN = set(ALL_CLASSES)
_txt = pathlib.Path(__file__).resolve().parents[1] / "classes.txt"          # the team's shared class list
DEFAULT = _txt.read_text().split() if _txt.exists() else ALL_CLASSES
ES_N0 = {"2FSK", "4FSK", "GMSK", "BPSK", "QPSK", "8PSK", "16QAM"}   # classes labelled with Es/N0
N = 1024                                                              # samples per clip

p = argparse.ArgumentParser()
p.add_argument("--classes", nargs="+", default=DEFAULT)
p.add_argument("--snr", nargs=3, type=float, default=[-10, 20, 2], metavar=("MIN", "MAX", "STEP"))
p.add_argument("--per-snr", type=int, default=1000)      # clips per class per SNR level
p.add_argument("--sps", nargs=2, type=int, default=[4, 12])  # samples-per-symbol range
p.add_argument("--fading", action="store_true")          # add light two-path multipath
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--out", default="ds1_v2.h5")
a = p.parse_args()
unknown = [c for c in a.classes if c not in KNOWN]
if unknown:
    raise SystemExit(f"No generator for {unknown}. Known classes: {ALL_CLASSES}")
rng = np.random.default_rng(a.seed)
SNRS = np.arange(a.snr[0], a.snr[1] + 1e-9, a.snr[2])

# ---------- pulse shapes and messages ----------
def rrc(sps, beta, span=8):
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

def gaussian(sps, bt=0.3, span=4):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    return g / g.sum()

def message(n):
    """A smooth random 'audio' message between -1 and +1 (bandwidth 0.5 to 3 % of the sample rate)."""
    cut = rng.uniform(0.005, 0.03)
    spec = np.fft.rfft(rng.standard_normal(n)); spec[int(cut * n):] = 0
    m = np.fft.irfft(spec, n)
    return m / (np.max(np.abs(m)) + 1e-12), cut

QAM16 = ((np.arange(16) % 4) * 2 - 3) + 1j * ((np.arange(16) // 4) * 2 - 3)

def clean(name):
    """One clean clip of N samples, average power 1. Returns (x, samples per symbol or 0, bandwidth or None)."""
    sps = int(rng.integers(a.sps[0], a.sps[1] + 1))
    nsym = N // sps + 40
    if name == "AM":
        m, cut = message(N)                           # AM band = 2 x message bandwidth (the carrier alone
        x = (1 + rng.uniform(0.3, 0.9) * m).astype(complex)    # would hold 99 % of the power)
        return x / np.sqrt(np.mean(np.abs(x) ** 2)), 0, 2 * cut
    elif name == "FM":
        x = np.exp(2j * np.pi * rng.uniform(0.01, 0.05) * np.cumsum(message(N)[0])); sps = 0
    elif name in ("2FSK", "4FSK"):
        M = 2 if name == "2FSK" else 4
        levels = 2 * rng.integers(0, M, nsym) - (M - 1)
        f = np.repeat(levels, sps) * rng.uniform(0.5, 1.0) / (2 * sps)   # modulation index 0.5 to 1
        x = np.exp(2j * np.pi * np.cumsum(f))[20 * sps:20 * sps + N]
    elif name == "GMSK":
        nrz = np.repeat(2 * rng.integers(0, 2, nsym) - 1, sps).astype(float)
        f = np.convolve(nrz, gaussian(sps), "same") * 0.5 / (2 * sps)     # BT 0.3, index 0.5
        x = np.exp(2j * np.pi * np.cumsum(f))[20 * sps:20 * sps + N]
    elif name in ("BPSK", "QPSK", "8PSK", "16QAM"):
        k = rng.integers(0, 16, nsym)
        s = {"BPSK": 2.0 * (k % 2) - 1, "QPSK": np.exp(1j * (np.pi / 4 + np.pi / 2 * (k % 4))),
             "8PSK": np.exp(2j * np.pi * (k % 8) / 8), "16QAM": QAM16[k] / np.sqrt(10)}[name]
        up = np.zeros(nsym * sps, complex); up[::sps] = s
        x = np.convolve(up, rrc(sps, rng.uniform(0.2, 0.5)), "same")[20 * sps:20 * sps + N]
    elif name == "OFDM":
        nfft = int(rng.choice([64, 128, 256])); used = int(nfft * rng.uniform(0.3, 0.7)) // 2
        bins = np.r_[1:used + 1, nfft - used:nfft]                          # subcarriers around 0 Hz, DC empty
        cp = nfft // int(rng.choice([4, 8])); qam = rng.random() < 0.5
        out = []
        while sum(len(o) for o in out) < 2 * N:
            X = np.zeros(nfft, complex)
            k = rng.integers(0, 16, len(bins))
            X[bins] = QAM16[k] / np.sqrt(10) if qam else np.exp(1j * (np.pi / 4 + np.pi / 2 * (k % 4)))
            sym = np.fft.ifft(X); out.append(np.r_[sym[-cp:], sym])
        x = np.concatenate(out); st = int(rng.integers(0, len(x) - N)); x = x[st:st + N]; sps = 0
    else:                                                                    # NOISE: no signal at all
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
        print(f"{name:6s} done ({i:,} / {total:,} clips)")
    split = np.zeros(total, np.int8); perm = rng.permutation(total)
    split[perm[int(0.7 * total):int(0.85 * total)]] = 1; split[perm[int(0.85 * total):]] = 2
    for key, val in dict(Y=Y, Z=Z, Z_inband=Zi, sps=SPS, bw99=BW, split=split).items():
        f[key] = val
    f.attrs.update(classes=a.classes, sample_rate=1.0, source="synthetic", version="v2",
                   snr_definition="Z: Es/N0 (dB) for " + ", ".join(sorted(ES_N0)) + "; in-band SNR (dB) for AM, FM, OFDM; NaN for NOISE. "
                                  "Z_inband: signal power / noise power inside the 99% occupied bandwidth (AM: 2 x message bandwidth).",
                   split_codes="0 train (70%), 1 validation (15%), 2 test (15%)",
                   settings=json.dumps(vars(a)), created=datetime.date.today().isoformat())
print(f"saved {total:,} clips to {a.out}")
