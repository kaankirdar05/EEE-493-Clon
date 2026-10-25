import numpy as np, common
from afsk_gen import *
from scipy.signal import welch

def disc(x, chan=7500.0):
    n, N = x.shape
    f = np.fft.fftfreq(N, 1 / FS)
    y = np.fft.ifft(np.fft.fft(x, axis=1) * (np.abs(f) <= chan)[None, :], axis=1)
    d = np.angle(y[:, 1:] * np.conj(y[:, :-1])) * FS / (2 * np.pi)
    return d - np.median(d, axis=1, keepdims=True)

def analytic_bp(d, lo, hi):
    N = d.shape[1]
    f = np.fft.fftfreq(N, 1 / FS)
    M = ((f >= lo) & (f <= hi)) * 2.0
    return np.fft.ifft(np.fft.fft(d, axis=1) * M[None, :], axis=1)

def line(v, f0=1200.0, half=40.0):
    v = v - v.mean(1, keepdims=True)
    N = v.shape[1]
    P = np.abs(np.fft.rfft(v * np.hanning(N), n=8 * N, axis=1)) ** 2
    f = np.arange(P.shape[1]) * FS / (8 * N)
    near = (np.abs(f - f0) < half)
    ref = (f > 300) & (f < 3000) & ~(np.abs(f - f0) < 150)
    return np.log10(P[:, near].max(1) / np.median(P[:, ref], axis=1))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for snr in (60, 10):
        for kind in ("afsk", "voice"):
            x = make_frames(kind, 40, snr, rng)
            d = disc(x)
            res = {}
            for bw in (300, 500, 700):
                e1 = np.abs(analytic_bp(d, 1200 - bw, 1200 + bw)) ** 2
                e2 = np.abs(analytic_bp(d, 2200 - bw, 2200 + bw)) ** 2
                res[f"env1_bw{bw}"] = line(e1)
                res[f"env1+2_bw{bw}"] = line(e1 + e2)
                res[f"env1-2_bw{bw}"] = line(e1 - e2)
                # edge detector on the BP envelope
                L = 6
                ed = np.abs(np.sqrt(e1[:, 2 * L:]) - np.sqrt(e1[:, :-2 * L])) + np.abs(np.sqrt(e2[:, 2 * L:]) - np.sqrt(e2[:, :-2 * L]))
                res[f"edge_bw{bw}"] = line(ed)
            res["d^2"] = line(d ** 2)
            res["|d|"] = line(np.abs(d))
            # freq-of-audio
            ag = analytic_bp(d, 600, 3000)
            g = np.angle(ag[:, 1:] * np.conj(ag[:, :-1])) * FS / (2 * np.pi)
            res["audio_instfreq_|g-1700|"] = line(np.abs(g - 1700))
            res["audio_env"] = line(np.abs(ag))
            print(snr, kind, {k: round(float(np.median(v)), 2) for k, v in res.items()})
