"""Makes practice_qpsk.cu8: a fake RTL-SDR recording (same bytes an RTL-SDR would write).
QPSK, 100 kBd, 300 kHz above a 150 MHz centre, 2.4 MS/s, 1 second. Only for practice."""
import numpy as np
rng = np.random.default_rng(1)
fs, rs, off, secs = 2.4e6, 100e3, 300e3, 1.0
sps = int(fs // rs); n = int(fs * secs)
beta = 0.35; t = np.arange(-4 * sps, 4 * sps + 1) / sps
h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2 + 1e-9)   # raised-cosine pulse
sym = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n // sps + 10)))
up = np.zeros(len(sym) * sps, complex); up[::sps] = sym
x = np.convolve(up, h, "same")[:n] * np.exp(2j * np.pi * off / fs * np.arange(n))
x = 0.25 * x / np.sqrt(np.mean(np.abs(x) ** 2)) + 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
raw = np.empty(2 * n, np.uint8)                                   # RTL-SDR style: I,Q,I,Q as 0..255
raw[0::2] = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255)
raw[1::2] = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
raw.tofile("practice_qpsk.cu8"); print("wrote practice_qpsk.cu8,", round(raw.nbytes / 1e6, 1), "MB")
