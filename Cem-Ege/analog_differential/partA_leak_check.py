"""Why is the noiseless SSB sideband ratio 'only' ~32 dB? Truncation leakage of the cropped frame, not a bug."""
import numpy as np, common
from analog_gen import _message
from common import save_json
rng = np.random.default_rng(1)
M = 8192
m, _ = _message(100, M, 0.03, rng, force_tone=False)
f = np.fft.fftfreq(M)
s = np.fft.ifft(2 * np.fft.fft(m, axis=1) * (f > 0), axis=1)
r = lambda P, f: float(np.median(10 * np.log10(P[:, f > 0].sum(1) / np.maximum(P[:, f < 0].sum(1), 1e-300))))
out = {"uncropped_circular_frame_dB": r(np.abs(np.fft.fft(s, axis=1)) ** 2, f)}
sc = s[:, :4096]; f2 = np.fft.fftfreq(4096)
out["cropped_4096_rectangular_dB"] = r(np.abs(np.fft.fft(sc, axis=1)) ** 2, f2)
out["cropped_4096_hann_dB"] = r(np.abs(np.fft.fft(sc * np.hanning(4096), axis=1)) ** 2, f2)
print(out)
save_json(out, "partA_ssb_leakage_check.json")
