"""Quick structural checks of ofdmlib (receiver correctness), run on the real code path."""
import sys; sys.dont_write_bytecode = True
import numpy as np
from ofdmlib import *

rng = np.random.default_rng(0)
x, tr = make_ofdm(256, 32, "QPSK", 10, rng)
print("z (mine) vs ofdm_params.z_stat:", cp_fold(x, 256, 32)[0], op.z_stat(x, 256, 32))
print("blind estimate:", op.estimate(x)[0], " true (256, 32)")
print("theta est", cp_fold(x, 256, 32)[1], "truth", tr["theta"], "period", tr["period"])
for (n, cp) in [(64, 16), (256, 32), (1024, 128)]:
    x, tr = make_ofdm(n, cp, "16QAM", 100, rng, cfo=False, noise=False)
    j = tr["first_sym"]; s0 = j * tr["period"] - tr["off"] + cp
    Y = np.fft.fft(x[s0:s0 + n])
    used = tr["used"]
    ratio = Y[used] / tr["sym"][j][used]
    print(f"N={n} cp={cp}: FFT/tx ratio over used bins: std/|mean| = {np.std(ratio)/abs(np.mean(ratio)):.2e}; null/used power = {np.mean(np.abs(Y[~used])**2)/np.mean(np.abs(Y[used])**2):.1e}")
