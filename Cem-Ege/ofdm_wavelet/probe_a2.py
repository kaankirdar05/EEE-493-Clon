"""Dev probe (seed 100+, NOT used for reporting): receiver EVM and constellation quality for AWGN / multipath."""
import sys; sys.dont_write_bytecode = True
import numpy as np
from ofdmlib import *

rng = np.random.default_rng(100)
cfgs = [(64, 16), (64, 2), (128, 16), (256, 64), (256, 8), (512, 128), (1024, 256), (1024, 32)]


def evm(Z, tr, n):
    """EVM of the extracted data symbols vs best-fitting transmitted constellation (nearest point after global-phase fit using 16QAM truth)"""
    c = MODS["16QAM"]
    # global phase: pick the rotation among 4 with least error
    best = 1e9
    for r in range(4):
        zr = Z * np.exp(1j * np.pi / 2 * r)
        d = np.min(np.abs(zr[:, None] - c[None, :]) ** 2, axis=1)
        best = min(best, np.mean(d))
    return 10 * np.log10(best)


for chan in (False, True):
    for snr in (100, 30):
        out = []
        for (n, cp) in cfgs:
            ev = []
            for _ in range(6):
                h = make_channel(cp, rng) if chan else None
                x, tr = make_ofdm(n, cp, "16QAM", snr, rng, channel=h)
                Z, info = subcarrier_symbols(x, n, cp, tr["theta"], equalize=True, return_info=True)
                # EVM wrt nearest 16QAM point (scale-normalised): crude but monotone in quality
                ev.append(evm(Z, tr, n))
            out.append(f"({n},{cp}):{np.mean(ev):.1f}")
        print(f"channel={chan} snr={snr}  EVM dB: " + "  ".join(out))
