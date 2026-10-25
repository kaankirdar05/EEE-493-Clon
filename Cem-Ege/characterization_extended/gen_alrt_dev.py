"""DEV copy (seed 1, 40 per cell, written to characterization_extended/data) of extra_methods/gen_alrt_torchsig.py. Original docstring follows.


constellation_modulator with known symbol rate (= nominal bandwidth), random SRRC
(roll-off 0.1-0.5) or rectangular pulses, 1,024 samples, 3-20 samples/symbol,
random phase, frequency offset up to +-0.002 cycles/sample, AWGN at -10..20 dB.

    ..\\.venv-torchsig\\Scripts\\python gen_alrt_torchsig.py   -> D:\\data\\alrt_torchsig.npz
"""

import os

import numpy as np
from torchsig.signals.builders.constellation import constellation_modulator

NAMES = ["bpsk", "qpsk", "8psk", "16qam", "64qam"]   # same order as practical_alrt.CLASSES
N = 1024


def main(per_cell=40, seed=1):
    rng = np.random.default_rng(seed)
    X, Y, S, SPS = [], [], [], []
    for k, name in enumerate(NAMES):
        for snr in range(-10, 21, 2):
            for _ in range(per_cell):
                sps = rng.uniform(3, 20)
                ps = "srrc" if rng.integers(0, 2) == 0 else "rectangular"
                x = np.asarray(constellation_modulator(name, ps, 1 / sps, 1.0, N, rng.uniform(0.1, 0.5) if ps == "srrc" else None, rng), complex)
                x /= np.sqrt(np.mean(np.abs(x) ** 2))
                x *= np.exp(1j * (rng.uniform(0, 2 * np.pi) + 2 * np.pi * rng.uniform(-0.002, 0.002) * np.arange(N)))
                x += np.sqrt(10 ** (-snr / 10) / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
                X.append(x.astype(np.complex64)); Y.append(k); S.append(snr); SPS.append(sps)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alrt_torchsig_dev.npz"), x=np.array(X), y=np.array(Y), snr=np.array(S), sps=np.array(SPS))
    print(len(X), "frames")


if __name__ == "__main__":
    main()
