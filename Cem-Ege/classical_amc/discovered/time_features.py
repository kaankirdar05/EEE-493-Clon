"""Single-process extraction time (ms/frame) of every feature family on the same 1820 real RadioML
frames (7 classes x 26 SNRs x 10 frames, seed 1), measured sequentially in ONE process with one
thread. base19 is included so that the numbers can be compared with its 0.50 ms/frame in the
original cache (the machine is shared, so absolute times may still be inflated).

    python time_features.py      -> results/timing_single_process.json
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import dcommon as dc  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import bispectrum as bs  # noqa: E402
import fractal as fd  # noqa: E402
import infotheory as it  # noqa: E402
import stransform as st  # noqa: E402
from amc.features import extract_features  # noqa: E402
from amc.radioml import DEFAULT_SUBSET, load_radioml2018  # noqa: E402
from wavelet import wavelet_features  # noqa: E402


def main():
    fr = load_radioml2018(common.RADIOML, classes=DEFAULT_SUBSET, per_group=10, seed=1)
    x = fr.x
    out = {"n_frames": int(len(x))}
    for name, fn in (("base19", extract_features), ("wav24", wavelet_features), ("bis30", bs.bispectrum_features),
                     ("fd22", fd.fractal_features), ("info24", it.info_features), ("st23", st.st_features)):
        fn(x[:50])                                                     # warm-up
        best = np.inf
        for _ in range(2):
            t0 = time.perf_counter()
            fn(x)
            best = min(best, time.perf_counter() - t0)
        out[name] = best / len(x) * 1e3
        print(f"{name:7s} {out[name]:8.2f} ms/frame", flush=True)
    # cost split inside info24 (SampEn/ApEn/RQA are O(N^2))
    t0 = time.perf_counter()
    for a in np.abs(dc.normalize(x[:300])):
        it.entropy_rqa_1d(a)
    out["info24_one_sampen_apen_rqa_call"] = (time.perf_counter() - t0) / 300 * 1e3
    print(out)
    dc.save_json(out, "timing_single_process.json")


if __name__ == "__main__":
    main()
