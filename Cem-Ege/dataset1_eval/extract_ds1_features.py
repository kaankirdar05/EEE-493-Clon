"""Features for every clip of ds1_v3.h5 with the Cem-Ege extractors (cached in results/features_<set>.npz).

    python extract_ds1_features.py 19      # amc.features.extract_features (19 features, ~0.5 ms/clip)
    python extract_ds1_features.py wav     # classical_amc/extended/wavelet.py (24 Haar features, ~4.6 ms/clip)
    python extract_ds1_features.py hist    # classical_amc/discovered/infotheory.py histogram entropies (8 features)
"""
import os, sys, time, numpy as np, h5py
import ds1common as dc

sys.path.insert(0, os.path.join(dc.ROOT, "classical_amc", "discovered"))


def get_fn(name):
    if name == "19":
        from amc.features import extract_features, FEATURE_NAMES
        return extract_features, FEATURE_NAMES
    if name == "wav":
        from extended.wavelet import wavelet_features, WAV_NAMES
        return wavelet_features, WAV_NAMES
    if name == "hist":
        import infotheory as it, dcommon
        def hist(x):
            xb = dcommon.normalize(x); amp, ph, fr = np.abs(xb), np.angle(xb), dcommon.inst_freq(xb)
            cols = [*it.hist_entropy(amp, 0.0, 3.0), *it.hist_entropy(ph, -np.pi, np.pi),
                    *it.hist_entropy(fr, -np.pi, np.pi), *it.iq_entropy(xb)]
            return dcommon.clean(np.stack(cols, axis=1))
        return hist, it.HIST_NAMES
    raise ValueError(name)


def main(name):
    out = os.path.join(dc.RESULTS, f"features_{name}.npz")
    fn, names = get_fn(name)
    t0 = time.perf_counter()
    with h5py.File(dc.DS1, "r") as f:
        n = f["X"].shape[0]
        F = None
        for s in range(0, n, 8000):
            blk = f["X"][s:s + 8000]
            x = blk[..., 0] + 1j * blk[..., 1]
            Fb = fn(x)
            if F is None:
                F = np.empty((n, Fb.shape[1]), np.float32)
            F[s:s + len(x)] = Fb
            print(f"{name}: {s + len(x)}/{n}  {(time.perf_counter() - t0) / (s + len(x)) * 1e3:.2f} ms/clip", flush=True)
    np.savez(out, F=F, names=np.array(names), sec_per_frame=(time.perf_counter() - t0) / n)
    print("saved", out, F.shape)


if __name__ == "__main__":
    main(sys.argv[1])
