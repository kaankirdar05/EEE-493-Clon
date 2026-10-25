"""Generate WidebandSig53-style detection scenes with TorchSig and save them to HDF5.

TorchSig 2.x no longer ships the `WidebandSig53` class from the 0.x releases; wideband
scenes are now built from dataset metadata. The defaults below follow WidebandSig53:
262,144 complex samples per scene (512 x 512 spectrogram), several co-existing
emitters per scene, TorchSig's full signal library (69 types), realistic impairments.

Run with the TorchSig environment (needs torch):
    ..\\.venv-torchsig\\Scripts\\python generate_wideband.py --n 300 --out D:\\data\\wideband\\scenes.h5

Output HDF5:
    iq      (N, 262144) complex64     one scene per row
    boxes   (M,) structured           scene, class_name, snr_db, t_start, t_stop (fraction of
                                      scene, 0..1), f_low, f_high (Hz, baseband, -fs/2..fs/2)
    attrs   sample_rate, num_iq_samples, fft_size, seed, impairment_level, snr range, ...
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np


def build_dataset(args):
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.transforms.impairments import Impairments
    from torchsig.utils.defaults import TorchSigDefaults

    n = args.num_iq_samples
    md = TorchSigDefaults().default_dataset_metadata
    md.update(
        num_iq_samples_dataset=n,
        fft_size=args.fft_size,
        fft_stride=args.fft_size,
        sample_rate=args.sample_rate,
        num_signals_min=args.signals_min,
        num_signals_max=args.signals_max,
        snr_db_min=args.snr_min,
        snr_db_max=args.snr_max,
        # bursts from 5 % to 100 % of the capture, bandwidths from fs/100 to fs/8
        signal_duration_in_samples_min=int(n * 0.05),
        signal_duration_in_samples_max=n,
        bandwidth_min=args.sample_rate / 100,
        bandwidth_max=args.sample_rate / 8,
        cochannel_overlap_probability=args.overlap_prob,
        # TorchSig defaults keep emitters inside +-2.5 MHz; use the whole band like WidebandSig53
        signal_center_freq_min=-args.sample_rate / 2,
        signal_center_freq_max=args.sample_rate / 2 - 1,
        frequency_min=-args.sample_rate / 2,
        frequency_max=args.sample_rate / 2 - 1,
    )
    transforms, component_transforms = [], []
    if args.impairments > 0:
        imp = Impairments(args.impairments)
        transforms, component_transforms = [imp.dataset_transforms], [imp.signal_transforms]
    ds = TorchSigIterableDataset(metadata=md, transforms=transforms, target_labels=None)
    for g in ds.signal_generators:
        try:
            g["transforms"] = component_transforms
        except Exception:
            pass  # generator without a transform slot
    ds.seed(args.seed)
    return ds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="number of scenes")
    p.add_argument("--out", default=r"D:\data\wideband\scenes.h5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-iq-samples", type=int, default=512 * 512)
    p.add_argument("--fft-size", type=int, default=512)
    p.add_argument("--sample-rate", type=float, default=10e6)
    p.add_argument("--signals-min", type=int, default=1)
    p.add_argument("--signals-max", type=int, default=6)
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=30.0)
    p.add_argument("--overlap-prob", type=float, default=0.0,
                   help="probability that emitters may overlap in time AND frequency")
    p.add_argument("--impairments", type=int, default=2, choices=[0, 1, 2],
                   help="TorchSig impairment level: 0 clean, 1 cabled, 2 over-the-air")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ds = build_dataset(args)
    box_dtype = np.dtype([("scene", "i4"), ("class_name", "S24"), ("snr_db", "f4"),
                          ("t_start", "f4"), ("t_stop", "f4"), ("f_low", "f8"), ("f_high", "f8")])
    boxes = []
    t0 = time.perf_counter()
    with h5py.File(args.out, "w") as f:
        iq = f.create_dataset("iq", (args.n, args.num_iq_samples), dtype="complex64",
                              chunks=(1, args.num_iq_samples))
        for i in range(args.n):
            s = next(ds)
            iq[i] = np.asarray(s.data, dtype=np.complex64)
            for c in s.component_signals:
                boxes.append((i, c.class_name.encode(), c.snr_db, c.start, c.stop, c.lower_freq, c.upper_freq))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.n} scenes, {len(boxes)} emitters, {time.perf_counter() - t0:.0f} s", flush=True)
        f.create_dataset("boxes", data=np.array(boxes, dtype=box_dtype))
        for k, v in vars(args).items():
            if k != "out":
                f.attrs[k] = v
        f.attrs["generator"] = "torchsig " + __import__("torchsig").__version__
    print(f"Wrote {args.n} scenes / {len(boxes)} emitters to {args.out}")


if __name__ == "__main__":
    main()
