"""Turn the I/Q clips of a dataset (our HDF5 format) into spectrogram images for image-based models.

  python make_spectrograms.py ds1_v2.h5 --out ds1_v2_spec.h5
  python make_spectrograms.py ds1_v2.h5 --out ds1_v2_spec224.h5 --size 224          # for ImageNet-style CNNs
  python make_spectrograms.py ds1_v2.h5 --out ds1_v2_spec.h5 --png-dir spec_png --png-per-class 300

What it does for every clip (1,024 complex samples):
  1. cut the clip into short overlapping pieces (nfft samples long, a new piece every hop samples)
  2. take the FFT of each piece (Hann window) -> power at nfft frequencies, 0 Hz in the middle
  3. convert to dB and keep the top `range` dB below the strongest pixel of that image
  4. scale to 0..255 and store as an 8-bit grayscale image

Image layout ("waterfall", the same view as our live display):
  rows    = time      (top = start of the clip)
  columns = frequency (left = -fs/2, middle = 0 Hz, right = +fs/2)

Output HDF5: S (N, T, F) uint8 images, plus Y, Z, Z_inband, split copied unchanged from the input file,
so image i belongs to clip i. Settings are stored as attributes.
"""
import argparse, json, pathlib
import numpy as np, h5py
from numpy.lib.stride_tricks import sliding_window_view

p = argparse.ArgumentParser()
p.add_argument("infile")
p.add_argument("--out", required=True)
p.add_argument("--nfft", type=int, default=64)          # frequency bins per row
p.add_argument("--hop", type=int, default=16)           # samples between rows -> 1024 / 16 = 64 rows
p.add_argument("--range", type=float, default=40.0)     # dynamic range kept below each image's peak, dB
p.add_argument("--size", type=int, default=0)           # resize to size x size (0 = keep T x nfft)
p.add_argument("--chunk", type=int, default=4096)       # clips processed at once (memory)
p.add_argument("--png-dir", default="")                 # also export PNG files to this folder
p.add_argument("--png-per-class", type=int, default=0)  # PNGs per class and split (-1 = all)
a = p.parse_args()

src = h5py.File(a.infile, "r")
N, L = src["X"].shape[:2]
T = L // a.hop                                          # rows (time steps)
win = np.hanning(a.nfft).astype(np.float32)
H, Wd = (a.size, a.size) if a.size else (T, a.nfft)

def spectrograms(X):
    """X: (B, L, 2) float32 I/Q clips -> (B, T, nfft) uint8 images."""
    x = X[..., 0] + 1j * X[..., 1]
    x = np.pad(x, ((0, 0), (a.nfft // 2, a.nfft // 2)), mode="reflect")         # centre the first window on sample 0
    frames = sliding_window_view(x, a.nfft, axis=1)[:, ::a.hop][:, :T]          # (B, T, nfft)
    P = np.abs(np.fft.fftshift(np.fft.fft(frames * win, axis=-1), axes=-1)) ** 2
    db = 10 * np.log10(P + 1e-12)
    db -= db.max(axis=(1, 2), keepdims=True)                                     # 0 dB = brightest pixel
    img = np.clip(db / a.range + 1, 0, 1)                                        # -range..0 dB -> 0..1
    if a.size:
        from scipy.ndimage import zoom
        img = zoom(img, (1, H / T, Wd / a.nfft), order=1)
    return np.round(img * 255).astype(np.uint8)

with h5py.File(a.out, "w") as f:
    S = f.create_dataset("S", (N, H, Wd), np.uint8, chunks=(min(256, N), H, Wd))
    for i in range(0, N, a.chunk):
        S[i:i + a.chunk] = spectrograms(src["X"][i:i + a.chunk])
        print(f"\r{min(i + a.chunk, N):,} / {N:,} images", end="", flush=True)
    for key in ("Y", "Z", "Z_inband", "split"):
        if key in src:
            f[key] = src[key][:]
    f.attrs.update({k: v for k, v in src.attrs.items()})
    f.attrs.update(source_file=pathlib.Path(a.infile).name, image_layout="rows = time (top = start), columns = frequency (-fs/2 .. +fs/2)",
                   spectrogram=json.dumps(dict(nfft=a.nfft, hop=a.hop, window="hann", rows=T, dynamic_range_db=a.range,
                                               scaling="per image: 0 = peak - range dB, 255 = peak", resized_to=a.size or None)))
print(f"\nsaved {N:,} images of {H} x {Wd} to {a.out}")

# optional PNG export: <png-dir>/<train|val|test>/<class>/<index>.png
if a.png_dir and a.png_per_class:
    from PIL import Image
    classes = [c.decode() if isinstance(c, bytes) else str(c) for c in src.attrs["classes"]]
    Y = src["Y"][:]; split = src["split"][:] if "split" in src else np.zeros(N, int)
    names = {0: "train", 1: "val", 2: "test"}
    with h5py.File(a.out, "r") as f:
        for s_code, s_name in names.items():
            for c, cname in enumerate(classes):
                idx = np.where((Y == c) & (split == s_code))[0]
                if a.png_per_class > 0:
                    idx = idx[:a.png_per_class]
                d = pathlib.Path(a.png_dir) / s_name / cname
                d.mkdir(parents=True, exist_ok=True)
                for i in idx:
                    Image.fromarray(f["S"][i]).save(d / f"{i:06d}.png")
    print(f"PNG files written under {a.png_dir}/ (train, val, test / class)")
