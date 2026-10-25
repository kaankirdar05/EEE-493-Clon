"""Loader for DeepSig RadioML 2018.01A (GOLD_XYZ_OSC.0001_1024.hdf5).

File layout (as distributed by DeepSig):
    X : (2555904, 1024, 2) float32   I/Q frames
    Y : (2555904, 24)      int64     one-hot class labels
    Z : (2555904, 1)       int64     SNR in dB (-20 ... +30, step 2)

The file is ~20 GB, so we never load X fully. We read the small Y/Z arrays,
pick `per_group` frames for every (class, SNR) pair we need, and read only
those rows from X (as contiguous slices where possible).

Class order warning: the `classes.txt` shipped next to the HDF5 file (and
widely copied as the "fixed" order) does NOT match the one-hot columns of
GOLD_XYZ_OSC.0001_1024.hdf5. The order listed in the RadioML 2018 paper does.
Verified on 2026-10-15 against the file itself: every column's physical
signature (x^2 / x^4 spectral lines, real-valued vs complex, constant envelope,
SSB spectral asymmetry) fits CLASSES_HDF5. Run
`python train_feature_classifier.py check-labels` on any new copy of the file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order in the shipped classes.txt: WRONG for the HDF5 columns, kept for comparison.
CLASSES_TXT = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM", "128APSK",
    "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM", "AM-DSB-WC", "OOK", "16QAM",
]

# Order of the one-hot columns in the HDF5 file (= the paper's list). Default.
CLASSES_HDF5 = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM", "128QAM",
    "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
]

# Classes relevant to V/UHF that match the candidate set on slide 17 of P1.
# GMSK stands in for GFSK/CPFSK, which 2018.01A does not contain.
DEFAULT_SUBSET = ["AM-DSB-WC", "FM", "BPSK", "QPSK", "8PSK", "16QAM", "GMSK"]


@dataclass
class Frames:
    """A labelled set of complex baseband frames."""

    x: np.ndarray        # (N, L) complex64
    y: np.ndarray        # (N,) int, index into `classes`
    snr: np.ndarray      # (N,) int, dB
    classes: list[str]


def class_order(name: str) -> list[str]:
    if name == "classes-txt":
        return list(CLASSES_TXT)
    if name == "hdf5":
        return list(CLASSES_HDF5)
    with open(name, encoding="utf-8") as f:  # a text file, one class per line
        return [line.strip() for line in f if line.strip()]


def _read_labels(f, chunk: int = 1 << 18) -> tuple[np.ndarray, np.ndarray]:
    """Return (label index, snr) for every row without loading Y in one go."""
    n = f["Y"].shape[0]
    labels = np.empty(n, dtype=np.int16)
    for start in range(0, n, chunk):
        labels[start:start + chunk] = np.argmax(f["Y"][start:start + chunk], axis=1)
    snr = np.asarray(f["Z"][:]).reshape(-1).astype(np.int16)
    return labels, snr


def _read_rows(xds, idx: np.ndarray) -> np.ndarray:
    """Read rows `idx` (sorted) of X.

    In the real file each (class, SNR) group is one contiguous block of 4096
    rows, so reading the whole span once and sub-selecting is far faster than
    many small reads. Falls back to per-run slices if the span is fragmented.
    """
    span = idx[-1] - idx[0] + 1
    if span <= 4 * idx.size:
        return xds[idx[0]:idx[-1] + 1][idx - idx[0]]
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.concatenate([xds[run[0]:run[-1] + 1] for run in np.split(idx, breaks)], axis=0)


def iter_radioml2018(
    path: str,
    classes: list[str] | None = None,
    snr_min: int = -20,
    snr_max: int = 30,
    per_group: int = 1000,
    order: str = "hdf5",
    seed: int = 0,
):
    """Yield (x, class_index, snr_db) for each (class, SNR) group.

    x is (n, 1024) complex64 with n = min(per_group, frames in group). Yielding
    group by group keeps memory flat: callers extract features and drop x.
    """
    import h5py

    all_classes = class_order(order)
    classes = list(classes or DEFAULT_SUBSET)
    unknown = [c for c in classes if c not in all_classes]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; valid: {all_classes}")

    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        labels, snr = _read_labels(f)
        found = False
        for k, c in enumerate(classes):
            ci = all_classes.index(c)
            for s in range(snr_min, snr_max + 1, 2):
                group = np.flatnonzero((labels == ci) & (snr == s))
                if group.size == 0:
                    continue
                found = True
                take = min(per_group, group.size)
                idx = np.sort(rng.choice(group, size=take, replace=False))
                raw = _read_rows(f["X"], idx)
                yield (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64), k, s
        if not found:
            raise ValueError("no frames matched the requested classes / SNR range")


def load_radioml2018(path: str, **kwargs) -> Frames:
    """Load everything `iter_radioml2018` yields into memory (small subsets only)."""
    xs, ys, ss = [], [], []
    for x, k, s in iter_radioml2018(path, **kwargs):
        xs.append(x)
        ys.append(np.full(len(x), k, dtype=np.int16))
        ss.append(np.full(len(x), s, dtype=np.int16))
    classes = list(kwargs.get("classes") or DEFAULT_SUBSET)
    return Frames(np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), classes)


def write_fake_radioml(path: str, frames_per_group: int, snrs, generator, seed: int = 0) -> None:
    """Write a small HDF5 file with the exact RadioML 2018.01A layout.

    `generator(name, n, snr_db, length, rng)` must return (n, length) complex
    frames for the classes it supports; other classes are filled with noise.
    Used only by the smoke test so the real loader code path gets exercised.
    """
    import h5py

    rng = np.random.default_rng(seed)
    n_cls, length = len(CLASSES_HDF5), 1024
    rows = n_cls * len(snrs) * frames_per_group
    with h5py.File(path, "w") as f:
        X = f.create_dataset("X", (rows, length, 2), dtype="float32")
        Y = f.create_dataset("Y", (rows, n_cls), dtype="int64")
        Z = f.create_dataset("Z", (rows, 1), dtype="int64")
        r = 0
        for ci, name in enumerate(CLASSES_HDF5):  # class-major, SNR-minor, like the real file
            for s in snrs:
                x = generator(name, frames_per_group, s, length, rng)
                X[r:r + frames_per_group] = np.stack([x.real, x.imag], axis=-1)
                onehot = np.zeros((frames_per_group, n_cls), dtype=np.int64)
                onehot[:, ci] = 1
                Y[r:r + frames_per_group] = onehot
                Z[r:r + frames_per_group] = s
                r += frames_per_group
