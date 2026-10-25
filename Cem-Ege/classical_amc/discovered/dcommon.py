"""Paths and shared helpers for classical_amc/discovered/.

Importing this module puts classical_amc/extended/ and classical_amc/ on sys.path (through
extended/common.py), so `common`, `symbol_classifiers`, `practical_alrt`, `amc.*` and the
extended feature modules can be reused unchanged. Nothing outside discovered/ is written.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), "extended")
if EXT not in sys.path:
    sys.path.insert(0, EXT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common  # noqa: E402,F401  (extended/common.py; adds classical_amc/ to sys.path)
import numpy as np  # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def normalize(x: np.ndarray) -> np.ndarray:
    """(N, L) complex -> zero-mean, unit-power complex128 frames (same as amc.features)."""
    x = np.asarray(x).astype(np.complex128)
    x = x - x.mean(axis=1, keepdims=True)
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    return x / np.sqrt(np.maximum(p, 1e-20))


def inst_freq(x: np.ndarray) -> np.ndarray:
    """Wrapped instantaneous frequency arg(x[n+1] x*[n]) in (-pi, pi], shape (N, L-1)."""
    return np.angle(x[:, 1:] * np.conj(x[:, :-1]))


def clean(F: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(F, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path
