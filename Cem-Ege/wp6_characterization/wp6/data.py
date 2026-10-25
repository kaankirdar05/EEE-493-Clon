"""Scene loading, class -> family mapping, and the WP4 noise-floor map."""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np

_WP4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wp4_detection")
if _WP4 not in sys.path:
    sys.path.insert(0, _WP4)

from wp4.energy import effective_looks, noise_floor, smooth  # noqa: E402
from wp4.spectrum import spectrogram  # noqa: E402

DATASETS = {
    "impaired": r"D:\data\wideband\scenes.h5",
    "clean": r"D:\data\wideband\scenes_clean.h5",
}

FAMILIES = ["psk", "qam", "ask/ook", "fsk/msk/gmsk", "ofdm/wifi", "am", "fm", "chirp/lfm/lora", "other"]


def family(name: str) -> str:
    """Group TorchSig class names into the WP6 reporting families.

    APSK (multi-ring) goes to qam; protocol signals (adsb, btle, dmr, p25, gsm,
    zigbee, dvbs2, tone) go to "other" even though some are FSK-like.
    """
    n = name.lower()
    if n.startswith("ofdm") or n == "80211a":
        return "ofdm/wifi"
    if "apsk" in n or "qam" in n:
        return "qam"
    if "psk" in n:
        return "psk"
    if n == "ook" or n.endswith("ask"):
        return "ask/ook"
    if any(k in n for k in ("fsk", "msk")):  # fsk, gfsk, msk, gmsk
        return "fsk/msk/gmsk"
    if n.startswith("am-"):
        return "am"
    if n == "fm":
        return "fm"
    if "lfm" in n or "chirp" in n or n == "lora":
        return "chirp/lfm/lora"
    return "other"


def envelope_group(name: str) -> str:
    """Grouping for the M2M4 estimator.

    ce_true : genuinely constant-envelope waveforms (FSK/GFSK/MSK/GMSK families, FM).
    ce_psk  : PSK. Constant modulus only at the symbol instants; TorchSig applies RRC
              pulse shaping, so the oversampled envelope is NOT constant.
    other   : everything else.
    The task's "constant-envelope" set is ce_true + ce_psk.
    """
    f = family(name)
    if f in ("fsk/msk/gmsk", "fm"):
        return "ce_true"
    if f == "psk":
        return "ce_psk"
    return "other"


def load(dataset: str):
    """Return (h5 file handle, boxes structured array, sample_rate)."""
    f = h5py.File(DATASETS[dataset], "r")
    return f, f["boxes"][:], float(f["iq"].attrs.get("sample_rate", f.attrs.get("sample_rate", 10e6)))


def noise_map(iq: np.ndarray, fft_size: int = 512, kt: int = 8, kf: int = 1) -> np.ndarray:
    """WP4 excision noise-power map, shape (T, F), units of noise power per sample."""
    spec = spectrogram(iq, fft_size)
    looks = effective_looks(kt, kf, fft_size)
    return noise_floor(smooth(spec, kt, kf), looks, "excision")
