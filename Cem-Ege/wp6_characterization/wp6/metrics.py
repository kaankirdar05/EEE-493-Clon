"""Error metrics and aggregation over the per-emitter table."""

from __future__ import annotations

import numpy as np

SNR_BINS = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30.01)]

CARRIER = ["fc_peak", "fc_centroid", "fc_mid10", "fc_obw99", "fc_noisecross"]
BANDWIDTH = ["bw_3db", "bw_10db", "bw_26db", "bw_obw99", "bw_noisecross"]
SNR = ["snr_psd_obw", "snr_psd_10db", "snr_psd_trueband", "snr_peak512", "snr_m2m4"]


def errors(rows: dict) -> dict:
    """Per-emitter error arrays. rows: dict of column -> np.array."""
    e = {}
    for k in CARRIER:
        e[k] = np.abs(rows[k] - rows["fc_true"]) / 1e3                      # kHz
        e[k + "/bw"] = np.abs(rows[k] - rows["fc_true"]) / rows["bw_true"]
    for k in BANDWIDTH:
        e[k] = (rows[k] - rows["bw_true"]) / rows["bw_true"]                # signed relative
    for k in SNR:
        e[k] = rows[k] - rows["snr_true"]                                    # signed dB
    return e


def _stats(x, signed: bool) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    a = np.abs(x)
    d = {"n": int(len(x)), "median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90))}
    if signed:
        d["median_signed"] = float(np.median(x))
    return d


def summarize(rows: dict, sel: np.ndarray | None = None) -> dict:
    """Overall + per-SNR-bin stats for every estimator. sel: boolean subset."""
    n = len(rows["snr_true"])
    sel = np.ones(n, bool) if sel is None else sel
    e = errors(rows)
    tone = rows["class_name"] == "tone"
    out = {}
    for k, v in e.items():
        is_bw = k.startswith("bw_") or k.endswith("/bw")
        s = sel & (~tone if is_bw else True)  # tone truth bandwidth is 1 Hz: no relative error
        signed = k.startswith("bw_") or k.startswith("snr_")
        d = {"overall": _stats(v[s], signed), "by_snr": {}}
        for lo, hi in SNR_BINS:
            b = s & (rows["snr_true"] >= lo) & (rows["snr_true"] < hi)
            d["by_snr"][f"[{lo},{int(round(hi))})"] = _stats(v[b], signed)
        out[k] = d
    return out


def by_group(rows: dict, key: str, groups, sel: np.ndarray) -> dict:
    """{estimator: {group: stats}} for the given column of group labels."""
    e = errors(rows)
    tone = rows["class_name"] == "tone"
    out = {}
    for k, v in e.items():
        is_bw = k.startswith("bw_") or k.endswith("/bw")
        signed = k.startswith("bw_") or k.startswith("snr_")
        out[k] = {}
        for g in groups:
            s = sel & (rows[key] == g) & (~tone if is_bw else True)
            out[k][g] = _stats(v[s], signed)
    return out


def best(summary: dict, names: list[str]) -> str:
    return min(names, key=lambda k: summary[k]["overall"].get("median_abs", np.inf))
