"""Plots: error vs SNR, estimated vs true SNR, PSD sanity plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import BANDWIDTH, CARRIER, SNR, SNR_BINS  # noqa: E402


def _centers():
    return [0.5 * (lo + min(hi, 30)) for lo, hi in SNR_BINS]


def error_vs_snr(summary: dict, path: str, title: str):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    x = _centers()
    panels = [
        (CARRIER, "", "carrier |error| median (kHz)", True),
        (BANDWIDTH, "", "bandwidth |rel. error| median", True),
        (SNR, "", "SNR error median, signed (dB)", False),
        (SNR, "", "SNR |error| median (dB)", True),
    ]
    for a, (names, suf, lab, absval) in zip(ax, panels):
        for k in names:
            st = summary[k + suf]["by_snr"]
            key = "median_abs" if absval else "median_signed"
            y = [v.get(key, np.nan) for v in st.values()]
            a.plot(x, y, "o-", label=k)
        a.set_xlabel("true SNR (dB, TorchSig snr_db)")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        if names is CARRIER or names is BANDWIDTH:
            a.set_yscale("log")
        if not absval:
            a.axhline(0, color="k", lw=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def snr_scatter(rows: dict, path: str, title: str):
    fig, ax = plt.subplots(1, len(SNR), figsize=(4 * len(SNR), 4))
    t = rows["snr_true"]
    for a, k in zip(ax, SNR):
        a.scatter(t, rows[k], s=5, alpha=0.5)
        a.plot([-5, 30], [-5, 30], "k--", lw=0.8)
        a.set_xlim(-7, 32)
        a.set_ylim(-20, 45)
        a.set_title(k, fontsize=9)
        a.set_xlabel("true SNR (dB)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("estimated SNR (dB)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def psd_example(res: dict, truth: dict, path: str):
    pl = res["_plot"]
    f = pl["f"] / 1e6
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, 10 * np.log10(np.maximum(pl["p"], 1e-30)), lw=0.8, label="Welch PSD")
    ax.plot(f, 10 * np.log10(pl["nz"]), "k--", lw=1, label="noise floor")
    ax.plot(f, 10 * np.log10(np.maximum(pl["ds"], 1e-2 * pl["nz"])), lw=0.8, alpha=0.7, label="PSD - noise (smoothed)")
    ax.axvspan(truth["f_low"] / 1e6, truth["f_high"] / 1e6, color="g", alpha=0.12, label="truth box")
    cols = {3: "r", 10: "m", 26: "c"}
    for x, (lo, hi) in pl["edges"].items():
        ax.axvline(lo / 1e6, color=cols[x], lw=1, ls=":")
        ax.axvline(hi / 1e6, color=cols[x], lw=1, ls=":", label=f"-{x} dB")
    ax.axvline(pl["obw"][0] / 1e6, color="orange", lw=1.2)
    ax.axvline(pl["obw"][1] / 1e6, color="orange", lw=1.2, label="99 % OBW")
    ax.axvline(pl["nc"][0] / 1e6, color="b", lw=1.2, ls="-.")
    ax.axvline(pl["nc"][1] / 1e6, color="b", lw=1.2, ls="-.", label="noise-crossing")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("power per bin (dB)")
    ax.set_title(f"{truth['class_name']}  SNR {truth['snr_true']:.1f} dB  | est: psd_obw {res['snr_psd_obw']:.1f}, "
                 f"peak512 {res['snr_peak512']:.1f}, m2m4 {res['snr_m2m4']:.1f}", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
