"""Overview figure for part A (reads results/*.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "results")
L = lambda n: json.load(open(os.path.join(R, n)))
det, dec, sanity = L("adsb_detect_test.json"), L("adsb_decode_test.json"), L("adsb_sanity.json")
interf = L("adsb_interf.json") if os.path.exists(os.path.join(R, "adsb_interf.json")) else None

col = {"energy": "tab:gray", "peak": "tab:brown", "mf": "tab:blue", "parhalf": "tab:orange"}
fig, ax = plt.subplots(3, 3, figsize=(15, 11))
for j, fs in enumerate(("10", "4", "2")):
    d = det[fs]
    a = ax[0, j]
    for k in ("energy", "peak", "mf", "parhalf"):
        a.plot(d["snrs"], np.array(d["pd"][k]) * 100, "o-", color=col[k], ms=3, label=k)
        a.plot(d["snrs"], np.array(d["pd_theory"][k]) * 100, "--", color=col[k], lw=0.8)
    a.set_title(f"Preamble detection, fs={fs} MS/s (dashed: theory)\n10 false alarms/s, n={d['trials']}/pt")
    a.set_xlabel("per-sample SNR (dB)"); a.set_ylabel("Pd (%)"); a.grid(alpha=.3)
    if j == 0:
        a.legend(fontsize=8)
    a = ax[1, j]
    p = dec[fs]
    for k, st in (("mf_energy_r10", ":"), ("mf_energy", "-"), ("mf_energy_fix1", "-"), ("mf_mlse", "-"),
                  ("mf_auto_fix1", "-"), ("parhalf_energy", "--")):
        a.plot(p["snrs"], np.array(p["pdec"][k]) * 100, st, marker="o", ms=3, label=k)
    a.set_title(f"Detect + decode (CRC ok, message equal), fs={fs} MS/s, n={p['trials']}/pt")
    a.set_xlabel("per-sample SNR (dB)"); a.set_ylabel("P(decoded) (%)"); a.grid(alpha=.3)
    if j == 0:
        a.legend(fontsize=7)
for j, fs in enumerate(("10", "2")):
    a = ax[2, j]
    rows = sanity["ber"][fs]
    snr = [r["snr"] for r in rows]
    a.semilogy(snr, [max(r["ber_meas"], 1e-6) for r in rows], "o", label="measured (genie timing)")
    a.semilogy(snr, [r["ber_theory"] for r in rows], "-", label="theory")
    a.set_ylim(1e-6, 1); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_title(f"Half-bit energy comparison BER, fs={fs} MS/s (400 msgs/pt)")
    a.set_xlabel("per-sample SNR (dB)"); a.set_ylabel("BER")
a = ax[2, 2]
if interf:
    key = "10MS/s_snr20"
    rows = interf[key]["rows"]
    names = list(rows)
    x = np.arange(len(names))
    for i, c in enumerate(("mf_energy", "mf_auto_fix1")):
        a.bar(x + i * 0.4, [rows[n][c] * 100 for n in names], 0.4, label=c)
    a.set_xticks(x + 0.2); a.set_xticklabels(names, rotation=60, fontsize=7)
    a.set_ylabel("P(decoded) (%)"); a.legend(fontsize=7)
    a.set_title("Interference, fs=10 MS/s, SNR 20 dB\n(garble SIR / DME 1 pair per frame)")
fig.tight_layout()
fig.savefig(os.path.join(R, "adsb_overview.png"), dpi=130)
print("saved")
