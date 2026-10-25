"""Print Markdown result tables from results/*.json (used to fill README.md)."""

from __future__ import annotations

import json
import os

import common


def table(path, title):
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    print(f"\n### {title}\n")
    print("| method | acc. all SNR | acc. SNR>=10 | first SNR >= 85 % | ms / frame |")
    print("|---|---|---|---|---|")
    for k, r in d["results"].items():
        ms = f"{r['ms_per_frame']:.1f}" if r.get("ms_per_frame") is not None else "-"
        s85 = "never" if r["snr_first_85"] is None else f"{r['snr_first_85']} dB"
        k = k.replace(" | ", " / ")
        print(f"| {k} | {r['accuracy_all'] * 100:.1f} % | {r['accuracy_snr_ge_10'] * 100:.1f} % | {s85} | {ms} |")


def replot(name, title, highlight):
    path = os.path.join(common.RESULTS, f"{name}.json")
    if os.path.exists(path):
        res = json.load(open(path))["results"]
        common.plot_overview(res, title, os.path.join(common.RESULTS, f"{name}.png"), highlight)


if __name__ == "__main__":
    R = common.RESULTS
    hl = {"hgb | base19", "hgb | base19+ext17", "hgb | base19+ext17+wav24"}
    replot("bench_features_7", "RadioML 2018, 7 classes (test split, 45,500 frames)", hl)
    replot("bench_features_24", "RadioML 2018, 24 classes (test split, 31,200 frames)", hl)
    table(os.path.join(R, "bench_features_7.json"), "RadioML 7 classes")
    table(os.path.join(R, "bench_features_24.json"), "RadioML 24 classes")
    table(os.path.join(R, "bench_symbol_radioml.json"), "RadioML 5 digital classes, sps known")
    table(os.path.join(R, "bench_symbol_torchsig.json"), "TorchSig 5 digital classes, sps known")
