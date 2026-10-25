"""Pick each detector's parameter on DEV scenes 0-99: best mean box F1 over the two
datasets (impaired, clean), Pfa target 1e-3. Writes results/dev_tuning/chosen.json
and results/dev_tuning/summary.md."""

from __future__ import annotations

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "results", "dev_tuning")


def main():
    rows = {}
    for p in glob.glob(os.path.join(DEV, "*_*", "metrics.json")):
        d = os.path.basename(os.path.dirname(p))
        cfg, ds = d.rsplit("_", 1)
        s = json.load(open(p, encoding="utf-8"))["main"]
        rows.setdefault(cfg, {})[ds] = (s["box_f1"] * 100, s["cell_pfa"])
    fam = {}
    for cfg, r in rows.items():
        if len(r) < 2:
            continue
        m = re.match(r"(agm|john|cond|med|tkeo_stft|tkeo_sb)(?:_(L|kt)(\d+))?$", cfg)
        fam.setdefault(m.group(1), []).append((cfg, m.group(3), (r["impaired"][0] + r["clean"][0]) / 2, r))
    chosen, lines = {"L": {}, "kt": None, "cfg": {}}, ["# Dev tuning (scenes 0-99, Pfa target 1e-3)\n",
                                                       "| config | impaired F1 | impaired Pfa | clean F1 | clean Pfa | mean F1 |",
                                                       "|---|---|---|---|---|---|"]
    for f, lst in sorted(fam.items()):
        lst.sort(key=lambda t: int(t[1] or 0))
        best = max(lst, key=lambda t: t[2])
        chosen["cfg"][f] = best[0]
        if f in ("agm", "john", "cond", "med"):
            chosen["L"][f] = int(best[1])
        elif f == "tkeo_stft":
            chosen["kt"] = int(best[1])
        for cfg, _, mf, r in lst:
            mark = " **(chosen)**" if cfg == best[0] else ""
            lines.append(f"| {cfg}{mark} | {r['impaired'][0]:.1f} | {r['impaired'][1]:.1e} | {r['clean'][0]:.1f} | "
                         f"{r['clean'][1]:.1e} | {mf:.1f} |")
    json.dump(chosen, open(os.path.join(DEV, "chosen.json"), "w"), indent=2)
    open(os.path.join(DEV, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(chosen)


if __name__ == "__main__":
    main()
