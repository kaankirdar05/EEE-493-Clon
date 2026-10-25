import numpy as np
from partA_analog import *
if __name__ == "__main__":
    Ftr, ytr, str_, _ = build(1500, 2048, (-5.0, 25.0), 1_000)
    t = HandTree(True).fit(Ftr, ytr)
    for r in t.log: print(f"{r['node']:14s} {r['feature']:13s} bin{r['snr_bin']} thr={r['thr']:.4g} Aabove={r['A_above']} err={r['train_err']:.3f}")
    Fd, yd, sd, td = build(100, 2048, [-5.0], 2_000)
    p = t.predict(Fd)
    print(confusion(yd, p))
