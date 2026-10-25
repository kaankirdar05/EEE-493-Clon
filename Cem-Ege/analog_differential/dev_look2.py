import numpy as np
from partA_analog import *
if __name__ == "__main__":
    F, y, s, t = build(300, 2048, [-5.0], 2_000)
    for fn in ["snr_est","sig_aa","sig_aa_c","kurt_amp","skew_amp","carrier_frac","line_x1","kurt_f","sig_f"]:
        print(fn)
        for i,c in enumerate(CLASSES):
            v=F[y==i, FI[fn]]; print(f"   {c:10s} {np.percentile(v,10):.3g} {np.median(v):.3g} {np.percentile(v,90):.3g}")
