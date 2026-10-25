import numpy as np
from analog_gen import *
from analog_feats import *
rng=np.random.default_rng(0)
sel=['snr_est','sig_aa_c','carrier_frac','line_x1','line_x2','asym_pk_abs','asym_nom_abs','bw90','occ','fm_slope','sig_f','kurt_f']
for snr in (100,10):
  print("SNR",snr)
  for c in CLASSES:
    x,meta=make_frames(c,300,2048,snr,rng)
    F=analog_features(x)
    print(f"{c:10s}", " ".join(f"{s}={np.percentile(F[:,FI[s]],10):.3g}/{np.median(F[:,FI[s]]):.3g}/{np.percentile(F[:,FI[s]],90):.3g}" for s in sel))
