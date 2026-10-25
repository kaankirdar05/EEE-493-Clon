import numpy as np
from modes import *
rng=np.random.default_rng(1)
fs=2
rx,dm=get_rx(fs,'mf',2000.0)
b=make_df17(rng); iq,s=synth_frame(fs,80,b,rng,start_fine=2005)
for sc in (1,2,3,4):
    p=np.abs(iq*sc)**2; n0=200; d=dm[5]
    sig2=local_sig2(d,p,n0,2.0); a2=d.amp2(p[n0:n0+d.Lp],sig2)
    bits=d.mlse(p[n0:n0+d.L],a2,sig2)
    print(sc,sig2,a2/sig2,(bits!=b).sum())
