import numpy as np
from modes import *
rng=np.random.default_rng(1)
for fs in (2,4,10):
    rx,dm=get_rx(fs,'mf',2000.0)
    for k in range(0,10 if fs==2 else (5 if fs==4 else 2)):
        errs=[]
        for t in range(5):
            b=make_df17(rng); iq,s=synth_frame(fs,30,b,rng,start_fine=2000+k*(1 if fs==2 else 1))
            p=np.abs(iq)**2; n0=s//(20//fs); kk=s%(20//fs)
            d=dm[kk]; sig2=1.0
            bits=d.mlse(p[n0:n0+d.L],d.amp2(p[n0:n0+d.Lp],sig2),sig2)
            be,_=d.bits(p[n0:n0+d.L])
            errs.append((int((bits!=b).sum()),int((be!=b).sum())))
        print(fs,k,errs)
