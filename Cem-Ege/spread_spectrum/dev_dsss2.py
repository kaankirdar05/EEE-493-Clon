import numpy as np, sys
from dsss_gen import *
import dsss_methods as m
from common import *
def run(args):
    L,snr,i,k,div=args
    rng=np.random.default_rng([SEED_DEV,56,L,i,int(snr*10+1000)])
    x,info=dsss_frame(L,snr,rng)
    z,a=m.acf_fluct(x,True); T=m.pn_period(z,a)
    B=m.occupied_bw(x,k=k)
    target=T*B/div
    c=2**np.arange(2,13)-1
    Lh=int(c[np.argmin(np.abs(np.log(c/max(target,1e-9))))])
    return (L,snr,k,div,Lh==L)
if __name__=="__main__":
    jobs=[(L,snr,i,k,div) for L in (31,127,511) for snr in (-12.5,-11,-10,-9,-7.5) for i in range(40) for k,div in ((4,1.35),(4,1.1),(3,1.1),(2.5,1.1),(3,1.0))]
    r=pool_map(run,jobs)
    import collections
    d=collections.defaultdict(list)
    for L,snr,k,div,ok in r: d[(k,div,snr)].append(ok)
    for key in sorted(d): print(key,np.mean(d[key]))
