import numpy as np
from modes import *
rng=np.random.default_rng(1)
fs=2
rx,dm=get_rx(fs,'mf',2000.0)
from collections import Counter
fail=Counter(); tot=Counter()
for i in range(600):
    b=make_df17(rng); iq,s=synth_frame(fs,80,b,rng)
    k=s%10; tot[k]+=1
    r=decode_modes_ex(iq,fs*1e6,decoder="auto")
    if not any(x['crc_ok'] and x['hex']==bits2hex(b) for x in r):
        fail[k]+=1
        if fail[k]<=1:
            sc=rx.scores(np.abs(iq)**2); n0=s//10
            ph,pos=np.unravel_index(sc.argmax(),sc.shape)
            print("k",k,"n0",n0,"best (phase,pos)",ph,pos,"score@true",sc[k,n0],"best",sc.max(), "cands",[(x['index'],x['phase_us']) for x in r][:4])
print({k:(fail[k],tot[k]) for k in sorted(tot)})
print([ (round(ph.phi,2),ph.pulses[0],ph.Np) for ph in rx.phases])
