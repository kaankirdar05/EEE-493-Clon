import numpy as np
from modes import *
rng=np.random.default_rng(0)
fs=2
for i in range(300):
    b=make_df17(rng); iq,s=synth_frame(fs,80,b,rng); sc=rng.uniform(.1,5); iq=iq*sc
    r=decode_modes_ex(iq,fs*1e6,decoder="auto")
    if not any(x['crc_ok'] and x['hex']==bits2hex(b) for x in r):
        print(i,s%10,sc,[(x['index'],x['phase_us'],x['crc_ok']) for x in r], s//10)
