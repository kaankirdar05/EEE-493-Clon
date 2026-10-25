import numpy as np
from modes import *
rng=np.random.default_rng(3)
fs=10
b=make_df17(rng); iq,s=synth_frame(fs,80,b,rng); iq=iq*3
p=np.abs(iq)**2
print("floor",noise_floor(p), "true", 9)
rx,dm=get_rx(fs,'mf',2000.0)
sc=rx.scores(p,noise_floor(p))
print(sc.shape, s//2, sc.max(), np.unravel_index(sc.argmax(),sc.shape))
print(candidates(rx,p,noise_floor(p))[:10], s//2)
