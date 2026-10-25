import numpy as np, time, radar_pri as R
rng=np.random.default_rng(5)
R._NULL={k:{"n":[5,1e5],"thr":[-1e9,-1e9]} for k in R.METHODS}
for kind in ("const",):
    t,lab,info=R.gen_train(rng,kind,n_true=100,T=523.7)
    for m in R.METHODS:
        t0=time.time(); r=R.estimate_pri(t,m)
        print(m, r['pri'], r['coarse'], r['stat'], r['ok'], f"{time.time()-t0:.3f}s")
t,lab,info=R.gen_train(rng,"jitter",n_true=300,T=800,jitter=0.05,miss=0.3,spur=0.5)
print(len(t))
for m in R.METHODS:
    t0=time.time(); r=R.estimate_pri(t,m)
    print(m, r['pri'], r['coarse'], r['stat'], r['ok'], f"{time.time()-t0:.3f}s")
