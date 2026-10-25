import json, numpy as np, sys
import dsss_bench as b
from common import *
if __name__=="__main__":
    cal=json.load(open(os.path.join(RES,"dsss_calibration.json"))); thr=cal["thr"]
    snrs=[-17.5,-15,-12.5,-10,-7.5,-5,10]
    n=40
    for L in (31,127,511):
        jobs=[(SEED_DEV,L,s,i,"dsss") for s in snrs for i in range(n)]
        r=pool_map(b._sig_job,jobs)
        print("L",L)
        for si,s in enumerate(snrs):
            rr=r[si*n:(si+1)*n]
            pd={k:np.mean([d[k]>thr[k] for d in rr]) for k in b.DET}
            T=rr[0]["T"]
            ex=np.mean([d["T_acf"]==T for d in rr]); exB=np.mean([d["T_acov"]==T for d in rr])
            chip=np.mean([abs(d["alpha_abs2"]-0.25)<0.005 for d in rr]); chipx=np.mean([abs(d["rc_x2"]-0.25)<0.005 for d in rr])
            print(f"{s:6.1f}", " ".join(f"{k[:8]}={v:.2f}" for k,v in pd.items()), f"| T_acf ok {ex:.2f} T_acov ok {exB:.2f} chip {chip:.2f} chipx2 {chipx:.2f}")
