import time, numpy as np
from partA_analog import *
if __name__ == "__main__":
    t=time.time(); Ftr, ytr, s, _ = build(3000, 2048, (-5.0, 25.0), 1_000); print("build", time.time()-t, flush=True)
    t=time.time(); h=HandTree(False).fit(Ftr,ytr); print("tree global", time.time()-t, flush=True)
    t=time.time(); h=HandTree(True).fit(Ftr,ytr); print("tree bin", time.time()-t, flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    own=[FI[n] for n in OWN]
    t=time.time(); HistGradientBoostingClassifier(max_iter=300,random_state=0).fit(Ftr[:,own],ytr); print("gbt", time.time()-t, flush=True)
