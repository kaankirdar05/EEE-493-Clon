"""Recover the per-clip carrier frequency offset of ds1_v3.h5 (not stored in the file).

make_synth_v3.py draws one uniform offset in [-0.002, 0.002] cycles/sample per clip but does not save it.
This script re-runs the generator's random stream unchanged (same seed, same calls, same order), records the
offset, writes a scratch copy of the dataset and checks that every clip is bit-identical to ds1_v3.h5.

    python gen_offsets.py            -> results/ds1_v3_offsets.npz  (fo, phase per clip)
"""
import os, sys, numpy as np, h5py
import ds1common as dc

SRC = os.path.join(dc.ROOT, "..", "synthetic", "make_synth_v3.py")
code = open(SRC, encoding="utf-8").read()
old = "    return x * np.exp(1j * (2 * np.pi * rng.random() + 2 * np.pi * rng.uniform(-0.002, 0.002) * n))"
new = ("    _ph = rng.random(); _fo = rng.uniform(-0.002, 0.002); _FO.append(_fo); _PH.append(_ph)\n"
       "    return x * np.exp(1j * (2 * np.pi * _ph + 2 * np.pi * _fo * n))")
assert code.count(old) == 1
code = code.replace(old, new)
tmp = os.path.join(os.path.dirname(dc.DS1), "ds1_v3_regen_check.h5")
sys.argv = ["make_synth_v3.py", "--per-snr", "1000", "--out", tmp]
g = {"__name__": "__main__", "__file__": os.path.abspath(SRC), "_FO": [], "_PH": []}
exec(compile(code, SRC, "exec"), g)
fo, ph = np.array(g["_FO"]), np.array(g["_PH"])
# generation order = file order (class blocks, SNR, repeat), so index i of the lists = clip i
with h5py.File(dc.DS1, "r") as a, h5py.File(tmp, "r") as b:
    same = True
    for s in range(0, a["X"].shape[0], 16000):
        same &= np.array_equal(a["X"][s:s + 16000], b["X"][s:s + 16000])
    same &= all(np.array_equal(a[k][:], b[k][:], equal_nan=True) for k in ("Y", "Z", "sps", "bw99", "split"))
print("regenerated file identical to ds1_v3.h5:", bool(same))
assert same and len(fo) == 224000
os.remove(tmp)
np.savez(os.path.join(dc.RESULTS, "ds1_v3_offsets.npz"), fo=fo.astype(np.float32), phase=ph.astype(np.float32))
print("saved", len(fo), "offsets; range", fo.min(), fo.max())
