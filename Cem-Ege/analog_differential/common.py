"""Shared helpers for the analog / differential-PSK add-on (nothing in classical_amc is modified)."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, r"D:\Bitirme - 493\classical_amc")      # read-only reuse of the baseline code

NCPU = 10            # machine is shared with other jobs; 24 cores total


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)


def wilson(k, n, z=1.96):
    """95 % Wilson interval half-width (for reporting number of trials)."""
    if n == 0: return 0.0
    p = k / n
    d = 1 + z * z / n
    return float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)
