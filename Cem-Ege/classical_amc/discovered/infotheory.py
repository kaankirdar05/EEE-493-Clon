"""Information-theoretic features.

Histogram entropies (fixed bins on fixed ranges so that they are comparable between frames,
normalised by log(#bins) so that a uniform histogram gives 1):
    Shannon  H1 = -sum p log p
    Renyi    H2 = -log sum p^2        (alpha = 2, collision entropy; Renyi 1961)
of the amplitude |x| (unit-power frame, range [0, 3], 32 bins), the phase arg x ([-pi, pi), 32 bins),
the wrapped instantaneous frequency ([-pi, pi), 32 bins) and the joint I/Q histogram
([-2, 2]^2, 16 x 16 bins): 8 features.

Permutation entropy (Bandt & Pompe 2002), order D = 4, delays 1 and 4, normalised by log(D!):
of the amplitude, I and instantaneous frequency: 6 features. White noise -> 1, monotone ramp -> 0.

Sample entropy SampEn(m=2, r=0.2 std) (Richman & Moorman 2000) and approximate entropy
ApEn(m=2, r=0.2 std) (Pincus 1991) of the amplitude and the instantaneous frequency: 4 features.
SampEn of iid Gaussian noise -> -ln P(|x-y| <= 0.2) = -ln erf(0.1) = 2.185; of a periodic signal -> 0.

Recurrence quantification (Webber & Zbilut 1994; Marwan et al. 2007) on the same m = 2 embedding and
radius (Chebyshev distance, main diagonal excluded): recurrence rate RR, determinism DET (share of
recurrent points on diagonal lines of length >= 2) and laminarity LAM (vertical lines >= 2), for
the amplitude and the instantaneous frequency: 6 features.
Total 24 features.
"""

from __future__ import annotations

import math

import numpy as np

import dcommon as dc

NB = 32
PE_D = 4
PE_DELAYS = (1, 4)
M_EMB = 2
R_FRAC = 0.2
HIST_NAMES = [f"ent_{s}_{h}" for s in ("amp", "phase", "freq", "iq2d") for h in ("shannon", "renyi2")]
PE_NAMES = [f"pe_{s}_d{t}" for s in ("amp", "i", "freq") for t in PE_DELAYS]
SE_NAMES = [f"{e}_{s}" for s in ("amp", "freq") for e in ("sampen", "apen")]
RQA_NAMES = [f"rqa_{s}_{q}" for s in ("amp", "freq") for q in ("rr", "det", "lam")]
INFO_NAMES = HIST_NAMES + PE_NAMES + SE_NAMES + RQA_NAMES


# ------------------------------------------------------------------ histogram entropies
def hist_entropy(v: np.ndarray, lo: float, hi: float, nb: int = NB):
    """Rows of v -> (normalised Shannon, normalised Renyi-2) of the nb-bin histogram on [lo, hi]."""
    v = np.atleast_2d(v)
    k = np.clip(((v - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    cnt = np.zeros((v.shape[0], nb))
    np.add.at(cnt, (np.repeat(np.arange(v.shape[0]), v.shape[1]), k.ravel()), 1)
    return _ent(cnt, nb)


def _ent(cnt, nb):
    p = cnt / cnt.sum(1, keepdims=True)
    h1 = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), 1) / np.log(nb)
    h2 = -np.log(np.sum(p ** 2, 1)) / np.log(nb)
    return h1, h2


def iq_entropy(x, nb=16, lim=2.0):
    i = np.clip(((x.real + lim) / (2 * lim) * nb).astype(int), 0, nb - 1)
    q = np.clip(((x.imag + lim) / (2 * lim) * nb).astype(int), 0, nb - 1)
    cnt = np.zeros((x.shape[0], nb * nb))
    np.add.at(cnt, (np.repeat(np.arange(x.shape[0]), x.shape[1]), (i * nb + q).ravel()), 1)
    return _ent(cnt, nb * nb)


# ------------------------------------------------------------------ permutation entropy
def permutation_entropy(v: np.ndarray, order: int = PE_D, delay: int = 1) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, float))
    n = v.shape[1] - (order - 1) * delay
    emb = np.stack([v[:, i * delay:i * delay + n] for i in range(order)], axis=2)    # (N, n, D)
    ranks = np.argsort(np.argsort(emb, axis=2, kind="stable"), axis=2, kind="stable")
    code = (ranks * (order ** np.arange(order))).sum(2)
    cnt = np.zeros((v.shape[0], order ** order))
    np.add.at(cnt, (np.repeat(np.arange(v.shape[0]), n), code.ravel()), 1)
    p = cnt / n
    h = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), 1)
    return h / math.log(math.factorial(order))


# ------------------------------------------------------------------ SampEn / ApEn / RQA (one frame)
def entropy_rqa_1d(v: np.ndarray, m: int = M_EMB, r_frac: float = R_FRAC):
    """Return (SampEn, ApEn, RR, DET, LAM) of a 1-D series. O(N^2) memory/time (float32)."""
    v = np.asarray(v, np.float32)
    N = len(v)
    r = r_frac * float(np.std(v))
    D1 = np.abs(v[:, None] - v[None, :])                       # Chebyshev distance, dimension 1
    Dm = D1[:N - m + 1, :N - m + 1].copy()                     # dimension m (N-m+1 templates)
    for k in range(1, m):
        np.maximum(Dm, D1[k:N - m + 1 + k, k:N - m + 1 + k], out=Dm)
    Dm1 = np.maximum(Dm[:N - m, :N - m], D1[m:N, m:N])         # dimension m+1 (N-m templates)
    Rm = Dm <= r
    Rm1 = Dm1 <= r
    # SampEn: N-m templates of length m and m+1, pairs i != j
    Bc = (np.count_nonzero(Rm[:N - m, :N - m]) - (N - m)) / 2
    Ac = (np.count_nonzero(Rm1) - (N - m)) / 2
    sampen = -math.log(Ac / Bc) if Ac > 0 and Bc > 0 else -math.log(1.0 / ((N - m) * (N - m - 1) / 2))
    # ApEn: self-matches included, N-m+1 templates of length m, N-m of length m+1
    phi_m = float(np.mean(np.log(Rm.sum(1) / (N - m + 1))))
    phi_m1 = float(np.mean(np.log(Rm1.sum(1) / (N - m))))
    apen = phi_m - phi_m1
    # RQA on Rm without the main diagonal (Theiler window 1)
    R = Rm.copy()
    np.fill_diagonal(R, False)
    n = R.shape[0]
    tot = np.count_nonzero(R)
    rr = tot / (n * (n - 1))
    if tot == 0:
        return sampen, apen, rr, 0.0, 0.0
    diag = np.zeros_like(R)
    diag[1:, 1:] |= R[:-1, :-1]
    diag[:-1, :-1] |= R[1:, 1:]
    det = np.count_nonzero(R & diag) / tot
    vert = np.zeros_like(R)
    vert[1:, :] |= R[:-1, :]
    vert[:-1, :] |= R[1:, :]
    lam = np.count_nonzero(R & vert) / tot
    return sampen, apen, rr, det, lam


def info_features(x: np.ndarray) -> np.ndarray:
    xb = dc.normalize(x)
    amp = np.abs(xb)
    ph = np.angle(xb)
    fr = dc.inst_freq(xb)
    cols = []
    cols += hist_entropy(amp, 0.0, 3.0)
    cols += hist_entropy(ph, -np.pi, np.pi)
    cols += hist_entropy(fr, -np.pi, np.pi)
    cols += iq_entropy(xb)
    for s in (amp, xb.real, fr):
        for t in PE_DELAYS:
            cols.append(permutation_entropy(s, PE_D, t))
    se = np.array([[*entropy_rqa_1d(a), *entropy_rqa_1d(f)] for a, f in zip(amp, fr)])  # (N, 10)
    # order: sampen_amp, apen_amp, sampen_freq, apen_freq, rqa_amp(rr,det,lam), rqa_freq(rr,det,lam)
    cols += [se[:, 0], se[:, 1], se[:, 5], se[:, 6], se[:, 2], se[:, 3], se[:, 4], se[:, 7], se[:, 8], se[:, 9]]
    return dc.clean(np.stack(cols, axis=1))
