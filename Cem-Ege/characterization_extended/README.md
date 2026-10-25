# characterization_extended — classical characterization: CRB checks, FSK symbol rate, blind ALRT fix

Extends the classical (non-deep-learning) characterization baseline of the EEE493 V/UHF project.
Nothing in `classical_amc/`, `extra_methods/` or `wp6_characterization/` was edited; those modules are
imported (`common.py` puts them on `sys.path`). Python: `D:\Bitirme - 493\.venv`; dev data generated with
`.venv-torchsig`.

| file | task | what it does |
|---|---|---|
| `carrier_crb.py` | 1 | carrier-frequency estimators vs the CRB (tone, RRC-QPSK; N = 1024, 4096) |
| `snr_crb.py` | 2 | SNR estimators vs DA / NDA CRBs (BPSK, QPSK, 16QAM; N = 256, 1024) |
| `fsk_rate.py` | 3 | band-limiting, FM discriminator, transition features, local-background line detector, CAF |
| `sr_bank.py` | 3, 4 | evaluates the bank of ~100 symbol-rate members (rate, prominence) on a frame set, cached in `data/bank_*.npz` |
| `consensus.py` | 3, 4 | prominence-normalised multi-estimator consensus + greedy member selection (DEV only) |
| `eval_symbol_rate.py` | 3 | tunes on DEV, reports on the held-out TorchSig frames |
| `blind_alrt.py` | 4 | re-runs `practical_alrt.classify` on the same RadioML / TorchSig test frames with the new sps estimate |
| `gen_symbol_rate_dev.py`, `gen_alrt_dev.py` | 3, 4 | copies of the extra_methods generators with seed 1 (DEV sets, `data/`) |
| `explore_dev.py` | 3 | scratch exploration on DEV only |

Results: `results/` (JSON + PNG + logs).

## Task 1 — carrier frequency vs the CRB (`carrier_crb.py`)

Tone `A exp(j(2π f0 n + φ))`, f0 ~ U(−0.4, 0.4); RRC QPSK (β 0.35, 4 sps), f0 ~ U(−0.1, 0.1); AWGN;
SNR per sample; 400 trials per SNR. Bound: `CRB = 6 / ((2π)² SNR N (N²−1))` cycles²/sample² (Rife–Boorstyn).
For QPSK this is the data-aided (modified) CRB, which NDA estimators cannot reach.
Threshold = the lowest SNR from which MSE/CRB stays below 3 at every higher SNR.

MSE / CRB for the tone (1.0 = efficient):

| estimator | N=1024 thr (dB) | 0 dB | 10 dB | 20 dB | 40 dB | N=4096 thr (dB) | 0 dB | 10 dB | 20 dB | 40 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| ML (periodogram max + scalar search) | −15 | 1.07 | 1.03 | 0.93 | 1.00 | −20 | 1.08 | 1.02 | 0.94 | 0.96 |
| Quinn 2nd | −10 | 1.33 | 1.33 | 1.21 | 1.27 | −20 | 1.53 | 1.24 | 1.24 | 1.27 |
| Jacobsen + Candan correction | −10 | 2.36 | 2.28 | 1.80 | 2.14 | −17.5 | 2.45 | 2.20 | 2.10 | 2.25 |
| Kay WPDE (raw) | 17.5 | 9e7 | 2e6 | 0.93 | 1.00 | 17.5 | 6e9 | 1e8 | 0.95 | 0.91 |
| Kay after coarse-FFT de-rotation | 10 | 1e5 | 1.11 | 0.93 | 1.00 | 10 | 2e6 | 1.05 | 0.95 | 0.91 |
| periodogram + parabola, 4N padding | – | 1.08 | 1.11 | 1.78 | 77 | – | 1.08 | 1.28 | 4.1 | 310 |
| periodogram + parabola, N points | – | 192 | 1.9e3 | 2e4 | 2e6 | – | 820 | 8e3 | 8e4 | 8e6 |
| MUSIC / ESPRIT, M = 16 | – | 107 / 93 | 25 / 20 | 13.5 / 11.8 | 12.6 / 11.2 | – | 1.7e3 / 1.5e3 | 226 / 195 | 63 / 54 | 50 / 43 |
| MUSIC / ESPRIT, M = 64 | – | 5.2 / 4.4 | 4.0 / 3.3 | 3.5 / 2.8 | 4.1 / 3.2 | – | 40 / 34 | 16 / 12.5 | 11 / 9.9 | 12 / 11 |

QPSK, MSE / MCRB: 4th power + ML gives 3.7 (10 dB), 7.7 (20 dB) and 664 (40 dB) for N = 1024, and 3.4 / 8.3 / 667 for N = 4096.
4th power + Jacobsen gives 8.7 / 23 / 2.2e3. Both have a threshold near 2.5–5 dB. The data-aided ML reference stays at 0.95–1.18 at every SNR.

CRB checks and interpretation
* Above threshold, the ML-type estimators reach the CRB: periodogram ML (ratio 0.93–1.08), Quinn (1.2–1.5),
  Kay above its threshold (0.9–1.1), and DA-ML for QPSK (≈ 1.0). The ML thresholds match theory at N·SNR ≈ 15 dB
  (−15 dB for N = 1024, −20 dB for N = 4096).
* Jacobsen/Candan sits at ≈ 2.2× the CRB: close to efficient, but not efficient (3 bins, rectangular window).
* Parabolic interpolation on N points has a bias floor, as expected. 4× zero-padding lowers the floor, but it still appears above 20–30 dB.
* Raw Kay shows a 17.5 dB "threshold", but this is ±π phase wrapping for |f0| near 0.5, not a true threshold. Removing a coarse
  FFT frequency first brings it down to ≈ 10 dB, which is the known lag-1 phase-difference threshold. This is genuine behaviour, not a bug.
* Single-signal MUSIC/ESPRIT with M ≪ N are consistent but not efficient. MSE/CRB grows roughly in proportion to N/M:
  it falls from 13 to 4 when M goes from 16 to 64 at N = 1024, and is ×4 higher at N = 4096. This is the known behaviour of
  short-window covariance-subspace estimators; I did not tune them further.
* 4th-power NDA QPSK has the expected threshold, then a self-noise floor: with an RRC pulse, x⁴ is not a clean tone, so
  MSE/MCRB grows with SNR.

Outputs: `results/carrier_crb.json`, `results/carrier_crb.png`, `results/carrier_crb.log`.

## Task 2 — SNR estimators vs the CRB (`snr_crb.py`)

ρ = Es/N0 at the matched-filter output. Reception is coherent (carrier phase known), as in Pauluzzi & Beaulieu (2000).
500 trials per SNR, Es/N0 from −10 to 30 dB. NMSE = E[(ρ̂−ρ)²]/ρ².

Bounds
* DA, symbol level: `CRB(ρ) = (2ρ + ρ²)/N`. Derived from the Fisher information of (A, σ²) for y = A c + n,
  n ~ CN(0, σ²): J_A = 2N/σ², J_σ² = N/σ⁴, ∂ρ/∂A = 2A/σ², ∂ρ/∂σ² = −A²/σ⁴.
* NDA, symbol level: Fisher information of the constellation-mixture likelihood, computed by Monte Carlo
  (4·10⁵ draws). **Code check:** with a one-point constellation it reproduces the closed-form DA CRB to within 0.1 %.
* Sample level (8 sps): DA is `(2ρ + ρ²/sps)/N`. NDA is the mixture FIM plus (sps−1)/σ⁴ of noise-only information.
  The "split" NDA bound is the mixture FIM plus 1/σ⁴, i.e. the bound for the two half-symbol sums that SSME uses.

NMSE / bound, with bias in % for DD / M2M4 / SSME / eig:

| case | Es/N0 (dB) | DA-ML / CRB_DA | DD-ML / CRB_DA | M2M4 / CRB_NDA | SSME / CRB_split | eig-MDL / CRB_NDA,sample | DA-ML(sample) / CRB_DA,sample | bias DD / M2M4 / SSME / eig (%) |
|---|---|---|---|---|---|---|---|---|
| BPSK N=256 | 0 | 1.06 | 5.7 | 6.7 | 2.1 | 6.2 | 1.09 | +23 / +2.5 / +1.0 / −17 |
| BPSK N=256 | 10 | 1.14 | 1.13 | 2.4 | 2.2 | 1.2 | 0.89 | +0.8 / +2.1 / +0.2 / −1.2 |
| BPSK N=256 | 20 | 1.05 | 1.03 | 2.1 | 2.1 | 1.3 | 1.03 | +0.5 / +1.1 / +0.2 / −0.7 |
| BPSK N=1024 | 10 | 0.95 | 0.95 | 2.3 | 1.8 | 1.6 | 1.02 | +0.1 / 0.0 / +0.1 / −1.1 |
| QPSK N=256 | 0 | 1.00 | 113 | 2.1 | 1.5 | 5.3 | 1.02 | +114 / +2.7 / +0.9 / −19 |
| QPSK N=256 | 10 | 1.07 | 1.05 | 2.6 | 2.3 | 1.2 | 1.01 | +0.9 / +0.8 / +0.2 / −0.9 |
| QPSK N=1024 | 20 | 1.06 | 1.05 | 2.1 | 2.2 | 1.7 | 1.00 | +0.3 / +0.3 / +0.1 / −0.8 |
| 16QAM N=256 | 10 | 1.06 | 58 | 60 | 1.4 | 1.4 | 0.96 | +52 / +24 / +0.8 / −0.7 |
| 16QAM N=1024 | 10 | 1.04 | 216 | 11.6 | 1.4 | 1.8 | 1.00 | +50 / +2.6 / +0.2 / −1.4 |
| 16QAM N=1024 | 20 | 1.07 | 1.07 | 1.8e8 | 2.7 | 3.8 | 0.98 | +0.2 / +1.8e4 / +0.4 / −0.6 |

Full curves are in `results/snr_crb_symbol_level.png` and `results/snr_crb_sample_level.png`; every number is in `results/snr_crb.json`.

Interpretation
* DA-ML attains the DA CRB at every SNR, at both symbol and sample level, so the estimator and the bound agree.
* DD-ML equals DA-ML once decisions are reliable (BPSK ≥ 5 dB, QPSK ≥ 8 dB, 16QAM ≥ 15 dB). Below that it is strongly
  over-optimistic, because decision errors shrink the residual. This is the classical Pauluzzi–Beaulieu result.
* M2M4 is ≈ 2× the NDA CRB at high SNR for BPSK and QPSK, with small bias. For 16QAM it is usable only near 0–5 dB.
  At high SNR its noise estimate M2 − Ŝ is swamped by the sample-kurtosis fluctuation of the 16QAM symbols: about 2–3 % of S
  for N ≤ 1024, which is larger than the noise power itself at 20 dB. The result is negative noise estimates, clipped at ρ̂ = 1e5,
  and that is why the MSE explodes. This is a genuine limitation, not a bug; `nmse_median_based` in the JSON shows the typical error.
  Below 0 dB, M2M4 can fall *below* the NDA CRB because it is biased (clipped at 0).
* The "kurtosis-based" estimator solves (κ−k_a)ρ² + (2κ−4)ρ + (κ−2) = 0 with κ = M4/M2². It is algebraically identical
  to M2M4: the positive root is r/(1−r), with r = √((2−κ)/(2−k_a)). I implemented it and checked this numerically
  (largest relative difference to M2M4: 2.4·10⁻¹³). It is therefore not claimed as a separate estimator.
* SSME (rectangular pulse, integrate-and-dump halves) is unbiased (|bias| ≤ 1 %) from −10 dB upward and sits at ≈ 1.0–2.8× the bound
  for its own data. At high SNR its NMSE tends to 1/N, against 1/(2N) for the bound. That factor of ≈ 2 is expected, because SSME
  ignores the decision-directed residual.
* Eigenvalue/MDL (RRC, 32×32 forward-backward covariance) is 1.2–1.8× the sample-level NDA CRB at 10 dB and 1.3–3.8× at 20 dB.
  It is biased low below ≈ 2 dB (−5 to −19 %): MDL under-estimates the signal-subspace dimension when the RRC band-edge
  eigenvalues sink into the noise, and returns ρ̂ = 0 when it picks dimension 0 at −10 dB.

## Task 3 — closing the FSK symbol-rate gap (`fsk_rate.py`, `sr_bank.py`, `consensus.py`, `eval_symbol_rate.py`)

DEV is `data/symbol_rate_torchsig_dev.npz` (same generator, seed 1). HELD-OUT is `D:\data\symbol_rate_torchsig.npz` (seed 0),
used only for the final report. A frame counts as a success if |R̂s − Rs|/Rs < 2 %.

What was built
1. **Band-limiting before the FM discriminator.** Welch PSD → noise floor → centroid and 99 % band → de-rotate →
   low-pass FIR at 0.6 × OBW. A narrower 0.4 × OBW variant ("n" suffix) is also used. This raises the discriminator SNR for narrow signals.
2. **Transition features of the discriminator output** at scales L = 1, 2, 3, 5, 8: |f_s − median|, the Haar wavelet of the
   frequency (a transition detector) and the squared increment. Also the cyclic autocorrelation |FFT(y[n+d] y*[n])|²,
   folded over ±α at lags 1–8; it is bilinear, so it has no FM threshold.
3. **Local-background line detection — the key fix.** Prominence is now line power divided by a running median of the spectrum
   (±0.006 cycles/sample). With the band-median used before, broad low-frequency content of the features won the argmax
   (for example CAF on GMSK, and the envelope on RadioML). On its own, this change raised an auto-scale discriminator feature
   from 47 % to 73 % on the DEV FSK frames.
4. **Consensus by normalised prominence.** Each member gets z_k = log prominence − (95th percentile of member k's log
   prominence on DEV frames where it is wrong). The "vote" rule clusters member rates (1 %) and sums the positive z values;
   members are chosen by greedy forward selection on DEV accuracy. The chosen set (DEV 90.9 %) is ifd3, ifd8n, haar5n, caf6,
   ifd2n, haar8n, ifd5n, ifd1, ifd8, ifd2, dsq5, dsq8, all with local background. Parameters: `results/consensus_params.json`.

Held-out results (3,000 frames, % within 2 %):

| estimator | all | PSK/QAM | FSK family | bpsk | qpsk | 8psk | 16qam | 64qam | 2fsk | 4fsk | 2gfsk | 2msk | 2gmsk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inst_freq_s3 (baseline) | 72.7 | 96.1 | 49.3 | 100 | 99 | 98 | 92 | 92 | 66 | 43 | 32 | 66 | 40 |
| best single new member (ifd3, local bg) | 77.6 | 93.3 | 61.9 | 95 | 96 | 95 | 91 | 90 | 59 | 55 | 58 | 75 | 62 |
| **consensus (DEV-tuned)** | **89.9** | **96.9** | **82.9** | 98 | 99 | 98 | 94 | 96 | 95 | 97 | 64 | 88 | 71 |
| FSK-only consensus on FSK frames (class family known) | 90.0 | 96.9 | 83.1 | 98 | 99 | 98 | 94 | 96 | 96 | 95 | 62 | 92 | 70 |

Consensus by SNR (all / FSK family): 0 dB 69.5 / 48.3, 5 dB 88.3 / 78.3, 10 dB 96.8 / 95.3, 15 dB 97.7 / 96.7, 20 dB 97.3 / 96.0.
DEV vs held-out is 90.9 % vs 89.9 %, so the member selection shows no sign of over-fitting.

Gap status
* **Closed for 2FSK (66 → 95), 4FSK (43 → 97) and MSK (66 → 88).**
* **Narrowed but not closed for GFSK (32 → 64) and GMSK (40 → 71).** By SNR (0/5/10/15/20 dB), GFSK scores 8/30/87/97/98 and
  GMSK 5/63/92/95/100: both are solved at ≥ 10 dB and fail at 0–5 dB.
* Bug check on those failures: they are not harmonic errors (only 13 % of all failures are 2×, 1 % are ½×), and at least one bank
  member is right on 93 % of the GFSK/GMSK frames. Their sps values are small (median 3.8–4.5), and TorchSig draws h = 0.1–0.5
  and BT = 0.1–0.5. With such small deviation and heavy Gaussian smoothing, the frequency transitions sit below the
  FM-discriminator threshold at 0–5 dB, and the bilinear CAF line is too weak. I treat this as a genuine limitation of
  transition/cyclic-line estimators on these frames, not a bug.
* The remaining 2FSK/4FSK misses have sps ≈ 18–24, at the edge of the 2.5–25 sps search range.
* Knowing the class family from AMC adds nothing (90.0 vs 89.9 %).

Outputs: `results/symbol_rate_extended.json`, `results/symbol_rate_extended.png`, `results/eval_symbol_rate.log`.
Banks (all members on all frames): `data/bank_{dev,heldout,alrt_dev,alrt_heldout,radioml}.npz`. Check: the bank member
`ifr3.g` reproduces `symbol_rate.ESTIMATORS["inst_freq_s3"]` exactly (maximum difference 0).

## Task 4 — blind likelihood classifier on RadioML (`blind_alrt.py`)

The frames, loader arguments (`per_group=120, order="hdf5", seed=5`, SNR −10…20 dB) and test split
(`train_test_split(..., test_size=0.5, stratify=y*100+(snr+50), random_state=0)`, 4,800 test frames) are the same as in
`practical_alrt.py`. `classify()` is imported unchanged; only the sps passed to it changes. The runner recomputes the bank
members for some test frames and asserts that the cached bank rows are in loader order. Variants:
* `inst_freq_s3`: the original blind step, re-run. It reproduces 59.0 / 80.4 % exactly.
* `cons_sr_dev`: the Task-3 consensus, tuned only on the TorchSig 4,096-sample DEV set (pure transfer).
* `cons_lin_dev`: the same method re-tuned on 1,024-sample linear-modulation DEV data, namely the TorchSig alrt DEV set
  (seed 1) plus the RadioML TRAIN half (true sps 8). Members: ifr3.g, caf6, ifd1n.g, ifd2.g, caf4, ifd2n, caf8, caf3n, ifd5.
* `sps_known`: 8.

RadioML (4,800 test frames):

| sps step | sps within 2 % (all / SNR ≥ 10) | ALRT all | ALRT SNR ≥ 10 | ALRT first SNR ≥ 85 % | HLRT all | HLRT SNR ≥ 10 | HLRT 85 % |
|---|---|---|---|---|---|---|---|
| inst_freq_s3 (baseline) | 42.4 / 70.6 | 59.0 | 80.4 | never | 59.3 | 81.1 | never |
| **consensus, TorchSig DEV** | 50.5 / 92.7 | **64.1** | **91.8** | **8 dB** | 64.1 | 91.7 | 8 dB |
| consensus, linear DEV | 51.1 / 91.7 | 62.8 | 90.2 | 8 dB | 63.2 | 90.3 | 8 dB |
| sps known (= 8) | 100 / 100 | 70.4 | 93.3 | 4 dB | 70.6 | 93.4 | 2 dB |

Symbol-rate accuracy on the RadioML test frames by SNR (consensus, TorchSig DEV): ≤ 4 % at ≤ 0 dB, 28 % at 2 dB, 53 % at 4 dB,
67 % at 6 dB and 92–94 % from 8 dB up.

The diagnosis holds:
* RadioML's true rate is 8 sps; all line estimators agree (median 8.00).
* The baseline failed on 16/64QAM because |inst. freq| has a weak line for QAM (29 % / 21 % correct on the train half at ≥ 10 dB).
* Envelope and CAF members with band-median prominence picked the low-frequency fading content (≈ 18–20 sps). The
  local-background detector removes this, and CAF reaches ≈ 88 % at ≥ 10 dB.

TorchSig `D:\data\alrt_torchsig.npz` (3,200 test frames, sps 3–20, SRRC or rectangular pulses):

| sps step | sps within 2 % (all / SNR ≥ 10) | ALRT all / ≥10 / 85 % | HLRT all / ≥10 / 85 % |
|---|---|---|---|
| inst_freq_s3 (baseline) | 59.4 / 90.6 | 69.1 / 91.3 / 6 dB | 69.0 / 90.9 / 6 dB |
| consensus, TorchSig DEV | 54.9 / 79.6 | 67.8 / 92.9 / 8 dB | 67.4 / 92.5 / 8 dB |
| consensus, linear DEV | 65.8 / 94.1 | 69.8 / 93.2 / 4 dB | 69.4 / 93.3 / 4 dB |
| sps known | 100 | 76.3 / 95.6 / 4 dB | 76.1 / 95.4 / 4 dB |

Gap status
* **Largely closed at SNR ≥ 10 dB:** RadioML goes from 80.4 to 91.8 %, against a sps-known ceiling of 93.3 %, and 85 % is
  now reached at 8 dB instead of never.
* The all-SNR figure (64.1 %) stays below sps-known (70.4 %) because the symbol rate cannot be estimated from 1,024 samples
  below ≈ 4 dB.
* The 4,096-sample DEV consensus does not transfer well to the 1,024-sample TorchSig frames: sps accuracy at ≥ 10 dB drops
  to 79.6 %, because the narrow-band FSK members misfire on short rectangular-pulse frames. The 1,024-sample linear-DEV
  consensus is the better choice there.

Outputs: `results/blind_alrt.json` (merged), `results/blind_alrt_parts/*.json`, `results/blind_alrt_*.log`.

## Caveats
* Symbol-rate success uses the 2 % criterion of the original study. For the ALRT, a 2 % error is already fatal: 0.5 % is
  ≈ 0.6 symbol of drift over 1,024 samples. In practice the estimates are either < 0.5 % off or grossly wrong (the
  within-0.5 % and within-2 % figures in the JSON are nearly equal).
* The consensus member selection and the μ_k normalisation are tuned on DEV. For `cons_lin_dev` they are also tuned on the
  RadioML TRAIN half, which is the same half the original feature classifier trains on. Held-out and test frames were never used.
* The DEV sets come from the same TorchSig generator as the held-out set, with a different seed. Held-out numbers therefore
  measure generalisation over random draws, not across generators.
* The NDA SNR CRBs assume a known carrier phase; they are still valid lower bounds for the phase-blind M2M4 and eig
  estimators. The sample-level NDA CRB is exact for the rectangular pulse and approximate for RRC.
* Tasks 1 and 2 use 400 and 500 Monte Carlo trials per point, so MSE/CRB ratios fluctuate by about ±10 %.
* The full bank costs ≈ 0.15–0.8 s per frame (about 100 FFT line searches). A deployed version would keep only the
  9–12 selected members.
