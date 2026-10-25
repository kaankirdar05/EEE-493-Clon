# detection_discovered: eigenvalue family, TKEO and stochastic-resonance detectors

This folder adds classical and statistical detection methods to the verified WP4 baseline. Nothing outside this folder was
edited. The WP4 framework (`wp4.detectors`, `segmentation`, `scoring`, `run_wp4.plot_*`) is imported unchanged.
- Parameters were tuned on dev scenes 0–99 or on separate synthetic seeds.
- Reported scene numbers are held-out scenes 100–299 of `D:\data\wideband\scenes.h5` (impaired) and `scenes_clean.h5`.
- Every method passed a theory or known-answer check before its benchmark numbers were produced.

```
disc/eigen.py       smoothed L x L sub-band covariance; AGM, MED, John, EMR, condition number; Tracy-Widom (beta=2)
                    CDF (Bornemann Fredholm determinant), Zeng-Liang and TW-Curtiss condition-number thresholds
disc/tkeo.py        complex TKEO; tkeo_stft (per STFT bin along time) and tkeo_sb (per sub-band block)
disc/sr.py          bistable stochastic-resonance pre-treatment + the paper's FFT-max metric
disc/signals.py     synthetic tone / RRC-QPSK generators
sanity.py           tw | rise | tkeo   theory / known-answer checks
run_scenes.py       calibrate | verify | run   (same loop as detection_extended/run_partA.py)
run_dev.sh, choose_dev.py, run_heldout.sh, compare.py   dev tuning -> held-out -> table
synth_bench.py      synthetic narrowband Pd vs SNR (N = 4096, Pfa 1e-2), all detectors + SR
sr_bench.py         SR in the paper's own setting (10 Hz tone, fs 100 Hz, N 512, FFT 256)
diag_sr.py, diag_tkeo.py   diagnostics behind two of the conclusions below
```
To run: use `..\.venv\Scripts\python.exe`. The scripts set `HDF5_USE_FILE_LOCKING=FALSE`.

## Methods and sources

All block detectors work on the WP4 coarse grid: 32 time blocks × 16 sub-bands of 512 complex samples (`wp4.detectors.subbands`).

**Covariance.** C is the smoothed sample covariance with smoothing factor L (Zeng & Liang 2009),
C[i,j] = 1/n Σ x[k+i] x*[k+j], with n = 513 − L. It is the same matrix as the reference `eigen_mme` builds. The two
implementations agree to 4e-15.

| name | statistic | source |
|---|---|---|
| `agm_L` | arithmetic / geometric mean of the eigenvalues. This is the GLRT (Mauchly sphericity LRT) of C = σ²I against an unstructured C | Mauchly 1940; Zhang, Mallik & Letaief 2010 |
| `med_L` | λmax / σ̂². σ̂² is a separable gain(t)×shape(sub-band) scene estimate of the block power tr C/L (25 % quantiles). The threshold is calibrated on whole white-noise scenes (400 scenes) | Zeng & Liang 2009 (MED) |
| `john_L` | U = L Σλ² / (Σλ)² − 1 | John 1971; Ledoit & Wolf 2002 |
| EMR | T = (1/L)Σλ² / ((1/L)Σλ)² = **U + 1** | Huang, Fang, Liu, So & Li, IEEE TVT 2015 |
| `cond_L` | λmax/λmin with an empirical threshold. It equals the reference `eigen_mme` at L = 8 | Zeng & Liang 2009 |
| `cond_zl_L` | λmax/λmin with an analytic threshold: TW2 for λmax, λmin at its Marchenko–Pastur edge (Zeng–Liang form, complex) | Zeng & Liang 2009; Johnstone 2001 |
| `cond_twc_L` | λmax/λmin with an analytic threshold: both edges TW2 (λmin with the Feldheim–Sodin scaling), combined with the Curtiss ratio integral | Penna, Garello & Spirito, IEEE Commun. Lett. 2009 |
| `tkeo_stft` | Complex TKEO Ψ = \|X(m)\|² − Re(X(m−1)X*(m+1)) along time for each STFT bin (non-overlapping Hann frames), then the `energy` kt = 8 smoothing and excision noise floor (looks = kt/1.5) | Kaiser 1990; Hamila et al., EURASIP JASP 2003 |
| `tkeo_sb` | block mean of the time-domain TKEO of each sub-band block, divided by a separable scene estimate built from the block means | as above |
| SR-ED | bistable pre-filter dx/dt = ax − bx³ + g·y + √D·ξ (a = b = 1, Euler step h), then energy or the paper's FFT-max metric | Zheng, Li, Djouadi & Wang, arXiv:0906.0739 |

**John vs EMR.** The two statistics are an affine map of each other (T = U + 1). With empirically calibrated thresholds
they make identical decisions, so the `john_L4` row *is* the EMR row. They differ only in how an analytic threshold
is derived.

**Thresholds.** Unless stated otherwise, each threshold is the (1 − Pfa) quantile on white complex Gaussian noise, as in
`Detector.calibrate`. Calibration files are in `results/calibration/`.

## Sanity checks (all before any benchmark)

### 1. Tracy–Widom implementation (`results/sanity/tracy_widom.json`)

**PASS: TW2 table.** The tabulated TW2 has mean −1.771087 (ref −1.7710868) and variance 0.81320 (ref 0.81319). Its 0.99
quantile is 0.4776.

**PASS: known answer on iid complex Wishart** (n = 513 − L independent columns, 50–100k trials). Measured Pfa of each
analytic threshold at a 1e-3 target:

| L | λmax TW (MED, σ² known) | λmin TW | cond ZL | cond TW-Curtiss | John asympt. N(−1/L, 2) |
|---|---|---|---|---|---|
| 4 | 7.1e-4 | 5.9e-4 | 1.4e-4 | 3.9e-4 | 5.5e-3 |
| 8 | 8.5e-4 | 7.7e-4 | 2.5e-4 | 5.9e-4 | 3.2e-3 |
| 16 | 8.6e-4 | 8.0e-4 | 2.3e-4 | 6.5e-4 | 1.9e-3 |
| 32 | 8.6e-4 | 9.4e-4 | 5.8e-4 | 1.2e-3 | 1.3e-3 |

- The edge laws behave as expected.
- TW-Curtiss is closer to target than Zeng–Liang. At 1e-1, ZL gives 0.007–0.011 and TWC gives 0.069–0.091: ZL ignores the
  fluctuation of λmin, so it is strongly conservative.
- John's Gaussian limit converges only slowly in L, because of skew at small L.
- **Bug found and fixed.** The first John threshold used N(0, 1). It gave Pfa 0.016–0.019 at a 1e-3 target. The
  measured moments of nU − L were mean −0.25/−0.13/−0.08 and variance 1.88–1.98, which pointed to the correct complex
  limit N(−1/L, 2): β = 2 halves the real Ledoit–Wolf variance of 4.

**The point of comparison: the same thresholds on the actual detector input.** Here the sub-band covariance is smoothed
(Hankel), which makes its columns overlapping and dependent. Measured Pfa at a 1e-3 target, on 200k fresh blocks:

| L | λmax TW | cond ZL | cond TWC | John asympt. | empirical cond thr. vs TWC thr. |
|---|---|---|---|---|---|
| 4 | 1.7e-2 | 2.6e-3 | 4.7e-3 | 1.9e-2 | 1.529 vs 1.463 |
| 8 | 6.1e-2 | 2.5e-2 | 3.4e-2 | 5.5e-2 | 1.944 vs 1.697 |
| 16 | 1.7e-1 | 1.2e-1 | 1.4e-1 | 1.2e-1 | 2.811 vs 2.110 |
| 32 | 3.8e-1 | 3.5e-1 | 3.7e-1 | 1.9e-1 | 4.826 vs 2.934 |

- The analytic thresholds are **5× (L = 4) to 370× (L = 32) too permissive**.
- This is **not an implementation error**: the identical code is on target for iid Wishart.
- It is a model mismatch. The random-matrix laws assume n independent columns, while the Hankel columns overlap, so the
  effective sample size is far below n.
- On the scene pipeline: `cond_twc_L8` gives a fresh-noise Pfa of 3.4e-2 at a 1e-3 target, and `cond_twc_L32` gives 0.37
  (`results/pfa_check/`).

### 2. Empirical calibration on fresh white noise (`results/pfa_check/*.json`)

Measured with a seed that differs from calibration. Each cell is block level / full `stat(spec, iq)` pipeline on 40 noise scenes.

| detector | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| agm_L4 | 1.0e-1 / 1.0e-1 | 9.8e-3 / 9.7e-3 | 1.1e-3 / 1.3e-3 | 1.1e-4 / 1.5e-4 |
| john_L4 (= EMR) | 1.0e-1 / 1.0e-1 | 9.8e-3 / 9.8e-3 | 1.0e-3 / 1.4e-3 | 1.3e-4 / 1.5e-4 |
| cond_L32 | 9.9e-2 / 1.0e-1 | 9.6e-3 / 1.1e-2 | 1.0e-3 / 1.2e-3 | 1.4e-4 / 9.8e-5 |
| cond_L8 | 9.9e-2 / 9.9e-2 | 1.0e-2 / 1.0e-2 | 1.0e-3 / 1.2e-3 | 1.0e-4 / 2.0e-4 |
| med_L4 | 9.9e-2 / 9.6e-2 | 9.7e-3 / 9.7e-3 | 1.0e-3 / 9.3e-4 | 6.3e-5 / 9.8e-5 |
| tkeo_sb | 1.0e-1 / 9.6e-2 | 1.0e-2 / 9.4e-3 | 1.0e-3 / 1.0e-3 | 1.3e-4 / 9.8e-5 |
| tkeo_stft | 1.0e-1 / 1.0e-1 | 9.8e-3 / 9.8e-3 | 1.1e-3 / 9.8e-4 | 1.0e-4 / 1.1e-4 |
| cond_twc_L8 (analytic) | 1.6e-1 / 1.6e-1 | 7.2e-2 / 7.3e-2 | 3.4e-2 / 3.2e-2 | 1.7e-2 / 1.6e-2 |
| cond_twc_L32 (analytic) | 5.7e-1 / 5.7e-1 | 4.5e-1 / 4.6e-1 | 3.7e-1 / 3.7e-1 | 3.1e-1 / 3.0e-1 |

**PASS** for every empirically calibrated detector. The full-pipeline 1e-4 values rest on 20k blocks, so ±50 % is
binomial noise.

### 3. Correlated signal in noise (`results/sanity/rise_correlated_signal.json`)

Setup: 512-sample blocks, RRC-QPSK at 8 samples/symbol, SNR −10/−5/0 dB, carrier offset 0.1 cycles/sample or centred.

**PASS.** Every statistic's mean rises under H1, for every L.

Example at L = 4, −10 dB, offset 0.1 (Pd at Pfa 1e-3 in brackets):

| statistic | H0 mean | H1 mean | Pd |
|---|---|---|---|
| AGM | 1.003 | 1.012 | 0.45 |
| John | 0.006 | 0.026 | 0.51 |
| cond | 1.22 | 1.49 | 0.39 |
| λmax/σ² | 1.10 | 1.38 | 0.75 |
| energy | 1.00 | 1.10 | 0.18 |

**Expected exception, as theory predicts:**
- For the *centred* signal the TKEO mean barely moves (1.000 → 1.010 at −10 dB, 1.107 at 0 dB, against 2.0 for energy).
- The TKEO weights the spectrum by (1 − cos 2ω), which is 0 at ω = 0 and ω = π. A narrowband signal at the centre of a
  sub-band is nearly invisible to `tkeo_sb`.

### 4. TKEO closed form (`results/sanity/tkeo_closed_form.json`)

For x = A e^{jωn}, Ψ = A²(1 − cos 2ω) = 2A² sin²ω. This was checked for A ∈ {0.5, 1, 3} and 7 values of ω. The maximum
error is 4e-12, and Ψ is constant in n.

> **Note: the formula in the brief, A²(1 − cos ω), is not correct for the operator as defined** (the maximum error is
> 18 = 2A² at ω = π). It would hold for a lag-1 cross term Re(x(n)x*(n+1)), or with ω measured over two samples.

For complex white noise, theory gives E Ψ = σ², Var Ψ = 1.5σ⁴ and no autocorrelation. Measured: mean 0.99975, variance
1.497, lag-1/2/3 autocovariance ≤ 5e-4. **PASS.**

### 5. Pipeline known answer

**PASS.** `cond_L8` (empirical) run through `run_scenes.py` reproduces the reference `eigen_mme` exactly:
- impaired: F1 4.7, Pd 87.2, Pfa 4.2e-1
- clean: F1 37.5, Pd 79.2, Pfa 8.9e-2

The synthetic harness (energy, tone) reaches Pd = 0.5 at −14.3 dB, against −14.4 dB from the noncentral-χ² closed form.

## Dev tuning (`results/dev_tuning/summary.md`)

L ∈ {4, 8, 16, 32} for each eigenvalue detector, and kt ∈ {4, 8, 16} for `tkeo_stft`. The choice is the best mean dev
box F1 over the two datasets:

| detector | chosen |
|---|---|
| AGM | L = 4 |
| John/EMR | L = 4 |
| MED | L = 4 |
| cond | L = 32 |
| tkeo_stft | kt = 8 |

Dev differences between L values are ±1–2 F1, which is within dev-to-held-out drift. `cond_L8` and `cond_twc_L8` were
also run held-out, for the reference check and the TW comparison.

## Held-out results (scenes 100–299, Pfa target 1e-3)

Sources: `results/heldout_comparison.md/.json`, `results/heldout_recall_vs_snr_{impaired,clean}.png`, and per-detector
folders `results/<det>_<dataset>/`. Reference rows are in italics. ms/scene is `stat()` time inside a pool worker on a
shared machine, so it is indicative only; the reference energy time was measured differently.

**Impaired**

| detector | F1 | P | R | rec 0–5 | rec 10–15 | emitter Pd | meas. Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|
| *energy* | 53.0 | 48.9 | 57.8 | 40 | 79 | 49.8 | 2.1e-2 | 318 |
| *eigen_mme* | 4.7 | 3.4 | 7.4 | 4 | 10 | 87.2 | 4.2e-1 | 298 |
| *cav_L4* | 6.6 | 4.6 | 12.2 | 5 | 13 | 82.6 | 3.2e-1 | 56 |
| *scd_fam_Np16* | 16.2 | 15.4 | 17.1 | 1 | 21 | 34.2 | 5.4e-2 | 4164 |
| agm_L4 | 5.9 | 4.1 | 10.5 | 5 | 12 | 83.5 | 3.3e-1 | 212 |
| john_L4 (= EMR) | 5.7 | 3.9 | 10.3 | 4 | 13 | 83.4 | 3.3e-1 | 68 |
| med_L4 | 8.5 | 6.2 | 13.5 | 11 | 17 | 86.2 | 2.6e-1 | 58 |
| cond_L32 (empirical) | 8.2 | 6.2 | 12.2 | 5 | 17 | 81.9 | 3.3e-1 | 564 |
| cond_twc_L32 (TW-Curtiss) | 0.0 | 0.0 | 0.0 | 0 | 0 | 97.3 | 8.0e-1 | 532 |
| cond_zl_L32 (Zeng–Liang TW) | 0.0 | 0.0 | 0.0 | 0 | 0 | 97.3 | 7.9e-1 | 469 |
| cond_L8 (empirical) | 4.7 | 3.4 | 7.4 | 4 | 10 | 87.2 | 4.2e-1 | 128 |
| cond_twc_L8 | 1.1 | 1.0 | 1.3 | 0 | 1 | 94.1 | 6.0e-1 | 129 |
| tkeo_sb | 20.9 | 15.7 | 31.2 | 24 | 41 | 60.9 | 6.7e-2 | 28 |
| **tkeo_stft** | **54.8** | 54.6 | 55.0 | 32 | 79 | 45.2 | 1.1e-2 | 206 |

**Clean**

| detector | F1 | P | R | rec 0–5 | rec 10–15 | emitter Pd | meas. Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|
| *energy* | **67.4** | 72.7 | 62.8 | 52 | 74 | 53.8 | 5.1e-3 | 316 |
| *eigen_mme* | 37.5 | 35.4 | 39.9 | 26 | 50 | 79.2 | 8.9e-2 | 308 |
| *cav_L4* | 36.8 | 33.3 | 41.1 | 27 | 55 | 78.3 | 7.1e-2 | 56 |
| *scd_fam_Np16* | 24.2 | 25.0 | 23.4 | 5 | 24 | 33.4 | 2.6e-2 | 4164 |
| agm_L4 | 35.3 | 32.7 | 38.4 | 24 | 48 | 79.6 | 9.4e-2 | 74 |
| john_L4 (= EMR) | 34.5 | 31.9 | 37.5 | 23 | 48 | 79.8 | 9.5e-2 | 76 |
| med_L4 | 37.1 | 33.1 | 42.3 | 28 | 46 | 88.0 | 1.0e-1 | 65 |
| cond_L32 (empirical) | 36.9 | 34.8 | 39.3 | 18 | 50 | 74.1 | 7.6e-2 | 563 |
| cond_twc_L32 | 0.0 | 0.0 | 0.0 | 0 | 0 | 96.6 | 4.4e-1 | 467 |
| cond_zl_L32 | 0.0 | 0.0 | 0.0 | 0 | 0 | 96.1 | 4.2e-1 | 480 |
| cond_L8 (empirical) | 37.5 | 35.4 | 39.9 | 26 | 50 | 79.2 | 8.9e-2 | 111 |
| cond_twc_L8 | 11.1 | 7.1 | 24.4 | 12 | 29 | 86.8 | 1.3e-1 | 409 |
| tkeo_sb | 35.7 | 30.5 | 43.0 | 37 | 49 | 68.1 | 5.0e-2 | 30 |
| tkeo_stft | 65.2 | 72.2 | 59.5 | 37 | 76 | 48.3 | 4.7e-3 | 174 |

## Synthetic narrowband benchmark (`results/synthetic/synthetic_bench.json`, `synthetic_pd_vs_snr.png`)

**Setup.**
- One signal in complex AWGN, N = 4096 samples, Pfa 1e-2 (empirical thresholds on 20000 noise frames, separate seed).
- 1000 trials per SNR.
- Signals: a tone at a random frequency and phase, and RRC-QPSK at 8 samples/symbol with a carrier in U(±0.4).
- Noise power is known to `energy`, `tkeo` and `med_known`. The others are blind.
- `fft_max` is the tone GLRT, included as a frequency-aware reference.

**SNR (dB) at Pd = 0.5:**

| detector | tone | QPSK |
|---|---|---|
| energy (analytic for tone: −14.4) | −14.3 | −14.4 |
| fft_max (reference) | −24.1 | −9.1 |
| tkeo | −13.5 (saturates: Pd 0.75 at −8 dB, 0.93 at +2 dB) | −14.3 |
| agm_L4 | −16.2 | −15.6 |
| john_emr_L4 | −16.3 | −15.7 |
| med_known_L4 | −16.9 | −16.8 |
| cond_L32 (empirical) | −19.6 | −14.9 |
| cond_TWC_L32 (analytic thr., measured Pfa **0.51**) | (−26, meaningless) | (−26, meaningless) |
| SR → energy (dev-tuned g=1, D=0, h=0.1) | never (Pd 0.35 at +2 dB) | +2.0 |
| SR → FFT-max | never (0.46 at +2 dB) | never (0.31) |

What the table shows:
- Eigenvalue detectors gain 1–5 dB over the radiometer on narrowband signals, because they exploit the signal's
  correlation. That gain does not survive on the impaired scenes, where the band-edge noise is coloured.
- TKEO matches energy on average but cannot see tones near ω = 0 or ω = π (theory: weight 1 − cos 2ω).

## Stochastic resonance

### Paper's own setting (`results/sr_paper_setting/`)

**Setup.**
- Real 10 Hz tone, fs 100 Hz, N = 512, FFT 256 (K = 2), unit noise.
- (g, D, h) grid-tuned on dev seeds at −15 dB, separately for each SR detector: 6 × 6 × 4 configurations, including the
  paper's values D = 0.43, a = b = 1.

**SNR (dB) at Pd = 0.5** (Pfa 1e-2 / 1e-1):

| detector | Pfa 1e-2 | Pfa 1e-1 |
|---|---|---|
| radiometer | −8.1 | −10.7 |
| FFT-max on the raw input (the paper's "ordinary" metric) | −10.5 | −12.2 |
| known-f periodogram (optimal) | −18.0 | −21.6 |
| SR → known-f | −17.7 | −21.3 |
| SR → FFT-max (paper detector; dev-tuned or paper parameters) | never (Pd = Pfa up to 0 dB) | never |
| SR → energy | never | never |

**Is it a bug?** `diag_sr.py` (`diag_sr_spectrum.json`) says no.
- The bistable system with time constant 1/a = 100 samples is a strong low-pass filter.
- Its output power in bins 1–5 (well hopping) exceeds the tone bin by about 3 orders of magnitude, and the argmax bin is
  bin 1 under both H0 and H1.
- The FFT-max of the SR output therefore never looks at the tone.
- Only with fast dynamics (g = 30, h = 0.03) does the tone become the argmax, and then only at about 0 dB.

**Effect of the added noise D.** At the best (g, h), the SR → known-f Pd at −15 dB is 0.824 for D = 0 and 0.825–0.829 for
D = 0.01–3. That difference is within Monte-Carlo noise of ±0.012, so **adding noise never helped**.

### Verdict

**We could not reproduce the paper's claim. With properly calibrated thresholds the SR pre-treatment does not beat the
energy detector, and is worse.**
- SR plus a frequency-matched statistic beats the radiometer by about 10 dB. The same statistic without SR does 0.3 dB
  better still.
- The gain therefore comes from using the known frequency, not from resonance. This is consistent with the
  data-processing argument: added noise cannot improve an optimal test (Zozor & Amblard 2002).
- In the generic test (unknown frequency) SR is more than 15 dB worse than energy.

## Reading the results

- **The eigenvalue family (AGM, MED, John/EMR, condition number) is no better than the existing whiteness tests.**
  - Clean: F1 34.5–37.5, against eigen_mme 37.5 and cav 36.8.
  - Impaired: F1 5.7–8.5, with measured Pfa 0.26–0.33.
  - This is the same genuine limitation documented in `detection_extended`. The receiver roll-off colours the band-edge
    noise, and every sphericity / eigenvalue-spread test reads that as signal.
  - Their synthetic advantage over energy (1–5 dB on narrowband signals) is real, but it needs white noise.
  - MED (a level test with a scene noise estimate) is marginally the best on impaired scenes (8.5, recall 0–5 dB of 11 %).
- **The Tracy–Widom thresholds are unusable for the smoothed single-antenna covariance.**
  - Their measured Pfa is 3.4e-2 (L = 8) to 0.37 (L = 32) against a 1e-3 target, which gives F1 near 0.
  - The same code is accurate (TWC within 2×, ZL conservative) on independent snapshots. The mismatch is the iid-column
    assumption, not the code.
  - Empirical calibration is required, which is what the framework already does.
- **`tkeo_stft` is on par with energy, not better.**
  - Impaired: 54.8 vs 53.0. Clean: 65.2 vs 67.4. Both differences are within the ±2 F1 dev-to-held-out drift.
  - Mechanism, from `diag_tkeo.py` on dev scenes 0–39 (`results/diag_tkeo_vs_energy.json`): the along-time TKEO cancels
    frame-to-frame coherent components. Impaired false-alarm cells at the DC ±3 bins fall from 3.2e-2 to 3.8e-3 (the
    DC/LO spur), and band-edge false alarms fall from 4.7e-2 to 3.0e-2.
  - The extra variance costs low-SNR recall: emitter Pd 0.48 vs 0.54, and recall 0–5 dB of 32 vs 40 (impaired) and
    37 vs 52 (clean).
  - Net effect: a precision-for-recall trade, with half the measured Pfa on impaired scenes.
- **`tkeo_sb` is the best coarse-grid detector on impaired scenes** (20.9, against SCD 16.2 and MED 8.5). It is a
  level test normalised per sub-band, so coloured noise does not trigger it. It is still far below energy, and it is
  blind to sub-band-centred narrowband signals.
- **Stochastic resonance: a negative result**, as above.

## Caveats

- Block detectors are limited by the 16 × 32-cell block granularity, the same as for the reference `eigen_mme`/`cav`.
  Their high emitter Pd comes with high measured Pfa.
- John and EMR are the same detector under empirical thresholds. Only one row is reported.
- The SR model is my reading of the paper: Euler-discretised, added noise per sample, FFT-max excluding only the DC bin.
  The paper gives neither the sampling rate nor the noise discretisation. The parameter grid was wide (g 0.1–30, D 0–3,
  h 0.01–0.3), but other SR systems (e.g. tuned a, b, or a mixer to a low frequency) were not tried.
- The John asymptotic threshold is only a large-L approximation (5.5e-3 at L = 4 on iid data). EMR's exact-moment
  threshold from Huang et al. 2015 was not implemented.
- Timing comes from a shared machine.
