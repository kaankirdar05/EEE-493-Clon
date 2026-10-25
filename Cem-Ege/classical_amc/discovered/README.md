# Additional classical / statistical AMC methods ("discovered")

This folder adds five method families to the verified baseline in `classical_amc/` and `classical_amc/extended/`:

1. bispectrum features
2. fractal-dimension features
3. information-theoretic features
4. S-transform features
5. Cramér–von Mises and Anderson–Darling goodness-of-fit classifiers

No file outside this folder was changed. Every method passed known-answer or theory checks before its benchmark numbers were produced. All numbers below are taken from the JSON files in `results/`.

| file | content |
|---|---|
| `dcommon.py` | paths; reuses `extended/common.py` |
| `bispectrum.py` | direct bispectrum and bicoherence estimator; 30 features (`bis30`) |
| `fractal.py` | Higuchi, Katz, Petrosian and box-counting dimensions plus a multi-scale Higuchi variant; 22 features (`fd22`) |
| `infotheory.py` | Shannon and Rényi-2 histogram entropies, permutation entropy, SampEn, ApEn and RQA; 24 features (`info24`) |
| `stransform.py` | discrete Stockwell transform and band / contour statistics; 23 features (`st23`) |
| `dist_classifiers.py` | two-sample CvM and AD statistics; classifier in the KS framework of `extended/symbol_classifiers.py` |
| `sanity_features.py`, `sanity_symbol.py` | known-answer checks, saved to `results/sanity_*.json` (all pass) |
| `extract_features.py` | extracts the new features from the same RadioML frames; output `results/features_new_{7,24}.npz` |
| `bench_features.py` | HGB benchmark: `results/bench_features_{7,24}.{json,png}`; cached test predictions in `results/preds_{7,24}/` |
| `bench_symbol.py` | CvM / AD on 5 digital classes: `results/bench_symbol_{radioml,torchsig}.{json,png}`; per-frame statistics in `symbol_stats_*.npz` |
| `diag_phase_cut.py` | **post-hoc** diagnostic of the AD failure on RadioML BPSK: `results/diag_phase_cut_*.json` |
| `diag_info_ablation.py` | which part of info24 carries the 24-class gain: `results/diag_info_ablation_24.json` |
| `time_features.py` | single-process extraction timing: `results/timing_single_process.json` |

Run the scripts from this folder with the project venv. Set `HDF5_USE_FILE_LOCKING=FALSE` first.

```
python sanity_features.py; python sanity_symbol.py
python extract_features.py --set 7; python extract_features.py --set 24
python bench_features.py --set 7; python bench_features.py --set 24
python bench_symbol.py --source radioml; python bench_symbol.py --source torchsig
python diag_phase_cut.py --source radioml; python diag_phase_cut.py --source torchsig
python diag_info_ablation.py --set 24; python time_features.py
```

`bench_features.py --shard i/n` fits a subset of the runs in parallel. A final call without `--shard` assembles the results from the cache. The cached predictions are identical either way.

## Protocol

### Frames and split

**RadioML feature benchmarks.** These use exactly the frames of the cached baseline:
- 7 classes × 26 SNRs × 1000 frames;
- 24 classes × 26 SNRs × 200 frames;
- seed 0, `order="hdf5"`, frames read through `iter_radioml2018`.

`extract_features.py` recomputes base19 as well. Labels and SNRs are identical to `extended/results/features_{7,24}.npz`, and max |base19 − cached| = **0.0** for both sets.

The split is the standard one: 25 % test, stratified by class × SNR, `random_state=0`. That gives 45,500 / 31,200 test frames. **No subsampling was used.**

The baselines reproduce exactly:

| features | 7 classes (all / SNR≥10) | 24 classes (all / SNR≥10) |
|---|---|---|
| HGB base19 | 71.4 / 99.95 % | 39.2 / 62.8 % |
| HGB base19+wav24 | 72.4 % (all) | 41.3 % (all) |

The HGB settings are those of `extended/bench_features.py`, and nothing was tuned.

**Paired comparison.** Every "X + family" model is compared with "X" on the same test frames using McNemar's z = (b − c)/√(b + c). The quantity b counts frames only the larger model gets right, and c counts frames only X gets right. With 31k–45k test frames, a 0.25-point difference is roughly 1 SE of an unpaired comparison. Read |z| < 2 as "no evidence of a change".

**Symbol-domain benchmark (item 5).** This uses the same frames, 50/50 split and 1,500-frame training subsample as `extended/bench_symbol.py`:
- RadioML: per_group 120, SNR −10…20 dB, seed 5.
- TorchSig: `alrt_torchsig.npz`.
- 4,800 test frames each, with the symbol rate known.

KS is recomputed here through the same code path and reproduces the extended JSON **exactly** on both sources (max difference 0.0). EM, features+HGB and ALRT/HLRT are quoted from `extended/results/bench_symbol_*.json`, which used the same test frames.

### Tuning

- The only tuned choice is the component set of each goodness-of-fit (GoF) classifier (I, Q, |z|, arg z, combinations). It was chosen by accuracy on the training subsample.
- The feature definitions and parameters were fixed before the benchmarks (bin ranges, m = 2, r = 0.2·std, permutation order 4, voices, nfft = 64).
- The one exception is the box-counting scale range j = 4…9. It was chosen on *synthetic* known-answer signals only (see the box-counting check below).

### Timing

The machine was shared with other agents' heavy jobs throughout. In particular, `extract_features.py` timings were taken inside 16 parallel workers and are inflated 3–8×: base19 measured 1.7 / 4.0 ms against 0.50 ms in the original cache.

Use the **single-process** numbers from `time_features.py`, measured on 1,820 real frames with one thread:

| family | ms/frame |
|---|---|
| base19 | 0.87 |
| wav24 | 2.5 |
| bis30 | 14.0 |
| fd22 | 1.2 |
| info24 | 36.7 |
| st23 | 2.3 |
| info: histogram entropies only | 0.41 |
| info: histogram + permutation entropies | 1.5 |

The ms/frame column of the result tables (and the JSON files) uses the inflated parallel-worker numbers, so compare it only between rows.

## Methods and sources

**1. Bispectrum (`bis30`).** Direct, segment-averaged estimator:
- B(f1, f2) = ⟨X_k(f1) X_k(f2) X_k*(f1+f2)⟩.
- 64-sample Hann segments with the mean removed, so K = 16 per frame.
- Kim–Powers bicoherence b².

The method follows Nikias & Petropulu (1993), the HOSA `bispecd` function and Kim & Powers (1979).

Ten statistics are computed on each of x, |x| and the instantaneous frequency:
- mean and max b²;
- the fraction of bins above the 5 % Gaussian-null threshold ln(20)/K (Elgar & Guza);
- the scale-free level mean|B|/P^1.5;
- the normalised bispectral entropies of |B| and |B|² (Chua et al. 2010);
- the diagonal-slice share;
- the radius of the peak location;
- the weighted centre of the bispectrum (Zhang et al. 2001);
- the peak-to-mean ratio of the axially integrated bispectrum (Tugnait 1994; Chandran & Elgar 1993).

**2. Fractal dimensions (`fd22`).** Four dimensions are computed on each of four signals: |x|, I and Q (averaged), the unwrapped detrended phase and the instantaneous frequency.
- Higuchi (1988), with k_max = 8.
- Katz (1988), in the scale-free variant with L = Σ|Δx| and d = max|x_i − x_0|.
- Petrosian (1995).
- Box counting of the graph, with ε = 2⁻⁴…2⁻⁹.

A multi-scale Higuchi dimension (coarse-graining by 2, 4 and 8, as in Costa et al. 2002) is added for |x| and I/Q.

**3. Information theory (`info24`).**
- **Histogram entropies:** normalised Shannon and Rényi-2 entropies of |x|, arg x, the instantaneous frequency and the joint I/Q histogram, all with fixed ranges and bins.
- **Permutation entropy:** Bandt & Pompe (2002), order 4, delays 1 and 4, on |x|, I and the instantaneous frequency.
- **SampEn and ApEn:** Richman & Moorman (2000) and Pincus (1991), with m = 2 and r = 0.2·std, on |x| and the instantaneous frequency.
- **RQA:** recurrence rate RR, determinism DET and laminarity LAM (Webber & Zbilut; Marwan et al. 2007), computed on the same m = 2 Chebyshev recurrence matrix.

**4. S-transform (`st23`).** Discrete Stockwell transform (Stockwell, Mansinha & Lowe 1996), computed as an inverse FFT per voice.
- 33 log-spaced signed voices: 0, ±1, 2, 4 … 384 of 1024.
- Statistics: energy share of 8 bands, coefficient of variation over time of each band envelope, ±frequency asymmetry, TF entropy, statistics of the peak-voice track (std, jump rate), and mean, CV and kurtosis of the max-over-voices contour.
- This set of statistics follows the S-transform feature practice in power-quality and modulation recognition. The exact set is our own choice.

**5. CvM and AD classifiers.** These use the same pipeline as the extended KS classifier (Wang & Wang 2010 style):
- class-conditional M2M4 SNR estimate;
- reference set at the nearest 1 dB (200 × 128 simulated symbols through the same phase estimator);
- class-conditional M-th-power de-rotation;
- the component set is chosen on training frames, and the decision is argmin of the summed statistic.

Only the distance changes:
- the two-sample Cramér–von Mises T (Anderson 1962; scipy's `cramervonmises_2samp` statistic);
- the two-sample Anderson–Darling A² (Pettitt 1976; the k = 2 case of Scholz & Stephens' A²_kN).

Background: Stephens (1974), and the distribution-test chapter of Zhu & Nandi (2015), *Automatic Modulation Classification*.

## Sanity checks (all pass: 19/19 in `sanity_features.json` (incl. 4 extraction smoke tests), 4/4 in `sanity_symbol.json`)

### Bispectrum

- **Gaussian noise** (K = 256 segments): mean b² = 0.0044 (real) and 0.0041 (complex), against 1/K = 0.0039. 6.8 % and 5.7 % of bins exceed the 5 % threshold. The normalised |B| level is 0.06, i.e. B ≈ 0 up to estimator noise.
- **Quadratic phase coupling** (tones at 10 and 17 of 64 bins, phases φ1, φ2, φ1+φ2 random per segment): b²(10, 17) = **0.997**. The arg-max is at an equivalent symmetric location, (27, 47) ≡ (10, 17) for a real signal. With the third phase independent, b²(10, 17) = 0.0002.
- **Real frames:** at frame length K = 16, the Gaussian floor of mean b² is 0.064 (theory 1/16). These features therefore carry a large estimator floor on real frames.

### Fractal dimensions

**Higuchi** (200 realisations, N = 1024):

| signal | estimate | theory |
|---|---|---|
| straight line | 1.0000 | 1 |
| white noise | 2.000 | 2 |
| Brownian motion | 1.505 | 1.5 |
| fBm, H = 0.3 (exact Davies–Harte) | 1.702 | 1.7 |
| fBm, H = 0.7 | 1.301 | 1.3 |

**Katz and Petrosian.**
- Katz on a straight line gives exactly 1.
- Petrosian on a straight line gives exactly 1.
- Petrosian on white noise gives 1.0352, against 1.0352 from the closed form with P(sign change) = 2/3.
- Katz on white noise gives 5.5. Katz is not bounded by 2 for noise, which is a known property of the formula, not a bug.

**Box counting.**
- Line: 1.008. Sine: 1.016.
- The estimate agrees with an independent rasterised box count to within 0.002.
- It orders fBm correctly: 1.07 / 1.20 / 1.34 / 1.49 / 1.62 for H = 0.9 … 0.1.
- **It is biased low for rough graphs at N = 1024.** Brownian motion gives 1.34 instead of 1.5, and white noise 1.69 instead of 2. This is a finite-resolution property of the estimator.
- **This check failed first** with ε = 2⁻²…2⁻⁷: the sine came out at 1.24 and the line at 1.03, because of the +1 box-per-column term at coarse scales. The scale range was then moved to 2⁻⁴…2⁻⁹, using synthetic signals only.

### Information theory

- **Permutation entropy:** white noise 0.997 (delays 1 and 4); monotone ramp exactly 0.
- **Agreement with brute-force code:** SampEn, ApEn, RR, DET and LAM are identical to direct double-loop implementations, including explicit run-length line counting (difference 0.0).
- **SampEn of white noise:** 2.180, against the theoretical −ln erf(0.1) = 2.185.
- **RR of white noise:** 0.01269, against erf(0.1)² = 0.01265.
- **Exactly periodic sine** (period 16, r = 0.05·std): SampEn = 0, ApEn = −6e-6, DET = 1.
- **First failure (not a bug).** With r = 0.2·std, the period-16 sine gave SampEn 0.23 and DET 0.80. Near the extrema, symmetric sample pairs match for 2 samples but not 3. The value → 0 as sampling gets denser: period 256 gives SampEn 0.057 and DET 0.9998. The check was changed to the exactly periodic case.
- **Histogram entropies:** uniform data gives Shannon 1.000 and Rényi 1.000; a constant gives 0.

### S-transform

- **Complex tone** at bin 37, amplitude 1.7: the peak voice is 37 at every time sample, |S| = 1.7000. All voices match the closed form A·exp(−2π²Δ²/n²) to 8e-15.
- **First failure (a bug in the check, not in the transform).** The first version of this check failed at 7e-8, because the closed form must use the DFT-periodic (wrapped) frequency difference Δ. With Δ wrapped it passes.
- **Real tone:** peaks at ±100.
- **Time integration:** Σ_k S[k, n] = X[n] to 2e-14.
- **Chirp:** the peak voice tracks the instantaneous frequency, with a median error of 0.27 bins.

### CvM and AD

- **Statistics match scipy.** CvM equals `cramervonmises_2samp` to 2e-14. AD, standardised with the Scholz–Stephens variance, equals `anderson_ksamp` (variants `continuous` and `right`) to 1e-14.
- **Null calibration with iid samples** (n = 120, m = 25,600):
  - mean T = 0.168, against theory 0.1667; rejection rate at 0.461 is 4.9 %.
  - mean A² = 1.014, against theory 1; rejection rate at 2.492 is 4.6 %.
- **With the classifier's per-frame normalisation:** mean T = 0.081 and mean A² = 0.56, with about 0 % rejections. The test becomes conservative (Lilliefors effect), exactly as for KS.
- **Synthetic accuracy** (random phase, AWGN):

  | symbols | SNR (dB) | KS (ap) | CvM (ap) | AD (ap) | oracle |
  |---|---|---|---|---|---|
  | 120 | 10 | 0.73 | 0.72 | 0.75 | 0.94 |
  | 1000 | 25 | 1.00 | 1.00 | 1.00 | 1.00 |

  The three tests are within sampling noise of each other.

## Results

### RadioML 7 classes (45,500 test frames)

| HGB features | all SNR | SNR≥10 | 85 % at | Δ vs reference (McNemar z) |
|---|---|---|---|---|
| base19 (baseline) | 71.4 % | 99.95 % | 2 dB | – |
| base19+wav24 | 72.4 % | 100.0 % | 0 dB | – |
| bis30 only | 63.9 % | 93.2 % | 6 dB | |
| fd22 only | 67.5 % | 97.7 % | 2 dB | |
| info24 only | 67.8 % | 100.0 % | 2 dB | |
| st23 only | 58.4 % | 77.3 % | never | |
| all 4 new (99) only | 72.6 % | 100.0 % | 2 dB | |
| base19+bis30 | 71.9 % | 100.0 % | 2 dB | +0.50 vs base19 (z 3.6) |
| base19+fd22 | 72.3 % | 100.0 % | 0 dB | +0.88 (z 6.2) |
| base19+info24 | 71.7 % | 100.0 % | 2 dB | +0.24 (z 1.9) |
| base19+st23 | 72.7 % | 100.0 % | 2 dB | +1.26 (z 8.7) |
| base19+all4 | 73.2 % | 100.0 % | 0 dB | +1.78 (z 11.8) |
| base19+wav24+bis30 | 73.1 % | 100.0 % | 0 dB | +0.68 vs base19+wav24 (z 5.3) |
| base19+wav24+fd22 | 72.6 % | 100.0 % | 0 dB | +0.21 (z 1.6) |
| base19+wav24+info24 | 72.4 % | 100.0 % | 0 dB | −0.04 (z −0.3) |
| base19+wav24+st23 | 73.2 % | 100.0 % | 0 dB | +0.84 (z 6.2) |
| **base19+wav24+all4** | **73.4 %** | 100.0 % | 0 dB | **+0.98 (z 7.0)** |

With 7 classes, SNR≥10 is already saturated. The base19+wav24+all4 gain is concentrated at −10 / −8 / −6 dB, with +6.8 / +9.2 / +5.3 points (about 1.1 points SE per SNR). It is about 0 above 0 dB.

### RadioML 24 classes (31,200 test frames; no method reaches 85 % at any SNR)

| HGB features | all SNR | SNR≥10 | Δ vs reference (McNemar z) |
|---|---|---|---|
| base19 (baseline) | 39.2 % | 62.8 % | – |
| base19+wav24 | 41.3 % | 64.6 % | – |
| bis30 only | 28.1 % | 45.0 % | |
| fd22 only | 33.2 % | 51.7 % | |
| info24 only | 38.9 % | 65.2 % | |
| st23 only | 24.4 % | 34.0 % | |
| all 4 new only | 41.5 % | 66.4 % | |
| base19+bis30 | 39.5 % | 63.0 % | +0.29 vs base19 (z 1.4) |
| base19+fd22 | 40.8 % | 64.1 % | +1.50 (z 6.8) |
| base19+info24 | 42.3 % | 69.7 % | +3.06 (z 13.5) |
| base19+st23 | 40.4 % | 62.9 % | +1.11 (z 5.1) |
| base19+all4 | 43.7 % | 69.8 % | +4.46 (z 18.2) |
| base19+wav24+bis30 | 41.2 % | 64.3 % | −0.02 vs base19+wav24 (z −0.1) |
| base19+wav24+fd22 | 41.6 % | 65.1 % | +0.35 (z 1.7) |
| base19+wav24+info24 | 43.8 % | 70.3 % | +2.56 (z 11.6) |
| base19+wav24+st23 | 41.6 % | 64.9 % | +0.38 (z 1.9) |
| **base19+wav24+all4** | **44.1 %** | **70.1 %** | **+2.88 (z 12.3)** |

**Where the info24 gain comes from (24 classes, `diag_info_ablation_24.json`).** Each subset is added to base19+wav24:

| info subset | Δ all | Δ SNR≥10 | McNemar z | ms/frame |
|---|---|---|---|---|
| histogram entropies (8) | +2.39 | +5.1 | 11.3 | 0.4 |
| permutation entropies (6) | +0.24 | +0.7 | 1.3 | |
| SampEn/ApEn (4) | +0.74 | +1.5 | 3.9 | |
| RQA (6) | +0.88 | +1.7 | 4.6 | |
| **histogram + permutation (14, cheap)** | **+2.54** | **+5.3** | 11.8 | **1.5** |
| full info24 | +2.56 | +5.7 | 11.6 | 36.7 |

So almost the entire gain comes from the cheap histogram entropies (amplitude, phase, instantaneous frequency, joint I/Q). The O(N²) SampEn/ApEn/RQA part adds nothing on top of them.

**Per-class change at SNR≥10, base19+wav24 → +info24.**

| class | change (points) |
|---|---|
| 8PSK | +37 |
| 16QAM | +15 |
| 16PSK | +13 |
| 64QAM | +10 |
| 32PSK, 16APSK, 32APSK | +9 each |
| analog classes and GMSK/OQPSK | about 0 |

### 5 digital classes, symbol rate known (4,800 test frames each)

| method | RadioML all | RadioML SNR≥10 | TorchSig all | TorchSig SNR≥10 | ms/frame |
|---|---|---|---|---|---|
| KS (chosen: ap / ap) | 60.2 % | 76.1 % | 66.2 % | 82.2 % | 69–98 |
| **CvM** (chosen: ap / iqap) | 60.3 % | 76.2 % | 67.4 % | 80.4 % | 70–98 |
| **AD** (chosen: ap / ap) | **55.9 %** | **62.0 %** | 67.0 % | 81.8 % | 21–28 |
| features + HGB, base19 (quoted) | 66.2 % | 87.1 % | 62.1 % | 85.4 % | 1–2 |
| EM constellation ML (quoted) | 71.8 % | 95.2 % | 77.2 % | 95.1 % | 700–800 |
| ALRT / HLRT, sps known (quoted) | 70.4 / 70.6 % | 93.3 / 93.4 % | 76.3 / 76.1 % | 95.6 / 95.4 % | ~40 |

**Paired tests against KS on the same frames:**

| comparison | RadioML | TorchSig |
|---|---|---|
| CvM vs KS | z = +0.3 | z = +1.9 |
| AD vs KS | z = −7.1 | z = +1.3 |

No CvM/AD row reaches 85 % at any SNR.

**AD on RadioML: diagnosis.** At SNR≥10, AD classifies only **32 of 360 RadioML BPSK frames** correctly; 193 go to QPSK and 135 to 8PSK. KS gets 313 of 360. The other classes are similar to KS.

The cause: in the phase ECDF, the BPSK cluster at ±π (also present for 8PSK) is split between the two ends of the (−π, π] range. AD's 1/(F(1−F)) weight is largest exactly there. On top of that, RadioML BPSK is much narrower in phase than the AWGN reference (see the extended README). AD amplifies this mismatch.

**Post-hoc check (`diag_phase_cut.py`, designed after seeing this failure on the test set, so treat it as a diagnosis, not a result).** The phase is measured from a branch cut placed in each constellation's widest angular gap, and the component set is re-chosen on training frames:

| | RadioML all | RadioML SNR≥10 | RadioML BPSK at SNR≥10 | TorchSig all | TorchSig SNR≥10 |
|---|---|---|---|---|---|
| AD with the cut | 60.9 % | 77.6 % | 96 % | 67.4 % | 80.8 % |
| KS with the cut | 59.8 % | 75.9 % | | 66.8 % | 80.1 % |
| CvM with the cut | 58.4 % | 74.8 % | | 66.7 % | 79.7 % |

So the original AD failure is an interaction between AD's tail weighting and the phase-component design inherited from the KS framework. It is not a bug in the AD statistic, which matches scipy and is calibrated. Even when repaired, AD is no better than KS beyond noise.

## Interpretation and honest caveats

- **Information-theoretic features are the only family with a clearly useful gain, and only on 24 classes.** The gain is +2.6 points over all SNRs and +5.7 at SNR≥10 over base19+wav24, at z ≈ 12. It comes almost entirely from the 8 histogram entropies, which cost 0.4 ms/frame. The 36 ms/frame full family is not worth its cost: SampEn, ApEn and RQA are O(N²) and redundant once the histograms are present.
  - **Recommendation:** add the histogram (+ permutation) entropies at 0.4–1.5 ms/frame and drop the rest.
  - With 7 classes, info24 adds nothing on top of base19+wav24 (−0.04, z −0.3).
- **S-transform and bispectrum help only with 7 classes, and only at low SNR.** They give +0.84 and +0.68 points over base19+wav24, at z 5–6, concentrated at −10…−6 dB. With 24 classes both are within noise of zero (z 1.9 and −0.1).
  - Alone they are weak (st23 58 % / 24 %). The S-transform statistics mostly re-describe the spectrum shape and the amplitude envelope, which base19 and wav24 already contain.
  - The bispectrum features on 1024-sample frames sit on a large estimator floor: K = 16 segments, Gaussian mean b² = 0.064.
- **Fractal dimensions** give +0.9 / +1.5 points over base19 alone (z 6–7). Over base19+wav24 they give only +0.2 / +0.35 (z < 2), i.e. **no evidence of a gain once the wavelet features are present.** Their information overlaps with the wavelet transition and roughness statistics.
- **All four families together** give the best numbers:
  - 7 classes: 73.4 %, +1.0 points over base19+wav24 and +2.0 over base19.
  - 24 classes: 44.1 % / 70.1 % at SNR≥10, +2.9 points over base19+wav24 and +4.9 over base19.

  On 24 classes this is not better than base19+wav24+info24 alone (43.8 % / 70.3 %). The 7-class SNR≥10 accuracy was already saturated at 99.95 %. The remaining 24-class errors are high-order QAM/APSK confusions, and 70 % at SNR≥10 is still far from 85 %.
- **CvM and AD are not improvements over KS.**
  - CvM equals KS within noise on both datasets.
  - AD is worse on RadioML (−4.3 / −14.1 points) because of the phase branch-cut interaction explained above, and equal on TorchSig.
  - All three distribution tests stay well below EM, ALRT/HLRT and features+HGB. The reasons are those already documented for KS: about 120 symbols per frame, per-hypothesis nuisance estimates, and an AWGN reference that does not match RadioML's distortion.
  - The one practical difference is cost: AD is about 3× faster than KS/CvM in this implementation, because it needs only one sorted merge.
- **Timing.** All timings were taken on a heavily shared machine. The extraction ms/frame stored in `bench_features_*.json` and `features_new_*.npz` are parallel-worker numbers and inflated 3–8×. Use `timing_single_process.json` for costs. Even those are at most about 1.7× high (base19 0.87 against 0.50 ms in the original cache).
- **Feature definitions are our own concrete choices** within the cited families: signals, statistics, bin ranges, voices and segment length. They were fixed before benchmarking and were not tuned on RadioML. Other reasonable choices could give somewhat different numbers.
