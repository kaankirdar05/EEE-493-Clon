# Extended classical AMC methods

This folder adds five classical or statistical method families to the 19-feature + boosted-trees baseline in `classical_amc/`. No file outside this folder was changed. Every method passed a sanity check with a known answer before its benchmark numbers were produced. All numbers below come from the JSON files in `results/`.

| file | content |
|---|---|
| `common.py` | paths, the standard splits, metrics, overview plot |
| `cumulants_ext.py` | general moment-to-cumulant formula, extended cumulants, cyclic moments and cumulants (17 features) |
| `wavelet.py` | Haar wavelet transform features (24 features) |
| `symbol_classifiers.py` | KS goodness-of-fit classifier, EM / Gaussian-mixture constellation classifier, coherent oracle ML (sanity checks only) |
| `sanity_features.py`, `sanity_symbol.py` | known-answer checks; results in `results/sanity_*.json` |
| `extract_radioml_features.py` | base19 + ext17 + wav24 features on the cached frames; output `results/features_{7,24}.npz` |
| `bench_features.py` | HGB / GNB / QDA / GMM on the feature sets; output `results/bench_features_{7,24}.{json,png}` |
| `bench_symbol.py` | KS and EM on 5 digital classes; output `results/bench_symbol_{radioml,torchsig}.{json,png}` |
| `make_tables.py` | rebuilds `results/tables.md` and the feature plots from the JSON files |

Run all scripts from this folder with the project venv:

```
python sanity_features.py; python sanity_symbol.py
python extract_radioml_features.py --set 7; python extract_radioml_features.py --set 24
python bench_features.py --set 7; python bench_features.py --set 24
python bench_symbol.py --source radioml; python bench_symbol.py --source torchsig
python make_tables.py
```

## Data, splits and protocol

- **Feature benchmarks (RadioML 2018.01A).** These use exactly the frames of the cached baseline: 7 classes × 26 SNRs × 1000 frames, and 24 classes × 26 SNRs × 200 frames, with seed 0 and `order="hdf5"`. No per-group subsampling was needed. The re-extracted labels and SNRs are identical to the cache, and the recomputed 19 baseline features differ from the cached ones by exactly 0.0. The split is the standard one (25 % test, stratified by class × SNR, `random_state=0`). The baseline is reproduced exactly: HGB on base19 gives 71.4 % / 99.95 % (7 classes) and 39.2 % / 62.8 % (24 classes).
- **Symbol-domain benchmarks.** These use the frames and the 50/50 split of `practical_alrt.py`, through its own `load()`: RadioML with per_group 120, SNR −10…20 dB and seed 5; and TorchSig from `D:\data\alrt_torchsig.npz`. The symbol rate is known in both cases. The ALRT/HLRT rows are copied from `results/practical_alrt/*.json`, which were computed on the same test frames. features+HGB was re-run here and reproduces those files (66.2 % RadioML, 62.1 % TorchSig).
- **Tuning on training data only.** The KS component set, the generative models' feature transform and QDA's `reg_param` were all chosen by accuracy on the training split (3-fold CV for the generative models). The GMM component count was chosen by BIC on training data.
- **Timing caveat.** The timings were measured on a machine shared with several other heavy jobs, so absolute values are inflated. The base19 features took 1.2–1.5 ms per frame here, against 0.50 ms in the original cache. Use the timings only for relative comparison. For the feature classifiers, ms/frame means feature extraction plus prediction. For KS and EM it means `to_symbols` plus classification, measured per frame inside a worker process.

## Methods

1. **Extended cumulants (`ext17`, cumulant part).** C61, C62, C63, C81, C82, C83 and C84 (magnitude, or real part for C63 and C84). They are computed with the general Leonov–Shiryaev partition formula: all 4140 partitions of an 8-set, collapsed to (#x, #x*) block types. The frame is first normalised to zero mean and unit power, so C21 = 1. Three ratios that do not depend on power or noise are added: |C40|/|C42|, |C63|/|C42|^1.5 and |C84|/|C42|^2.
   **Cyclic part.** The peaks over cycle frequency α of |⟨x²e^{−j2παn}⟩|, |⟨x⁴…⟩| and |⟨x⁸…⟩|. The peaks are found with a zero-padded FFT and then a fine local search. Also included:
   - The 4th-order cyclic **cumulant** at the x⁴ peak α₄, |M40(α₄) − 3·M20(α₄/2)²|. This equals |C40| after CFO correction.
   - The conjugate cyclic autocorrelation at the strongest non-zero conjugate cycle frequency in [1/32, 1/2], which is the symbol rate for linear modulations. It is taken at lag 0 and at a half-symbol lag, and the cycle frequency itself is also a feature.
2. **Generative classifiers on the features.** Gaussian naive Bayes, regularised QDA, and a per-class full-covariance GaussianMixture with K ∈ {1, 2, 4, 8, 16} chosen by BIC, used as a Bayes classifier with the class priors. The features are standardised or Yeo-Johnson transformed (chosen by CV on the training split).
3. **KS goodness-of-fit (Wang & Wang 2010 style).** Each frame goes through `to_symbols` (symbol rate known). Then, for each candidate class:
   - the SNR is estimated with M2M4 using that class's kurtosis;
   - the phase is estimated with the M-th power method for that class, and the symbols are de-rotated;
   - KS distances are computed for I, Q, |z| and arg z against reference CDFs. The references are 200 simulated frames × 128 symbols of that class at the nearest 1-dB grid SNR, passed through the same phase estimator.

   The decision is the class with the smallest summed distance over a component set chosen on the training split. Amplitude + phase ("ap") was chosen on both datasets.
4. **EM constellation classifier.** For each candidate, z ~ (1/M) Σ CN(g·c_m, σ²), where the complex gain g (amplitude and phase) and σ² are fitted by EM with a closed-form M-step and 4 phase restarts. Every candidate has the same 3 parameters, so the maximum-likelihood decision is also the BIC decision. The free-GMM/BIC variant was not implemented: with 120 symbols per RadioML frame, BIC can never support K = 64.
5. **Haar wavelet features (`wav24`, after Ho, Prakasam & Chan).** The Haar CWT is computed at scales a = 2, 4, 8 and 16. For each scale the features are:
   - mean |HWT|;
   - the normalised variance of |HWT|, both raw and after a (2a+1)-sample median filter that removes the transition spikes;
   - the rate of transition peaks;
   - the median-filtered variance of |HWT| of x/|x| (separates FSK from PSK);
   - mean |HWT| of |x| (amplitude steps).

## Sanity checks (all pass: 7/7 in `sanity_features.json`, 5/5 in `sanity_symbol.json`)

**Cumulant formula.**
- The general formula matches the hard-coded C20, C40, C41, C42, C60 and C80 of `amc/features.py` to a relative error of 1e-15. This was tested on frames whose odd moments are exactly zero, where those hand-written formulas are exact.
- The exact theoretical values match the literature (Swami & Sadler 2000) to within 2e-4 relative. For example, C63 = 16 / 4 / 4 / 2.08 / 1.797 and C80 = −272 / −34 / 1 / −13.98 / −11.50 for BPSK / QPSK / 8PSK / 16QAM / 64QAM.
- **New cumulants.** These were estimated on 8 × 200k noiseless symbols for BPSK, QPSK, 8PSK, 16QAM, 64QAM, 4ASK and 16PSK. Every estimate is within 4.3 standard errors of the theory, with a maximum relative error of 1.05 % where the theoretical value is non-zero. The full table is in the JSON file.
- On Gaussian noise, all cumulants above order 2 come out at about 0.

**Cyclic features.** Test case: a carrier offset of 0.0123 cycles/sample.
- Plain |C40| collapses to 0.007–0.05.
- cyc_C40 gives 1.996 / 0.995 / 0.089 / 0.672, against the theoretical 2 / 1 / 0 / 0.68 for BPSK / QPSK / 8PSK / 16QAM. The 0.089 for 8PSK is the positive bias of picking a maximum out of noise.
- The BPSK x² line sits at 2f₀ with |M20| = 0.999.
- The conjugate cycle frequency of RRC(0.35) QPSK falls within one frame bin (1/L) of 1/sps for 100 / 100 / 95 / 82 % of frames at sps = 4 / 5 / 8 / 10. The misses are gross picks of the low-frequency |x|² continuum, not a bias: the median error is 0.02–0.04 bin.

**Haar wavelet.**
- |HWT| of a tone matches the closed form (2/√a)|sin(ωa/4)|·|sin(ωa/4)/sin(ω/2)| to 1e-14.
- For rectangular QPSK, the number of non-zero |HWT| runs equals the number of symbol transitions (98 = 98), and the median-filtered variance is 0.
- The median-filtered variance of x/|x| is 0.0007 for PSK with a carrier offset and 0.104 for 2-FSK.
- The check initially failed with an (a+1)-sample median window, because a transition spike is a−1 samples wide. The window was widened to 2a+1. This was a real bug in the first version.

**KS.**
- The statistic is identical to `scipy.stats.ks_2samp` (difference 2e-16).
- **Null calibration with iid samples.** The mean of √(nm/(n+m))·D is 0.856, against 0.869 in theory, and the rejection rate at 5 % is 4.65 %.
- **With per-frame power normalisation**, as used by the classifier, the mean drops to 0.68 and the rejection rate to 0.2 %. This is the expected Lilliefors effect of estimating parameters from the same frame: the test becomes conservative.
- **Bug found and fixed.** The first version used only I, Q and |z|, and the synthetic check showed BPSK being called QPSK at 25 dB. The cause is that the QPSK phase estimator rotates BPSK to ±45°, where its I and Q *marginals* equal those of QPSK. Adding arg z fixed this.

**EM.**
- At 15 dB with 1000 symbols, the median relative amplitude error is 0.03–0.7 %, the phase error 0.002–0.008 rad and the noise-variance error 1.5–8.6 %.
- The log-likelihood never decreases across EM iterations.

**Synthetic accuracy compared with the coherent oracle ML.** The oracle knows the true gain, phase and noise variance. Neither blind method ever beats the oracle.

| symbols | SNR (dB) | oracle | EM | KS (ap) |
|---|---|---|---|---|
| 120 | 10 | 0.95 | 0.89 | 0.78 |
| 120 | 25 | 1.00 | 1.00 | 0.95 |
| 1000 | 10 | 1.00 | 0.98 | 0.77 |
| 1000 | 25 | 1.00 | 1.00 | 1.00 |

## Results (test split)

### RadioML 7 classes (45,500 test frames)

| method / features | all SNR | SNR≥10 | first SNR ≥ 85 % | ms/frame |
|---|---|---|---|---|
| HGB / base19 (baseline) | 71.4 % | 99.95 % | 2 dB | 1.2 |
| HGB / ext17 only | 67.6 % | 99.8 % | 2 dB | 2.2 |
| HGB / wav24 only | 69.0 % | 98.7 % | 2 dB | 4.0 |
| HGB / base19+ext17 | 71.4 % | 100.0 % | 2 dB | 3.4 |
| HGB / base19+wav24 | 72.4 % | 100.0 % | 0 dB | 5.2 |
| HGB / base19+ext17+wav24 | **72.6 %** | 100.0 % | 0 dB | 7.5 |
| GNB / base19 | 64.0 % | 98.1 % | 4 dB | 1.2 |
| QDA / base19 | 70.1 % | 99.9 % | 2 dB | 1.2 |
| GMM-Bayes / base19 | 70.8 % | 100.0 % | 2 dB | 1.3 |
| GNB / base19+ext17 | 63.1 % | 97.0 % | 4 dB | 3.4 |
| QDA / base19+ext17 | 70.1 % | 100.0 % | 2 dB | 3.4 |
| GMM-Bayes / base19+ext17 | 70.4 % | 100.0 % | 2 dB | 3.5 |
| GNB / all 60 | 65.1 % | 99.2 % | 4 dB | 7.4 |
| QDA / all 60 | 70.9 % | 100.0 % | 0 dB | 7.5 |
| GMM-Bayes / all 60 | 71.6 % | 100.0 % | 2 dB | 7.5 |

### RadioML 24 classes (31,200 test frames; no method reaches 85 % at any SNR)

| method / features | all SNR | SNR≥10 | ms/frame |
|---|---|---|---|
| HGB / base19 (baseline) | 39.2 % | 62.8 % | 1.5 |
| HGB / ext17 only | 35.5 % | 58.8 % | 2.9 |
| HGB / wav24 only | 35.7 % | 55.4 % | 4.9 |
| HGB / base19+ext17 | 39.7 % | 63.5 % | 4.3 |
| HGB / base19+wav24 | 41.3 % | 64.6 % | 6.4 |
| HGB / base19+ext17+wav24 | **41.5 %** | **64.6 %** | 9.2 |
| GNB / base19 | 28.4 % | 48.6 % | 1.5 |
| QDA / base19 | 35.5 % | 58.6 % | 1.5 |
| GMM-Bayes / base19 | 37.5 % | 60.4 % | 1.5 |
| GNB / base19+ext17 | 27.8 % | 46.9 % | 4.3 |
| QDA / base19+ext17 | 36.3 % | 59.7 % | 4.3 |
| GMM-Bayes / base19+ext17 | 34.2 % | 54.7 % | 4.4 |
| GNB / all 60 | 30.8 % | 51.8 % | 9.3 |
| QDA / all 60 | 38.5 % | 63.0 % | 9.3 |
| GMM-Bayes / all 60 | 37.9 % | 61.2 % | 9.3 |

### 5 digital classes, symbol rate known (4,800 test frames each)

| method | RadioML all | RadioML SNR≥10 | RadioML 85 % at | TorchSig all | TorchSig SNR≥10 | TorchSig 85 % at | ms/frame |
|---|---|---|---|---|---|---|---|
| features + HGB (base19) | 66.2 % | 87.1 % | 6 dB | 62.1 % | 85.4 % | 16 dB | 1–2 |
| features + HGB (base19+ext17) | 66.7 % | 87.3 % | 6 dB | 63.9 % | 85.2 % | 16 dB | 2–7 |
| KS, amplitude + phase (chosen on train) | 60.2 % | 76.1 % | never | 66.2 % | 82.2 % | never | ~63 |
| KS, I+Q+amp+phase (not chosen; reference) | 53.1 % | 60.7 % | never | 67.0 % | 80.0 % | never | ~63 |
| **EM constellation ML** | **71.8 %** | **95.2 %** | **2 dB** | **77.2 %** | 95.1 % | **2 dB** | 700–800 |
| ALRT, sps known (practical_alrt.py) | 70.4 % | 93.3 % | 4 dB | 76.3 % | 95.6 % | 4 dB | ~40 |
| HLRT, sps known (practical_alrt.py) | 70.6 % | 93.4 % | 2 dB | 76.1 % | 95.4 % | 4 dB | ~40 |

The plots are `results/bench_features_7.png`, `bench_features_24.png`, `bench_symbol_radioml.png` and `bench_symbol_torchsig.png`.

## Interpretation and honest caveats

- **Extended cumulants and cyclic features add almost nothing on top of the 19 features.** The gain is +0.0 points with 7 classes and +0.5 points with 24 classes, and the features are weaker on their own. The information is largely redundant: |C40|, |C42|, |C60|, |C80| and the power-law lines already capture what cumulants can, and RadioML's carrier offsets are small enough that the CFO-robust cyclic C40 is not needed. 8th-order estimates from 1024 oversampled samples are also very noisy.
- **The Haar wavelet features give the largest gain: +1.0 / +2.1 points (7 / 24 classes).** Combined with ext17 the gains are +1.2 / +2.3 points. With 31k–45k test frames the standard error is about 0.2–0.3 points, so these gains are larger than the split noise. They are still modest, and on 24 classes the SNR≥10 accuracy (64.6 %) remains far from 85 %. The remaining confusions are among the high-order QAM/APSK classes, which none of these statistics resolve.
- **The generative classifiers are below HGB on every feature set.** The order is GNB < QDA < GMM ≤ HGB. This is a genuine limitation, not a bug: the features are heavy-tailed and strongly dependent, and the SNR mixture makes each class multimodal.
  - On 24 classes, GMM on base19+ext17 (34.2 %) is worse than on base19 (37.5 %). A post-hoc diagnostic, *not* used for model selection, confirms over-fitting: 43 % train vs 34–37 % test accuracy. The Yeo-Johnson transform picked by QDA's CV is also worse for GMM than plain standardisation (34.2 % vs 36.7 % test). Density estimation in 36 dimensions with about 3,900 frames per class is simply hard.
  - The reported numbers keep the transform chosen on the training split.
- **EM is on par with ALRT/HLRT,** which is expected because it is essentially the HLRT with continuous ML nuisance estimates instead of grids. It is +1.4 / +0.9 points over all SNRs and +1.9 / −0.5 points at SNR≥10 on RadioML / TorchSig; at ~0.65 points standard error, only the RadioML all-SNR difference is about 2 SE. It is the best method here on symbol-domain frames, beating features+HGB by +5.6 / +15.1 points. It costs about 20× the ALRT time in this unoptimised NumPy implementation (5 classes × 4 restarts × up to 100 iterations).
- **KS is the weakest symbol-domain method, and this is a limitation, not a bug.** Its implementation passes every check: the statistic is exact, the null distribution is calibrated, and it reaches 100 % on synthetic data at 25 dB with 1000 symbols. On real frames it plateaus at about 75–80 %, for three reasons:
  1. **Low power with about 120 symbols.** Even on ideal synthetic frames it reaches only 78 % at 10 dB, against 89 % for EM.
  2. **Nuisance parameters estimated per hypothesis.** The SNR comes from M2M4. A 64QAM hypothesis at a higher fitted SNR mimics the marginals of 16QAM. With 1000 synthetic symbols at 10 dB, the 3 PSK classes are 100 % correct but 16QAM/64QAM are only 58 % / 75 %.
  3. **Reference-model mismatch.** The reference assumes circular AWGN. On RadioML BPSK at SNR ≥ 14 dB, the residual distortion after `to_symbols` is almost entirely in-phase: in-phase variance 0.039 against quadrature variance below 1e-4. The phase ECDF is therefore much narrower than the reference, and 24 of 116 such BPSK frames went to 8PSK. A shape-based test penalises this mismatch; a likelihood ratio is more tolerant of it.
- **Symbol rate.** KS and EM were run only with the symbol rate known. They inherit the blind-symbol-rate degradation documented for ALRT in `practical_alrt.py` (93 → 81 % at SNR ≥ 10 on RadioML).
- **Selection and test frames.** Only the existing feature and frame sets were used. The KS component set is the only choice made on symbol-domain data, and it was made on a 1,500-frame training subsample. The other KS component sets are listed in the JSON files for transparency, but were not used to select anything.
