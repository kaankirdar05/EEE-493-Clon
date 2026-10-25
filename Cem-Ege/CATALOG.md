# Catalogue of classical / statistical methods — coverage of the baseline

Every method family named in the standard surveys, and where it was tested. A literally exhaustive list
does not exist; this catalogue is built from the taxonomies of:

- **Spectrum sensing / detection:** Axell, Leus, Larsson, *Overview of spectrum sensing for cognitive radio*
  (energy, fundamental limits / SNR wall, feature detection: cyclostationarity, autocorrelation, covariance
  eigenvalues, cooperative detection); Ahmad, *A survey of wideband spectrum sensing algorithms* (arXiv 2001.02574):
  coherent, energy, cyclostationary, covariance detectors, Nyquist multiband (filter bank, wavelet edge),
  compressive sub-Nyquist (multicoset, modulated wideband converter), alias-free sampling, cooperative sensing.
- **Modulation classification:** Dobre et al., *Survey of AMC techniques: classical approaches and new trends*
  (IET Commun. 2007): likelihood-based and feature-based; Zhu & Nandi, *Automatic Modulation Classification*
  (Wiley 2015): likelihood-based, distribution-test-based, feature-based, machine-learning-assisted, blind.
- **Parameter estimation:** the classical frequency-estimation (periodogram/ML, interpolated FFT, Kay, subspace),
  SNR-estimation (Pauluzzi & Beaulieu 2000) and cyclostationary symbol-rate literature.
- **Time–frequency and sequential analysis:** STFT, Wigner–Ville family, fractional Fourier / Radon–Wigner,
  S-transform, Hilbert–Huang; CUSUM, Page–Hinkley, Shiryaev–Roberts, GLR, Bayesian online change-point.

Status: **T** = tested (result and file given) · **N/A** = not applicable to this project, with the reason.
Results marked **independently re-run** were reproduced from scratch by a second implementation or a fresh run, not just read from the original output.
All tests are on synthetic data (RadioML 2018.01A, TorchSig, or purpose-built signals with known answers).

## A. Detection

| Family | Method | Status | Where / result |
|---|---|---|---|
| Energy | Energy / radiometer (several noise-floor models) | T | BASELINE §1: best detector, box F1 67 % clean / 53 % impaired |
| Energy | Multitaper (Thomson) energy | T | §1 |
| Energy | FCME, LAD (double threshold) | T | §1 |
| Energy | PSD occupancy (frequency only) | T | §1 |
| Energy | SNR wall under noise uncertainty | T | §4: energy Pfa 0.43 instead of 0.01 at ±1 dB uncertainty |
| CFAR | CA, GO, SO, OS, 2-D CFAR | T | §1, tuned guard/training sizes |
| Coherent | Matched filter / preamble correlator (plain, segmented for frequency offset) | T | §4: 6 dB better than energy, matches theory |
| Feature: cyclostationary | 2nd-order cyclic lines | T | §1 (`cyclo`) |
| Feature: cyclostationary | Spectral correlation density (FAM/SSCA) | T | `detection_extended/`: FAM-based SCD per block; sanity: BPSK features at α = Rs and 2fc. Box F1 16.2 % impaired (best block detector, lowest Pfa 5.4e-2) / 24.2 % clean; recall 0–5 dB only 1–5 %; ~4 s per scene |
| Feature: autocorrelation | OFDM cyclic-prefix detector (known and blind FFT size) | T | §4: robust to noise uncertainty |
| Feature: autocorrelation | Generic normalised-autocorrelation detector | T | `detection_extended/`: box F1 7.3 % impaired (Pfa 0.34) / 38.7 % clean. Impaired failure = receiver filter colours the noise (whiteness test fires); prewhitening helps impaired, hurts clean |
| Feature: covariance | Max/min eigenvalue (MME) | T | §1 |
| Feature: covariance | Covariance absolute value (CAV) | T | `detection_extended/`: box F1 6.6 % impaired (Pfa 0.32) / 36.8 % clean; same coloured-noise limitation. **Independent re-calibration on fresh noise hits target (1e-3 → 1.0e-3); subset re-run consistent** |
| Feature: covariance | Covariance-index detectors: Gini (GID), Gerschgorin (GRCR), Pietra–Ricci (PRIDe), Theil (TID), Atkinson (AID) | T | `extra_methods/cov_index_detectors.py` (Sensors 2026, doi 10.3390/s26082557): fresh-noise Pfa 0.8–1.2e-2 for a 1e-2 target, correlated signal raises every statistic; box F1 6.4–7.3 % impaired (Pfa 0.30–0.32) / 35.4–36.9 % clean (Pfa 0.07), the same coloured-noise limit as CAV and MME; GRCR and PRIDe give near-identical results |
| Distribution | Anderson–Darling goodness of fit | T | §1 |
| Distribution | Spectral kurtosis | T | §1 |
| Information | Entropy-based detection | T | `detection_extended/`: spectral entropy F1 7.5 % / 39.1 %; histogram (Zhang) entropy 6.4 % / 12.5 %. **Independently re-run: consistent** |
| Multiband | Filter-bank / channelised detection | T | used inside the block detectors (16 sub-bands) |
| Multiband | Wavelet band-edge detection (Tian & Giannakis) | T | §4: finds physical edges; worse vs TorchSig labels (definition mismatch) |
| Time–frequency | STFT spectrogram | T | basis of all WP4 detectors |
| Time–frequency | S-transform (Stockwell) | T | `extra_methods/stransform_detect.py`: no gain for 64-sample bursts (Pd 90 % at 0 dB vs −4 dB for a burst-matched STFT); its window shrinks with frequency |
| Time–frequency | Wigner–Ville / smoothed pseudo-WVD, fractional Fourier / Radon–Wigner (chirps) | T | `detection_extended/` (synthetic chirps, 4,096 samples, Pfa 1e-2), SNR for Pd 50 %: radiometer −14.6, spectrogram cell −13.4, spectrogram line sum −20.7, PWVD −21.7, SPWVD −21.4, **FrFT/dechirp −23.1 dB** (theory ≈ −23.7 dB: checked); WVD energy 99 % on the IF line |
| Time–frequency | Hough transform on the spectrogram (chirps) | T | §4: energy + Hough raises chirp recall to 77 % / 84 % |
| Sequential | CUSUM edge refinement | T | §4 |
| Sequential | CUSUM, Page–Hinkley, Shiryaev–Roberts, GLR, Bayesian online change-point (burst onset) | T | `detection_extended/` (burst onset, ARL0 = 1000 frames, −15 dB): Shiryaev–Roberts 70.9 and CUSUM 72.1 frames mean delay (oracle-tuned), GLR 103.5, Bayesian online 110.8, mis-tuned CUSUM 168.8 (9.5 % miss), Page–Hinkley 170.5 (29.5 % miss). CUSUM ARL0 and delay within 3.7 % / 1.0 % of Siegmund's theory |
| Model order | MDL / AIC number of signals | T | §4: MDL consistent, AIC over-counts |
| Sub-Nyquist | Compressive sensing (multicoset, MWC), alias-free sampling | N/A | Our SDRs sample the whole band at Nyquist; these methods matter only when the ADC cannot. |
| Cooperative | Soft / hard fusion across receivers | N/A | Single receiver in this project (possible future work with several SDRs). |
| Multi-antenna | Array GLRT / blind multi-antenna detectors, DOA | N/A | Single-channel receiver. |

## B. Identification (modulation classification)

| Family | Method | Status | Where / result |
|---|---|---|---|
| Likelihood | ALRT, HLRT, ideal conditions | T | §2: 85 % at 8 dB (upper bound) |
| Likelihood | ALRT, HLRT, practical (all parameters blind) | T | §2: beats features when the symbol rate is right |
| Likelihood | Quasi-HLRT / GLRT variants | T | HLRT with noise level maximised (practical_alrt.py) is the GLRT over phase and noise |
| Distribution test | Kolmogorov–Smirnov classifier | T | `classical_amc/extended/`: 5 digital classes, symbol rate known: RadioML 60.2 % / 76.1 % (SNR ≥ 10), TorchSig 66.2 / 82.2, never 85 %; below features (87.1) and ALRT (93.3). Limitation, not bug: low test power with ~120 symbols, 64QAM mimics 16QAM |
| Feature: HOS | Cumulants C20…C80 (19-feature set) | T | §2 |
| Feature: HOS | Extended and cyclic cumulants | T | `classical_amc/extended/`: 17 extra (C6x, C8x, cyclic) cumulants, checked vs theory; +0.0 (7 cls) / +0.5 (24 cls) points: no real gain. Cyclic C40 survives frequency offset where plain C40 collapses |
| Feature: spectral / instantaneous | Azzouz–Nandi amplitude, phase, frequency features | T | inside the 19 features |
| Feature: power-law lines | x², x⁴, x⁸ spectral lines | T | inside the 19 features |
| Feature: wavelet | Haar wavelet features | T | `classical_amc/extended/`: 24 Haar features; boosted trees 71.4 → 72.4 % (7 cls), 39.2 → 41.3 % (24 cls; 62.8 → 64.6 % at SNR ≥ 10). **Re-run independently: identical** |
| Feature: Hilbert–Huang | EMD / IMF instantaneous-amplitude and frequency features | T | `extra_methods/hht_features.py`: EMD check passes (two sines separated, corr ≥ 0.999). Alone 48.9 % / 67.8 % (SNR ≥ 10), never 85 %; added to the 19 features 71.4 % vs 71.1 % (no real gain) at 18.7 ms/frame extra |
| Classifier | Decision tree (≈ Azzouz–Nandi thresholds), k-NN, SVM, LDA, random forest, boosted trees, nearest centroid | T | §2 |
| Classifier | Training-free cumulant matching (Swami–Sadler) | T | §2 |
| Classifier | Hierarchical cumulant classifier on recovered symbols (Swami–Sadler, training-free) | T | `classical_amc/requirements_methods.py`: noiseless check 100/100; RadioML 5 digital classes 49.4 % / 64.4 % (SNR ≥ 10); limited by the M2M4 noise estimate (88 % with true noise on clean frames) |
| Classifier | Rule tree, one feature per node, thresholds learned (Azzouz–Nandi) | T | `classical_amc/requirements_methods.py`: 61.2 % / 98.6 % (SNR ≥ 10), 85 % from 6 dB, vs 99.4 % learned tree (first version 66.1 % because of two badly chosen node features; fixed) |
| Classifier: Bayesian | Gaussian naive Bayes, QDA, per-class Gaussian mixtures | T | `classical_amc/extended/`: on the 19 features, naive Bayes 64.0 / QDA 70.1 / per-class GMM 70.8 % (7 cls); 28.4 / 35.5 / 37.5 % (24 cls): below boosted trees (71.4 / 39.2) |
| Blind | EM / Gaussian-mixture constellation classification | T | `classical_amc/extended/`: symbol rate known, RadioML 71.8 % / **95.2 %** (SNR ≥ 10), TorchSig 77.2 / 95.1, 85 % at 2 dB: level with ALRT, ~18× slower. **Spot-checked independently: 95.3 % on 300 frames** |
| Blind | MCMC (Gibbs) Bayesian classification | N/A | Targets the same posterior as EM at far higher cost; EM is the representative of this family. |

## C. Characterization

| Quantity | Method | Status | Where / result |
|---|---|---|---|
| Carrier | PSD peak, centroid, x-dB midpoints, 99 % OBW midpoint | T | §3 |
| Carrier | Interpolated FFT (Jacobsen/Quinn), Kay, MUSIC, ESPRIT, M-th power, vs Cramér–Rao bound | T | `characterization_extended/`: periodogram ML, Quinn, Kay and data-aided ML reach the CRB above threshold (MSE/CRB 0.9–1.5); Jacobsen ≈2.2×; MUSIC/ESPRIT (M = 16/64) 3–50× (never reach it); QPSK 4th-power floors from self-noise. **Independent ML re-implementation: MSE/CRB 0.90–1.11** |
| Bandwidth | −3/−10/−26 dB, 99 % occupied (ITU-R SM.443), noise crossing | T | §3 |
| SNR | PSD-based (in-band, peak), M2M4 | T | §3 |
| SNR | Split-symbol, data-aided / decision-directed ML, eigenvalue-based, vs CRB | T | `characterization_extended/`: CRB code reproduces the closed form to 0.1 %; data-aided ML 0.9–1.1× its bound; split-symbol 1–2.8×; M2M4 ≈2× (fails for 16QAM above 10 dB); eigenvalue 1.2–3.8× and biased low below 2 dB; kurtosis estimator shown identical to M2M4 |
| Symbol rate | Envelope, 4th-power envelope, delay-multiply, Haar, instantaneous frequency (two test generators) | T | §3 |
| Symbol rate | FSK-specific estimators (closing the 32–66 % gap) | T | `characterization_extended/`: background-normalised line measurement + dev-tuned vote of 12 estimators: overall 72.7 → **89.9 %**, FSK family 49 → **83 %** (2FSK 95, 4FSK 97, MSK 88, GMSK 71, GFSK 64); GFSK/GMSK still weak below 5 dB. **Independent re-run on 200 random held-out frames: 94.5 %** |
| Symbol rate | MSK/GMSK squared-signal line pair | T | `extra_methods/msk_line_pair.py`: held-out GMSK at 0 dB 93 % (vote: 5 %), MSK at 0 dB 100 % (vote: 45 %); does not apply to GFSK with random index (25 %) |
| Symbol rate | Band-limiting before the FM discriminator; gated by constant envelope; diagnosis vs in-band SNR | T | `extra_methods/fsk_prefilter.py`: always band-limiting trades FSK (+2.8 dev / +5.7 held-out points) against PSK/QAM (−3.4 / −4.6), so the rule chosen on dev is not to add it; gate σ_aa < 0.375 (chosen on dev): held-out 89.9 → 91.3 %, FSK family 82.9 → 86.3 %, PSK/QAM 96.9 → 96.3 %, GMSK at 5 dB 63 → 97 %, GFSK 30 → 65 %. At 0 dB no gain (GFSK 8 %, GMSK 5 %): vote accuracy for GFSK/GMSK collapses below about 12 dB in-band SNR (2 % at 3–6 dB, 83 % at 12–15 dB) vs 3–6 dB for 2FSK/4FSK/MSK |
| Symbol rate → classification | Blind likelihood classifier with improved symbol rate on RadioML | T | `characterization_extended/`: same 4,800 RadioML test frames, blind ALRT 80.4 → **91.8 %** at SNR ≥ 10 (sps-known bound 93.3), 85 % now at 8 dB; symbol rate within 2 % rises 70.6 → 92.7 %. **Independent re-run on 150 random frames: 90.0 %** |
| Number of signals | MDL / AIC | T | §4 |
| OFDM parameters (FFT size, cyclic prefix) | Blind cyclic-prefix correlation search over 20 candidates (van de Beek / Sandell style) | T | `extra_methods/ofdm_params.py`: clean frames 100/100; standardised noise statistic comparable across candidates (mean 1.9–2.4, std 0.3–0.4). Exact (N, cp) 90 % from −2 dB per-sample SNR; cp = N/4 from −6 dB, cp = N/32 needs about +2 dB (less correlated energy, as theory says) |

## D. Discovery log (methods found after the first catalogue)

A targeted search for classical methods outside the survey taxonomies found these families. Status at the time of writing:

| Found | Source | Status |
|---|---|---|
| Bispectrum features | arXiv 2604.10317 (review) | T: checks pass (Gaussian ≈ 0, coupled pair 0.997). 7 cls 71.4 → 73.1 %; 24 cls no gain (z −0.1) |
| Fractal-dimension features (Higuchi, Katz, box, multi-scale) | cumulants + fractal AMC paper; SPIE multi-scale fractal paper | T: Higuchi matches theory (line 1.000, white 2.000, Brownian 1.505). Small, not significant gains (z 1.6–1.7) |
| Information-theoretic features (Shannon / Rényi / permutation / sample entropy) | Gazi Univ. information-theoretic AMC paper | T: **24 classes 62.8 → 70.3 % at SNR ≥ 10** (8PSK +37, 16QAM +15 points), almost all from 8 cheap histogram entropies (0.4 ms/frame). **Independently re-run: 69.7 %** |
| S-transform features for classification | S-transform AMC paper | T: 7 cls 71.4 → 73.2 % (low-SNR gain); 24 cls small (z 1.9) |
| Cramér–von Mises and Anderson–Darling classifiers | EDF-test family (KS already tested) | T: match scipy to 1e-14, null calibrated. CvM ≈ KS (76.2 / 80.4 % at SNR ≥ 10); AD worse on RadioML (62.0 %: phase cut splits the BPSK cluster) |
| AGM, MED, John's test, eigenvalue moment ratio, condition number with Tracy–Widom thresholds | Huang & Fang 2015; Sedighi et al. 2017; small-sample sensing (arXiv 1404.5331) | T (`detection_discovered/`): box F1 6–9 % impaired / 35–37 % clean (coloured-noise limitation, like MME); Tracy–Widom thresholds miscalibrated on smoothed covariances (Pfa 5e-3–0.37 for 1e-3); EMR identical to John's test |
| Teager–Kaiser energy operator detector | MIT LL transient detection; arXiv 1504.08177 | T: on the spectrogram ≈ energy detector (54.8 vs 53.0 impaired, 65.2 vs 67.4 clean) with half the false alarms on impaired scenes; weaker at 0–5 dB. **Independent re-run (scenes 100–159): 54.0 vs 53.7 / 68.8 vs 72.0** |
| Stochastic-resonance pre-treated energy detector | arXiv 0906.0739 | T: paper's gain not reproduced; never better than plain energy detection (synthetic, 4,096 samples) |
| Covariance-index detectors (Gini, Gerschgorin, Pietra–Ricci, Theil, Atkinson) | Sensors 2026 (doi 10.3390/s26082557), found in the 2020–2026 sweep | T (row A): no better than CAV |
| OFDM blind parameter estimation (FFT size, cyclic prefix) | 2022 blind-OFDM parameter-estimation literature, found in the 2020–2026 sweep | T (row C) |
| Hough trade-off | – | full held-out curve done: thinness rule 0.5 → 73 % / 81 % chirp recall at 0.10–0.17 false boxes/scene; ≥ 2 or off → 77 % / 84 % at 1.4–1.5 |

## E. How the catalogue was built, and what it does not claim

A literally exhaustive list of "all statistical methods" does not exist, so this catalogue cannot prove completeness.
What it shows is **saturation of the published taxonomies**: the search ran in rounds and each round added fewer families.

| Round | Sources and queries | Families added |
|---|---|---|
| 1 | Survey taxonomies: Axell, Leus & Larsson spectrum-sensing overview; Ahmad wideband sensing survey (arXiv 2001.02574); Dobre et al. modulation-classification survey; Zhu & Nandi book; classical estimation literature (Pauluzzi & Beaulieu, Kay, Quinn) | 52 rows (48 tested, 4 not applicable) |
| 2 | Four targeted queries: bispectrum / fractal / entropy features; eigenvalue detectors (AGM, EMR, Tracy–Widom); CvM / AD distribution tests; Teager–Kaiser detection (the stochastic-resonance paper came from this search) | 8 families, all tested |
| 3 | Three sweep queries (extended search): modulation-classification surveys 2023–24; spectrum-sensing surveys 2022–24; blind parameter-estimation surveys 2021–23 | 2 families (covariance-index detectors, OFDM parameter estimation), both tested; every other method the results named was already in the catalogue |

**Stopping rule:** repeat targeted rounds until one returns no family that is missing. Round 3 still returned two, so the rule is not yet met; one more round with new queries would be the next step. The yield 52 → 8 → 2 shows diminishing returns, not an end.

**Known untested areas (from the sweeps and from the requirements document), so nothing here is hidden:**

| Area | Why not tested |
|---|---|
| Subcarrier modulation classification inside OFDM (survey doi 10.3390/s22031020) | Needs OFDM signals with known subcarrier mappings; RadioML has none, TorchSig only gives the OFDM label |
| Morlet-wavelet symbol-rate estimator | Haar-wavelet and the other estimators were tested; the Morlet variant was not run separately |
| Detectors for DSSS, FHSS, PPM (ADS-B), PM, VSB, AFSK, π/4-DQPSK, D8PSK listed in the requirements document | Signal-specific; not in RadioML, and TorchSig covers only a few of them |
| Deep-learning methods | Out of scope: this is the classical baseline the AI work is compared against |
| Real-data coverage | One public recording; our own SDR recordings (job 2.1) are not made yet |
