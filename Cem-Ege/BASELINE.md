# Classical baseline — detection, identification, characterization

EEE493 P1 · V/UHF signal AI · TÜBİTAK BİLGEM İLTAREN · built 2026-10-15

Every classical / statistical method we know of for each pipeline stage, run on the same data with
the same scoring, so the AI models (5.2+, 4.4, 6.3) have one fixed reference to beat.
**All numbers are on synthetic data** (no real SDR recordings yet).

| Stage | Data | Methods | Code |
|---|---|---|---|
| Detection (WP4) | 300 + 300 TorchSig wideband scenes (impaired / clean), 1–6 emitters, 69 signal types | 14 detectors | `wp4_detection/` |
| Identification (WP5) | RadioML 2018.01A (7-class subset and all 24); symbol-level synthetic for the likelihood bound; TorchSig modulators | 8 classifiers + ALRT/HLRT (ideal and practical, blind) | `classical_amc/` |
| Characterization (WP6) | same TorchSig scenes (truth boxes give carrier, bandwidth, SNR); synthetic for symbol rate | 5 carrier, 5 bandwidth, 5 SNR, 6 symbol-rate estimators | `wp6_characterization/`, `classical_amc/symbol_rate.py` |

**Coverage:** [CATALOG.md](CATALOG.md) lists every method family named in the standard surveys (48 tested, 4 not applicable with reasons). Section 5 summarises the families added last.

**Rule used everywhere:** tune on scenes 0–99 / training split; report on scenes 100–299 / held-out split.

---

## 1. Detection (WP4) — 14 detectors

All thresholds are calibrated on pure white noise (target Pfa is exact on noise for every detector), then the
mask is cut into boxes by the same segmentation (4.3) and scored the same way.
Box = Hungarian match with IoU ≥ 0.5. Numbers at target Pfa 10⁻³, scenes 100–299.

### Impaired scenes (TorchSig level 2: passband shape, gain steps / AGC, IQ imbalance, IMD, fading)

| Detector | Box F1 | Precision | Recall | Recall 0–5 dB | Recall 10–15 dB | ms/scene |
|---|---|---|---|---|---|---|
| **energy** (excision noise floor) | **53.0** | 48.9 | 57.8 | 40 | 79 | 318 |
| energy, multitaper | 42.7 | 34.1 | 56.9 | 38 | 79 | 406 |
| cfar_os (ordered statistic) | 39.8 | 36.7 | 43.4 | 33 | 71 | 209 |
| fcme | 35.6 | 36.9 | 34.5 | 16 | 51 | 92 |
| cfar_go | 34.5 | 33.9 | 35.2 | 24 | 60 | 20 |
| lad (double threshold) | 30.5 | 20.7 | 57.5 | 38 | 78 | 366 |
| cfar_2d | 28.8 | 22.2 | 41.0 | 27 | 73 | 14 |
| cfar_ca | 28.6 | 21.6 | 42.3 | 31 | 69 | 20 |
| cfar_so | 15.6 | 9.9 | 36.7 | 26 | 51 | 20 |
| gof_ad (Anderson–Darling)* | 13.7 | 11.7 | 16.6 | 8 | 23 | 28 |
| cyclo (cyclostationary)* | 12.3 | 9.2 | 18.4 | 2 | 21 | 106 |
| psd_occupancy (no time resolution)* | 9.9 | 6.1 | 26.0 | 17 | 34 | 98 |
| spectral_kurtosis | 5.8 | 6.2 | 5.5 | 2 | 9 | 18 |
| eigen_mme (eigenvalue)* | 4.7 | 3.4 | 7.4 | 4 | 10 | 298 |

### Clean scenes

| Detector | Box F1 | Precision | Recall | Recall 0–5 dB | Recall 10–15 dB |
|---|---|---|---|---|---|
| **energy** | **67.4** | 72.7 | 62.8 | 52 | 74 |
| lad | 66.6 | 68.1 | 65.2 | 57 | 79 |
| fcme | 65.1 | 71.8 | 59.5 | 40 | 70 |
| cfar_so | 64.1 | 60.3 | 68.5 | 71 | 77 |
| energy, multitaper | 60.7 | 58.3 | 63.3 | 58 | 77 |
| cfar_ca | 52.1 | 52.5 | 51.7 | 59 | 72 |
| cfar_os | 48.8 | 47.8 | 49.8 | 61 | 60 |
| cfar_2d | 45.6 | 43.6 | 47.7 | 49 | 70 |
| cfar_go | 41.2 | 41.9 | 40.5 | 51 | 47 |
| eigen_mme* | 37.5 | 35.4 | 39.9 | 26 | 50 |
| gof_ad* | 36.2 | 31.2 | 43.0 | 30 | 51 |
| psd_occupancy* | 23.8 | 27.1 | 21.3 | 16 | 22 |
| cyclo* | 19.6 | 19.0 | 20.2 | 3 | 18 |
| spectral_kurtosis | 6.3 | 7.3 | 5.5 | 3 | 2 |

\* Decides per block (16 frames × 32 bins) or per whole frequency column, so boxes are coarse by construction;
judge these on emitter Pd vs Pfa (`wp4_detection/results/compare_operating_*.png`), not box F1.

**Findings**
- **An energy detector with a good noise-floor model is the strongest classical detector.** The noise model
  matters more than the test: a single global noise level gave Pfa 0.28 instead of 10⁻³ on impaired scenes;
  modelling noise as gain(t) × shape(f) and excising signal blobs fixed it.
- **Impairments cost ~15 F1 points** (energy 67 → 53). Gain steps that also change the noise shape mid-scene
  and unlabelled real energy (IQ images, intermodulation, DC spike) are the main causes; on pure noise every
  detector meets its Pfa exactly, so the extra "false alarms" are mostly real but unlabelled emissions.
- **1-D CFAR self-masks wide signals.** With guard 4 / train 16 it found 5 % of emitters; guard 16 / train 32
  (tuned on dev) gives 25–57 % emitter Pd. Recall also drops above 15 dB (strong skirts leak into training cells).
  cfar_so is the best CFAR on clean data, cfar_os on impaired data.
- **Low SNR (0–5 dB) is the gap:** at most 40–71 % box recall. This is where AI detection (4.4) has to win.
- Eigenvalue and goodness-of-fit tests assume white noise; the receiver passband makes noise correlated, so on
  impaired data they fire everywhere (Pfa 0.36–0.42). Spectral kurtosis only sees non-Gaussian signals; dense
  TorchSig signals look Gaussian, so it finds ~2 %. Cyclostationary features are blind to constant-envelope
  signals without x² lines.

## 2. Identification (WP5)

### Classifiers on the same 19 classical features (RadioML 2018.01A, held-out 25 %, same split for all)

| Model | 7 classes: all SNRs | 7 classes: SNR ≥ 10 | 85 % reached at | 24 classes: all | 24 classes: SNR ≥ 10 | Predict ms/frame |
|---|---|---|---|---|---|---|
| **gradient-boosted trees** | **71.4** | **99.95** | **2 dB** | 39.2 | 62.8 | 0.04 |
| random forest | 71.3 | 99.97 | 2 dB | **39.9** | **63.1** | 0.04 |
| SVM (RBF) | 71.3 | 99.96 | 2 dB | 37.6 | 60.0 | 2.7 |
| k-NN (k=15) | 70.4 | 99.9 | 2 dB | 35.4 | 56.0 | 0.13 |
| LDA | 68.6 | 99.5 | 4 dB | 28.7 | 46.8 | 0.0001 |
| decision tree, depth 6 (≈ Azzouz–Nandi) | 63.8 | 99.4 | 6 dB | 24.2 | 41.1 | 0.0004 |
| nearest centroid | 48.3 | 70.4 | not sustained | 16.0 | 23.4 | 0.0002 |
| cumulant theory, training-free (Swami–Sadler)† | 40.9 | 54.2 | never | 11.4 | 14.0 | 0.0002 |

Feature extraction adds 0.5–0.6 ms/frame for all. 7 classes = AM-DSB-WC, FM, BPSK, QPSK, 8PSK, 16QAM, GMSK.
† Scored only on the digital classes it can describe (4 of 7, 12 of 24).

### Likelihood-ratio classifier — the classical upper bound (symbol-level synthetic, 128 symbols)

Assumes what RadioML does not give: perfect timing, known SNR, no pulse shaping (only the phase is unknown).
Classes BPSK, QPSK, 8PSK, 16QAM, 64QAM; same frames for all methods.

| Method | All SNRs (−10…20 dB) | SNR ≥ 10 | 85 % reached at | ms/frame |
|---|---|---|---|---|
| **ALRT** (phase-averaged likelihood) | **70.2** | **99.2** | **8 dB** | 15 |
| HLRT (phase-maximised) | 69.1 | 99.1 | 8 dB | 15 |
| 19 features + boosted trees | 61.1 | 89.2 | 14 dB | 0.05 |
| cumulant theory, training-free | 45.8 | 81.1 | 18 dB | 0.05 |

### Practical likelihood classifier — no oracle knowledge (`classical_amc/practical_alrt.py`)

Every nuisance parameter is estimated per frame: symbol rate (smoothed instantaneous-frequency estimator),
resampling to 8 samples/symbol, RRC matched filter (roll-off 0.35 assumed), timing (max-energy phase), carrier
offset (4th/8th-power line); phase averaged (ALRT) or maximised (HLRT); noise level maximised over a grid.
Same test frames for all methods; the feature classifier is trained on the other half of the same source.
Classes BPSK, QPSK, 8PSK, 16QAM, 64QAM, −10…20 dB.

| Source | Method | All SNRs | SNR ≥ 10 | 85 % at | ms/frame |
|---|---|---|---|---|---|
| RadioML 2018.01A (4,800 test frames) | 19 features + boosted trees | **66.2** | 87.1 | 6 dB | 0.05 |
| | HLRT, symbol rate known (8 sps) | 70.6 | **93.4** | **2 dB** | 42 |
| | ALRT, symbol rate known (8 sps) | 70.4 | 93.3 | 4 dB | 42 |
| | ALRT, fully blind | 59.0 | 80.4 | never | 42 |
| TorchSig modulators (3,200 test frames; 3–20 sps, SRRC or rectangular) | 19 features + boosted trees | 62.1 | 85.4 | 16 dB | 0.05 |
| | ALRT, symbol rate known | **76.3** | **95.6** | **4 dB** | 38 |
| | ALRT, fully blind | 69.1 | 91.3 | 6 dB | 38 |
| | HLRT, fully blind | 69.0 | 90.9 | 6 dB | 38 |

Unit check: on clean 25 dB RRC frames the pipeline classifies 48/50 correctly.

- **With a correct symbol rate the likelihood classifier beats the features on real-world-style frames**
  (RadioML: 85 % at 2–4 dB instead of 6 dB; TorchSig: 4 dB instead of 16 dB).
- **Fully blind, it depends on the symbol-rate estimate.** On TorchSig it still wins by 10 dB; on RadioML it
  loses, because the estimator mis-reads RadioML QAM frames (median 11–12 instead of 8 samples/symbol), and a
  wrong symbol rate ruins the likelihood. Better symbol-rate estimation would transfer directly to accuracy.
- **Fixed with a better symbol-rate step (section 5):** the blind ALRT on the same RadioML test frames rises from 80.4 % to **91.8 %** at SNR ≥ 10 dB and reaches 85 % at 8 dB (symbol-rate-known bound: 93.3 %).
- Cost: ~40 ms per frame vs 0.05 ms for the feature classifier (800×), and it only covers classes whose
  constellation is known.

**Findings**
- On 7 well-separated classes the classifier choice barely matters (HGB ≈ RF ≈ SVM ≈ k-NN); **the features are
  the limit**. Boosted trees are the best accuracy / speed trade-off.
- **All 24 classes: ~63 % at high SNR, never 85 %.** Failures: 8/16/32PSK (identical cumulants), high-order
  QAM/APSK, AM-SSB-WC vs SC, OOK vs BPSK. The 85 % İLTAREN target is only reachable classically on a reduced,
  well-separated class set.
- **Feature classifiers sit ~6 dB below the likelihood bound** (85 % at 14 dB vs 8 dB). That 6 dB is the room an
  AI model can realistically claim without channel knowledge.
- Theoretical cumulant values do not survive pulse shaping and noise (BPSK |C40| 1.45 instead of 2), so
  training-free cumulant classification is weak; learned thresholds are required.

## 3. Characterization (WP6)

Oracle detection (truth box gives the region), TorchSig scenes 100–299; errors are median / 90th percentile.

| Quantity | Best on clean | Best on impaired | Other estimators (median, clean / impaired) |
|---|---|---|---|
| Carrier (error ÷ true BW) | power centroid: 0.004 / 0.088 | −10 dB-edge midpoint: 0.015 / 0.35 | −10 dB mid 0.007 · centroid 0.026 (imp.) · 99 % mid 0.010 / 0.044 · peak 0.091 / 0.124 |
| Bandwidth (relative error) | noise-crossing‡: 0.077 / 0.49 | noise-crossing‡: 0.19 / 0.88 | −26 dB 0.15 / 0.25 · 99 % OBW 0.18 / 0.30 · −10 dB 0.26 / 0.44 · −3 dB 0.52 / 0.76 |
| SNR (dB error) | peak-PSD: 0.45 / 3.8, bias −0.15 | peak-PSD: 2.1 / 13.5, bias −0.13 | in-band PSD 2.1–3.1 (bias −1.6…−3.3) · M2M4 9.5 / 19.5 |

‡ Reproduces TorchSig's labelling rule, so it partly measures the label itself; the best *physical* bandwidth
is −26 dB (0.14 / 0.24).

### Symbol rate (synthetic RRC-shaped PSK/QAM + GMSK, 3–20 samples/symbol, success = within 2 %)

| Estimator | All | BPSK | QPSK | 8PSK | 16QAM | GMSK | 0 dB | 20 dB |
|---|---|---|---|---|---|---|---|---|
| **inst. frequency, smoothed** | **79** | 92 | 94 | 95 | 85 | **30** | 62 | **95** |
| inst. frequency | 76 | **99** | **95** | **96** | 84 | 6 | 59 | 83 |
| squared envelope | 67 | 69 | 88 | 90 | 83 | 2 | 44 | 72 |
| 4th-power envelope | 46 | 34 | 61 | 62 | 70 | 2 | 29 | 48 |
| Haar wavelet | 36 | 44 | 45 | 43 | 45 | 3 | 2 | 79 |
| delay-and-multiply | 36 | 26 | 50 | 47 | 56 | 1 | 20 | 40 |

With the modulation known (from 5.1) and the best estimator chosen per class: **81 %**.

**Second, independent generator: TorchSig's own modulators** (`extra_methods/gen_symbol_rate_torchsig.py`,
`symbol_rate_torchsig_eval.py`). TorchSig builds PSK/QAM at 4 samples/symbol and resamples so the symbol rate
equals the nominal bandwidth; for FSK the symbol rate is bandwidth ÷ a random factor, captured during generation.
Labels were checked on clean rectangular BPSK (phase jumps fall on whole symbol periods). 3,000 frames,
4,096 samples, SRRC or rectangular pulses, 0–20 dB:

| Estimator | All | BPSK | QPSK | 8PSK | 16QAM | 64QAM | 2FSK | 4FSK | GFSK | MSK | GMSK |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **inst. frequency, smoothed** | **73** | **100** | **99** | **98** | **92** | **92** | **66** | **43** | **32** | **66** | **40** |
| inst. frequency | 49 | 88 | 89 | 90 | 82 | 83 | 1 | 1 | 20 | 6 | 27 |
| Haar wavelet | 43 | 75 | 78 | 76 | 72 | 76 | 2 | 0 | 17 | 6 | 23 |
| squared envelope | 21 | 37 | 46 | 43 | 39 | 33 | 1 | 1 | 4 | 3 | 2 |
| 4th-power envelope | 13 | 15 | 27 | 26 | 29 | 25 | 1 | 1 | 2 | 3 | 2 |
| delay-and-multiply | 11 | 25 | 22 | 21 | 19 | 20 | 1 | 5 | 0 | 1 | 0 |

The ranking matches the first test. Envelope methods fail on rectangular pulses (squared envelope: 70 % on
SRRC, 9 % on rectangular), as theory predicts: a rectangular-pulse PSK signal has an almost constant envelope.
**FSK family symbol rate was the open gap (32–66 %); a background-normalised, dev-tuned vote of 12 estimators raises it to 83 % (overall 73 → 90 %), see section 5.** On RadioML the smoothed estimator puts all 200 high-SNR
BPSK/QPSK/8PSK frames at 7.5–8 samples/symbol: **RadioML 2018.01A uses 8 samples/symbol.**

**Findings**
- **TorchSig's `snr_db` is a peak-PSD SNR**, not in-band power (verified in `torchsig/utils/dsp.py`, lines
  1465–1469). In-band estimators read 2–3 dB low; chirp/LFM/LoRa labels are 25–48 dB below their physical
  SNR. **Anyone training an SNR regressor (6.3) on TorchSig labels learns this definition.**
- TorchSig's bandwidth label grows with SNR (≈2× wider above 25 dB), so fixed "x dB" bandwidths are biased in an
  SNR-dependent way.
- M2M4 only works on truly constant-envelope signals (FSK/GMSK/FM); PSK is pulse-shaped in TorchSig, so even
  there it is 7 dB low.
- GMSK / FSK symbol rate is the hard case: the frequency only changes at bit transitions.

---

## 4. Waveform-specific and auxiliary methods

These need a known waveform, or work on top of a detector, so they are tested where that assumption holds.
Code in `extra_methods/` and `wp4_detection/` (`refine_wp4.py`, `hough_wp4.py`). Each was first checked
against theory or a synthetic case with known answer.

| Method | Test | Result | Check |
|---|---|---|---|
| **Matched filter** (known 256-sample preamble) vs energy | Synthetic bursts in 4,096-sample frames, Pfa 10⁻³ per frame | Pd 90 % at **−10 dB** vs −4 dB for energy (**6 dB gain**); exact burst timing (energy: ~260 samples off). With ±0.002 cycles/sample frequency offset: −8 dB (plain and segmented) | Theory −10.6 dB ✓ |
| **OFDM cyclic-prefix detector** (Nfft 64, CP 16) vs energy | Synthetic OFDM, Pfa 10⁻², noise power known or ±1 dB uncertain | Known noise: energy −12 dB, CP −8 dB. **±1 dB noise uncertainty: energy false-alarm rate 0.43 (target 0.01), CP keeps 0.01** and still reaches 90 % at −8 dB (−6 dB blind over 4 FFT sizes) | Energy theory −12.4 dB ✓; SNR-wall effect as in Tandra & Sahai ✓ |
| **MDL / AIC signal counting** (single channel, 16-sample embedding) | 0–4 tones in noise, 1,024 samples | **MDL: 90 % correct from −12 dB per tone, 100 % correct on noise only, no over-count at +10 dB.** AIC never reaches 90 %: 71 % on noise only, over-counts 32 % at +10 dB | MDL consistent, AIC over-estimates: as theory says ✓ |
| **CUSUM time-edge refinement** of energy boxes | TorchSig scenes 100–299 | Box F1 clean 67.4 → **68.1**, IoU 0.81 → 0.83; impaired 53.0 → **53.9**, IoU 0.79 → 0.82 | Synthetic boxes: time-edge error 10.5 → 0 frames ✓ |
| **Wavelet band-edge refinement** (Tian & Giannakis) | TorchSig scenes 100–299 | Box F1 clean 67.4 → 53.1, impaired 53.0 → 44.7 (**worse against TorchSig labels**) | Synthetic: edge error 6 → 1 bin ✓. On TorchSig the refined boxes are 0.71× the truth width (85 % narrower): the wavelet finds the physical edge, TorchSig labels extend to where the signal is 0.5 dB above noise. A label-definition mismatch, not a bug |
| **Hough transform** for chirp / LFM / LoRa | TorchSig scenes 100–299, chirp-family emitters | Chirp recall at IoU ≥ 0.5: Hough alone 27 % (impaired) / 41 % (clean) vs energy segmentation 69 % / 78 %; **energy + Hough 77 % / 84 %**, at the cost of 1.4–1.6 extra false boxes per scene | Synthetic sawtooth chirps: 18/20 exactly one correct box, 0 false boxes on 20 non-chirp masks ✓. Tuned on dev scenes (32 chirps): TorchSig chirps are slow sweeps 5–20° from vertical. **With the line-thinness rule on** (held-out too): Hough alone 8 % / 6 %, energy + Hough 73 % / 81 %, only 0.10–0.17 false boxes per scene; the rule rejects dense LoRa sawtooths |

## 5. Further method families (from the literature catalogue)

Added after enumerating the survey taxonomies (CATALOG.md). Implemented by helper agents in
`classical_amc/extended/`, `detection_extended/`, `characterization_extended/`, plus `extra_methods/hht_features.py` and
`extra_methods/stransform_detect.py`; every method passed a known-answer or theory check first, and the key numbers
were re-run independently (section 6).

**Identification** (RadioML; boosted trees unless stated)

| Method | 7 classes all / SNR ≥ 10 | 24 classes all / SNR ≥ 10 | Note |
|---|---|---|---|
| 19 features (reference) | 71.4 / 99.95 | 39.2 / 62.8 | |
| + 17 extended & cyclic cumulants | 71.4 / 99.97 | 39.7 / 63.5 | no real gain |
| **+ 24 Haar wavelet features** | **72.4** / 99.97 | **41.3 / 64.6** | small real gain; reproduced exactly |
| + 32 Hilbert–Huang (EMD) features | 71.4 / 99.9 | – | no gain; HHT alone 48.9 / 67.8 |
| Naive Bayes / QDA / per-class GMM | 64.0 / 70.1 / 70.8 (all) | 28.4 / 35.5 / 37.5 (all) | below boosted trees |

5 digital classes, symbol rate known: **EM constellation classifier 95.2 % (RadioML) / 95.1 % (TorchSig) at SNR ≥ 10**,
level with ALRT (93.3 / 95.6); Kolmogorov–Smirnov classifier 76.1 / 82.2 (weak with ~120 symbols).

**Detection**

| Method | Result |
|---|---|
| Spectral correlation density (FAM) | box F1 16.2 % impaired / 24.2 % clean; best block detector on impaired scenes, but 0–5 dB recall 1–5 %, ~4 s/scene |
| Covariance absolute value, autocorrelation, entropy | box F1 6–8 % impaired (false-alarm rate ~0.3: coloured receiver noise breaks whiteness tests) / 37–39 % clean |
| Chirps (synthetic): spectrogram cell / line sum / PWVD / SPWVD / **FrFT-dechirp** | 50 % detection at −13.4 / −20.7 / −21.7 / −21.4 / **−23.1 dB** (theory ≈ −23.7 dB) |
| S-transform vs STFT (64-sample bursts) | no gain: 90 % at 0 dB vs −4 dB for a burst-matched STFT |
| Burst onset: Shiryaev–Roberts / CUSUM / GLR / Bayesian online / Page–Hinkley | mean delay 70.9 / 72.1 / 103.5 / 110.8 / 170.5 frames at equal false-alarm run length (−15 dB); CUSUM within 1–4 % of Siegmund's theory |

**Characterization**

| Method | Result |
|---|---|
| Carrier: periodogram ML, Quinn, Kay, data-aided ML | reach the Cramér–Rao bound above threshold (MSE/CRB 0.9–1.5) |
| Carrier: Jacobsen; MUSIC / ESPRIT | ≈2.2× CRB; 3–50× CRB (never reach it) |
| SNR: data-aided ML / split-symbol / M2M4 / eigenvalue | 0.9–1.1× / 1–2.8× / ≈2× / 1.2–3.8× their bounds; M2M4 fails for 16QAM above 10 dB |
| Symbol rate: consensus of 12 estimators | **73 → 90 %** overall, FSK family **49 → 83 %** (GFSK/GMSK still weak below 5 dB) |
| Blind likelihood classifier with the consensus symbol rate | RadioML **80.4 → 91.8 %** at SNR ≥ 10 dB, 85 % at 8 dB |

## 5b. Final additions (requirements document, gaps, verification)

| Item | Result |
|---|---|
| Azzouz–Nandi rule tree (one feature per node, thresholds learned at design SNR) | 61.2 % / **98.6 %** at SNR ≥ 10 (7 classes), 85 % from 6 dB (learned tree 99.4 %). First version had two badly chosen node features (66.1 %); fixed |
| Swami–Sadler hierarchical cumulant classifier (symbols, training-free) | 49.4 % / 64.4 % at SNR ≥ 10 (5 digital classes). Limited by my blind noise estimate (M2M4, biased for QAM): with the true noise level it reaches 88 % on clean 120-symbol frames |
| MSK/GMSK symbol rate, squared-signal line pair | GMSK at 0 dB 93 % (vote 5 %), MSK at 0 dB 100 % (vote 45 %). Not valid for GFSK with random modulation index (25 %): **GFSK below 5 dB remains the one open symbol-rate gap** |
| Hough thinness trade-off (held-out) | rule 0.5: chirp recall 73 % / 81 % at 0.10–0.17 false boxes per scene; rule ≥ 2 or off: 77 % / 84 % at 1.4–1.5. Recommended: 0.5 |
| Chirp benchmark verification | All six 50 %-detection SNRs and the sanity values re-read from `detection_extended/results/partB/chirp_bench.json` and `sanity_chirp.json`: match the report |
| Newly found identification families (bispectrum, fractal, entropy, S-transform features; CvM / AD classifiers) | Done, all checks pass. **Histogram-entropy features: 24 classes 62.8 → 70.3 % at SNR ≥ 10 dB** (re-run independently: 69.7 %), the biggest classical gain on 24 classes; bispectrum / S-transform help only on 7 classes at low SNR; fractal small; CvM ≈ KS, AD worse |
| Newly found detection families (AGM / MED / John / EMR / Tracy–Widom eigen detectors, Teager–Kaiser, stochastic resonance) | Done, all checks pass. None beats energy detection. Teager–Kaiser ≈ energy (54.8 vs 53.0 impaired, 65.2 vs 67.4 clean) with half the false alarms on impaired scenes (re-run independently); eigenvalue family 6–9 % impaired / 35–37 % clean (coloured noise); Tracy–Widom thresholds miscalibrated on smoothed covariances; stochastic-resonance gain not reproduced |
| Real (over-the-air) data | One recording tested, see section 5c |

## 5c. First test on real over-the-air data

Recording: Allen Telescope Array RFI survey, Zenodo 8242048 (CC-BY-4.0), file `2023-08-03-15-38-39`:
ADALM-Pluto + Maia SDR, 61.44 Msps, centre 753.52 MHz, 2.28 s. Ground truth from the author's
description: LTE downlink carriers at 742.5 MHz (5 MHz), 751 MHz (10 MHz), 763 MHz (10 MHz);
checked against the long-integration spectrum (`real_data/results/atarfi_psd.png`): the three
plateaus sit exactly on those bands. Code: `real_data/analyze_atarfi.py`.

| Finding | Result |
|---|---|
| Signal strength | Carriers only ~1.5–2.5 dB above the noise even after long integration |
| Real receiver artefacts | Strong comb of spurious lines at the band edges (722–740, 767–784 MHz) and single lines in band (Pluto / phone / USB cable, per the author) |
| Noise model | Quiet-bin cells deviate measurably from the Exp model (KS distance 0.021); flagged-cell rate in carrier-free bins 1–4 × 10⁻² instead of 10⁻³, mostly the real spurs |
| Time-frequency detectors (4.3 ms segments, 20 segments) | 0 % of carriers detected by every detector (energy, multitaper, FCME, LAD, CFAR CA/OS/SO/2-D): per-cell SNR is below 0 dB |
| Frequency-only energy detection vs integration time | 5 MHz carrier: detected from 4.3 ms. 10 MHz carriers: 21 % / 24 % of bins at 4.3 ms → 47 % / 56 % at 342 ms: the running-median noise estimate (101 bins) sits partly on these ~75-bin-wide carriers |
| Carrier / bandwidth estimation (long PSD) | 742.5 MHz carrier: −97 kHz error; 10 MHz carriers biased by in-band spurs (−1.2 / −1.8 MHz); 99 % OBW over-estimated (6.5 vs 4.5, 12 vs 9 MHz) because spurs add power |

**What this means:** on real data the bottleneck is weak signals plus receiver spurs, not the detector
choice. Needed next: longer integration, spur suppression (notch known receiver lines), and a
noise-floor estimate that does not sit on wide signals. Real SDR recordings of our own (job 2.1) remain the priority.

## 5d. Gap checks, second round

| Item | Result |
|---|---|
| Covariance-index detectors (Gini, Gerschgorin, Pietra–Ricci, Theil, Atkinson), from the 2020–26 literature sweep | Pfa calibrated on noise; box F1 6.4–7.3 % impaired, 35.4–36.9 % clean: no better than CAV |
| OFDM blind parameter estimation (FFT size and cyclic prefix, 20 candidates) | Exact match 90 % from −2 dB per-sample SNR; cp = N/4 from −6 dB; cp = N/32 needs about +2 dB |
| GFSK/GMSK symbol rate below 5 dB: diagnosis | Vote accuracy vs in-band SNR (per-sample SNR + filter gain): GFSK/GMSK 2 % (3–6 dB), 27 % (6–9), 34 % (9–12), 83 % (12–15), 94 % (≥ 15); 2FSK/4FSK/MSK 77 / 87 / 94 / 100 / 96 %. A 6–9 dB higher threshold, consistent with the weaker frequency swing of Gaussian-filtered signals (mechanism not tested separately) |
| Band-limiting every frame before the vote | dev 90.9 → 90.6 %, held-out 89.9 → 90.4 %: FSK family +2.8 / +5.7 points, PSK/QAM −3.4 / −4.6 points. Rule chosen on dev: do not apply to every frame |
| Gate: band-limit only when σ_aa < 0.375 (threshold chosen on dev, held-out scored once) | Held-out 89.9 → 91.3 %; FSK family 82.9 → 86.3 %, PSK/QAM 96.9 → 96.3 %; GMSK at 5 dB 63 → 97 %, GFSK 30 → 65 %. At 0 dB no change (noise lifts σ_aa of FSK frames to about 0.47, above the gate): GFSK 8 %, GMSK 5 %, **still open: a threshold limit** |
| Completeness of the method list | Cannot be proven. CATALOG.md section E: search rounds yielded 52 → 8 → 2 new families, the stopping rule is not yet met, and the untested areas are listed |

## 6. Verification of delegated work

WP5 classifier comparison and WP6 estimators were written by helper agents; their key results were re-derived
independently:
- **WP5:** boosted trees reproduce the 5.1 run exactly (71.4 / 99.95 %); LDA (68.6 / 99.5 %) and k-NN
  (70.4 / 99.9 %) re-run from scratch on the same split give identical numbers.
- **WP6:** summary statistics recomputed from the raw per-emitter files match the agent's report, except the
  impaired SNR 90th percentile (13.5 dB, not 10.5 dB; corrected above). A from-scratch re-implementation of
  the centroid carrier and peak-PSD SNR estimators gives 0.005 × BW and 0.52 dB median error on clean scenes
  (agent: 0.004 and 0.45 dB). The TorchSig SNR definition was confirmed in its source code.
- Every WP4 detector was checked to hit its target false-alarm rate on fresh white noise.
- **Section 5 (re-run independently):** Haar-feature gain reproduced exactly (71.4 → 72.4 %, 39.2 → 41.3 %); EM classifier
  95.3 % on 300 random high-SNR RadioML frames (agent 95.2 %); symbol-rate consensus 94.5 % on 200 random held-out frames
  with a fresh estimator-bank computation (agent 89.9 % on all); blind ALRT 90.0 % on 150 random frames (agent 91.8 %);
  my own ML tone-frequency estimator gives MSE/CRB 0.90–1.11 (agent 0.93–1.08); CAV and spectral-entropy detectors
  re-calibrated from scratch hit their target false-alarm rates on fresh noise, and a re-run on scenes 100–159 shows the
  same pattern as the agent's table.

## Not covered

| Item | Reason |
|---|---|
| Real SDR recordings | Not recorded yet (2.1 / 2.4). Every number here is synthetic. |
| Matched filter / CP detector on TorchSig scenes | TorchSig does not expose preambles or OFDM parameters per emitter; tested on synthetic signals with known waveforms instead (section 4). Apply per standard once İLTAREN's signal list is fixed (M1). |
| Sub-Nyquist / compressive sensing, cooperative sensing, multi-antenna detection, MCMC classification | Not applicable: Nyquist-rate single-receiver, single-antenna system; MCMC targets the same posterior as the tested EM classifier (CATALOG.md). |
| Further untested areas: subcarrier modulation classification inside OFDM, Morlet-wavelet symbol rate, detectors for DSSS / FHSS / PPM / PM / VSB / AFSK / π/4-DQPSK / D8PSK, learning-based methods | Not tested; see CATALOG.md section E for the reasons |
| MDL / AIC for emitter counting in wideband scenes | With one channel, MDL/AIC count signal-subspace dimensions (tones), not emitters; a wideband emitter spans many dimensions. Tested on its valid use case (section 4). |

## Reproduce

```bash
# WP4: all detectors, both datasets, comparison table + plots
cd wp4_detection && ..\.venv\Scripts\python run_all_wp4.py
# WP5
cd classical_amc && ..\.venv\Scripts\python compare_classifiers.py --features results\radioml_hgb\features.npz --out results\compare_7
cd classical_amc && ..\.venv\Scripts\python likelihood_bound.py
cd classical_amc && ..\.venv\Scripts\python practical_alrt.py --source radioml   # and --source torchsig (after extra_methods\gen_alrt_torchsig.py in .venv-torchsig)
# WP6
cd wp6_characterization && ..\.venv\Scripts\python run_wp6.py --dataset clean
cd classical_amc && ..\.venv\Scripts\python symbol_rate.py
# Section 4: waveform-specific and auxiliary methods
cd extra_methods && ..\.venv-torchsig\Scripts\python gen_symbol_rate_torchsig.py && ..\.venv\Scripts\python symbol_rate_torchsig_eval.py
cd extra_methods && ..\.venv\Scripts\python matched_filter.py && ..\.venv\Scripts\python ofdm_cp.py && ..\.venv\Scripts\python mdl_aic.py
# Section 5 (see each folder's README for the full command list)
cd classical_amc\extended && ..\..\.venv\Scripts\python bench_features.py --set 7 && ..\..\.venv\Scripts\python bench_features.py --set 24
cd classical_amc\extended && ..\..\.venv\Scripts\python bench_symbol.py --source radioml && ..\..\.venv\Scripts\python bench_symbol.py --source torchsig
cd detection_extended && bash run_heldout.sh                     # part A (calibrate first: see its README)
cd detection_extended && ..\.venv\Scripts\python partB_chirp.py bench && ..\.venv\Scripts\python partC_changepoint.py bench && ..\.venv\Scripts\python partC_changepoint.py report
cd characterization_extended && ..\.venv\Scripts\python carrier_crb.py && ..\.venv\Scripts\python snr_crb.py && ..\.venv\Scripts\python eval_symbol_rate.py && ..\.venv\Scripts\python blind_alrt.py merge
cd extra_methods && ..\.venv\Scripts\python hht_features.py && ..\.venv\Scripts\python stransform_detect.py
cd wp4_detection && ..\.venv\Scripts\python refine_wp4.py && ..\.venv\Scripts\python hough_wp4.py && ..\.venv\Scripts\python hough_wp4.py --max-side-ratio 0.5
```

Data: `D:\data\radioml2018\` (RadioML, use the paper's class order, not `classes.txt`), `D:\data\wideband\`
(TorchSig scenes; regenerate with `wp4_detection/generate_wideband.py` in `.venv-torchsig`).
