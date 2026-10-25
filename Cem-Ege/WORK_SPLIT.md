# Work split: classical identification and characterization (jobs 5.1 + 6.1)

**One joint package by Cem, Ege and Ertuğ**, for Lab Meeting 1 (15 October). Detection is Kağan's
(energy detector, CFAR, YOLO) and is not part of this package. Our own detection runs
(`wp4_detection/`, `detection_extended/`, `detection_discovered/`) can go to Kağan as comparison material.

Inside the package each of us has one role, so every contribution is visible. Today is a talk with one or
two visuals, no slides.

## Roles

### Cem: features and classifiers (5.1)
- 19 handcrafted features: cumulants (C40, C42, …), power-law spectral lines (x², x⁴, x⁸),
  amplitude and instantaneous-frequency statistics, spectrum shape.
- 8 classifiers on the same features (boosted trees, random forest, SVM, k-NN, LDA, decision tree, …).
- Results (RadioML 2018.01A): 7 classes ~100 % at SNR ≥ 10 dB, 85 % from 2 dB; 24 classes 63 %,
  **70 % with entropy features**.
- **Say:** the classifier barely matters, the features are the limit.
- **Show:** `classical_amc/results/compare_7/acc_vs_snr_all_models.png`
- Code: `classical_amc/train_feature_classifier.py`, `compare_classifiers.py`, `classical_amc/extended/`, `classical_amc/discovered/`

### Ege: theoretical limit and statistical classifiers (5.1)
- Likelihood-ratio classifier (ALRT / HLRT): ideal bound and fully blind version
  (estimates symbol rate, timing and frequency offset itself).
- EM constellation fitting, Kolmogorov–Smirnov / Cramér–von Mises tests, Bayesian classifiers.
- Symbol-rate estimation, which the blind likelihood classifier needs (90 % within 2 %).
- Results: likelihood reaches 85 % at 8 dB vs 14 dB for features (**~6 dB headroom for AI**);
  blind version 91.8 % at SNR ≥ 10 dB; EM 95 %.
- **Say:** likelihood is the best classical method if the symbol rate is right; this is the bar AI must beat.
- **Show:** `classical_amc/results/likelihood_bound/acc_vs_snr.png`
- Code: `classical_amc/likelihood_bound.py`, `practical_alrt.py`, `symbol_rate.py`, `characterization_extended/`

### Ertuğ: characterization (6.1) and robustness / real data
- Carrier (spectrum peak / centroid, Quinn / Kay refinement), bandwidth (−x dB, 99 % occupied, ITU-R SM.443),
  SNR (spectrum-based, M2M4), compared with the Cramér–Rao bound.
- Rule-based classifiers from the requirements document: Azzouz–Nandi tree (98.6 % at SNR ≥ 10 dB),
  Swami–Sadler hierarchy (64 %).
- What breaks the classical methods: receiver impairments, TorchSig's SNR label definition (peak-PSD, not
  in-band), and the real Pluto recording (LTE only ~2 dB above noise; receiver spurs dominate).
- Results: carrier error 0.4 % of bandwidth, SNR 0.45 dB median error (clean).
- **Say:** near-optimal on synthetic data; on real data weak signals and receiver spurs are the problem.
- **Show:** `real_data/results/atarfi_psd.png` (and `wp6_characterization/results/clean/error_vs_snr.png`)
- Code: `wp6_characterization/`, `characterization_extended/`, `classical_amc/requirements_methods.py`, `real_data/`

## Against the CM1 deliverables (November, week 8, P1 deck slide 19)

| CM1 item | Owner | Status today |
|---|---|---|
| Cumulant classifier: accuracy vs SNR (5.1) | Cem, Ege, Ertuğ | Done |
| First carrier / bandwidth estimates (6.1) | Cem, Ege, Ertuğ | Done |
| Shared evaluation scripts (8.1) | team | Done (scoring + `acc_vs_snr.csv` format) |
| Energy detector with Pd / Pfa per SNR (4.1) | Kağan | His part |
| Signed-off requirements + literature summary (1.1, 1.2) | team | Requirements document drafted; needs İLTAREN sign-off |
| Labeled synthetic dataset v1 (2.2) | open | **Not done**: we used RadioML and TorchSig, not our own generator |
| SDR recordings with spectrum / waterfall plots (2.1) | open | **Not done**: only one public recording tested |

Between now and CM1 the open items are our own dataset v1 and the first SDR recordings.

## Shared, not anyone's share
- Data setup (RadioML loader and class-order fix, TorchSig scene generator) and the method catalogue (`CATALOG.md`).
- Full write-up for ourselves: `Classical_Baseline_Report.pdf`, `BASELINE.md`.
