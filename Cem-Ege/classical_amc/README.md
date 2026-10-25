# Job 5.1 — Feature-based modulation classifier (classical baseline)

Classical automatic modulation classification on **RadioML 2018.01A**:
19 hand-crafted features per frame → gradient-boosted trees (default) or RBF SVM.
Output is the accuracy-vs-SNR curve and confusion matrix that milestone M3 asks for,
in a format the 5.2 CNN can be plotted against.

## Features (`amc/features.py`)

| Group | Features | What they separate |
|---|---|---|
| Higher-order cumulants | \|C20\|, \|C40\|, \|C41\|, C42, \|C60\|, \|C80\| | PSK order, QAM vs PSK (BPSK \|C40\|=2, QPSK 1, 8PSK 0, 16QAM 0.68) |
| Power-law spectral lines | peak/mean of \|FFT(x^k)\|, k = 1, 2, 4, 8 | BPSK / QPSK / 8PSK, robust to frequency offset; AM carrier |
| Amplitude | σ_aa, γ_max, amplitude kurtosis | AM vs constant-envelope signals |
| Instantaneous frequency | σ_f, σ_af, kurtosis | FM and GMSK vs linear modulations |
| Spectrum shape | flatness, 99 % bandwidth fraction, asymmetry | SSB vs DSB, narrow vs wide signals |

Cumulant formulas are checked against theory (BPSK 2 / −2 / 16 / 272, etc.).

## Setup

```bash
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download `GOLD_XYZ_OSC.0001_1024.hdf5` (~20 GB) from DeepSig (deepsig.ai/datasets, free sign-up).
Licence is CC BY-NC-SA: fine for the course; tell İLTAREN before reusing models trained on it.

## Run

```bash
# 1. No dataset needed: whole pipeline on fake data with the real file layout
python train_feature_classifier.py smoke

# 2. Once, on the real file: confirm which one-hot column is which class
python train_feature_classifier.py check-labels --data D:\data\GOLD_XYZ_OSC.0001_1024.hdf5

# 3. Train + evaluate (default: 7 classes x 26 SNRs x 1000 frames)
python train_feature_classifier.py train --data D:\data\GOLD_XYZ_OSC.0001_1024.hdf5
python train_feature_classifier.py train --data ... --model svm
```

Useful options: `--classes BPSK,QPSK,8PSK,16QAM`, `--per-group 300` (quick run),
`--snr-min -10 --snr-max 20`, `--rebuild` (ignore cached features).

**Class-order warning.** The `classes.txt` shipped with the dataset (often copied around
as the "fixed" order) does **not** match the HDF5 one-hot columns. The list in the RadioML
2018 paper does. We checked all 24 columns of the file against their physical signatures
(x² / x⁴ spectral lines, real vs complex, constant envelope, SSB asymmetry), so the default
is `--order hdf5`. `check-labels` repeats the BPSK/QPSK test on any copy of the file; run it
before trusting results, and use this order in the 5.2 CNN loader too.

## Outputs (`results/<model>/`)

- `metrics.json`: overall accuracy, accuracy at SNR ≥ 10 dB, per-SNR, per-class, timing (ms/frame)
- `acc_vs_snr.csv`: `snr_db,accuracy`. **5.2 should write the same file** so both curves go on one M3 plot.
- `acc_vs_snr.png`, `confusion_all.png`, `confusion_high_snr.png`
- `features.npz` (cache), `model.joblib`

## Default class subset

`AM-DSB-WC, FM, BPSK, QPSK, 8PSK, 16QAM, GMSK`. This is the slide-17 candidate set mapped
onto RadioML 2018: GMSK stands in for GFSK/CPFSK, which this dataset does not contain.

## Status / caveats

- Tested only on the built-in synthetic smoke data (99.5 % at SNR ≥ 10 dB, 75 % overall).
  That data is much easier than RadioML; expect clearly lower numbers on the real file.
- RadioML is a stand-in until our own dataset v1 (job 2.2) exists. `extract_features`
  takes any `(N, L)` complex array, so switching datasets only changes the loader.
