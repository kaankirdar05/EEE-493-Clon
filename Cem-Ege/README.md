# Cem-Ege: classical baseline for detection, identification and characterization

Classical (non-deep-learning) baseline for the V/UHF project: signal **detection**, modulation
**identification** and parameter **characterization**. Results were produced between 2026-10-15
and 2026-10-25. This is the reference the AI models (CNN / ResNet / YOLO) will be compared against.

Start with **`Classical_Baseline_Report.pdf`** (14 pages) or **`BASELINE.md`** (same content as text).
`CATALOG.md` lists every method family we looked at and whether it was tested; `WORK_SPLIT.md` is how the
work is divided between Cem, Ege and Ertuğ for jobs 5.1 + 6.1.

## What is in this folder

| Folder | Contents |
|---|---|
| `classical_amc/` | Identification (5.1). 19 standard statistical features (cumulants, spectral lines, amplitude/frequency statistics), 8 classifiers, ALRT/HLRT likelihood classifiers (ideal and blind), symbol-rate estimators, rule-based Azzouz–Nandi tree and Swami–Sadler hierarchy. Subfolders `extended/` (EM, KS, Bayesian, wavelet, extended cumulants) and `discovered/` (bispectrum, fractal, entropy, S-transform features, CvM/AD tests) |
| `wp6_characterization/` | Characterization (6.1). Carrier, bandwidth and SNR estimators scored on TorchSig scenes |
| `characterization_extended/` | Estimators vs the Cramér–Rao bound, symbol-rate vote of 12 estimators, blind likelihood classifier with the improved symbol rate |
| `wp4_detection/` | Detection framework and scoring: 14 detectors (energy, multitaper, FCME, LAD, CFAR variants, eigenvalue, cyclostationary, …), segmentation, Hough/wavelet/CUSUM refinement, TorchSig scene generator |
| `detection_extended/`, `detection_discovered/` | Further detectors: covariance / autocorrelation / entropy / spectral-correlation, chirp (WVD, FrFT), change-point family; eigenvalue family, Teager–Kaiser, stochastic resonance |
| `extra_methods/` | Waveform-specific and auxiliary methods: matched filter, OFDM cyclic prefix and blind OFDM parameter estimation, MDL/AIC, S-transform, Hilbert–Huang features, covariance-index detectors, MSK line-pair symbol rate, FSK band-limiting study, TorchSig symbol-rate frame generators |
| `real_data/` | First test on a real over-the-air recording (Allen Telescope Array, Pluto SDR, 753 MHz LTE) |

Each folder has its own `README.md` or docstrings with the run commands; the full list is in the
"Reproduce" section of `BASELINE.md`.

## Headline results (all held-out; see the report for tables)

| Area | Result |
|---|---|
| Identification, RadioML 7 classes | about 100 % at SNR ≥ 10 dB, 85 % from 2 dB (boosted trees on 19 features) |
| Identification, RadioML 24 classes | 63 % at SNR ≥ 10 dB, 70 % with histogram-entropy features |
| Likelihood classifier | optimal bound reaches 85 % at 8 dB vs 14 dB for features (6 dB gap); blind version 91.8 % at SNR ≥ 10 dB |
| Detection (TorchSig scenes) | energy detector box F1 67 % clean / 53 % with receiver impairments |
| Characterization | carrier error 0.4 % of the bandwidth, SNR error 0.45 dB median (clean) |
| Symbol rate | 91 % within 2 % (vote of 12 estimators, band-limited when the envelope is constant); GFSK/GMSK at 0 dB is a diagnosed threshold limit |
| Real recording | LTE carriers only ~2 dB above noise; receiver spurs dominate |

## Data (not in Git)

| Dataset | Where it comes from |
|---|---|
| RadioML 2018.01A | DeepSig; we used the Hugging Face mirror `Katherinezml/RML2018.01A`. **Use the class order of the RadioML 2018 paper, not the shipped `classes.txt`** (it does not match the HDF5 columns; checked against all 24 classes) |
| TorchSig wideband scenes | generated with `wp4_detection/generate_wideband.py` (TorchSig 2.2.0, needs PyTorch; separate virtual environment) |
| TorchSig modulator frames | `extra_methods/gen_symbol_rate_torchsig.py`, `gen_alrt_torchsig.py` |
| Real recording | Zenodo record 8242048, file `2023-08-03-15-38-39_atarfi.sigmf` (CC-BY-4.0) |

Large files (`.npz`, `.h5`, models, per-scene example images, calibration caches, virtual environments) are
left out, as in the repo's `.gitignore`; the scripts regenerate them.

## Environment

Python 3.14; `numpy`, `scipy`, `scikit-learn`, `h5py`, `matplotlib` (`classical_amc/requirements.txt`).
Scene generation needs `torchsig==2.2.0` (installs PyTorch) in its own environment.

**Caveat:** many scripts have the data locations written in the code (`D:\data\radioml2018\`,
`D:\data\wideband\`, `D:\data\real\`). Change them to your own paths before running.

## Notes for the team

- This baseline uses RadioML's 24 classes and TorchSig's signal list, **not** the team's 14-class
  Dataset 1 (`../classes.txt`, `../synthetic/`). Re-running the classical methods on Dataset 1 is a next step.
- TorchSig's `snr_db` is the spectrum **peak** over the noise level, not the in-band SNR (verified in its
  source, `utils/dsp.py`). It matches Es/N0 for PSK/QAM but is off by tens of dB for chirps and multi-tone FSK.
- Everything is synthetic except one public real recording.

## Open items

- **GFSK/GMSK symbol rate at 0 dB** is a diagnosed limit, not a bug: the estimators need about 12 dB in-band SNR for
  these low-deviation signals (`extra_methods/fsk_prefilter.py`). The constant-envelope gate closes the 5 dB case
  (GMSK 63 → 97 %), chosen on a dev set and scored once on held-out frames.
- **Completeness of the method list cannot be proven.** `CATALOG.md` section E documents how the list was built
  (search rounds that added 52, then 8, then 2 new families), why that shows saturation of the published taxonomies
  rather than completeness, and what is untested (subcarrier classification inside OFDM, DSSS/FHSS/PPM/PM/VSB/AFSK
  detectors, learning-based methods).
- **Real data:** one public recording only; our own SDR recordings (job 2.1) are still to do.
