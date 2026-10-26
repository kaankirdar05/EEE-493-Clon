# Cem-Ege classical methods on Dataset 1 v3 (`ds1_v3.h5`)

Runs the classical baseline (identification, detection, characterization) on our own 14-class Dataset 1 instead
of RadioML / TorchSig. **No Cem-Ege code is modified**; every script here imports it from `../`.
Results produced 2026-10-26 on the test split (split 2, 33,600 clips); everything is trained or tuned on the
train split (split 0, 156,800 clips) only.

## Headline results

| Area | Best method | Result (test split) |
|---|---|---|
| Identification, 14 classes | 19 features + 24 Haar wavelet features, boosted trees | 90.2 % at Z >= 10 dB, 85 % from 12 dB (19 features alone: 78.7 %, 85 % from 16 dB) |
| Identification, BPSK/QPSK/8PSK/16QAM | ALRT likelihood, symbol rate known | 96.2 % at Z >= 10 dB, 85 % from 10 dB (features + HGB: 94.2 %, 12 dB) |
| Detection (signal vs NOISE per clip) | John / CAV covariance-eigenvalue detectors | Pd 90 % from 0 dB at Pfa 1e-2 (energy detector: 6 dB) |
| Carrier | power centroid | median error 2.1 % of the bandwidth |
| Bandwidth | 99 % occupied bandwidth | median relative error 15 % (2-4 % from 10 dB) |
| SNR | in-band PSD | median error 0.58 dB (true band) / 0.85 dB (estimated band) |
| Symbol rate | 12-estimator consensus (frozen) | 25 % within 2 % overall; PSK 95-98 % from 16 dB; 2FSK/4FSK/OOK never above 70 % |

Biggest confusions (Z >= 10 dB, 19 features): WBFM->NBFM 27 %, NBFM->WBFM 23 %, QPSK->PI4DQPSK 20 %,
4FSK<->NBFM 18 %. With wavelet features: NBFM->WBFM 19 %, WBFM->NBFM 13 %, QPSK->PI4DQPSK 11 %.

## Run (in this order)

```bash
# 0. Dataset 1 (not in Git): seed 2026, 224,000 clips, 1.84 GB
python ../../synthetic/make_synth_v3.py --per-snr 1000 --out <data>/ds1_v3.h5
export DS1=<data>/ds1_v3.h5                       # default in ds1common.py: /home/claude/data/ds1_v3.h5

python gen_offsets.py                             # carrier offset per clip (file does not store it); checks bit-identical regeneration
python extract_ds1_features.py 19                 # 19 features, ~0.5 ms/clip
python extract_ds1_features.py hist               # 8 histogram entropies
python extract_ds1_features.py wav                # 24 Haar wavelet features, ~5 ms/clip
python run_identification.py classifiers          # 8 models of compare_classifiers.py
python run_identification.py extended             # boosted trees on 19 + wavelet / entropy features
python run_detection.py                           # 16 detectors, Pfa 1e-2 and 1e-3
python run_characterization.py wp6                # carrier, bandwidth, SNR (wp6/estimators.py, oracle box)
python run_characterization.py sr                 # symbol rate (inst_freq_s3 + frozen consensus); ~90 ms/clip
python run_likelihood.py                          # ALRT/HLRT (needs the sr run's bank); ~150 ms/clip/variant
python run_identification.py rules                # Azzouz-Nandi tree, Swami-Sadler hierarchy
python make_figures.py                            # results/figures/*.png, results/tables/*.csv
```

Runtime on a 2-core, 8 GB machine: about 1.5 hours in total. Random forest needs > 4.5 GB on the full train
split, so it is trained on 60,000 clips (`RF_MAX_TRAIN`); SVM on 40,000 as in `compare_classifiers.py`.

## How each method was applied

- **SNR axis** = the dataset's `Z` label (Es/N0 for single-carrier digital classes, in-band SNR for AM, NBFM,
  WBFM, OFDM). NOISE clips have no SNR: they count in "all clips" and get their own accuracy.
- **Identification:** classifiers exactly as `compare_classifiers.make_model`; scoring adds per-class x SNR
  tables. Azzouz-Nandi uses its 7 RadioML classes mapped to AM, NBFM (as FM), GMSK, BPSK, QPSK, 8PSK, 16QAM.
  Swami-Sadler and ALRT use the 4 PSK/QAM classes; ALRT's candidate set still includes 64QAM (counted wrong).
  PI4DQPSK has no likelihood hypothesis; its clips are run to see what they are called (QPSK).
  Likelihood runs use 20 test clips per class and SNR (too slow for more: up to 340 symbols per clip).
- **Detection:** per-clip H0 (NOISE) vs H1. Block detectors' `block_stat` is applied to the whole 1,024-sample
  clip; the WP4 energy and PSD-occupancy detectors run on a 16 x 64 spectrogram of the clip (statistic = max
  cell). Thresholds from 20,000 fresh white-noise clips scaled to power 1 (how make_synth_v3 makes NOISE);
  measured Pfa on the 2,382 test NOISE clips is reported. CFAR, FCME, LAD, spectral kurtosis, multitaper and
  Anderson-Darling are not applicable: they need scenes with idle time or a known noise power, and every
  Dataset-1 clip is one continuous signal scaled to power 1.
- **Characterization:** `wp6.estimators.characterize` with default settings, oracle box = true carrier
  +- bw99/2, sample rate given as 10 MHz (their design value) with all frequencies scaled to match.
  Symbol rate success = within 2 % of 1/sps; consensus parameters are the frozen
  `../characterization_extended/results/consensus_params.json` (not re-tuned). 2FSK/4FSK clips with more than
  25 samples/symbol (16 % / 14 %) are outside every estimator's search range.

## Files

| Path | Contents |
|---|---|
| `results/identification/*.json` | full metrics, per class x SNR, confusion matrices |
| `results/detection/detection.json`, `detection_class_snr.csv` | thresholds, measured Pfa, Pd per class x SNR, both Pfa |
| `results/characterization/*.json`, `*_class_snr.csv` | median errors per estimator, class, SNR; symbol-rate success |
| `results/tables/*.csv` | compact tables (class rows x SNR columns), incl. `id_acc_vs_snr.csv` in the 5.1 format |
| `results/figures/*.png` | accuracy / Pd vs SNR, class x SNR heatmaps, confusion matrices |
| `results/*.log` | console output of every run |

Large caches (`features_*.npz`, `ds1_v3_offsets.npz`, `symbol_rate_bank.npz`, `wp6_errors.npz`, `pred_*.npy`)
are not included; the scripts regenerate them.
