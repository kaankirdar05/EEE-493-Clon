# V/UHF AI signal detection, identification and characterization

EEE 493/494 project with TÜBİTAK BİLGEM İLTAREN. Bilkent University, 2026-2027.

## Folders

| Folder | What is in it |
|---|---|
| `synthetic/` | Dataset 1 generator (`make_synth_v3.py`), checker (`check_synth.py`), spectrogram images (`make_spectrograms.py`, `preview_spectrograms.py`), dataset card (`README.md`) |
| `sdr/` | Recording and preprocessing: raw SDR file -> SigMF -> HDF5 training clips, SNR measurement, practice kit |
| `classes.txt` | The shared class list (14 classes). Its order defines the class numbers in every dataset. Change it only together. |

## Quick start

```bash
pip install -r requirements.txt
cd synthetic
python make_synth_v3.py --per-snr 1000 --out ds1_v3.h5        # 224,000 clips, 1.84 GB
python check_synth.py ds1_v3.h5 --baseline                    # format checks, SNR check, test classifier
python make_spectrograms.py ds1_v3.h5 --out ds1_v3_spec.h5    # 64 x 64 spectrogram images, 0.92 GB
python preview_spectrograms.py ds1_v3_spec.h5                 # spectrogram_examples.png
```

## Data

Datasets are **not** stored in Git. Download them from the shared drive, or regenerate them with the commands above
(fixed seed, so everyone gets the same clips). Recordings (`.cu8`, `.sigmf-*`) also go to the shared drive.
