# V/UHF AI signal detection, identification and characterization

EEE 493/494 project with TÜBİTAK BİLGEM İLTAREN. Bilkent University, 2026-2027.

## Folders

| Folder | What is in it |
|---|---|
| `synthetic/` | Dataset 1 generator (`make_synth_v2.py`), checker (`check_synth.py`), dataset card (`README.md`) |
| `sdr/` | Recording and preprocessing: raw SDR file -> SigMF -> HDF5 training clips, SNR measurement, practice kit |
| `classes.txt` | The shared class list. Its order defines the class numbers in every dataset. Change it only together. |

## Quick start

```bash
pip install -r requirements.txt
cd synthetic
python make_synth_v2.py --per-snr 1000 --out ds1_v2.h5     # about 1 minute, 1.45 GB
python check_synth.py ds1_v2.h5 --baseline                 # format checks, SNR check, quick baseline
```

## Data

Datasets are **not** stored in Git. Download `ds1_v2.h5` from the shared drive, or regenerate it with the command above
(fixed seed, so everyone gets the same clips). Recordings (`.cu8`, `.sigmf-*`) also go to the shared drive.
