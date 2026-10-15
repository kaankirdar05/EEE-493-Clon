# Dataset 1 (synthetic), version 2

**176,000 labelled I/Q clips, 11 classes, 16 SNR levels.** File `ds1_v2.h5`, 1.45 GB, made in about 1 minute with
`python make_synth_v2.py --per-snr 1000 --out ds1_v2.h5` (seed 2026). No SDR involved.

## Classes and how each is made

| Class | How it is generated |
|---|---|
| AM | double-sideband with carrier, modulation depth 0.3-0.9, random smooth message (0.5-3 % of the sample rate wide) |
| FM | same kind of message, peak deviation 1-5 % of the sample rate |
| 2FSK, 4FSK | continuous-phase FSK, modulation index 0.5-1.0 |
| GMSK | Gaussian filter BT = 0.3, modulation index 0.5 |
| BPSK, QPSK, 8PSK, 16QAM | root-raised-cosine pulses, roll-off 0.2-0.5 |
| OFDM | 64, 128 or 256 subcarriers, 30-70 % used, QPSK or 16QAM, cyclic prefix 1/4 or 1/8 |
| NOISE | no signal, noise only |

Every clip: 1,024 samples, 4-12 samples per symbol, random phase, frequency offset up to 0.2 % of the sample rate,
white Gaussian noise, scaled to average power 1. Sample rate is normalised to 1. `--fading` adds a light two-path echo.

## SNR labels

- `Z`: **Es/N0** for 2FSK, 4FSK, GMSK, BPSK, QPSK, 8PSK, 16QAM; **in-band SNR** for AM, FM, OFDM; NaN for NOISE.
- `Z_inband`: in-band SNR for every clip = signal power / noise power inside the 99 % occupied bandwidth (AM: 2 x message bandwidth).
- Levels: -10 to +20 dB in 2 dB steps. Will be adjusted once İLTAREN answers which SNR definition they use.

## File layout

| Name | Shape, type | Meaning |
|---|---|---|
| `X` | (N, 1024, 2) float32 | clips; `[:, :, 0]` = I, `[:, :, 1]` = Q |
| `Y` | (N,) int64 | class number, in the order of `attrs["classes"]` |
| `Z`, `Z_inband` | (N,) float32 | SNR labels in dB (see above) |
| `sps` | (N,) float32 | samples per symbol (0 for AM, FM, OFDM, NOISE) |
| `bw99` | (N,) float32 | occupied bandwidth as a fraction of the sample rate |
| `split` | (N,) int8 | 0 train (70 %), 1 validation (15 %), 2 test (15 %) |

Attributes: `classes`, `sample_rate`, `source`, `version`, `snr_definition`, `split_codes`, `settings`, `created`.

```python
import h5py, numpy as np
with h5py.File("ds1_v2.h5", "r") as f:
    classes = list(f.attrs["classes"]); Y, Z, split = f["Y"][:], f["Z"][:], f["split"][:]
    train = np.where(split == 0)[0]
    X_train = f["X"][train[:50000]]          # read only what you need: the full X is 1.45 GB
```

## Checks (`python check_synth.py ds1_v2.h5 --baseline`)

- Format: all checks pass (shapes, types, attributes, power 1, no NaN, 16,000 clips per class).
- SNR: measured in-band SNR from each clip's spectrum agrees with the stored label within 0.1-0.8 dB at 10 dB for every class.
- Sanity baseline (9 simple features + random forest, trained on 40,000 clips): 84 % at 20 dB, 58 % at 10 dB, 17 % at 0 dB.
  Accuracy rises with SNR as it should. Below about -2 dB it falls under chance (9 %) because weak signals are called NOISE.
  This only shows the labels are learnable; it is not our classifier.

## Known limits (planned for version 3)

- Class list, symbol rates and SNR definition are provisional until the WP1.2 answers.
- One signal per clip; no real hardware effects (IQ imbalance, DC spike, 8-bit rounding); fading only with `--fading`.
- Wideband scenes for the detector are not included yet.
