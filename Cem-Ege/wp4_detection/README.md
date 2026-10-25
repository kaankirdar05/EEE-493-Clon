# WP4 — Detection: energy detector (4.1), segmentation (4.3), shared scoring

> **Update:** 14 classical detectors are now in `wp4/detectors.py` (energy, multitaper, FCME, LAD,
> PSD occupancy, CA/GO/SO/OS/2-D CFAR, spectral kurtosis, eigenvalue MME, cyclostationary,
> Anderson–Darling), all with thresholds calibrated on white noise. Run them with
> `python run_all_wp4.py`; results are in `results/comparison.md` and `results/compare_*.png`,
> and summarised in `../BASELINE.md`. The sections below describe the energy detector in detail.

Detects transmissions in wideband spectrograms and scores them the same way for every
detector, so **energy (4.1) vs CA-CFAR (4.2, Kağan) vs YOLO (4.4)** is a fair comparison.

```
I/Q scene ──► spectrogram (512×512) ──► detector ──► mask ──► segmentation ──► boxes ──► scoring
              wp4/spectrum.py            wp4/energy.py           wp4/segmentation.py     wp4/scoring.py
```

## Data: WidebandSig53-style scenes from TorchSig

TorchSig 2.2 no longer has a `WidebandSig53` class (that was the 0.x API), so
`generate_wideband.py` builds the same kind of scenes from TorchSig's current generator:
262,144 samples at 10 MS/s (512×512 spectrogram), 1–6 emitters per scene from 69 signal
types, SNR −5…30 dB, bursts 5–100 % of the scene, bandwidths fs/100…fs/8, no overlap.

| File | Impairments | Scenes |
|---|---|---|
| `D:\data\wideband\scenes.h5` | level 2 (TX/RX hardware, AGC, gain steps, IQ imbalance, fading) | 300 |
| `D:\data\wideband\scenes_clean.h5` | none | 300 |

Each file has `iq` (N × 262144 complex64) and a `boxes` table (scene, class, SNR, start/stop
time as 0–1 fractions, low/high frequency in Hz). **Scenes 0–99 were used to tune segmentation;
all reported numbers use scenes 100–299.**

## Setup

TorchSig (needs PyTorch) is only needed to *generate* data:

```bash
py -3 -m venv .venv-torchsig
.venv-torchsig\Scripts\python -m pip install torchsig==2.2.0
.venv-torchsig\Scripts\python generate_wideband.py --n 300 --out D:\data\wideband\scenes.h5
.venv-torchsig\Scripts\python generate_wideband.py --n 300 --impairments 0 --out D:\data\wideband\scenes_clean.h5
```

Detection and scoring need only numpy / scipy / h5py / matplotlib (same env as `classical_amc`):

```bash
python run_wp4.py --data D:\data\wideband\scenes.h5 --out results\energy_impaired
python run_wp4.py --data D:\data\wideband\scenes_clean.h5 --out results\energy_clean
```

## Energy detector (4.1): how it works

1. **Spectrogram** — non-overlapping Hann frames, scaled so a noise cell is σ²·Exp(1).
2. **Smoothing** — average `kt = 8` frames in time → Gamma(8) under noise. (Averaging adjacent
   frequency bins is supported, but Hann-windowed bins are correlated, so the code uses an
   *effective* number of looks; `kf = 1` is exactly calibrated.)
3. **Noise floor** (`--noise-method excision`, default) — noise = gain(t) × shape(f).
   Real receivers (and TorchSig impairments) have a curved passband and gain steps / AGC,
   so one global noise level fails badly (measured Pfa 0.28 instead of 0.001). The shape and
   gain are estimated, then re-estimated from noise-only cells after removing signal-sized
   blobs, so wide long-lasting signals don't raise the floor under themselves.
4. **Threshold** — σ̂²(t,f) · Gamma⁻¹(L, 1 − Pfa) / L for the requested Pfa.

On pure white noise the measured Pfa matches the target: 10⁻² → 1.0×10⁻², 10⁻³ → 0.96×10⁻³,
10⁻⁴ → 0.9×10⁻⁴.

## Segmentation (4.3)

Closing (17 time × 5 freq) to bridge holes, opening (3×3) to drop specks, connected components,
bounding boxes, minimum 120 cells (just under the smallest possible TorchSig emitter).

## Scoring (shared)

- **Cell Pfa** — flagged cells outside all truth boxes ÷ cells outside all truth boxes
- **Emitter Pd** — emitter counted as detected if ≥ 50 % of its box cells are flagged
- **Box precision / recall / F1** — one-to-one Hungarian matching, hit if IoU ≥ 0.5
- Per-SNR curves, a Pfa sweep (10⁻¹ … 10⁻⁶) and example images with truth / detected boxes

## Results — energy detector, Pfa target 10⁻³, scenes 100–299

| | Impaired (level 2) | Clean |
|---|---|---|
| Box precision / recall / F1 | 48.8 / 58.1 / **53.0 %** | 72.5 / 62.8 / **67.3 %** |
| Box recall, SNR ≥ 10 dB | 75–80 % | 74–83 % |
| Box recall, 0–5 dB | 40 % | 52 % |
| Mean IoU of matched boxes | 0.79 | 0.81 |
| Measured cell Pfa | 2.1×10⁻² | 5.2×10⁻³ |
| Time per scene (detector + segmentation) | ~90 ms | ~95 ms |

### Why measured Pfa is above target on scenes (but exact on noise)

Most "false alarms" are real energy that the truth boxes don't label:
- **Unlabelled emissions from impairments**: IQ-imbalance images (a mirrored copy of a chirp on
  the other side of the band), intermodulation products, amplifier spectral regrowth, DC spike.
- **Signal skirts** beyond TorchSig's nominal bandwidth (pulse shaping, FM/FSK spread). This is
  why even the clean set sits at 5×10⁻³.
- **Gain steps that also change the noise shape** mid-scene (scenes 98, 94, 50). The separable
  noise model assumes one shape per scene. A local detector such as **CA-CFAR should do better
  here** — worth checking in the comparison.
- **Band edges** (first/last few bins): the receiver roll-off is steeper than the noise-shape
  smoothing can follow.

Emitter Pd plateaus around 80 % at high SNR because sparse signals (ADS-B pulses, LoRa/chirp
sweeps, bursty OFDM, frequency-hopping BTLE) fill only part of their bounding box. That
reflects how the box labels are defined, not missed signals.

## Adding CA-CFAR (Kağan) or any other detector

Write a function with this signature and register it in `DETECTORS` in `run_wp4.py`:

```python
def ca_cfar(spec, pfa, **kw):
    # spec: (T, F) power spectrogram from wp4.spectrum.spectrogram
    # return mask (T, F) bool, info dict
    ...
    return mask, {}
```

```bash
python run_wp4.py --data D:\data\wideband\scenes.h5 --detector ca_cfar --out results\cfar_impaired
```

Segmentation, scoring, curves and example images are then identical, so the results tables
line up directly with the energy detector's.
