# WP6: classical signal characterization (carrier, bandwidth, SNR)

Oracle detection: each TorchSig truth box sets the time span and a frequency search region (the box widened by 50 % of its width on each side, clipped to ±fs/2). Scenes 0–99 were used for tuning and for picking the "best" estimator. **All reported numbers are for scenes 100–299.**

```
..\.venv\Scripts\python run_wp6.py --dataset clean      # -> results/clean/
..\.venv\Scripts\python run_wp6.py --dataset impaired   # -> results/impaired/
..\.venv\Scripts\python run_wp6.py --dataset impaired --dev-only --noise map --out <dir>   # tuning
```

Outputs per dataset:
- `metrics.json`: overall, per-SNR-bin, per-family and per-envelope-group results, plus the estimator chosen on dev.
- `per_emitter.csv` (test) and `per_emitter_dev.csv`.
- `error_vs_snr.png` and `snr_scatter.png`.
- `psd_*.png`: dev-scene PSDs with the estimated edges, drawn as a sanity check.

## Method (`wp6/estimators.py`)
- **PSD.** Welch over the box's time span (Hann, 50 % overlap). The FFT size gives about 64 bins across the box, limited to 256–16384 and to at least 8 segments. Scaling: white noise of power s² gives s² per bin.
- **Noise floor.** Default `oob_lin`: a robust mean of the region on each side of the box (with a 10 % guard), interpolated linearly across the box. This was chosen on dev. The WP4 excision map (`--noise map`) was clearly worse on clean dev data: its local level varies by about ±20–70 % from cell to cell (SNR p90 8.6 dB vs 3.7 dB).
- **Carrier.** `fc_peak`, `fc_centroid` (power centroid of the noise-subtracted PSD inside the −26 dB edges), `fc_mid10` (midpoint of the −10 dB edges), `fc_obw99` (midpoint of the 99 % OBW) and `fc_noisecross`.
- **Bandwidth.** `bw_3db`, `bw_10db` and `bw_26db` are measured from the peak outward on the smoothed, noise-subtracted PSD. `bw_obw99` is the ITU-R SM.443 99 % band. `bw_noisecross` reproduces TorchSig's labelling rule (see below).
- **SNR.**
  - `snr_psd_obw` and `snr_psd_10db`: noise-subtracted power divided by noise power in the estimated band.
  - `snr_psd_trueband`: the same with the truth band.
  - `snr_peak512`: peak of the 512-point Blackman time-averaged PSD minus noise, divided by noise. This is the TorchSig definition.
  - `snr_m2m4`: M2M4 on the signal masked by FFT to the estimated 99 % band. It is clipped to [−30, 60] dB, and invalid moments are counted.

## TorchSig definitions found (torchsig 2.2.0, `utils/dsp.py:update_signal_snr_bandwidth`)
- **snr_db is a peak-PSD SNR, not an in-band power ratio.** It is computed as max over frequency of the time-mean of the *dB* spectrogram of the noise-free signal, minus the time- and frequency-mean of the dB noise spectrogram. Both spectrograms use a 512-point Blackman window and are non-overlapping. This measurement is taken before RX impairments are applied. Three consequences follow:
  1. In-band SNR estimators are biased by roughly −2 to −3 dB.
  2. Log averaging lowers noise-like spectra by about 2.5 dB, but the noise floor is lowered by the same 2.5 dB, so the two cancel. A tone is not lowered, so its label is about 2.5 dB too high: we measured −2.7 dB on tones.
  3. Chirps, LFM and LoRa sweep across the band. In each frame most bins sit near the −100 dB floor, so their labels come out about 25–48 dB *below* their real peak-PSD SNR. FSK with many tones (8/16-FSK) shows the same effect to a smaller degree (+4 to +10 dB). This is a labelling artifact, not an estimator error.
- **Bandwidth is the extent of the bins where the noise-free signal's max-hold spectrogram exceeds the log-mean noise + 3 dB, widened by half a bin.** In practice this is about +0.5 dB above the mean noise PSD, quantized to fs/512. The truth bandwidth therefore grows with SNR: the median is about 2× wider above 25 dB than below 5 dB. Fixed "x dB" bandwidths are consequently biased in an SNR-dependent way.
- The box is centred on the nominal carrier. Tones have bw = 1 Hz, so they are excluded from the bandwidth metrics and from the carrier-error/bandwidth metric.
- Emitters near the band edge are anti-alias filtered and clipped: about 14 % of emitters. They are flagged as `band_edge`, and `metrics.json` also reports results without them.
- In the impaired set, RX impairments (nonlinearity, IMD, gain steps, AGC) are applied after the labels are fixed.

`bw_noisecross` and `fc_noisecross` exploit the labelling rule, so they are partly circular. They score well because they mimic the label, not because they measure a physical bandwidth.
