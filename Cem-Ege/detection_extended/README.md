# detection_extended: more classical detection methods

This folder extends the WP4 classical detection baseline (`../wp4_detection`). Nothing outside this
folder was edited. The WP4 framework (`wp4.detectors`, `segmentation`, `scoring`, `run_wp4.plot_*`) is
imported unchanged. Parameters were tuned on dev data only (scenes 0–99, or synthetic seeds). All
reported numbers use held-out scenes 100–299.

```
ext/block_detectors.py   Part A: CAV, spectral entropy, Zhang histogram entropy, autocorrelation,
                         FAM spectral-correlation (SCD) detector, optional prewhitening wrapper
ext/chirp.py             Part B: (S)PWVD, spectrogram, IF-line Radon sums, dechirp / FrFT
run_partA.py             calibrate | verify | run  (same loop and outputs as run_wp4.py, parallel)
calibrate_scd.py         parallel white-noise calibration for the SCD detector (400k blocks)
run_heldout.sh           held-out runs with the dev-chosen parameters; compare_partA.py builds the table
sanity_scd.py            Part A.4 sanity check (BPSK SCD)
diag_noise_blocks.py     diagnostic: false alarms on scene blocks with no signal
time_partA.py            single-process timing
partB_chirp.py           sanity | bench   (chirp detection)
diag_wvd_deflection.py   diagnostic: WVD line statistic vs lag window and time sampling
partC_changepoint.py     sanity | bench | report   (sequential change-point detection)
results/                 all JSON, CSV, PNG and markdown outputs (paths listed below)
```

Python: `..\.venv\Scripts\python.exe`, with `HDF5_USE_FILE_LOCKING=FALSE` set by the scripts.

---

## Part A: new block detectors in the calibrated framework

Each detector subclasses `wp4.detectors.BlockDetector`. It uses the same coarse grid as `eigen_mme`
and `cyclo`: 32 time blocks × 16 sub-bands, each block a 512-sample complex sub-band signal.
Thresholds are the (1 − Pfa) quantile of the statistic on white complex Gaussian blocks, the same as
`Detector.calibrate`. That means 200k blocks (400k for SCD), cached in `results/calibration/`.

| name | statistic |
|---|---|
| `cav` | Covariance absolute value (Zeng & Liang 2009), T1/T2 of the Toeplitz L×L sample covariance, L = 4 |
| `acorr` | Portmanteau statistic N·Σ_{l=1..L} \|ρ(l)\|², L = 32. Gamma(L,1) under H0 |
| `entropy_spec` | log m − H(normalised Welch PSD), Hann segments m = 64 |
| `entropy_hist` | Zhang et al. (2010): entropy of a 16-bin histogram of \|FFT\| |
| `scd_fam` | FFT Accumulation Method SCD. Max squared spectral coherence over (f, α), non-conjugate α ≠ 0 (\|α\| ≥ 4/N) and conjugate. Channeliser Np = 16, hop Np/4 |
| `*_pw` | Same statistic after scene-adaptive prewhitening per sub-band (mitigation attempt, see below) |

**Dev tuning** (`results/dev_tuning/summary.md`): L ∈ {4,8,16,32}, m ∈ {32,64,128}, bins ∈ {8,16,32},
Np ∈ {16,32}. The choice was the best mean dev F1 over the two datasets.

### Sanity checks

1. **Pfa on fresh white noise** (`results/pfa_check/*.json`). The seed differs from the calibration
   seed. Results are block-level on 200k–400k fresh blocks, plus the full `stat(spec, iq)` pipeline
   on 40 noise scenes.

   | target | cav_L4 | acorr_L32 | entropy_spec | entropy_hist | scd_fam_Np16 |
   |---|---|---|---|---|---|
   | 1e-1 | 9.9e-2 | 9.9e-2 | 1.0e-1 | 1.0e-1 | 9.9e-2 |
   | 1e-2 | 1.0e-2 | 1.0e-2 | 9.9e-3 | 1.0e-2 | 9.7e-3 |
   | 1e-3 | 9.0e-4 | 9.0e-4 | 9.7e-4 | 8.8e-4 | 9.2e-4 |
   | 1e-4 | 1.2e-4 | 7.5e-5 | 8.5e-5 | 7.5e-5 | 9.0e-5 |

   At 1e-4 there are only 15–40 exceedances, so ±25 % is binomial noise. The `acorr` thresholds also
   match theory: Gamma(8,1) at 1e-3 gives 19.6, and the calibrated value is 19.7. For the `_pw`
   variants the measured Pfa is 0.93–1.2e-3 at block level but 1.5–1.6e-3 on the 40-scene full
   pipeline. Whitening makes the blocks of one scene correlated, so their calibration is less precise.
2. **SCD of synthetic BPSK** (`results/sanity_scd_bpsk.png/.json`), with rectangular pulses,
   Rs = 1/8 and fc = 0.05 cycles/sample at 20 dB. Checked at N = 4096 (Np = 64) and N = 512
   (Np = 32 and 16).
   - The non-conjugate SCD peaks at α = ±0.125 = ±Rs, with coherence 0.89–0.97.
   - The conjugate SCD peaks at α = 0.1 = 2fc and at 2fc + kRs.
   - All three strongest peaks lie on the theoretical set in every case: **PASS**.
   - The maximum on white noise is 0.09–0.37.

   A first version of the check accepted only the strongest harmonic. It falsely flagged Np = 16,
   whose strongest conjugate peak was 2fc + 4Rs: a valid feature for rectangular pulses.

### Held-out results (scenes 100–299, Pfa target 1e-3)

The full table with reference rows is in `results/partA_comparison.md`. Recall-vs-SNR plots are in
`results/partA_recall_vs_snr_{impaired,clean}.png`. Per-detector folders are `results/<det>_<dataset>/`,
each with `metrics.json`, `pd_vs_snr.csv/png`, `roc.png` and examples.

| detector | impaired F1 / P / R | rec 0–5 / 10–15 | Pd | meas. Pfa | clean F1 / P / R | rec 0–5 / 10–15 | Pd | meas. Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|---|
| ref energy | 53.0 / 48.9 / 57.8 | 40 / 79 | 49.8 | 2.1e-2 | 67.4 / 72.7 / 62.8 | 52 / 74 | 53.8 | 5.1e-3 | 318 |
| ref eigen_mme | 4.7 / 3.4 / 7.4 | 4 / 10 | 87.2 | 4.2e-1 | 37.5 / 35.4 / 39.9 | 26 / 50 | 79.2 | 8.9e-2 | 298 |
| ref cyclo | 12.3 / 9.2 / 18.4 | 2 / 21 | 29.0 | 4.0e-2 | 19.6 / 19.0 / 20.2 | 3 / 18 | 28.0 | 2.3e-2 | 106 |
| cav_L4 | 6.6 / 4.6 / 12.2 | 5 / 13 | 82.6 | 3.2e-1 | 36.8 / 33.3 / 41.1 | 27 / 55 | 78.3 | 7.1e-2 | 148 |
| acorr_L32 | 7.3 / 5.5 / 10.8 | 4 / 15 | 82.3 | 3.4e-1 | 38.7 / 36.0 / 41.8 | 23 / 55 | 71.1 | 6.4e-2 | 665 |
| entropy_spec | 7.5 / 5.7 / 11.1 | 3 / 16 | 81.3 | 3.3e-1 | **39.1** / 36.6 / 42.0 | 20 / 57 | 69.7 | 6.2e-2 | 117 |
| entropy_hist | 6.4 / 4.5 / 10.7 | 2 / 12 | 36.4 | 1.3e-1 | 12.5 / 11.0 / 14.4 | 3 / 10 | 27.9 | 3.5e-2 | 112 |
| scd_fam_Np16 | **16.2** / 15.4 / 17.1 | 1 / 21 | 34.2 | 5.4e-2 | 24.2 / 25.0 / 23.4 | 5 / 24 | 33.4 | 2.6e-2 | 4164 |
| cav_L4_pw | 8.7 / 6.1 / 15.2 | 8 / 19 | 57.2 | 1.3e-1 | 15.5 / 12.2 / 21.3 | 15 / 29 | 57.3 | 8.9e-2 | 197 |
| entropy_spec_pw | 11.8 / 9.1 / 16.8 | 3 / 24 | 52.9 | 1.3e-1 | 19.7 / 16.3 / 25.0 | 7 / 32 | 49.3 | 6.1e-2 | 166 |

ms/scene is single-process `stat()` time on a machine shared with other jobs, so treat it as an upper
bound. SCD fell from 16 s to 4 s after replacing an `einsum` with a matmul (identical output,
max diff 6e-16).

### Reading the Part A results

- **None of the new block detectors comes near energy or CFAR on box F1.** All decide on coarse
  16×32-cell blocks, so boxes are blocky. Mean IoU of matched boxes is 0.65–0.71.
- **The whiteness tests (CAV, autocorrelation, spectral entropy) behave like `eigen_mme`.**
  - Clean: F1 37–39, against 37.5 for MME.
  - Impaired: F1 7–8, with measured Pfa 0.32–0.34 instead of 1e-3.
  - **This is a genuine limitation, not a bug** (`results/diag_noise_blocks.json`). On scene blocks
    that touch no truth box (1-block margin, dev scenes):
    - Clean scenes: the block Pfa is 1.8e-3 for CAV and autocorrelation and 4.4e-3 for spectral entropy.
    - Impaired scenes: the block Pfa is 0.26–0.36.
    - The impaired false alarms are concentrated in the band-edge sub-bands 0, 1, 14 and 15, where the
      CAV false-alarm rate is 0.73–0.94. Outside them it is 0.07–0.23.
    - In dev scenes 0–2 the median noise |ρ(1)| is 0.37–0.74 in sub-bands 0 and 15 and 0.12–0.17 in
      sub-bands 1 and 14, against 0.04 for white noise. Sub-band 0/15 power is up to about 10 dB lower.
    - This is the TorchSig level-2 receiver filter roll-off colouring the noise. Every whiteness test
      reads coloured noise as signal, as Zeng & Liang note: their tests need prewhitening.
- **Prewhitening (`*_pw`) only partly helps, so treat it as a negative result.** The noise PSD shape
  per sub-band was estimated as the 25 % quantile over time blocks of the raw periodograms.
  - It cuts impaired free-block Pfa from 0.34 to 0.11 and raises impaired F1 (CAV 6.6 → 8.7,
    entropy 7.5 → 11.8).
  - It costs a lot on clean scenes: F1 drops 37–39 → 16–20, and free-block Pfa rises to 0.03–0.04.
  - The cause: a time-invariant noise-shape estimate absorbs long-lasting signals, and the noise-only
    blocks of that sub-band then look coloured. A synthetic test with a 30 % duty-cycle signal showed
    the earlier per-block-normalised versions failing completely (block Pfa 1.0).
  - Two earlier normalisations (by block mean, then by block median) were bugs of this kind. They were
    replaced, and are documented in `whiten_subbands`.
- **`entropy_hist` (Zhang) is weak.** The \|FFT\| histogram changes only when a few bins dominate, so
  it reacts mainly to strong narrowband signals.
- **SCD-FAM is the best coarse detector on impaired scenes.**
  - F1 16.2, against 12.3 for `cyclo` and 13.7 for `gof_ad`, with the lowest measured Pfa among the
    coarse detectors (5.4e-2). Spectral coherence is normalised per channel, so stationary coloured
    noise does not create cyclic features.
  - On clean scenes it beats `cyclo` (24.2 vs 19.6) but loses to the whiteness tests.
  - It is blind at low SNR (recall 1–5 % at 0–5 dB). Features need Rs below the 625 kHz sub-band width
    and enough cycles in 512 samples.
  - Constant-envelope FM/FSK and OFDM with no cyclic prefix in the block show weak 2nd-order features:
    a known limitation, not a bug.
  - It is about 30× slower than energy.

---

## Part B: linear-chirp detection (synthetic)

**Model.** Frame N = 4096 in unit complex AWGN. Chirp s(n) = A exp(j(2πf0n′ + πkn′²)).
- Parameters are unknown to every detector: f0 ~ U[−0.2, 0.2], sweep B = |k|N ~ U[0.05, 0.5] with a
  random sign, and a random phase.
- Each detector maximises over its own grid.
- Thresholds are the empirical 99 % quantile over 2000 noise frames. Measured Pfa is 1.0e-2 for all.
- 300 frames per SNR.

**Statistics:**
- radiometer
- spectrogram max cell (128-Hann, hop 16, 512 bins)
- spectrogram IF-line Radon sum over 257 rates × 512 offsets
- PWVD Radon sum: Hann lag window of 255 lags of the 2×-upsampled signal (avoids discrete-WVD aliasing)
- SPWVD Radon sum: plus 64-sample time smoothing
- FrFT / dechirp peak: 2049 rates, step 2/N² (equivalently the FrFT angle α = arccot(−kN)), 2N-point
  FFT, parabolic refinement of the rate

### Sanity checks (`results/partB/sanity_chirp.json/.png`): PASS

- **WVD concentration.** For a clean chirp (B = 0.3) the PWVD and SPWVD put 99.3 % and 98.9 % of their
  positive energy within ±2 bins of the true IF line. The spectrogram puts 58.6 % there. The per-time
  peak is ≤ 0.52 bin from the IF.
- **FrFT angle.** Over a 401-point angle scan the |FrFT| peak sits exactly at the theoretical
  α = arccot(−kN) = 1.8621 rad (step 2.5e-4). For an on-grid chirp the dechirp gives k̂/k = 1.000 and
  peak/N = 1.000, i.e. full coherent gain. An off-grid chirp loses 0.82 (−0.85 dB) to rate and
  frequency straddle.
- **Radon–Wigner identity.** The line integral of the full WVD (rectangular lag window over the whole
  frame, summed at every sample) gives the same rate argmax as the dechirp at −15 dB. The two rate
  profiles have correlation 0.99995.

### Bug found and fixed

The first benchmark run (`results/partB/chirp_bench_v1_decimated_wvd.*`, kept for the record) put the
PWVD Radon statistic *below* the spectrogram Radon statistic (Pd = 0.5 at −16.5 dB vs −20.7 dB).
`diag_wvd_deflection.py` (`results/partB/diag_wvd_deflection.json`, −20 dB, true line, deflection)
found the cause:
- Summing the WVD only on the hop-16 time grid uses just 1/16 of the sample pairs.
- Deflection with this hop-16-grid-only sum: full-frame WVD 8.5, 255-lag PWVD 2.2.
- Deflection when every sample is summed: full-frame WVD 38.2, 1023-lag 23.0, 255-lag 12.8.
- For comparison, the dechirp gives 51 (theory N·SNR = 41, with 100 trials) and the spectrogram 7.6.

The PWVD now sums every sample within each hop (boxcar) before decimating. All Part B numbers below
come from the fixed version.

### Results (`results/partB/chirp_bench.json/.png`)

| detector | SNR for Pd = 0.5 at Pfa 1e-2 | Pd at −22 dB | Pd at −20 dB | median \|B̂ − B\| at −20 dB (cycles/sample) |
|---|---|---|---|---|
| radiometer | −14.6 dB | 0.03 | 0.04 | – |
| spectrogram max cell | −13.4 dB | 0.01 | 0.02 | – |
| spectrogram Radon | −20.7 dB | 0.18 | 0.69 | 1.9e-3 |
| PWVD Radon | −21.7 dB | 0.43 | 0.89 | 1.1e-3 |
| SPWVD Radon | −21.4 dB | 0.36 | 0.87 | 1.1e-3 |
| **FrFT / dechirp** | **−23.1 dB** | 0.75 | 0.99 | **7.6e-5** |

- The FrFT is about 1.4 dB ahead of the PWVD and 2.4 dB ahead of the spectrogram Radon sum. Exploiting
  the chirp structure gains about 8.5 dB over a radiometer.
- The TF-Radon rate error floors at about 1e-3. This is the 257-rate grid quantisation (step 1/256 in
  B): a grid limit, not an estimator limit. The FrFT error is below 1e-4 above −22 dB.
- Below about −24 dB every method fails, and the rate estimates are random (error about 0.3).
- PWVD Radon beats SPWVD Radon slightly here. Time smoothing reduces cross-terms, but a single chirp
  has no auto–auto cross-terms to remove.

**TorchSig chirp emitters: not done.** Held-out scenes contain only 26 chirp-family emitters
(chirpss 9, lfm-data 9, lfm-radar 4, lora 4), with median SNR 9–21 dB, where energy already works.
LoRa and chirpss are repeated, wrapping chirps, so turning a dechirp search into masks and boxes needs
its own segmentation design. 26 emitters also give ±10 pp confidence intervals. The synthetic
advantage is at negative per-sample SNR, a regime these scenes barely contain.

---

## Part C: sequential change-point detection of burst onset (synthetic)

**Model.**
- Frame energy E_t of M = 64 complex samples, with known noise power. E_t is exactly Gamma(M, 1)
  before the change and Gamma(M, 1 + ρ) after it, for a Gaussian signal.
- Onset ν ~ U[100, 300]. 1000 sequences per SNR.
- Each procedure's threshold is set by bisection to ARL0 = 1000 frames on 1000 × 8000 noise frames.
- ARL0 was re-measured on 500 independent sequences: 939–1069 for every procedure, within about 1.5
  standard errors (`arl0_check` in the JSON).
- Delay = alarm frame − first post-change frame, conditioned on no alarm before ν.
- Miss = no alarm within 500 frames.
- About 16 % of runs alarm before ν. This is consistent with 1 − e^(−200/1000) ≈ 0.18 and is reported
  as `early_alarm_rate`.

Procedures:
- `cusum_oracle`, `sr_oracle` (Shiryaev–Roberts): know the true ρ.
- `cusum_fixed`: designed for −6 dB.
- `page_hinkley`: running-mean form with δ = ρ_d/2, so it does not assume the pre-change level.
- `glr`: window-limited, 400 frames, MLE of the post-change power.
- `bocpd`: Adams–MacKay with a known-noise initial segment, a Gamma(1,1) prior on the post-change
  rate, hazard 1e-3 and run lengths truncated at 600. Its statistic is the posterior log-odds that a
  change has occurred.

### Sanity check (`results/partC/sanity_cusum.json`): PASS

For a Gaussian mean shift N(0,1) → N(μ,1), μ ∈ {0.5, 1}, h ∈ {2, 4, 6}, Monte-Carlo CUSUM ARL0 and
ARL1 match Siegmund's approximation within **3.7 %** and **1.0 %**. Example: μ = 0.5, h = 4 gives
ARL0 763 ± 17 vs 738, and ARL1 28.96 ± 0.27 vs 28.75. Lorden's asymptote log(ARL0)/I is far off at
these moderate ARLs: 53 vs 29. Use the Siegmund form, not the asymptote.

### Results (`results/partC/changepoint_bench.json/.png`, `changepoint_table.md`)

Each cell is mean delay in frames / miss rate.

| SNR dB | CUSUM oracle | CUSUM fixed −6 dB | SR oracle | Page–Hinkley | GLR (unknown ρ) | BOCPD | Siegmund (CUSUM oracle, worst case) |
|---|---|---|---|---|---|---|---|
| −15 | 72.1 / 0 % | 168.8 / 9.5 % | **70.9** / 0 % | 170.5 / 29.5 % | 103.5 / 0 % | 110.8 / 0 % | 81.7 |
| −12 | 27.3 / 0 % | 68.9 / 0.1 % | **26.4** / 0 % | 101.3 / 8.1 % | 33.0 / 0 % | 37.5 / 0 % | 30.3 |
| −9 | 9.8 | 13.1 | **9.4** | 17.9 | 10.4 | 11.9 | 10.0 |
| −6 | 2.8 | 2.8 | 2.7 | 2.9 | 2.9 | 3.5 | 2.8 |
| −3 | 0.5 | 0.6 | 0.5 | 0.6 | 0.6 | 0.8 | 0.4 |

- **SR and CUSUM with the true SNR are best.** SR is about 2–4 % faster, as expected for a
  conditional-average-delay criterion.
- **GLR is the best procedure that does not know the SNR.** It costs about 1.4× the oracle delay at
  −15 dB and about 1.2× at −12 dB, with no misses.
- **BOCPD is close behind GLR.**
- **A mis-designed CUSUM** (−6 dB design at −15 dB) is 2.3× slower and misses 9.5 %.
- **Page–Hinkley**, which must also track the unknown noise level through a running mean, is worst at
  low SNR.
- The simulated conditional delay of the oracle CUSUM sits just below Siegmund's worst-case (S₀ = 0)
  value, as it should.

---

## Caveats

- Box F1 for block detectors is limited by block granularity, which is shared with the reference
  `eigen_mme`, `cyclo` and `gof_ad`. Their emitter Pd is high mainly because of their high measured
  Pfa, so compare at measured Pfa, not target Pfa.
- Dev → held-out drift is visible: raw CAV impaired F1 is 10.3 on dev and 6.6 held-out. Dev differences
  between parameter settings (±2 F1) are within this noise.
- Timing was taken on a machine shared with other jobs.
- Parts B and C are synthetic, idealised models: AWGN, known noise power, a single full-length chirp,
  a Gaussian signal.
