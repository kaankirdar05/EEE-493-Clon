# WP4 detector comparison (scenes 100-299, Pfa target 1e-3 unless noted)

Thresholds are calibrated on white noise, so on pure noise every detector has the target Pfa. On scenes the measured Pfa is higher because of unlabelled real energy (see README). eigen_mme and cyclo decide per 16-frame x 32-bin block, so their boxes are coarse.


## impaired

| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|---|
| energy | 53.0 | 48.9 | 57.8 | 40 | 79 | 49.8 | 0.79 | 2.1e-02 | 318 |
| energy_mt | 42.7 | 34.1 | 56.9 | 38 | 79 | 59.4 | 0.79 | 2.8e-02 | 406 |
| fcme | 35.6 | 36.9 | 34.5 | 16 | 51 | 62.6 | 0.79 | 1.9e-01 | 92 |
| lad | 30.5 | 20.7 | 57.5 | 38 | 78 | 55.9 | 0.77 | 3.7e-02 | 366 |
| psd_occupancy | 9.9 | 6.1 | 26.0 | 17 | 34 | 75.6 | 0.66 | 2.9e-01 | 98 |
| cfar_ca | 28.6 | 21.6 | 42.3 | 31 | 69 | 25.1 | 0.71 | 1.0e-02 | 20 |
| cfar_go | 34.5 | 33.9 | 35.2 | 24 | 60 | 19.0 | 0.70 | 4.1e-03 | 20 |
| cfar_so | 15.6 | 9.9 | 36.7 | 26 | 51 | 57.1 | 0.73 | 1.5e-01 | 20 |
| cfar_os | 39.8 | 36.7 | 43.4 | 33 | 71 | 27.3 | 0.73 | 5.9e-03 | 209 |
| cfar_2d | 28.8 | 22.2 | 41.0 | 27 | 73 | 19.2 | 0.71 | 9.0e-03 | 14 |
| spectral_kurtosis | 5.8 | 6.2 | 5.5 | 2 | 9 | 1.8 | 0.71 | 1.6e-02 | 18 |
| eigen_mme | 4.7 | 3.4 | 7.4 | 4 | 10 | 87.2 | 0.66 | 4.2e-01 | 298 |
| cyclo | 12.3 | 9.2 | 18.4 | 2 | 21 | 29.0 | 0.68 | 4.0e-02 | 106 |
| gof_ad | 13.7 | 11.7 | 16.6 | 8 | 23 | 80.2 | 0.66 | 3.6e-01 | 28 |

## clean

| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|---|
| energy | 67.4 | 72.7 | 62.8 | 52 | 74 | 53.8 | 0.81 | 5.1e-03 | 316 |
| energy_mt | 60.7 | 58.3 | 63.3 | 58 | 77 | 65.1 | 0.82 | 1.3e-02 | 407 |
| fcme | 65.1 | 71.8 | 59.5 | 40 | 70 | 50.8 | 0.79 | 3.5e-03 | 90 |
| lad | 66.6 | 68.1 | 65.2 | 57 | 79 | 61.9 | 0.81 | 8.9e-03 | 356 |
| psd_occupancy | 23.8 | 27.1 | 21.3 | 16 | 22 | 70.2 | 0.67 | 1.3e-01 | 97 |
| cfar_ca | 52.1 | 52.5 | 51.7 | 59 | 72 | 25.6 | 0.74 | 1.8e-03 | 20 |
| cfar_go | 41.2 | 41.9 | 40.5 | 51 | 47 | 20.5 | 0.73 | 1.6e-03 | 20 |
| cfar_so | 64.1 | 60.3 | 68.5 | 71 | 77 | 52.9 | 0.81 | 2.8e-03 | 20 |
| cfar_os | 48.8 | 47.8 | 49.8 | 61 | 60 | 27.9 | 0.75 | 1.9e-03 | 210 |
| cfar_2d | 45.6 | 43.6 | 47.7 | 49 | 70 | 21.1 | 0.72 | 1.5e-03 | 14 |
| spectral_kurtosis | 6.3 | 7.3 | 5.5 | 3 | 2 | 3.3 | 0.74 | 6.2e-03 | 18 |
| eigen_mme | 37.5 | 35.4 | 39.9 | 26 | 50 | 79.2 | 0.70 | 8.9e-02 | 308 |
| cyclo | 19.6 | 19.0 | 20.2 | 3 | 18 | 28.0 | 0.68 | 2.3e-02 | 105 |
| gof_ad | 36.2 | 31.2 | 43.0 | 30 | 51 | 76.5 | 0.69 | 7.7e-02 | 28 |
