# Part A - additional block detectors vs WP4 reference (scenes 100-299, Pfa target 1e-3)

Reference rows (italic) are read from wp4_detection/results. All block detectors decide per 16-frame x 32-bin block. ms/scene for the new rows is single-process time of stat() on 5 scenes (time_partA.py) on a machine shared with other jobs: indicative upper bounds only.


## impaired

| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|---|
| *energy* | 53.0 | 48.9 | 57.8 | 40 | 79 | 49.8 | 0.79 | 2.1e-02 | 318 |
| *cfar_so* | 15.6 | 9.9 | 36.7 | 26 | 51 | 57.1 | 0.73 | 1.5e-01 | 20 |
| *cfar_os* | 39.8 | 36.7 | 43.4 | 33 | 71 | 27.3 | 0.73 | 5.9e-03 | 209 |
| *eigen_mme* | 4.7 | 3.4 | 7.4 | 4 | 10 | 87.2 | 0.66 | 4.2e-01 | 298 |
| *cyclo* | 12.3 | 9.2 | 18.4 | 2 | 21 | 29.0 | 0.68 | 4.0e-02 | 106 |
| *gof_ad* | 13.7 | 11.7 | 16.6 | 8 | 23 | 80.2 | 0.66 | 3.6e-01 | 28 |
| **cav_L4** | 6.6 | 4.6 | 12.2 | 5 | 13 | 82.6 | 0.66 | 3.2e-01 | 148 |
| **acorr_L32** | 7.3 | 5.5 | 10.8 | 4 | 15 | 82.3 | 0.66 | 3.4e-01 | 665 |
| **entropy_spec** | 7.5 | 5.7 | 11.1 | 3 | 16 | 81.3 | 0.66 | 3.3e-01 | 117 |
| **entropy_hist** | 6.4 | 4.5 | 10.7 | 2 | 12 | 36.4 | 0.65 | 1.3e-01 | 112 |
| **scd_fam_Np16** | 16.2 | 15.4 | 17.1 | 1 | 21 | 34.2 | 0.68 | 5.4e-02 | 4164 |
| **cav_L4_pw** | 8.7 | 6.1 | 15.2 | 8 | 19 | 57.2 | 0.65 | 1.3e-01 | 197 |
| **entropy_spec_pw** | 11.8 | 9.1 | 16.8 | 3 | 24 | 52.9 | 0.66 | 1.3e-01 | 166 |

## clean

| detector | box F1 | precision | recall | recall 0-5 dB | recall 10-15 dB | emitter Pd | mean IoU | measured Pfa | ms/scene |
|---|---|---|---|---|---|---|---|---|---|
| *energy* | 67.4 | 72.7 | 62.8 | 52 | 74 | 53.8 | 0.81 | 5.1e-03 | 316 |
| *cfar_so* | 64.1 | 60.3 | 68.5 | 71 | 77 | 52.9 | 0.81 | 2.8e-03 | 20 |
| *cfar_os* | 48.8 | 47.8 | 49.8 | 61 | 60 | 27.9 | 0.75 | 1.9e-03 | 210 |
| *eigen_mme* | 37.5 | 35.4 | 39.9 | 26 | 50 | 79.2 | 0.70 | 8.9e-02 | 308 |
| *cyclo* | 19.6 | 19.0 | 20.2 | 3 | 18 | 28.0 | 0.68 | 2.3e-02 | 105 |
| *gof_ad* | 36.2 | 31.2 | 43.0 | 30 | 51 | 76.5 | 0.69 | 7.7e-02 | 28 |
| **cav_L4** | 36.8 | 33.3 | 41.1 | 27 | 55 | 78.3 | 0.71 | 7.1e-02 | 148 |
| **acorr_L32** | 38.7 | 36.0 | 41.8 | 23 | 55 | 71.1 | 0.70 | 6.4e-02 | 665 |
| **entropy_spec** | 39.1 | 36.6 | 42.0 | 20 | 57 | 69.7 | 0.71 | 6.2e-02 | 117 |
| **entropy_hist** | 12.5 | 11.0 | 14.4 | 3 | 10 | 27.9 | 0.67 | 3.5e-02 | 112 |
| **scd_fam_Np16** | 24.2 | 25.0 | 23.4 | 5 | 24 | 33.4 | 0.69 | 2.6e-02 | 4164 |
| **cav_L4_pw** | 15.5 | 12.2 | 21.3 | 15 | 29 | 57.3 | 0.65 | 8.9e-02 | 197 |
| **entropy_spec_pw** | 19.7 | 16.3 | 25.0 | 7 | 32 | 49.3 | 0.67 | 6.1e-02 | 166 |
