# Dev tuning (scenes 0-99), Pfa target 1e-3

| variant | dataset | box F1 | emitter Pd | measured Pfa |
|---|---|---|---|---|
| acorr_L=16 | impaired | 11.5 | 84.5 | 3.6e-01 |
| acorr_L=32 | clean | 38.2 | 71.4 | 6.0e-02 |
| acorr_L=32 | impaired | 12.2 | 82.5 | 3.2e-01 |
| acorr_L=4 | impaired | 8.9 | 88.9 | 4.4e-01 |
| acorr_L=8 | clean | 37.1 | 78.2 | 6.6e-02 |
| acorr_L=8 | impaired | 9.5 | 86.5 | 4.0e-01 |
| cav_L=16 | impaired | 7.8 | 85.7 | 3.6e-01 |
| cav_L=32 | clean | 36.9 | 70.5 | 6.0e-02 |
| cav_L=32 | impaired | 11.1 | 82.2 | 3.1e-01 |
| cav_L=4 | clean | 39.4 | 81.5 | 6.8e-02 |
| cav_L=4 | impaired | 10.3 | 85.4 | 3.1e-01 |
| cav_L=8 | impaired | 7.9 | 88.3 | 3.7e-01 |
| cav_pw_L=4 | clean | 13.2 | 61.2 | 8.5e-02 |
| cav_pw_L=4 | impaired | 8.4 | 58.5 | 1.4e-01 |
| entropy_hist_bins=16 | clean | 11.6 | 28.9 | 3.5e-02 |
| entropy_hist_bins=16 | impaired | 7.6 | 37.4 | 1.2e-01 |
| entropy_hist_bins=32 | impaired | 7.8 | 37.1 | 1.2e-01 |
| entropy_hist_bins=8 | impaired | 6.8 | 35.4 | 1.1e-01 |
| entropy_spec_m=128 | clean | 36.8 | 67.1 | 5.8e-02 |
| entropy_spec_m=128 | impaired | 15.9 | 76.3 | 2.7e-01 |
| entropy_spec_m=32 | impaired | 13.1 | 82.5 | 3.1e-01 |
| entropy_spec_m=64 | clean | 38.6 | 68.3 | 6.2e-02 |
| entropy_spec_m=64 | impaired | 14.5 | 80.7 | 2.9e-01 |
| entropy_spec_pw_m=64 | clean | 17.8 | 50.8 | 6.4e-02 |
| entropy_spec_pw_m=64 | impaired | 14.5 | 53.2 | 1.2e-01 |
| scd_fam_Np=16 | clean | 23.4 | 33.2 | 3.1e-02 |
| scd_fam_Np=16 | impaired | 16.9 | 30.1 | 4.3e-02 |
| scd_fam_Np=32 | clean | 18.2 | 29.5 | 2.6e-02 |
| scd_fam_Np=32 | impaired | 15.8 | 23.7 | 3.5e-02 |

Chosen (mean dev F1 over the two datasets where both exist): cav L=4, acorr L=32, entropy_spec m=64, entropy_hist bins=16, scd_fam Np=16.
