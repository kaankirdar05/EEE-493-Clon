
### RadioML 7 classes

| method | acc. all SNR | acc. SNR>=10 | first SNR >= 85 % | ms / frame |
|---|---|---|---|---|
| hgb / base19 | 71.4 % | 100.0 % | 2 dB | 1.2 |
| hgb / ext17 | 67.6 % | 99.8 % | 2 dB | 2.2 |
| hgb / wav24 | 69.0 % | 98.7 % | 2 dB | 4.0 |
| hgb / base19+ext17 | 71.4 % | 100.0 % | 2 dB | 3.4 |
| hgb / base19+wav24 | 72.4 % | 100.0 % | 0 dB | 5.2 |
| hgb / base19+ext17+wav24 | 72.6 % | 100.0 % | 0 dB | 7.5 |
| gnb / base19 | 64.0 % | 98.1 % | 4 dB | 1.2 |
| qda / base19 | 70.1 % | 99.9 % | 2 dB | 1.2 |
| gmm / base19 | 70.8 % | 100.0 % | 2 dB | 1.3 |
| gnb / base19+ext17 | 63.1 % | 97.0 % | 4 dB | 3.4 |
| qda / base19+ext17 | 70.1 % | 100.0 % | 2 dB | 3.4 |
| gmm / base19+ext17 | 70.4 % | 100.0 % | 2 dB | 3.5 |
| gnb / base19+ext17+wav24 | 65.1 % | 99.2 % | 4 dB | 7.4 |
| qda / base19+ext17+wav24 | 70.9 % | 100.0 % | 0 dB | 7.5 |
| gmm / base19+ext17+wav24 | 71.6 % | 100.0 % | 2 dB | 7.5 |

### RadioML 24 classes

| method | acc. all SNR | acc. SNR>=10 | first SNR >= 85 % | ms / frame |
|---|---|---|---|---|
| hgb / base19 | 39.2 % | 62.8 % | never | 1.5 |
| hgb / ext17 | 35.5 % | 58.8 % | never | 2.9 |
| hgb / wav24 | 35.7 % | 55.4 % | never | 4.9 |
| hgb / base19+ext17 | 39.7 % | 63.5 % | never | 4.3 |
| hgb / base19+wav24 | 41.3 % | 64.6 % | never | 6.4 |
| hgb / base19+ext17+wav24 | 41.5 % | 64.6 % | never | 9.2 |
| gnb / base19 | 28.4 % | 48.6 % | never | 1.5 |
| qda / base19 | 35.5 % | 58.6 % | never | 1.5 |
| gmm / base19 | 37.5 % | 60.4 % | never | 1.5 |
| gnb / base19+ext17 | 27.8 % | 46.9 % | never | 4.3 |
| qda / base19+ext17 | 36.3 % | 59.7 % | never | 4.3 |
| gmm / base19+ext17 | 34.2 % | 54.7 % | never | 4.4 |
| gnb / base19+ext17+wav24 | 30.8 % | 51.8 % | never | 9.3 |
| qda / base19+ext17+wav24 | 38.5 % | 63.0 % | never | 9.3 |
| gmm / base19+ext17+wav24 | 37.9 % | 61.2 % | never | 9.3 |

### RadioML 5 digital classes, sps known

| method | acc. all SNR | acc. SNR>=10 | first SNR >= 85 % | ms / frame |
|---|---|---|---|---|
| features_hgb (base19) | 66.2 % | 87.1 % | 6 dB | 0.8 |
| features_hgb (base19+ext17) | 66.7 % | 87.3 % | 6 dB | 2.3 |
| ks_amp [not chosen; for reference] | 32.7 % | 39.6 % | never | 63.5 |
| ks_iq [not chosen; for reference] | 46.2 % | 57.3 % | never | 63.5 |
| ks_iqa [not chosen; for reference] | 49.3 % | 60.8 % | never | 63.5 |
| ks_ap [chosen on train] | 60.2 % | 76.1 % | never | 63.5 |
| ks_iqap [not chosen; for reference] | 53.1 % | 60.7 % | never | 63.5 |
| em_gmm_ml | 71.8 % | 95.2 % | 2 dB | 805.9 |
| alrt_sps_known (practical_alrt.py) | 70.4 % | 93.3 % | 4 dB | 41.5 |
| hlrt_sps_known (practical_alrt.py) | 70.6 % | 93.4 % | 2 dB | 41.5 |

### TorchSig 5 digital classes, sps known

| method | acc. all SNR | acc. SNR>=10 | first SNR >= 85 % | ms / frame |
|---|---|---|---|---|
| features_hgb (base19) | 62.1 % | 85.4 % | 16 dB | 2.0 |
| features_hgb (base19+ext17) | 63.9 % | 85.2 % | 16 dB | 6.8 |
| ks_amp [not chosen; for reference] | 38.6 % | 47.0 % | never | 62.6 |
| ks_iq [not chosen; for reference] | 52.1 % | 61.3 % | never | 62.6 |
| ks_iqa [not chosen; for reference] | 56.7 % | 67.5 % | never | 62.6 |
| ks_ap [chosen on train] | 66.2 % | 82.2 % | never | 62.6 |
| ks_iqap [not chosen; for reference] | 67.0 % | 80.0 % | never | 62.6 |
| em_gmm_ml | 77.2 % | 95.1 % | 2 dB | 700.9 |
| alrt_sps_known (practical_alrt.py) | 76.3 % | 95.6 % | 4 dB | 38.0 |
| hlrt_sps_known (practical_alrt.py) | 76.1 % | 95.4 % | 4 dB | 38.0 |
