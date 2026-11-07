# Results (test period)

| model | crps | power_crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6374 | 0.0269 | 3.8472 | 0.6374 | 0.8892 | 0.0040 | 0.0093 | 0.0021 | 0.0037 | 0.0000 | - | - | - |
| climatology | 0.7334 | 0.0324 | 4.2178 | 1.0415 | 1.3200 | 0.0258 | 0.1089 | 0.0077 | 0.0137 | 1.6785 | 0.9604 | 0.0036 | 0.9030 |
| ridge_bootstrap | 0.4179 | 0.0175 | 2.4937 | 0.5619 | 0.7554 | 0.0390 | 0.0825 | 0.0023 | 0.0268 | 0.9251 | 0.9000 | 0.0088 | 0.8957 |
| diffusion | 0.3968 | 0.0170 | 2.3799 | 0.5448 | 0.7350 | 0.0222 | 0.0491 | 0.0076 | 0.0472 | 0.9056 | 0.8900 | 0.0054 | 0.8998 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 55.8%
- wind_direction: 10.2%
- radiation: 9.6%
- temperature: 7.9%
- calendar: 4.7%
- humidity: 4.3%
- pressure: 3.9%
- precipitation: 3.6%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.164 m/s
- temperature +2.0 (K): mean wind change +0.113 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +0.894 m/s
- humidity +10.0 (% points): mean wind change +0.271 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change +0.061 m/s

## Explanation quality

- shap_deletion_spearman: 0.651
- shap_top1_deletion_ratio: 3.165
- shap_stability_spearman: 0.833
