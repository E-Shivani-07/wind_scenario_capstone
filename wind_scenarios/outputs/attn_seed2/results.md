# Results (test period)

| model | crps | power_crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6374 | 0.0269 | 3.8472 | 0.6374 | 0.8892 | 0.0040 | 0.0093 | 0.0021 | 0.0037 | 0.0000 | - | - | - |
| climatology | 0.7334 | 0.0324 | 4.2178 | 1.0415 | 1.3200 | 0.0258 | 0.1089 | 0.0077 | 0.0137 | 1.6785 | 0.9604 | 0.0036 | 0.9030 |
| ridge_bootstrap | 0.4179 | 0.0175 | 2.4937 | 0.5619 | 0.7554 | 0.0390 | 0.0825 | 0.0023 | 0.0268 | 0.9251 | 0.9000 | 0.0088 | 0.8957 |
| diffusion | 0.4110 | 0.0177 | 2.4556 | 0.5655 | 0.7576 | 0.0196 | 0.0490 | 0.0070 | 0.0449 | 0.9256 | 0.8831 | 0.0023 | 0.8965 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 60.3%
- wind_direction: 12.8%
- calendar: 10.7%
- radiation: 5.7%
- pressure: 3.1%
- temperature: 2.9%
- humidity: 2.5%
- precipitation: 2.0%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.152 m/s
- temperature +2.0 (K): mean wind change -0.188 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +0.882 m/s
- humidity +10.0 (% points): mean wind change +0.098 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change -0.026 m/s

## Explanation quality

- shap_deletion_spearman: 0.613
- shap_top1_deletion_ratio: 3.647
- shap_stability_spearman: 0.762
- attention_occlusion_spearman: 0.725

## Attention (top history hours)

- 1 h before origin: 12.2%
- 2 h before origin: 8.3%
- 7 h before origin: 5.8%
- 3 h before origin: 5.5%
- 8 h before origin: 5.1%
