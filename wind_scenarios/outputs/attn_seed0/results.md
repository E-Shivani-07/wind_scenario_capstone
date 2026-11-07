# Results (test period)

| model | crps | power_crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6075 | 0.0259 | 3.6829 | 0.6075 | 0.8382 | 0.0065 | 0.0174 | 0.0011 | 0.0029 | 0.0000 | - | - | - |
| climatology | 0.7222 | 0.0316 | 4.1546 | 1.0237 | 1.3043 | 0.0306 | 0.1002 | 0.0025 | 0.0143 | 1.6454 | 0.9512 | 0.0006 | 0.8993 |
| ridge_bootstrap | 0.4023 | 0.0165 | 2.4099 | 0.5401 | 0.7273 | 0.0472 | 0.0922 | 0.0025 | 0.0311 | 0.9228 | 0.9309 | 0.0009 | 0.8987 |
| diffusion | 0.3965 | 0.0169 | 2.3767 | 0.5426 | 0.7273 | 0.0252 | 0.0496 | 0.0044 | 0.0475 | 0.9180 | 0.9095 | 0.0075 | 0.9010 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 70.3%
- wind_direction: 9.9%
- radiation: 6.4%
- calendar: 6.2%
- humidity: 2.4%
- temperature: 2.4%
- pressure: 1.4%
- precipitation: 1.1%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.265 m/s
- temperature +2.0 (K): mean wind change +0.002 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +1.093 m/s
- humidity +10.0 (% points): mean wind change +0.023 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change -0.004 m/s

## Explanation quality

- shap_deletion_spearman: 0.685
- shap_top1_deletion_ratio: 4.386
- shap_stability_spearman: 0.976
- attention_occlusion_spearman: 0.819

## Attention (top history hours)

- 1 h before origin: 11.7%
- 2 h before origin: 7.9%
- 3 h before origin: 5.8%
- 4 h before origin: 4.7%
- 5 h before origin: 4.5%
