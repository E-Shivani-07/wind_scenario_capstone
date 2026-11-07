# Results (test period)

| model | crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6075 | 3.6829 | 0.6075 | 0.8382 | 0.0065 | 0.0174 | 0.0011 | 0.0029 | 0.0000 | - | - | - |
| climatology | 0.7222 | 4.1546 | 1.0237 | 1.3043 | 0.0306 | 0.1002 | 0.0025 | 0.0143 | 1.6454 | 0.9512 | 0.0006 | 0.8993 |
| ridge_bootstrap | 0.4023 | 2.4099 | 0.5401 | 0.7273 | 0.0472 | 0.0922 | 0.0025 | 0.0311 | 0.9228 | 0.9309 | 0.0009 | 0.8987 |
| diffusion | 0.3849 | 2.3084 | 0.5248 | 0.7115 | 0.0289 | 0.0590 | 0.0079 | 0.0506 | 0.8937 | 0.9068 | 0.0095 | 0.9010 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 63.8%
- radiation: 9.0%
- temperature: 8.6%
- wind_direction: 7.1%
- pressure: 4.3%
- humidity: 3.5%
- calendar: 2.8%
- precipitation: 0.8%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.331 m/s
- temperature +2.0 (K): mean wind change +0.003 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +0.888 m/s
- humidity +10.0 (% points): mean wind change +0.064 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change +0.007 m/s
