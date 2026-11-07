# Results (test period)

| model | crps | power_crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6275 | 0.0263 | 3.8048 | 0.6275 | 0.8682 | 0.0112 | 0.0240 | 0.0012 | 0.0030 | 0.0000 | - | - | - |
| climatology | 0.7348 | 0.0322 | 4.2137 | 1.0404 | 1.3197 | 0.0326 | 0.1270 | 0.0049 | 0.0142 | 1.6737 | 0.9581 | 0.0025 | 0.9009 |
| ridge_bootstrap | 0.4174 | 0.0173 | 2.4983 | 0.5578 | 0.7568 | 0.0467 | 0.0919 | 0.0018 | 0.0314 | 0.9202 | 0.8931 | 0.0087 | 0.8883 |
| diffusion | 0.3953 | 0.0167 | 2.3753 | 0.5401 | 0.7322 | 0.0296 | 0.0615 | 0.0080 | 0.0515 | 0.9022 | 0.8907 | 0.0075 | 0.8998 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 56.9%
- wind_direction: 8.5%
- radiation: 8.4%
- calendar: 6.4%
- pressure: 6.1%
- temperature: 6.0%
- humidity: 4.0%
- precipitation: 3.7%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.217 m/s
- temperature +2.0 (K): mean wind change -0.043 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +0.986 m/s
- humidity +10.0 (% points): mean wind change +0.039 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change +0.036 m/s

## Explanation quality

- shap_deletion_spearman: 0.657
- shap_top1_deletion_ratio: 3.559
- shap_stability_spearman: 0.952
