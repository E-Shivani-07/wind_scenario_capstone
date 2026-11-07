# Results (test period)

| model | crps | power_crps | energy_score | mae | rmse | ks | wasserstein | acf_distance | ramp_wasserstein | diversity | spread_skill | calibration_error | coverage_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.6275 | 0.0263 | 3.8048 | 0.6275 | 0.8682 | 0.0112 | 0.0240 | 0.0012 | 0.0030 | 0.0000 | - | - | - |
| climatology | 0.7348 | 0.0322 | 4.2137 | 1.0404 | 1.3197 | 0.0326 | 0.1270 | 0.0049 | 0.0142 | 1.6737 | 0.9581 | 0.0025 | 0.9009 |
| ridge_bootstrap | 0.4174 | 0.0173 | 2.4983 | 0.5578 | 0.7568 | 0.0467 | 0.0919 | 0.0018 | 0.0314 | 0.9202 | 0.8931 | 0.0087 | 0.8883 |
| diffusion | 0.4102 | 0.0176 | 2.4564 | 0.5612 | 0.7599 | 0.0271 | 0.0632 | 0.0159 | 0.0490 | 0.9328 | 0.8884 | 0.0056 | 0.8966 |

## Feature-group importance (mean |SHAP| share)

- wind_speed: 64.1%
- wind_direction: 12.9%
- radiation: 8.0%
- calendar: 5.0%
- humidity: 3.7%
- pressure: 2.3%
- precipitation: 2.0%
- temperature: 2.0%

## Counterfactuals (example window)

- pressure -500.0 (Pa (100 Pa = 1 hPa)): mean wind change -0.156 m/s
- temperature +2.0 (K): mean wind change +0.048 m/s
- wind_speed +0.2 (relative change (0.1 = +10 %)): mean wind change +1.089 m/s
- humidity +10.0 (% points): mean wind change +0.064 m/s
- radiation -0.3 (relative change (0.1 = +10 %)): mean wind change +0.059 m/s

## Explanation quality

- shap_deletion_spearman: 0.546
- shap_top1_deletion_ratio: 3.696
- shap_stability_spearman: 0.905
- attention_occlusion_spearman: 0.895

## Attention (top history hours)

- 1 h before origin: 9.6%
- 2 h before origin: 7.0%
- 3 h before origin: 5.8%
- 8 h before origin: 5.1%
- 7 h before origin: 4.9%
