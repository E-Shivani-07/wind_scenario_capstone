# Explainable Generative AI for Wind-Speed Scenario Generation

Conditional diffusion model (DDPM) that generates many plausible 24 h wind-speed
trajectories from the previous 24 h of meteorology (wind speed, wind direction, temperature,
relative humidity, pressure, precipitation, solar radiation) plus calendar features, with
SHAP, cross-attention, temporal-attribution and counterfactual explanations, explanation-quality
metrics, a turbine power conversion and an interactive Streamlit demo.

Two denoiser backbones: `mlp` (history flattened into an MLP) and `attn` (the history hours are
read **only** through multi-head cross-attention, so the attention weights are a real explanation).

Pure NumPy/pandas/SciPy/Matplotlib - no GPU or deep-learning framework required.

## Run
```bash
pip install -r requirements.txt
# put wind.csv, temperature.csv, pressure.csv, radiation.csv in data/raw/
python run_pipeline.py --data-dir data/raw --out-dir outputs                  # one run, ~1 min on a laptop CPU
python run_pipeline.py --data-dir data/raw --backbones attn mlp --seeds 0 1 2  # full study, ~6 min
python run_pipeline.py --synthetic --quick --out-dir /tmp/demo                # smoke test, no data needed
python -m unittest discover -s tests -t . -v                                  # 85 tests
streamlit run app.py                                                          # interactive demo
```
Each run writes `results.md/json` (all metrics), `diffusion_model.npz`, `test_windows.npz` and figures
(scenarios, training loss, rank histograms, SHAP importance, temporal attribution, counterfactual,
attention for `attn`). Several runs also write `summary.md/json` (mean ± std over seeds).

The app (`app.py`) needs only `outputs/`: pick a test day, generate up to 100 scenarios, see them as
wind and turbine power (P10/P50/P90 daily energy), move what-if sliders (pressure, temperature,
humidity, past wind, direction), and view per-forecast SHAP and attention maps.

## Layout
| file | purpose |
|---|---|
| `windgen/data.py` | loading/joining the 4 csv files, features (speed, direction sin/cos, RH via Magnus), leakage-free windows, normaliser |
| `windgen/nn.py` | NumPy MLP (SiLU) and cross-attention denoiser with manual backprop + Adam |
| `windgen/diffusion.py` | conditional DDPM (cosine schedule, EMA, DDPM sampling), `ScenarioGenerator` |
| `windgen/baselines.py` | persistence, month/hour climatology bootstrap, ridge + residual bootstrap |
| `windgen/metrics.py` | CRPS, energy score, MAE/RMSE, KS, Wasserstein, ACF and ramp distance, diversity, rank histogram, coverage, calibration error, spread-skill |
| `windgen/explain.py` | group Shapley values, attention by lag, occlusion attribution, counterfactuals + search, explanation-quality metrics |
| `windgen/power.py` | hub-height extrapolation and generic turbine power curve |
| `windgen/pipeline.py` | experiment runner, report/plots, multi-seed summary |
| `app.py` | Streamlit demo |

## Design notes and limitations (please read)
* Data are single-point ERA5 (14.6N, 79.6E). `ssrd` and `tp` in your files are per-hour values, so no de-accumulation is applied.
* Split by time: train < 2016, validation 2016-2020, test >= 2021. Windows crossing time gaps are dropped.
* Wind speed here is 10 m wind (mean ~2.7 m/s). `power_crps` and the app convert it to a generic turbine's
  capacity factor at 100 m (power law, alpha = 1/7; cut-in 3, rated 12, cut-out 25 m/s). It is illustrative,
  not a specific turbine or site assessment.
* Attention analysis uses the `attn` backbone: averaged over scenarios, denoising steps and windows.
  Its agreement with the model-agnostic occlusion attribution is reported (`attention_occlusion_spearman`).
* Explanation quality: `shap_deletion_spearman` (do larger |SHAP| groups change the output more when removed?),
  `shap_top1_deletion_ratio` (effect of removing the top group / average group), `shap_stability_spearman`
  (importance ranking with a different background sample and permutations).
* SHAP is a permutation-sampling Shapley estimator over feature groups (own implementation, no `shap` dependency).
  The explained value is the mean generated wind speed; common random numbers make it deterministic.
* Counterfactuals are model-based (what the model generates under altered inputs), not causal claims about the atmosphere.
