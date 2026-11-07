"""Interactive demo: pick a test day, generate wind scenarios, explain them, try what-ifs.

    streamlit run app.py            # after run_pipeline.py has written outputs/
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from windgen.data import Normalizer, Windows  # noqa: E402
from windgen.diffusion import ConditionalDDPM, ScenarioGenerator  # noqa: E402
from windgen.explain import ScenarioExplainer, shift_group  # noqa: E402
from windgen.metrics import crps_ensemble  # noqa: E402
from windgen.power import wind_to_power  # noqa: E402

st.set_page_config(page_title="Wind Scenario Generator", layout="wide")
# Responsive layout: columns wrap onto new rows instead of squeezing (metrics >= 8rem, charts >= 24rem).
st.markdown("""<style>
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 1rem; }
[data-testid="stColumn"] { min-width: min(24rem, 100%); }
.st-key-metrics [data-testid="stColumn"] { min-width: 8rem; }
[data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2vw, 2.1rem); }
h1 { font-size: clamp(1.6rem, 3.2vw, 2.6rem) !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_run(run_dir):
    run_dir = Path(run_dir)
    z = np.load(run_dir / "diffusion_model.npz")
    norm = Normalizer.from_state({k: z[f"x_{k}"] for k in ("feat_mean", "feat_std", "t_mean", "t_std")})
    gen = ScenarioGenerator(ConditionalDDPM.load(run_dir / "diffusion_model.npz"), norm)
    t = np.load(run_dir / "test_windows.npz")
    w = Windows(t["hist"], t["cal"], t["target"], t["origin"])
    # ponytail: background for SHAP = test windows (train windows are not saved); fine for a demo
    return gen, w, ScenarioExplainer(gen, w, n_scenarios=8, seed=0)


runs = sorted(p.parent for p in (ROOT / "outputs").glob("**/diffusion_model.npz"))
if not runs:
    st.error("No trained model found. Run `python run_pipeline.py --data-dir <csv dir> --backbones attn mlp --seeds 0 1 2` first.")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
run = st.sidebar.selectbox("Trained model", runs, format_func=lambda p: p.relative_to(ROOT / "outputs").as_posix() or "outputs")
gen, w, expl = load_run(str(run))
backbone = gen.model.cfg.get("backbone", "mlp")
dates = pd.DatetimeIndex(w.origin)
windy = int(np.argmax(w.target.mean(axis=1)))
i = st.sidebar.selectbox("Forecast start (test period)", range(len(w)), index=windy,
                         format_func=lambda k: dates[k].strftime("%Y-%m-%d %H:00"))
n = st.sidebar.slider("Number of scenarios", 5, 100, 30, step=5)
seed = st.sidebar.number_input("Random seed", 0, 9999, 0)
hub = st.sidebar.slider("Turbine hub height (m)", 10, 150, 100, step=10)

st.sidebar.subheader("What-if (counterfactual)")
shifts = {
    "pressure": st.sidebar.slider("Pressure change (hPa)", -15.0, 15.0, 0.0, 0.5) * 100,
    "temperature": st.sidebar.slider("Temperature change (K)", -5.0, 5.0, 0.0, 0.5),
    "humidity": st.sidebar.slider("Humidity change (% points)", -30.0, 30.0, 0.0, 5.0),
    "wind_speed": st.sidebar.slider("Past wind speed change (%)", -50, 50, 0, 5) / 100,
    "wind_direction": st.sidebar.slider("Wind direction rotation (deg)", -90, 90, 0, 15),
}
hist, cal, obs = w.hist[i:i + 1].astype(np.float64), w.cal[i:i + 1], w.target[i]
cf_hist = hist
for g, d in shifts.items():
    if d:
        cf_hist = shift_group(cf_hist, g, d)
what_if = cf_hist is not hist

# ------------------------------------------------------------------ main
st.title("Explainable generative wind-speed scenarios")
st.caption(f"Conditional diffusion model ({backbone} backbone) · ERA5 14.6°N 79.6°E · "
           f"24 h ahead from the previous 24 h of weather")

scen = gen.scenarios_from_arrays(hist, cal, n=n, seed=int(seed))[0]
cf = gen.scenarios_from_arrays(cf_hist, cal, n=n, seed=int(seed))[0] if what_if else None
power = wind_to_power(scen, hub=hub)
energy = power.sum(axis=1)  # MWh per MW installed over the 24 h

c = st.container(key="metrics").columns(5)
c[0].metric("Observed wind", f"{obs.mean():.2f} m/s", help="Mean observed 10 m wind over the 24 h")
c[1].metric("Scenario wind", f"{scen.mean():.2f} m/s",
            None if cf is None else f"{cf.mean() - scen.mean():+.2f} m/s what-if",
            help="Mean over all generated scenarios")
c[2].metric("CRPS (this day)", f"{crps_ensemble(obs[None], scen[None]):.3f} m/s")
c[3].metric("Energy P50", f"{np.median(energy):.1f}", help="MWh per MW installed over the 24 h")
c[4].metric("Energy P10–P90", f"{np.percentile(energy, 10):.1f}–{np.percentile(energy, 90):.1f}",
            help="MWh per MW installed over the 24 h")

tab_s, tab_x, tab_r = st.tabs(["Scenarios", "Explain this forecast", "Study results"])
hrs = pd.date_range(dates[i], periods=len(obs), freq="h")

with tab_s:
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s in scen:
        ax.plot(hrs, s, color="tab:blue", alpha=0.15, lw=0.8)
    ax.fill_between(hrs, *np.percentile(scen, [10, 90], axis=0), color="tab:blue", alpha=0.2, label="P10–P90")
    ax.plot(hrs, np.median(scen, 0), color="navy", lw=2, label="median scenario")
    ax.plot(hrs, obs, color="red", lw=2, label="observed")
    if cf is not None:
        ax.plot(hrs, np.median(cf, 0), color="tab:orange", lw=2, ls="--", label="what-if median")
    ax.set_ylabel("10 m wind speed (m/s)"); ax.legend(); ax.set_title("Wind-speed scenarios")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    left.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(hrs, *np.percentile(power, [10, 90], axis=0), color="tab:green", alpha=0.25, label="P10–P90")
    ax.plot(hrs, np.median(power, 0), color="darkgreen", lw=2, label="median")
    ax.plot(hrs, wind_to_power(obs, hub=hub), color="red", lw=2, label="from observed wind")
    ax.set_ylabel("capacity factor"); ax.legend(); ax.set_title(f"Turbine output at {hub} m hub height")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    right.pyplot(fig)
    st.caption("Power uses a generic turbine (cut-in 3, rated 12, cut-out 25 m/s) and power-law "
               "height extrapolation (α = 1/7). Energy is MWh per MW installed over the 24 h.")

with tab_x:
    left, right = st.columns(2)
    with left:
        st.subheader("Which inputs drove this forecast? (SHAP)")
        with st.spinner("computing Shapley values ..."):
            sh = expl.shapley(w.hist[i], w.cal[i], n_perm=8, seed=int(seed))
        order = np.argsort(np.abs(sh["phi"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(np.array(sh["names"])[order], sh["phi"][order],
                color=["tab:red" if v < 0 else "tab:green" for v in sh["phi"][order]])
        ax.set_xlabel("contribution to mean wind (m/s)")
        ax.set_title(f"baseline {sh['base_value']:.2f} → forecast {sh['prediction']:.2f} m/s")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        if backbone == "attn":
            st.subheader("Which past hours did the model attend to?")
            a = gen.attention_from_arrays(w.hist[i:i + 1], w.cal[i:i + 1], n=8, seed=int(seed))[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            L = a.shape[1]
            im = ax.imshow(a, aspect="auto", cmap="viridis", extent=[L + 0.5, 0.5, len(a) - 0.5, -0.5])
            ax.set_yticks(range(len(a)))
            ax.set_xlabel("hours before forecast start"); ax.set_ylabel("attention head")
            fig.colorbar(im, ax=ax, label="weight")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Pick an `attn_*` model in the sidebar to see attention maps.")

with tab_r:
    summary = ROOT / "outputs" / "summary.json"
    if summary.exists():
        stats = json.loads(summary.read_text())
        fmt = lambda v: "–" if v is None else f"{v[0]:.3g} ± {v[1]:.1g}"
        short = {"persistence": "persist.", "climatology": "climat.", "ridge_bootstrap": "ridge",
                 "diffusion_attn": "diff. attn", "diffusion_mlp": "diff. MLP"}
        # metrics as rows, models as columns; st.table wraps cells, so it fits narrow screens
        table = lambda rows: pd.DataFrame({short.get(n, n): {k.replace("_", " "): fmt(v) for k, v in m.items()}
                                           for n, m in rows})
        st.subheader("Test-set metrics (mean ± std over seeds)")
        st.caption("Lower is better, except coverage_90 (target 0.90), spread_skill (target ≈ 1) and diversity.")
        st.table(table((n, m) for n, m in stats.items() if not n.startswith("explanations_")))
        st.subheader("Explanation quality")
        st.table(table((n.removeprefix("explanations_"), m) for n, m in stats.items()
                       if n.startswith("explanations_")))
    imgs = sorted(Path(run).glob("*.png"))
    for col, img in zip(st.columns(2) * (len(imgs) // 2 + 1), imgs):
        col.image(str(img), caption=img.stem.replace("_", " "))
