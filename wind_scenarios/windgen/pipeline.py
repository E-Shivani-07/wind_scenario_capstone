"""End-to-end experiment: data -> baselines + diffusion -> evaluation -> explainability -> report."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .baselines import ClimatologyBaseline, PersistenceBaseline, RidgeResidualBaseline
from .data import FEATURES, Normalizer, build_features, load_raw, make_windows
from .diffusion import ConditionalDDPM, ScenarioGenerator
from .explain import DELTA_UNITS, ScenarioExplainer, rank_agreement
from .metrics import crps_ensemble, evaluate_scenarios, rank_histogram
from .power import wind_to_power


@dataclass
class Config:
    lookback: int = 24
    horizon: int = 24
    T: int = 100
    hidden: int = 256
    n_layers: int = 3
    backbone: str = "mlp"  # "mlp" or "attn" (cross-attention over history hours)
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    n_scenarios: int = 20
    train_stride: int = 3
    test_stride: int = 25
    max_train: int = 60000
    max_val: int = 3000
    max_test: int = 1000
    val_start: str = "2016-01-01"
    test_start: str = "2021-01-01"
    n_explain: int = 12
    n_perm: int = 10
    n_explain_scen: int = 8
    seed: int = 0
    log_every: int = 250
    make_plots: bool = True


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_pipeline(raw, out_dir, cfg: Config = None, verbose=True):
    cfg = cfg or Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    feats = build_features(raw)
    L, H = cfg.lookback, cfg.horizon
    train = make_windows(feats, L, H, end=cfg.val_start, stride=cfg.train_stride,
                         max_windows=cfg.max_train, seed=cfg.seed)
    val = make_windows(feats, L, H, start=cfg.val_start, end=cfg.test_start,
                       stride=cfg.test_stride, max_windows=cfg.max_val, seed=cfg.seed)
    test = make_windows(feats, L, H, start=cfg.test_start, stride=cfg.test_stride,
                        max_windows=cfg.max_test, seed=cfg.seed)
    log(f"windows: train {len(train)}  val {len(val)}  test {len(test)}")
    if train.origin.max() >= val.origin.min() or val.origin.max() >= test.origin.min():
        raise RuntimeError("time splits overlap")

    norm = Normalizer().fit(train)
    c_tr, c_va = norm.cond(train.hist, train.cal), norm.cond(val.hist, val.cal)
    y_tr, y_va = norm.norm_target(train.target), norm.norm_target(val.target)

    # ---- diffusion model
    model = ConditionalDDPM(H, c_tr.shape[1], T=cfg.T, hidden=cfg.hidden,
                            n_layers=cfg.n_layers, seed=cfg.seed, backbone=cfg.backbone, lookback=L)
    log(f"training conditional diffusion model ({cfg.backbone}) ...")
    history = model.fit(y_tr, c_tr, steps=cfg.steps, batch_size=cfg.batch_size, lr=cfg.lr,
                        seed=cfg.seed, log_every=cfg.log_every, val=(y_va, c_va), verbose=verbose)
    model.save(out / "diffusion_model.npz", extra=norm.state())
    np.savez(out / "test_windows.npz", hist=test.hist, cal=test.cal, target=test.target,
             origin=test.origin.astype("datetime64[h]"))
    gen = ScenarioGenerator(model, norm)

    # ---- generators
    gens = {
        "persistence": PersistenceBaseline().fit(train),
        "climatology": ClimatologyBaseline().fit(train),
        "ridge_bootstrap": RidgeResidualBaseline(norm).fit(train),
    }
    scen = {k: g.scenarios(test, cfg.n_scenarios, cfg.seed) for k, g in gens.items()}
    log("sampling diffusion scenarios on the test set ...")
    scen["diffusion"] = gen.scenarios(test, cfg.n_scenarios, cfg.seed)
    metrics = {k: evaluate_scenarios(test.target, v) for k, v in scen.items()}
    obs_power = wind_to_power(test.target)
    for k, v in scen.items():  # CRPS in capacity-factor units (generic turbine at 100 m)
        metrics[k]["power_crps"] = crps_ensemble(obs_power, wind_to_power(v))
    rank_hists = {k: rank_histogram(test.target, v) for k, v in scen.items() if v.shape[1] > 1}

    # ---- explainability
    log("explainability ...")
    expl = ScenarioExplainer(gen, train, n_scenarios=cfg.n_explain_scen, seed=cfg.seed)
    imp = expl.global_importance(test, n_instances=cfg.n_explain, n_perm=cfg.n_perm, seed=cfg.seed)
    temporal = expl.temporal_attribution(test, n_instances=cfg.n_explain)
    faith = expl.deletion_faithfulness(test, imp)
    expl2 = ScenarioExplainer(gen, train, n_scenarios=cfg.n_explain_scen, seed=cfg.seed + 1000)
    imp2 = expl2.global_importance(test, n_instances=cfg.n_explain, n_perm=cfg.n_perm, seed=cfg.seed + 1000)
    quality = {"shap_deletion_spearman": faith["spearman"],
               "shap_top1_deletion_ratio": faith["top1_deletion_ratio"],
               "shap_stability_spearman": rank_agreement(imp["share"], imp2["share"])}
    attention = None
    if cfg.backbone == "attn":
        attention = expl.attention_by_lag(test, n_instances=cfg.n_explain)
        quality["attention_occlusion_spearman"] = rank_agreement(attention["share"], temporal["share"])
    ex_i = int(np.argmax(test.target.mean(axis=1)))  # a windy example
    cfs = {}
    for group, delta in {"pressure": -500.0, "temperature": 2.0, "wind_speed": 0.2,
                         "humidity": 10.0, "radiation": -0.3}.items():
        r = expl.counterfactual(test.hist[ex_i], test.cal[ex_i], group, delta)
        cfs[group] = {k: r[k] for k in ("delta", "unit", "mean_shift", "wasserstein")}
    search = expl.find_counterfactual(test.hist[ex_i], test.cal[ex_i], "pressure", target_shift=-0.2)
    search_out = None if search is None else {k: search[k] for k in ("group", "delta", "unit", "mean_shift")}

    results = {
        "config": asdict(cfg),
        "n_windows": {"train": len(train), "val": len(val), "test": len(test)},
        "features": FEATURES,
        "metrics": metrics,
        "explainability": {
            "group_importance_share": dict(zip(imp["names"], imp["share"])),
            "group_mean_abs_shap_ms": dict(zip(imp["names"], imp["mean_abs_phi"])),
            "temporal_attribution_share": dict(zip(map(int, temporal["lag_hours_before_origin"]),
                                                   temporal["share"])),
            "counterfactuals_example_window": cfs,
            "counterfactual_search_pressure_minus_0.2ms": search_out,
            "explanation_quality": quality,
            "attention_share": None if attention is None else dict(
                zip(map(int, attention["lag_hours_before_origin"]), attention["share"])),
            "attention_per_head": None if attention is None else attention["per_head"],
            "delta_units": DELTA_UNITS,
        },
        "train_history": history,
        "runtime_seconds": time.time() - t0,
    }
    (out / "results.json").write_text(json.dumps(_clean(results), indent=2))
    _write_markdown(out / "results.md", results)
    if cfg.make_plots:
        _plots(out, test, scen, rank_hists, history, imp, temporal, ex_i, expl, attention)
    log(f"done in {results['runtime_seconds']:.0f}s -> {out}")
    return results


def _write_markdown(path, res):
    cols = ["crps", "power_crps", "energy_score", "mae", "rmse", "ks", "wasserstein", "acf_distance",
            "ramp_wasserstein", "diversity", "spread_skill", "calibration_error", "coverage_90"]
    lines = ["# Results (test period)\n", "| model | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, m in res["metrics"].items():
        row = [("-" if m[c] is None else f"{m[c]:.4f}") for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    ex = res["explainability"]
    lines += ["\n## Feature-group importance (mean |SHAP| share)\n"]
    lines += [f"- {k}: {v:.1%}" for k, v in sorted(ex["group_importance_share"].items(), key=lambda kv: -kv[1])]
    lines += ["\n## Counterfactuals (example window)\n"]
    lines += [f"- {k} {v['delta']:+} ({v['unit']}): mean wind change {v['mean_shift']:+.3f} m/s"
              for k, v in ex["counterfactuals_example_window"].items()]
    lines += ["\n## Explanation quality\n"]
    lines += [f"- {k}: {v:.3f}" for k, v in ex["explanation_quality"].items()]
    if ex["attention_share"]:
        top = sorted(ex["attention_share"].items(), key=lambda kv: -kv[1])[:5]
        lines += ["\n## Attention (top history hours)\n"]
        lines += [f"- {k} h before origin: {v:.1%}" for k, v in top]
    Path(path).write_text("\n".join(lines) + "\n")


def _plots(out, test, scen, rank_hists, history, imp, temporal, ex_i, expl, attention=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = test.target.shape[1]
    hrs = np.arange(H)
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in scen["diffusion"][ex_i]:
        ax.plot(hrs, s, color="tab:blue", alpha=0.25, lw=1)
    ax.plot(hrs, scen["diffusion"][ex_i].mean(0), color="navy", lw=2, label="scenario mean")
    ax.plot(hrs, test.target[ex_i], color="red", lw=2, label="observed")
    ax.set_xlabel("hours ahead"); ax.set_ylabel("wind speed (m/s)"); ax.legend()
    ax.set_title("Generated wind-speed scenarios (test window)")
    fig.tight_layout(); fig.savefig(out / "scenarios.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["step"], history["train_loss"], label="train")
    ax.plot(history["step"], history["val_loss"], label="val")
    ax.set_xlabel("step"); ax.set_ylabel("noise-prediction MSE"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "training_loss.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(rank_hists), figsize=(4 * len(rank_hists), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, h) in zip(axes, rank_hists.items()):
        ax.bar(range(len(h)), h); ax.axhline(1 / len(h), color="k", ls="--"); ax.set_title(k)
    fig.suptitle("Rank histograms (flat = calibrated)"); fig.tight_layout()
    fig.savefig(out / "rank_histograms.png", dpi=130); plt.close(fig)

    order = np.argsort(imp["share"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(imp["names"])[order], imp["share"][order])
    ax.set_xlabel("share of mean |SHAP|"); ax.set_title("Feature-group importance")
    fig.tight_layout(); fig.savefig(out / "shap_importance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(temporal["lag_hours_before_origin"], temporal["share"])
    ax.invert_xaxis(); ax.set_xlabel("hours before forecast origin"); ax.set_ylabel("importance share")
    ax.set_title("Temporal attribution (occlusion)")
    fig.tight_layout(); fig.savefig(out / "temporal_attribution.png", dpi=130); plt.close(fig)

    r = expl.counterfactual(test.hist[ex_i], test.cal[ex_i], "pressure", -500.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hrs, r["base"].mean(0), label="factual", color="tab:blue")
    ax.plot(hrs, r["counterfactual"].mean(0), label="pressure -5 hPa", color="tab:orange")
    ax.set_xlabel("hours ahead"); ax.set_ylabel("mean scenario wind (m/s)"); ax.legend()
    ax.set_title("Counterfactual: lower pressure")
    fig.tight_layout(); fig.savefig(out / "counterfactual_pressure.png", dpi=130); plt.close(fig)

    if attention is not None:
        lags = attention["lag_hours_before_origin"]
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(attention["per_head"], aspect="auto", cmap="viridis",
                       extent=[lags[0] + 0.5, lags[-1] - 0.5, len(attention["per_head"]) - 0.5, -0.5])
        ax.set_xlabel("hours before forecast origin"); ax.set_ylabel("attention head")
        ax.set_title("Cross-attention over history hours"); fig.colorbar(im, ax=ax, label="weight")
        fig.tight_layout(); fig.savefig(out / "attention.png", dpi=130); plt.close(fig)


def run_from_csv(data_dir, out_dir, cfg: Config = None, nrows=None, verbose=True):
    return run_pipeline(load_raw(data_dir, nrows=nrows), out_dir, cfg, verbose)


def summarize_runs(run_dirs, out_path):
    """Mean +- std of every metric over several runs (seeds); writes summary.md and summary.json."""
    runs = [json.loads((Path(d) / "results.json").read_text()) for d in run_dirs]
    groups = {}
    for d, r in zip(run_dirs, runs):
        bb = r["config"].get("backbone", "mlp")
        for model, m in r["metrics"].items():
            name = f"diffusion_{bb}" if model == "diffusion" else model
            for k, v in m.items():
                groups.setdefault(name, {}).setdefault(k, []).append(v)
        for k, v in r["explainability"]["explanation_quality"].items():
            groups.setdefault(f"explanations_{bb}", {}).setdefault(k, []).append(v)
    stats = {n: {k: (None if any(x is None for x in v) else [float(np.mean(v)), float(np.std(v)), len(v)])
                 for k, v in m.items()} for n, m in groups.items()}
    cols = ["crps", "power_crps", "energy_score", "mae", "rmse", "ks", "wasserstein", "acf_distance",
            "ramp_wasserstein", "diversity", "spread_skill", "calibration_error", "coverage_90"]
    fmt = lambda c: "-" if c is None else f"{c[0]:.4f} ± {c[1]:.4f}"
    lines = [f"# Summary over {len(runs)} runs (mean ± std across seeds)\n",
             "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {n} | " + " | ".join(fmt(m.get(c)) for c in cols) + " |"
              for n, m in stats.items() if not n.startswith("explanations_")]
    lines += ["\n## Explanation quality\n"]
    for n, m in stats.items():
        if n.startswith("explanations_"):
            lines += [f"- {n[len('explanations_'):]} {k}: {fmt(v)}" for k, v in m.items()]
    out_path = Path(out_path)
    (out_path / "summary.md").write_text("\n".join(lines) + "\n")
    (out_path / "summary.json").write_text(json.dumps(_clean(stats), indent=2))
    return stats
