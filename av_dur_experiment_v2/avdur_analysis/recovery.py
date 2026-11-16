"""Parameter and model recovery with the v2 design (v1 article figures 18 and 19).

    python -m avdur_analysis.recovery --n 50            # both, 50 datasets per generating model
    python -m avdur_analysis.recovery --kind model --n 20 --workers 6

Each synthetic dataset has the cells (stimuli and trial counts) of a randomly chosen complete participant,
with answers drawn from a generating model. Its parameters come from group-level distributions of the real
joint fits (fitting.py): sigmas log-normal around the participants' log values (SD at least SIGMA_SD_MIN),
lapse uniform on LAPSE_RANGE, probabilities logit-normal (SD at least LOGIT_SD_MIN), so the simulations
cover a range wider than the few fitted participants. The two auditory sigmas are ordered (low noise <= high).
    parameter recovery: fitted with the generating model; true against recovered
    model recovery:     fitted with every model; the best AIC is the recovered model
Every finished dataset is appended to results/recovery/<kind>/<model>.jsonl, so a run can be stopped and
resumed; running again adds datasets until --n are there.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D
from . import fitting as F
from . import models as M
from . import style as St

OUT = D.RESULTS / "recovery"
FIG_DIR = D.RESULTS / "figures" / "recovery"
SIGMA_SD_MIN = 0.25
LOGIT_SD_MIN = 0.5
LAPSE_RANGE = (0.0, 0.06)
N_STARTS = 3


def group_distribution(model):
    """{parameter: (kind, centre, spread)} from the saved real fits of `model`."""
    table = F.fit_all(models=[model], verbose=False)
    dist = {}
    for name in F.param_names(model):
        v = table[name].to_numpy(float)
        if name.startswith("lapse"):
            dist[name] = ("uniform", *LAPSE_RANGE)
        elif name.startswith("sigma"):
            lv = np.log(v)
            dist[name] = ("lognormal", float(lv.mean()), float(max(lv.std(ddof=1) if len(v) > 1 else 0, SIGMA_SD_MIN)))
        else:
            lv = np.log(np.clip(v, 0.02, 0.98) / (1 - np.clip(v, 0.02, 0.98)))
            dist[name] = ("logitnormal", float(lv.mean()), float(max(lv.std(ddof=1) if len(v) > 1 else 0, LOGIT_SD_MIN)))
    return dist


def draw(dist, rng):
    out = {}
    for name, (kind, a, b) in dist.items():
        if kind == "uniform":
            out[name] = float(rng.uniform(a, b))
        elif kind == "lognormal":
            out[name] = float(np.clip(np.exp(rng.normal(a, b)), 0.03, 3.0))
        else:
            out[name] = float(np.clip(1 / (1 + np.exp(-rng.normal(a, b))), 0.01, 0.99))
    if out.get("sigma_a_low", 0) > out.get("sigma_a_high", np.inf):      # louder noise, larger sigma
        out["sigma_a_low"], out["sigma_a_high"] = out["sigma_a_high"], out["sigma_a_low"]
    return out


def complete_designs():
    pids = [p for p in D.participants() if D.status(p)["complete"]]
    return {p: F.design(p) for p in pids}


def _one(kind, gen, seed, dist, fit_models):
    rng = np.random.default_rng(seed)
    designs = complete_designs()
    pid = list(designs)[rng.integers(len(designs))]
    true = draw(dist, rng)
    sim = F.simulate_design(designs[pid], gen, true, rng)
    fits = {}
    for m in fit_models:
        t0 = time.time()
        prm, nll, starts = F.fit_design(sim, m, n_starts=N_STARTS, seed=seed)
        k = len(prm)
        fits[m] = dict(params=prm, nll=nll, aic=2 * k + 2 * nll, bic=k * np.log(sim.n_trials) + 2 * nll,
                       starts=starts, seconds=round(time.time() - t0, 1))
    return dict(kind=kind, generating=gen, seed=int(seed), design=pid, true=true, fits=fits)


def run(kind="parameter", n=50, models=M.MODELS, workers=None, seed0=1000):
    """Add datasets until each generating model has n of them (results/recovery/<kind>/<model>.jsonl)."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    (OUT / kind).mkdir(parents=True, exist_ok=True)
    jobs = []
    for gi, gen in enumerate(models):
        path = OUT / kind / f"{gen}.jsonl"
        done = {json.loads(line)["seed"] for line in path.read_text().splitlines()} if path.exists() else set()
        dist = group_distribution(gen)
        fit_models = [gen] if kind == "parameter" else list(models)
        seeds = [s for s in range(seed0 + 10000 * gi, seed0 + 10000 * gi + 10 * n) if s not in done][: max(0, n - len(done))]
        jobs += [(kind, gen, s, dist, fit_models) for s in seeds]
    print(f"[recovery] {kind}: {len(jobs)} datasets on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            with open(OUT / kind / f"{res['generating']}.jsonl", "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[recovery] {kind} {i}/{len(jobs)} ({res['generating']}, design {res['design']})", flush=True)


def load(kind):
    rows = []
    for path in sorted((OUT / kind).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

LABELS = {"lapse": "λ", "sigma_a_low": "σ_A low", "sigma_a_high": "σ_A high", "sigma_v": "σ_V", "p_c": "p_c",
          "p_switch_low": "p_switch low", "p_switch_high": "p_switch high"}


def parameter_table():
    rows = []
    for r in load("parameter"):
        fit = r["fits"][r["generating"]]["params"]
        for name, tv in r["true"].items():
            rows.append(dict(model=r["generating"], parameter=name, true=tv, recovered=fit[name], seed=r["seed"]))
    return pd.DataFrame(rows)


def fig_parameter_recovery(save=True):
    """Rows: generating model; columns: parameters. Dashed: identity; grey band: +/-5%; r and RMSE inset."""
    t = parameter_table()
    if t.empty:
        raise FileNotFoundError("no parameter-recovery results: python -m avdur_analysis.recovery --kind parameter")
    models = [m for m in M.MODELS if m in set(t.model)]
    cols = list(dict.fromkeys(p for m in models for p in F.param_names(m)))
    fig, axes = plt.subplots(len(models), len(cols), figsize=(3.0 * len(cols), 3.0 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, p in enumerate(cols):
            ax = axes[i, j]
            s = t[(t.model == m) & (t.parameter == p)]
            if s.empty:
                ax.set_visible(False)
                continue
            lo, hi = min(s.true.min(), s.recovered.min()), max(s.true.max(), s.recovered.max())
            pad = 0.05 * (hi - lo + 1e-9)
            xx = np.linspace(lo - pad, hi + pad, 50)
            ax.fill_between(xx, xx * 0.95, xx * 1.05, color="0.85", lw=0)
            ax.plot(xx, xx, "k--", lw=1)
            ax.scatter(s.true, s.recovered, s=14, color=St.MODEL_STYLE[m]["color"], alpha=0.7)
            r = np.corrcoef(s.true, s.recovered)[0, 1] if len(s) > 2 else np.nan
            rmse = np.sqrt(np.mean((s.true - s.recovered) ** 2))
            ax.text(0.04, 0.96, f"r = {r:.2f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(fc="white", ec="0.7", lw=0.5))
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"true {LABELS.get(p, p)}", fontsize=11)
            ax.set_ylabel("recovered", fontsize=10)
            ax.tick_params(labelsize=9)
        axes[i, 0].annotate(St.MODEL_STYLE[m]["label"], xy=(-0.55, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=13, fontweight="bold")
    n = t.groupby("model").seed.nunique().to_dict()
    fig.suptitle("Parameter recovery (" + ", ".join(f"{St.MODEL_STYLE[m]['label']}: N = {n[m]}" for m in models) + ")",
                 y=1.01, fontsize=14)
    fig.tight_layout()
    return _save(fig, "fig_parameter_recovery", save)


def confusion(crit="aic"):
    """Generating model x best-fitting model: counts."""
    rows = [dict(generating=r["generating"], best=min(r["fits"], key=lambda m: r["fits"][m][crit])) for r in load("model")]
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return pd.crosstab(t.generating, t.best).reindex(index=M.MODELS, columns=M.MODELS, fill_value=0).dropna(how="all")


def fig_model_recovery(crit="aic", save=True):
    c = confusion(crit)
    if c.empty:
        raise FileNotFoundError("no model-recovery results: python -m avdur_analysis.recovery --kind model")
    c = c[c.sum(1) > 0]
    prop = c.div(c.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(prop.to_numpy(), cmap="Greys", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(prop.to_numpy()):
        ax.text(j, i, f"{v:.2f}\n({c.iloc[i, j]})", ha="center", va="center", fontsize=12,
                color="white" if v > 0.6 else "black")
    ax.set_xticks(range(prop.shape[1]))
    ax.set_xticklabels([St.MODEL_STYLE[m]["label"] for m in prop.columns], rotation=15)
    ax.set_yticks(range(prop.shape[0]))
    ax.set_yticklabels([St.MODEL_STYLE[m]["label"] for m in prop.index])
    ax.set_xlabel(f"Best model ({crit.upper()})")
    ax.set_ylabel("Generating model")
    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    return _save(fig, f"fig_model_recovery_{crit}", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}.{ext}")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("parameter", "model", "both"), default="both")
    ap.add_argument("--n", type=int, default=50, help="datasets per generating model")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=list(M.MODELS), choices=list(M.MODELS))
    args = ap.parse_args(argv)
    for kind in (("parameter", "model") if args.kind == "both" else (args.kind,)):
        run(kind, args.n, tuple(args.models), args.workers)


if __name__ == "__main__":
    main()
