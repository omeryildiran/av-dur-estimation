"""Generate plotConflictvsPSE_betterRefit.ipynb (valid ipynb JSON)."""
import json
import uuid

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "id": uuid.uuid4().hex[:12],
                  "source": text.splitlines(keepends=True)})


md("""# PSE vs Conflict — improved free μ/σ/λ psychometric refits

Replicates `aggregated_mu_vs_models_freeMuSigmaLambda_sem_clean` but **refits each
participant better**:

* **Lognormal psychometric** — cumulative Gaussian on `log(testDurS / standardDur)`:
  `p(choose test) = λ/2 + (1−λ)·Φ((x − μ)/σ)`, with **μ, σ, λ all free per condition**.
* **Canonical conflict snapping** — recorded `conflictDur` is jittered per participant
  (up to 13 near-duplicate values around the 7 design levels). We snap each trial to the
  nearest canonical level so every level pools all its trials into one curve, and data /
  models / participants share identical conflict labels. This is the main accuracy gain.
* **Stronger optimizer** — curated + random multi-starts of `L-BFGS-B` on a binomial NLL,
  then a Nelder–Mead polish (no `curve_fit`).
* Same procedure for **real data** and **model-simulated** choices, then mean ± SEM across
  participants (n = 11).

Fits are cached to `psychometric_fits_freeMuSigmaLambda_better_{real,simulated}/`.""")

code("""import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from plot_style import setup_style, FONT_SIZE_LABEL

setup_style()

import loadData
from free_psychometric_refits import (
    pf_input_from_durations,   # log(test/standard) for pf_mode='lognormal'
    psychometric_pf,           # lambda/2 + (1-lambda)*norm.cdf((x-mu)/sigma)
    mu_to_shift_s,             # standard_s*(exp(mu)-1) in lognormal mode
    sigma_to_plot_units,
    simulated_csv_path,
    DEFAULT_MAIN_MODELS,
    DEFAULT_SIM_VARIANTS,
)""")

code("""FONT = FONT_SIZE_LABEL""")

code("""# ── Configuration ────────────────────────────────────────────
STANDARD_S = 0.5
PF_MODE = 'lognormal'                       # log cumulative Gaussian on log(test/standard)
PIDS = ['as', 'dt', 'hh', 'ip', 'ln2', 'mh', 'ml', 'mt', 'oy', 'qs', 'sx']  # n = 11
MAIN_MODELS = DEFAULT_MAIN_MODELS           # ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']

# Model predictions come from the FREE-sigma generative fits (model_fits/P0x), simulated
# into simulated_data_freeSigma/ by _gen_freesigma_sims.py. With free sensory sigma the
# fits use a single shared lapse (LapseFix), matching the manuscript's single-lambda
# model-comparison table; p_c is no longer pinned by the fixed-sigma parameterization.
SIM_DIR = 'simulated_data_freeSigma'
MODEL_VARIANTS = ['LapseFix_sharedPrior']

CANONICAL_CONFLICTS = np.array([-0.25, -0.17, -0.08, 0.0, 0.08, 0.17, 0.25])
NOISE_LEVELS = [0.1, 1.2]

OUT_REAL = Path('psychometric_fits_freeMuSigmaLambda_better_real')
OUT_SIM = Path('psychometric_fits_freeMuSigmaLambda_better_freeSigmaModel_simulated')
FORCE_REFIT = True                          # recompute the better fits from scratch

# Optimizer budget for the improved fit (more starts -> fewer jagged local optima).
N_RANDOM_STARTS = 40
RNG_SEED = 0""")

code("""# ── Lognormal PF fit per condition (free mu, sigma, lambda) ──────────────────
PF_BOUNDS = [(-1.0, 1.0), (0.01, 3.0), (0.0, 0.25)]   # mu, sigma, lambda (lognormal mode)


def snap_conflict(values):
    \"\"\"Snap recorded conflictDur to the nearest canonical design level.\"\"\"
    values = np.asarray(values, dtype=float)
    idx = np.abs(values[:, None] - CANONICAL_CONFLICTS[None, :]).argmin(axis=1)
    return CANONICAL_CONFLICTS[idx]


def condition_counts(sub):
    \"\"\"Aggregate one condition's trials into (unique pf_x, n_chose_test, n_total).\"\"\"
    df = sub.copy()
    if 'chose_test' in df.columns:
        chose = (df['chose_test'].astype(float) > 0.5).astype(float)
    elif 'responses' in df.columns:
        chose = (df['responses'].astype(float) == 2).astype(float)
    else:
        raise ValueError('Data must contain chose_test or responses.')
    pf_x = pf_input_from_durations(df['testDurS'], df['standardDur'], pf_mode=PF_MODE)
    chose = chose.to_numpy()
    mask = np.isfinite(pf_x) & np.isfinite(chose)
    pf_x, chose = pf_x[mask], chose[mask]
    ux, inv = np.unique(np.round(pf_x, 12), return_inverse=True)
    n_total = np.bincount(inv).astype(float)
    n_chose = np.bincount(inv, weights=chose).astype(float)
    return ux, n_chose, n_total


def _binom_nll(params, pf_x, n_chose, n_total):
    mu, sigma, lam = params
    if sigma <= 0 or lam < 0 or lam >= 0.5:
        return 1e10
    p = psychometric_pf(pf_x, mu, sigma, lam)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -float(np.sum(n_chose * np.log(p) + (n_total - n_chose) * np.log1p(-p)))


def _clip(x0):
    return np.array([min(max(v, lo + 1e-6), hi - 1e-6)
                     for v, (lo, hi) in zip(x0, PF_BOUNDS)], dtype=float)


def fit_pf_counts_better(pf_x, n_chose, n_total, n_random=N_RANDOM_STARTS, seed=RNG_SEED):
    \"\"\"Free mu/sigma/lambda lognormal PF fit with strong multi-start + polish.\"\"\"
    pf_x = np.asarray(pf_x, float); n_chose = np.asarray(n_chose, float)
    n_total = np.asarray(n_total, float)
    m = np.isfinite(pf_x) & np.isfinite(n_chose) & np.isfinite(n_total) & (n_total > 0)
    pf_x, n_chose, n_total = pf_x[m], n_chose[m], n_total[m]
    if len(np.unique(pf_x)) < 3:
        return dict(mu=np.nan, sigma=np.nan, lambda_=np.nan, nll=np.nan,
                    success=False, n_trials=float(n_total.sum()))

    prop = np.clip(n_chose / n_total, 0, 1)
    order = np.argsort(pf_x)
    xs, ps = pf_x[order], prop[order]
    # mu guess: linear-interpolate where the proportion crosses 0.5
    if ps.min() <= 0.5 <= ps.max():
        mu_guess = float(np.interp(0.5, ps, xs))
    else:
        mu_guess = float(xs[np.argmin(np.abs(ps - 0.5))])
    sigma_guess = float(max(np.nanstd(pf_x), 0.05))

    rng = np.random.default_rng(seed)
    # deterministic grid over (mu, sigma, lambda) + the data-driven guess
    mu_seeds = [mu_guess, 0.0]
    sigma_seeds = [0.05, 0.10, 0.20, 0.40, 0.80, 1.20]
    lam_seeds = [0.01, 0.05]
    curated = [[mu_guess, sigma_guess, 0.02]]
    curated += [[mu, sig, lam] for mu in mu_seeds for sig in sigma_seeds for lam in lam_seeds]
    randoms = [[rng.uniform(-0.5, 0.5), rng.uniform(0.05, 1.2), rng.uniform(0.0, 0.15)]
               for _ in range(n_random)]

    best = None
    for x0 in curated + randoms:
        try:
            res = minimize(_binom_nll, _clip(x0), args=(pf_x, n_chose, n_total),
                           bounds=PF_BOUNDS, method='L-BFGS-B')
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return dict(mu=np.nan, sigma=np.nan, lambda_=np.nan, nll=np.nan,
                    success=False, n_trials=float(n_total.sum()))

    # Nelder-Mead polish around the best L-BFGS-B solution.
    try:
        pol = minimize(_binom_nll, best.x, args=(pf_x, n_chose, n_total),
                       method='Nelder-Mead',
                       options=dict(xatol=1e-7, fatol=1e-7, maxiter=2000))
        if np.isfinite(pol.fun) and pol.fun < best.fun:
            cand = _clip(pol.x)
            if _binom_nll(cand, pf_x, n_chose, n_total) <= best.fun:
                best = pol
    except Exception:
        pass

    mu, sigma, lam = _clip(best.x)
    return dict(mu=float(mu), sigma=float(sigma), lambda_=float(lam),
                nll=float(best.fun), success=bool(best.success),
                n_trials=float(n_total.sum()))


def fit_free_table(data):
    \"\"\"Per (noise x canonical-conflict) free PF fit -> tidy DataFrame.\"\"\"
    df = data.copy()
    df['conflict_snap'] = snap_conflict(df['conflictDur'].astype(float))
    rows = []
    for noise in NOISE_LEVELS:
        nm = np.isclose(df['audNoise'].astype(float), noise)
        for conflict in CANONICAL_CONFLICTS:
            sub = df[nm & (df['conflict_snap'] == conflict)]
            if sub.empty:
                continue
            ux, nc, nt = condition_counts(sub)
            fit = fit_pf_counts_better(ux, nc, nt)
            rows.append(dict(
                audioNoise=float(noise), conflict=float(conflict),
                conflict_ms=int(round(conflict * 1000)),
                n_trials=int(fit['n_trials']),
                mu=fit['mu'], sigma=fit['sigma'], lambda_=fit['lambda_'],
                nll=fit['nll'], success=fit['success'],
                mu_shift_s=mu_to_shift_s(fit['mu'], STANDARD_S, PF_MODE),
                mu_shift_ms=mu_to_shift_s(fit['mu'], STANDARD_S, PF_MODE) * 1000,
                sigma_plot=sigma_to_plot_units(fit['sigma'], STANDARD_S, PF_MODE),
            ))
    return pd.DataFrame(rows)""")

code("""# ── Cached fit drivers (real + simulated) ────────────────────────────
def _save(path, fit_df, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {'metadata': metadata, 'fitParamsByCondition': fit_df.to_dict(orient='records')},
        indent=2))


def _load(path):
    return pd.DataFrame(json.loads(path.read_text())['fitParamsByCondition'])


def fit_real_pid(pid, force=FORCE_REFIT):
    out = OUT_REAL / pid / f'{pid}_psychometricFits.json'
    if out.exists() and not force:
        return _load(out)
    # loadData builds log(conflictDur)/log(standardDur) helper columns we never use;
    # conflictDur == 0 (and negatives) make those log(0)/log(<0) -> harmless inf/nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        data, data_name = loadData.loadData(f'{pid}_all.csv', verbose=False)
    fit_df = fit_free_table(data)
    fit_df.insert(0, 'pid', pid)
    _save(out, fit_df, dict(participantID=pid, dataName=data_name, pf_mode=PF_MODE,
                            parameterization='free_mu_sigma_lambda_per_condition',
                            conflict='snapped_to_canonical', fitter='multistart_lbfgsb+neldermead'))
    return fit_df


def fit_sim_pid(pid, model, force=FORCE_REFIT):
    out = OUT_SIM / pid / f'{pid}_{model}_psychometricFits.json'
    if out.exists() and not force:
        return _load(out)
    sim_path = simulated_csv_path(pid, model, variants=MODEL_VARIANTS, sim_dir=SIM_DIR)
    if sim_path is None:
        return pd.DataFrame()
    fit_df = fit_free_table(pd.read_csv(sim_path))
    fit_df.insert(0, 'pid', pid)
    fit_df.insert(1, 'model', model)
    _save(out, fit_df, dict(participantID=pid, modelType=model, simulatedData=str(sim_path),
                            pf_mode=PF_MODE, parameterization='free_mu_sigma_lambda_per_condition',
                            conflict='snapped_to_canonical', fitter='multistart_lbfgsb+neldermead'))
    return fit_df


def aggregate(fit_df, value_col='mu_shift_s'):
    keys = ['audioNoise', 'conflict']
    if 'model' in fit_df.columns:
        keys = ['model'] + keys
    agg = fit_df.groupby(keys)[value_col].agg(['mean', 'std', 'count']).reset_index()
    agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    return agg""")

code("""# ── Run the refits ─────────────────────────────────────────────
data_frames, model_frames, missing = [], [], []
for pid in PIDS:
    data_frames.append(fit_real_pid(pid))
    for model in MAIN_MODELS:
        mf = fit_sim_pid(pid, model)
        (model_frames if not mf.empty else missing).append(mf if not mf.empty else (pid, model))
    print(f'  fitted {pid}')

data_pp = pd.concat(data_frames, ignore_index=True)
model_pp = pd.concat([m for m in model_frames], ignore_index=True)

data_agg = aggregate(data_pp, 'mu_shift_s')
model_agg = aggregate(model_pp, 'mu_shift_s')
sigma_agg = aggregate(data_pp, 'sigma_plot')
lambda_agg = aggregate(data_pp, 'lambda_')

print(f'\\nParticipants: {len(PIDS)} | data rows: {len(data_pp)} | model rows: {len(model_pp)}')
if missing:
    print('Missing simulated CSVs:', missing)
print('\\nData PSE shift (ms) mean ± SEM:')
print((data_agg.assign(mean_ms=data_agg['mean']*1000, sem_ms=data_agg['sem']*1000)
       [['audioNoise', 'conflict', 'mean_ms', 'sem_ms', 'count']]).to_string(index=False))
print('\\nData PF sigma (log units) mean ± SEM:')
print(sigma_agg[['audioNoise', 'conflict', 'mean', 'sem']].to_string(index=False))""")

code("""# ── Plot: free μ/σ/λ lognormal PF, mean ± SEM (replicates the clean figure) ──────
DATA_BASE = {0.1: '#d62728', 1.2: '#1f77b4'}
MODEL_DISPLAY = {'lognorm': 'Causal inference', 'fusionOnlyLogNorm': 'Forced fusion',
                 'switchingFree': 'Cue switching'}
MODEL_STYLE = {'lognorm': dict(ls='-', mk='o'), 'fusionOnlyLogNorm': dict(ls='--', mk='s'),
               'switchingFree': dict(ls=':', mk='^')}
MODEL_COLORS = {'fusionOnlyLogNorm': '#2ca02c', 'lognorm': '#600c70', 'switchingFree': '#080529'}

audio_levels = sorted(data_agg['audioNoise'].unique())
panel_titles = ['A', 'B']
model_names = [m for m in MAIN_MODELS if m in model_agg['model'].unique()]
all_series = ['__data__'] + model_names
mid = (len(all_series) - 1) / 2
offsets = {s: (i - mid) * 15.0 for i, s in enumerate(all_series)}

fig, axes = plt.subplots(1, len(audio_levels), figsize=(12, 6), sharey=True)
if len(audio_levels) == 1:
    axes = [axes]

from matplotlib.lines import Line2D
handles, labels, models_done = [], [], set()

for pi, (ax, noise) in enumerate(zip(axes, audio_levels)):
    sd = data_agg[data_agg['audioNoise'] == noise].sort_values('conflict')
    x = sd['conflict'].to_numpy(float) * 1000
    y = sd['mean'].to_numpy(float) * 1000
    yerr = sd['sem'].to_numpy(float) * 1000
    base_rgb = mcolors.to_rgb(DATA_BASE[noise])
    alphas = np.linspace(0.25, 1.0, len(x))
    for k in range(len(x)):
        rgba = (*base_rgb, alphas[k])
        ax.errorbar(x[k] + offsets['__data__'], y[k], yerr=[[yerr[k]], [yerr[k]]],
                    color=rgba, fmt='o', capsize=6, lw=1.8, capthick=1.5,
                    markersize=7, zorder=6, markeredgecolor=rgba)
    if pi == 0:
        r, b = mcolors.to_rgb(DATA_BASE[0.1]), mcolors.to_rgb(DATA_BASE[1.2])
        handles += [Line2D([0], [0], color=(*r, 0.7), marker='o', lw=0, markersize=7,
                           markeredgecolor=(*r, 0.7)),
                    Line2D([0], [0], color=(*b, 0.7), marker='o', lw=0, markersize=7,
                           markeredgecolor=(*b, 0.7))]
        labels += ['Data (low noise)', 'Data (high noise)']

    for mname in model_names:
        sm = model_agg[(model_agg['audioNoise'] == noise) &
                       (model_agg['model'] == mname)].sort_values('conflict')
        if sm.empty:
            continue
        c = MODEL_COLORS.get(mname, 'gray')
        ls = MODEL_STYLE.get(mname, {}).get('ls', '-')
        mk = MODEL_STYLE.get(mname, {}).get('mk', 'o')
        xm = sm['conflict'].to_numpy(float) * 1000 + offsets.get(mname, 0)
        ym = sm['mean'].to_numpy(float) * 1000
        m_yerr = sm['sem'].to_numpy(float) * 1000
        ax.errorbar(xm, ym, yerr=m_yerr, color=c, linestyle=ls, marker=mk,
                    markerfacecolor=c, markeredgewidth=1.6, capsize=4, lw=2.5,
                    markersize=7, zorder=4)
        if mname not in models_done:
            handles.append(Line2D([0], [0], color=c, ls=ls, marker=mk, markerfacecolor=c,
                                  markeredgewidth=1.3, lw=2.5, markersize=7))
            labels.append(MODEL_DISPLAY.get(mname, mname))
            models_done.add(mname)

    ax.text(0, 200, 'Low auditory noise' if pi == 0 else 'High auditory noise',
            fontsize=FONT, ha='center', va='center')
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=.7)
    ax.axvline(0, color='gray', ls='--', lw=1, alpha=.7)
    ax.set_title(panel_titles[pi], fontsize=FONT, loc='left', pad=12)
    ax.set_xticks([-250, -170, -80, 0, 80, 170, 250])
    ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_xticklabels([str(t) for t in [-250, -170, -80, 0, 80, 170, 250]], fontsize=FONT - 2)
    ax.tick_params(axis='both', labelsize=FONT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('PSE shift (ms)', fontsize=FONT)
fig.supxlabel('Cue conflict (ms)', fontsize=FONT, x=0.5)
handles.append(Line2D([], [], color='none'))
labels.append(f'free lambda/mu/sigma, ±SEM (n={len(PIDS)})')
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5),
           fontsize=FONT - 4, frameon=True, edgecolor='black', fancybox=False)
plt.tight_layout(rect=(0, 0.05, 0.82, 1))

stem = 'aggregated_mu_vs_models_freeMuSigmaLambda_sem_better_freeSigmaModel'
plt.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{stem}.pdf', dpi=900, format='pdf', bbox_inches='tight')
plt.savefig(f'{stem}.svg', format='svg', bbox_inches='tight')
plt.show()
print(f'Saved {stem}.png/.pdf/.svg')""")

md("""## Model discrimination

The PSE-shift panel collapses each model into per-condition μ, so the three models look
near-identical there. The panels below separate them: the **discrimination threshold (σ)**
and **lapse (λ)** the descriptive PF recovers from each model's simulated choices, plus a
**per-participant RMSE** of each model's PSE-shift against the real data.""")

code("""# ── Aggregate sigma & lambda for data and each model ────────────
data_sigma = aggregate(data_pp, 'sigma_plot')
model_sigma = aggregate(model_pp, 'sigma_plot')
data_lambda = aggregate(data_pp, 'lambda_')
model_lambda = aggregate(model_pp, 'lambda_')


def two_panel_metric(d_agg, m_agg, ylabel, stem, yticks=None, scale=1.0):
    \"\"\"Generic two-panel (low/high noise) metric vs conflict, data + models, mean±SEM.\"\"\"
    levels = sorted(d_agg['audioNoise'].unique())
    mnames = [m for m in MAIN_MODELS if m in m_agg['model'].unique()]
    series = ['__data__'] + mnames
    mmid = (len(series) - 1) / 2
    off = {s: (i - mmid) * 15.0 for i, s in enumerate(series)}

    fig, axes = plt.subplots(1, len(levels), figsize=(12, 6), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    from matplotlib.lines import Line2D
    hs, ls_, done = [], [], set()
    for pi, (ax, noise) in enumerate(zip(axes, levels)):
        sd = d_agg[d_agg['audioNoise'] == noise].sort_values('conflict')
        x = sd['conflict'].to_numpy(float) * 1000
        y = sd['mean'].to_numpy(float) * scale
        ye = sd['sem'].to_numpy(float) * scale
        ax.errorbar(x + off['__data__'], y, yerr=ye, color='black', fmt='o',
                    capsize=6, lw=1.8, capthick=1.5, markersize=7, zorder=6)
        if pi == 0:
            hs.append(Line2D([0], [0], color='black', marker='o', lw=1.8, markersize=7))
            ls_.append('Data')
        for mname in mnames:
            sm = m_agg[(m_agg['audioNoise'] == noise) & (m_agg['model'] == mname)].sort_values('conflict')
            if sm.empty:
                continue
            c = MODEL_COLORS.get(mname, 'gray')
            sty = MODEL_STYLE.get(mname, {})
            xm = sm['conflict'].to_numpy(float) * 1000 + off.get(mname, 0)
            ym = sm['mean'].to_numpy(float) * scale
            yem = sm['sem'].to_numpy(float) * scale
            ax.errorbar(xm, ym, yerr=yem, color=c, linestyle=sty.get('ls', '-'),
                        marker=sty.get('mk', 'o'), markerfacecolor=c, markeredgewidth=1.6,
                        capsize=4, lw=2.5, markersize=7, zorder=4)
            if mname not in done:
                hs.append(Line2D([0], [0], color=c, ls=sty.get('ls', '-'), marker=sty.get('mk', 'o'),
                                 markerfacecolor=c, markeredgewidth=1.3, lw=2.5, markersize=7))
                ls_.append(MODEL_DISPLAY.get(mname, mname))
                done.add(mname)
        ax.text(0.5, 1.01, 'Low auditory noise' if pi == 0 else 'High auditory noise',
                transform=ax.transAxes, fontsize=FONT, ha='center', va='bottom')
        ax.axvline(0, color='gray', ls='--', lw=1, alpha=.7)
        ax.set_title(['A', 'B'][pi], fontsize=FONT, loc='left', pad=12)
        ax.set_xticks([-250, -170, -80, 0, 80, 170, 250])
        ax.set_xticklabels([str(t) for t in [-250, -170, -80, 0, 80, 170, 250]], fontsize=FONT - 2)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=FONT)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=FONT)
    fig.supxlabel('Cue conflict (ms)', fontsize=FONT, x=0.5, y=0.04)
    hs.append(Line2D([], [], color='none'))
    ls_.append(f'±SEM (n={len(PIDS)})')
    fig.legend(hs, ls_, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=FONT - 4,
               frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout(rect=(0, 0.05, 0.82, 1))
    for ext, kw in [('png', dict(dpi=200)), ('pdf', dict(dpi=900, format='pdf')), ('svg', dict(format='svg'))]:
        plt.savefig(f'{stem}.{ext}', bbox_inches='tight', **kw)
    plt.show()
    print(f'Saved {stem}.png/.pdf/.svg')


two_panel_metric(data_sigma, model_sigma, 'σ of psychometric function\\n(log-duration unit)',
                 'aggregated_sigma_vs_models_freeMuSigmaLambda_sem_better_freeSigmaModel')""")

code("""# ── Lapse rate (lambda) vs conflict ─────────────────────────────
two_panel_metric(data_lambda, model_lambda, 'Lapse rate lambda',
                 'aggregated_lambda_vs_models_freeMuSigmaLambda_sem_better_freeSigmaModel')""")

code("""# ── Per-participant model–data agreement (RMSE of PSE shift) ─────────
merged = model_pp.merge(
    data_pp[['pid', 'audioNoise', 'conflict', 'mu_shift_s']],
    on=['pid', 'audioNoise', 'conflict'], suffixes=('', '_data'))
merged['sq_err_ms2'] = ((merged['mu_shift_s'] - merged['mu_shift_s_data']) * 1000) ** 2

per_pid = (merged.groupby(['pid', 'model'])['sq_err_ms2'].mean()
           .pow(0.5).reset_index(name='rmse_ms'))
rmse_agg = per_pid.groupby('model')['rmse_ms'].agg(['mean', 'std', 'count']).reset_index()
rmse_agg['sem'] = rmse_agg['std'] / np.sqrt(rmse_agg['count'])
rmse_agg = rmse_agg.set_index('model').loc[[m for m in MAIN_MODELS if m in rmse_agg['model'].values]].reset_index()

print('PSE-shift RMSE vs data (ms), mean ± SEM across participants:')
print(rmse_agg[['model', 'mean', 'sem', 'count']].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 6))
xpos = np.arange(len(rmse_agg))
colors = [MODEL_COLORS.get(m, 'gray') for m in rmse_agg['model']]
ax.bar(xpos, rmse_agg['mean'], yerr=rmse_agg['sem'], color=colors, alpha=0.85,
       capsize=6, edgecolor='black', linewidth=1.2, zorder=3)
# overlay individual participants
for j, m in enumerate(rmse_agg['model']):
    pts = per_pid[per_pid['model'] == m]['rmse_ms'].to_numpy()
    ax.scatter(np.full_like(pts, xpos[j]) + np.random.default_rng(j).uniform(-0.12, 0.12, len(pts)),
               pts, color='black', s=22, alpha=0.5, zorder=4)
ax.set_xticks(xpos)
ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in rmse_agg['model']], fontsize=FONT - 2)
ax.set_ylabel('PSE-shift RMSE vs data (ms)', fontsize=FONT)
ax.set_title(f'Model-data agreement (lower = better, n={len(PIDS)})', fontsize=FONT - 2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
stem = 'model_vs_data_pse_rmse_better_freeSigmaModel'
for ext, kw in [('png', dict(dpi=200)), ('pdf', dict(dpi=900, format='pdf')), ('svg', dict(format='svg'))]:
    plt.savefig(f'{stem}.{ext}', bbox_inches='tight', **kw)
plt.show()
print(f'Saved {stem}.png/.pdf/.svg')""")

md("""## Formal model discrimination — trial-level AIC/BIC

The PSE-shift means are model-invariant (all three models fit the same data reproduce the
same mean bias), so they cannot rank the models. The discrimination lives in the
**trial-level likelihood** of the free-sigma generative fits (`model_fits/P0x`). Below:
per-participant ΔAIC/ΔBIC relative to Causal inference, with the Monte-Carlo noise floor
(AIC SD ≈ 1.9 at nSimul=2000, from `aic_noise_floor_seeds.json`) shown as a reference band.""")

code("""# ── Load free-sigma generative AIC/BIC for the 3 models x 11 participants ─────
PID_TO_ANON = {'as': 'P01', 'dt': 'P02', 'hh': 'P03', 'ip': 'P04', 'ln2': 'P07',
               'mh': 'P08', 'ml': 'P09', 'mt': 'P10', 'oy': 'P11', 'qs': 'P12', 'sx': 'P13'}
GEN_VARIANT = 'LapseFix_sharedPrior'

gen_rows = []
for pid, anon in PID_TO_ANON.items():
    for model in MAIN_MODELS:
        fp = Path('model_fits') / anon / f'{anon}_{model}_{GEN_VARIANT}_fit.json'
        d = json.loads(fp.read_text())
        gen_rows.append(dict(pid=pid, model=model, AIC=d['AIC'], BIC=d['BIC'],
                             logLik=d['logLikelihood'], k=len(d['fittedParams'])))
gen = pd.DataFrame(gen_rows)

aic = gen.pivot(index='pid', columns='model', values='AIC')[MAIN_MODELS]
bic = gen.pivot(index='pid', columns='model', values='BIC')[MAIN_MODELS]

# AIC Monte-Carlo noise floor (free-sigma) from the seed-resampling cache, if present.
AIC_NOISE_FLOOR = 1.9
try:
    nf = pd.DataFrame(json.loads(Path('aic_noise_floor_seeds.json').read_text()))
    nf = nf[nf['fit_source'] == 'Free sensory noise']
    if not nf.empty:
        AIC_NOISE_FLOOR = float(nf['AIC_sd'].mean())
except Exception:
    pass

# ΔAIC relative to Causal inference (lognorm); negative = better than causal inference.
REF = 'lognorm'
dAIC = aic.sub(aic[REF], axis=0)
dBIC = bic.sub(bic[REF], axis=0)
wins_aic = aic.idxmin(axis=1).value_counts().reindex(MAIN_MODELS).fillna(0).astype(int)
wins_bic = bic.idxmin(axis=1).value_counts().reindex(MAIN_MODELS).fillna(0).astype(int)

summary = pd.DataFrame({
    'mean_dAIC_vs_causal': dAIC.mean(), 'sem_dAIC': dAIC.sem(),
    'mean_dBIC_vs_causal': dBIC.mean(), 'sem_dBIC': dBIC.sem(),
    'AIC_wins': wins_aic, 'BIC_wins': wins_bic,
}).loc[MAIN_MODELS]
print(f'MC AIC noise floor (SD): {AIC_NOISE_FLOOR:.2f}')
print('\\nΔAIC/ΔBIC vs Causal inference (negative = better than causal):')
print(summary.round(2).to_string())""")

code("""# ── Plot: model discrimination (ΔAIC per participant + win counts) ──────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [2, 1]})

order = MAIN_MODELS
xpos = np.arange(len(order))
# noise-floor reference band around 0 (causal inference reference)
axL.axhspan(-AIC_NOISE_FLOOR, AIC_NOISE_FLOOR, color='gray', alpha=0.15, zorder=0,
            label=f'±MC noise floor ({AIC_NOISE_FLOOR:.1f})')
axL.axhline(0, color=MODEL_COLORS['lognorm'], ls='-', lw=2, alpha=0.8, zorder=1)

rng = np.random.default_rng(1)
for j, m in enumerate(order):
    vals = dAIC[m].to_numpy()
    jit = rng.uniform(-0.12, 0.12, len(vals))
    axL.scatter(np.full(len(vals), xpos[j]) + jit, vals, s=40,
                color=MODEL_COLORS.get(m, 'gray'), alpha=0.55, zorder=3,
                edgecolor='black', linewidth=0.5)
    mean, sem = dAIC[m].mean(), dAIC[m].sem()
    axL.errorbar(xpos[j], mean, yerr=sem, fmt='_', color='black', capsize=8,
                 markersize=22, lw=2.5, zorder=4)
axL.set_xticks(xpos)
axL.set_xticklabels([MODEL_DISPLAY[m] for m in order], fontsize=FONT - 4,
                    rotation=20, ha='right')
axL.set_ylabel('ΔAIC vs Causal inference', fontsize=FONT)
axL.set_title('A   Per-participant ΔAIC (negative = beats causal inference)',
              fontsize=FONT - 3, loc='left')
axL.axhline(0, color='gray', lw=0.5)
axL.legend(fontsize=FONT - 6, loc='upper right')
axL.spines['top'].set_visible(False); axL.spines['right'].set_visible(False)

# Right: AIC win counts
wc = wins_aic.loc[order]
axR.bar(xpos, wc.to_numpy(), color=[MODEL_COLORS.get(m, 'gray') for m in order],
        alpha=0.85, edgecolor='black', linewidth=1.2)
for j, v in enumerate(wc.to_numpy()):
    axR.text(xpos[j], v + 0.1, str(int(v)), ha='center', va='bottom', fontsize=FONT)
axR.set_xticks(xpos)
axR.set_xticklabels([MODEL_DISPLAY[m] for m in order], fontsize=FONT - 5,
                    rotation=20, ha='right')
axR.set_ylabel(f'# participants best by AIC (n={len(PIDS)})', fontsize=FONT - 2)
axR.set_title('B   Best-fitting model count', fontsize=FONT - 3, loc='left')
axR.set_ylim(0, len(PIDS))
axR.spines['top'].set_visible(False); axR.spines['right'].set_visible(False)

plt.tight_layout()
stem = 'model_discrimination_AIC_freeSigmaModel'
for ext, kw in [('png', dict(dpi=200)), ('pdf', dict(dpi=900, format='pdf')), ('svg', dict(format='svg'))]:
    plt.savefig(f'{stem}.{ext}', bbox_inches='tight', **kw)
plt.show()
print(f'Saved {stem}.png/.pdf/.svg')

# Paired Wilcoxon: is each alternative better than causal inference across participants?
from scipy.stats import wilcoxon
for m in [x for x in MAIN_MODELS if x != REF]:
    try:
        stat, p = wilcoxon(aic[m], aic[REF])
        print(f'{MODEL_DISPLAY[m]:14} vs Causal inference: Wilcoxon p={p:.3f} '
              f'(mean ΔAIC={dAIC[m].mean():+.2f})')
    except Exception as e:
        print(m, 'wilcoxon failed', e)""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open('plotConflictvsPSE_betterRefit.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote plotConflictvsPSE_betterRefit.ipynb with', len(cells), 'cells')
