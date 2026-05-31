from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import loadData


STANDARD_S = 0.5

DEFAULT_MAIN_MODELS = ["lognorm", "fusionOnlyLogNorm", "switchingFree"]
DEFAULT_ALL_MODELS = [
    "fusionOnlyLogNorm",
    "fusionOnly",
    "gaussian",
    "logLinearMismatch",
    "lognorm",
    "probabilityMatchingLogNorm",
    "probabilityMatching",
    "selection",
    "switchingFree",
    "switching",
    "switchingWithConflict",
]
DEFAULT_SIM_VARIANTS = [
    "LapseFree_sharedPrior",
    "LapseFix_sharedPrior",
    "LapseFree_contextualPrior",
]


def pf_input_from_durations(test_durs, standard_durs, pf_mode="lognormal"):
    test_durs = np.asarray(test_durs, dtype=float)
    standard_durs = np.asarray(standard_durs, dtype=float)
    if pf_mode == "lognormal":
        valid = (test_durs > 0) & (standard_durs > 0)
        out = np.full(test_durs.shape, np.nan, dtype=float)
        out[valid] = np.log(test_durs[valid] / standard_durs[valid])
        return out
    if pf_mode == "normal":
        return test_durs - standard_durs
    raise ValueError("pf_mode must be 'lognormal' or 'normal'")


def psychometric_pf(pf_x, mu, sigma, lambda_):
    return lambda_ / 2.0 + (1.0 - lambda_) * norm.cdf((pf_x - mu) / sigma)


def mu_to_shift_s(mu, standard_s=STANDARD_S, pf_mode="lognormal"):
    if pf_mode == "lognormal":
        return standard_s * (np.exp(mu) - 1.0)
    return mu


def sigma_to_plot_units(sigma, standard_s=STANDARD_S, pf_mode="lognormal"):
    if pf_mode == "lognormal":
        return sigma
    return sigma / standard_s


def sigma_plot_label(pf_mode="lognormal"):
    if pf_mode == "lognormal":
        return "PF sigma (log units)"
    return "PF sigma / standard duration"


def _bounds_for_mode(pf_mode):
    if pf_mode == "lognormal":
        return [(-1.0, 1.0), (0.01, 3.0), (0.0, 0.25)]
    return [(-0.6, 0.6), (1e-3, 1.5), (0.0, 0.4)]


def _clip_x0(x0, bounds):
    return np.array([
        min(max(v, lo + 1e-6), hi - 1e-6)
        for v, (lo, hi) in zip(x0, bounds)
    ], dtype=float)


def _binom_nll(params, pf_x, n_chose, n_total):
    mu, sigma, lambda_ = params
    if sigma <= 0 or lambda_ < 0 or lambda_ >= 0.5:
        return 1e10
    p = psychometric_pf(pf_x, mu, sigma, lambda_)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -float(np.sum(n_chose * np.log(p) + (n_total - n_chose) * np.log1p(-p)))


def fit_pf_counts(pf_x, n_chose, n_total, pf_mode="lognormal"):
    pf_x = np.asarray(pf_x, dtype=float)
    n_chose = np.asarray(n_chose, dtype=float)
    n_total = np.asarray(n_total, dtype=float)
    mask = np.isfinite(pf_x) & np.isfinite(n_chose) & np.isfinite(n_total) & (n_total > 0)
    pf_x, n_chose, n_total = pf_x[mask], n_chose[mask], n_total[mask]

    if len(np.unique(pf_x)) < 3:
        return dict(mu=np.nan, sigma=np.nan, lambda_=np.nan, nll=np.nan, success=False)

    prop = np.clip(n_chose / n_total, 0, 1)
    mu_guess = float(pf_x[np.argmin(np.abs(prop - 0.5))])
    sigma_guess = float(max(np.nanstd(pf_x), 0.05 if pf_mode == "lognormal" else 0.01))
    bounds = _bounds_for_mode(pf_mode)
    starts = [
        [mu_guess, sigma_guess, 0.02],
        [0.0, 0.10 if pf_mode == "lognormal" else 0.05, 0.02],
        [0.05, 0.20 if pf_mode == "lognormal" else 0.10, 0.05],
        [-0.05, 0.30 if pf_mode == "lognormal" else 0.20, 0.02],
    ]

    best = None
    for x0 in starts:
        try:
            res = minimize(
                _binom_nll,
                _clip_x0(x0, bounds),
                args=(pf_x, n_chose, n_total),
                bounds=bounds,
                method="L-BFGS-B",
            )
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return dict(mu=np.nan, sigma=np.nan, lambda_=np.nan, nll=np.nan, success=False)
    return dict(
        mu=float(best.x[0]),
        sigma=float(best.x[1]),
        lambda_=float(best.x[2]),
        nll=float(best.fun),
        success=bool(best.success),
    )


def _condition_counts(trials_df, pf_mode="lognormal"):
    df = trials_df.copy()
    if "chose_test" not in df.columns:
        if "responses" not in df.columns:
            raise ValueError("Data must contain chose_test or responses.")
        df["chose_test"] = (df["responses"].astype(float) == 2).astype(float)

    pf_x = pf_input_from_durations(df["testDurS"], df["standardDur"], pf_mode=pf_mode)
    chose = df["chose_test"].astype(float).to_numpy()
    mask = np.isfinite(pf_x) & np.isfinite(chose)
    pf_x, chose = pf_x[mask], chose[mask]

    unique_x, inv = np.unique(np.round(pf_x, 12), return_inverse=True)
    n_total = np.bincount(inv).astype(float)
    n_chose = np.bincount(inv, weights=chose).astype(float)
    return unique_x, n_chose, n_total


def fit_free_psychometrics(data, pf_mode="lognormal"):
    rows = []
    for noise in sorted(data["audNoise"].dropna().unique()):
        for conflict in sorted(data["conflictDur"].dropna().unique()):
            sub = data[
                np.isclose(data["audNoise"].astype(float), float(noise))
                & np.isclose(data["conflictDur"].astype(float), float(conflict))
            ]
            if sub.empty:
                continue
            pf_x, n_chose, n_total = _condition_counts(sub, pf_mode=pf_mode)
            fit = fit_pf_counts(pf_x, n_chose, n_total, pf_mode=pf_mode)
            rows.append(
                dict(
                    audioNoise=float(noise),
                    conflict=float(conflict),
                    conflict_ms=int(round(float(conflict) * 1000)),
                    n_trials=int(n_total.sum()),
                    mu=fit["mu"],
                    sigma=fit["sigma"],
                    lambda_=fit["lambda_"],
                    nll=fit["nll"],
                    success=fit["success"],
                    mu_shift_s=mu_to_shift_s(fit["mu"], pf_mode=pf_mode),
                    mu_shift_ms=mu_to_shift_s(fit["mu"], pf_mode=pf_mode) * 1000,
                    sigma_plot=sigma_to_plot_units(fit["sigma"], pf_mode=pf_mode),
                )
            )
    return pd.DataFrame(rows)


def bootstrap_free_psychometrics(data, n_boot=200, pf_mode="lognormal", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for noise in sorted(data["audNoise"].dropna().unique()):
        for conflict in sorted(data["conflictDur"].dropna().unique()):
            sub = data[
                np.isclose(data["audNoise"].astype(float), float(noise))
                & np.isclose(data["conflictDur"].astype(float), float(conflict))
            ].reset_index(drop=True)
            if len(sub) < 3:
                continue
            boot = {k: [] for k in ("mu", "sigma", "lambda_", "mu_shift_s", "sigma_plot")}
            for _ in range(n_boot):
                idx = rng.integers(0, len(sub), size=len(sub))
                fit_df = fit_free_psychometrics(sub.iloc[idx], pf_mode=pf_mode)
                if fit_df.empty:
                    continue
                row = fit_df.iloc[0]
                for key in boot:
                    boot[key].append(float(row[key]))
            out = dict(audioNoise=float(noise), conflict=float(conflict), n_boot=len(boot["mu"]))
            for key, vals in boot.items():
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    out[f"{key}_lo"] = np.nan
                    out[f"{key}_hi"] = np.nan
                    out[f"{key}_med"] = np.nan
                else:
                    out[f"{key}_lo"], out[f"{key}_hi"] = np.percentile(vals, [2.5, 97.5])
                    out[f"{key}_med"] = np.median(vals)
            rows.append(out)
    return pd.DataFrame(rows)


def save_fit_table(path, fit_df, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "fitParamsByCondition": fit_df.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_fit_table(path):
    payload = json.loads(Path(path).read_text())
    return pd.DataFrame(payload["fitParamsByCondition"]), payload.get("metadata", {})


def load_real_data(pid):
    data, data_name = loadData.loadData(f"{pid}_all.csv", verbose=False)
    return data, data_name


def simulated_csv_path(pid, model_name, variants=None, sim_dir="simulated_data"):
    variants = variants or DEFAULT_SIM_VARIANTS
    base_dir = Path(sim_dir) / pid
    for variant in variants:
        path = base_dir / f"{pid}_{model_name}_{variant}_simulated.csv"
        if path.exists():
            return path
    return None


def fit_real_pid(pid, out_dir="psychometric_fits_freeMuSigmaLambda_real",
                 pf_mode="lognormal", force=False):
    out_path = Path(out_dir) / pid / f"{pid}_psychometricFits.json"
    if out_path.exists() and not force:
        fit_df, _ = load_fit_table(out_path)
        return fit_df
    data, data_name = load_real_data(pid)
    fit_df = fit_free_psychometrics(data, pf_mode=pf_mode)
    fit_df.insert(0, "pid", pid)
    save_fit_table(
        out_path,
        fit_df,
        metadata=dict(participantID=pid, dataName=data_name, pf_mode=pf_mode,
                      parameterization="free_mu_sigma_lambda_per_condition"),
    )
    return fit_df


def fit_simulated_pid(pid, model_name,
                      out_dir="psychometric_fits_freeMuSigmaLambda_simulated",
                      pf_mode="lognormal", force=False, variants=None):
    out_path = Path(out_dir) / pid / f"{pid}_{model_name}_psychometricFits.json"
    if out_path.exists() and not force:
        fit_df, _ = load_fit_table(out_path)
        return fit_df

    sim_path = simulated_csv_path(pid, model_name, variants=variants)
    if sim_path is None:
        return pd.DataFrame()
    sim_data = pd.read_csv(sim_path)
    fit_df = fit_free_psychometrics(sim_data, pf_mode=pf_mode)
    fit_df.insert(0, "pid", pid)
    fit_df.insert(1, "model", model_name)
    save_fit_table(
        out_path,
        fit_df,
        metadata=dict(participantID=pid, modelType=model_name, simulatedData=str(sim_path),
                      pf_mode=pf_mode,
                      parameterization="free_mu_sigma_lambda_per_condition"),
    )
    return fit_df


def collect_free_fits(pids, model_names=DEFAULT_MAIN_MODELS, pf_mode="lognormal",
                      force=False, variants=None):
    data_frames = []
    model_frames = []
    missing = []
    for pid in pids:
        data_frames.append(fit_real_pid(pid, pf_mode=pf_mode, force=force))
        for model_name in model_names:
            fit_df = fit_simulated_pid(pid, model_name, pf_mode=pf_mode,
                                       force=force, variants=variants)
            if fit_df.empty:
                missing.append((pid, model_name))
            else:
                model_frames.append(fit_df)
    data_pp = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    model_pp = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    return data_pp, model_pp, missing


def aggregate_free_fits(fit_df, value_col="mu_shift_s"):
    group_cols = ["audioNoise", "conflict"]
    if "model" in fit_df.columns:
        group_cols = ["model"] + group_cols
    agg = (
        fit_df.groupby(group_cols)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg


def value_label(value_col, pf_mode="lognormal"):
    labels = {
        "mu_shift_s": "PSE shift (ms)",
        "mu": "PF mu (log units)" if pf_mode == "lognormal" else "PF mu (s)",
        "sigma_plot": sigma_plot_label(pf_mode),
        "sigma": "PF sigma",
        "lambda_": "Lapse lambda",
    }
    return labels.get(value_col, value_col)
