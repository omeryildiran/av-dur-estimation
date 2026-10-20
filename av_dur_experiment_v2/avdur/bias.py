"""Cross-modal bias correction, fitted automatically from the participant's cross-modal data.

The cross-modal task compares an auditory test with a 500 ms visual standard. If the
auditory duration that matches the visual standard (the PSE) is A*, vision is perceived
as (0.5 - A*) s shorter than audition, so every visual flash in the main experiment is
lengthened by correction = 0.5 - A* (the v1 `VisualPSE`; negative values shorten it).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import experiment_settings as S
from avdur import psychometrics as pm
from avdur import storage


def load_crossmodal(pid, root=None):
    df = storage.load_trials("crossmodal", pid, root)
    return df[df["responses"].isin([1, 2])] if len(df) else df


def estimate(df, method=None, rng=None, n_boot=None):
    """Fit both noise levels jointly and convert each PSE into a visual correction (s)."""
    cfg = S.BIAS
    method = method or cfg["method"]
    rng = rng or np.random.default_rng(0)
    n_boot = cfg["n_bootstrap"] if n_boot is None else n_boot
    scale = "lognormal" if method == "lognormal" else "linear"
    lapse_max = cfg["lapse_max"] if method == "lognormal" else 0.2      # v1 notebook used 0.2

    result = pm.fit(df, "audNoise", scale, lapse_max=lapse_max)
    boot = pm.bootstrap(df, "audNoise", scale, n_boot, rng, lapse_max=lapse_max) if n_boot else None
    per_noise = {}
    for i, row in result["params"].iterrows():
        a_star = S.STANDARD_DUR_S * pm.pse_ratio(row.mu, scale)
        entry = dict(correction_s=round(S.STANDARD_DUR_S - a_star, 4), pse_auditory_s=round(a_star, 4),
                     mu=row.mu, sigma=row.sigma, lapse=row.lapse, n_trials=row.n)
        if boot is not None:
            corr = S.STANDARD_DUR_S - S.STANDARD_DUR_S * np.array([pm.pse_ratio(m, scale) for m in boot[:, i]])
            entry["ci95_s"] = [round(float(np.percentile(corr, 2.5)), 4), round(float(np.percentile(corr, 97.5)), 4)]
        per_noise[str(float(row.condition))] = entry
    return dict(method=method, per_noise=per_noise, at_bound=result["at_bound"],
                converged=result["success"], n_trials=int(len(df)))


def quality_warnings(est):
    cfg, warnings = S.BIAS, []
    if est["n_trials"] < cfg["min_trials"]:
        warnings.append(f"only {est['n_trials']} cross-modal trials (expected at least {cfg['min_trials']})")
    if not est["converged"]:
        warnings.append("the fit did not converge")
    if est["at_bound"]:
        warnings.append(f"parameters at a fitting bound: {', '.join(est['at_bound'])}")
    for noise, e in est["per_noise"].items():
        if abs(e["correction_s"]) > cfg["max_abs_correction_s"]:
            warnings.append(f"noise {noise}: correction {e['correction_s']:+.3f} s exceeds "
                            f"{cfg['max_abs_correction_s']} s (v1 excluded such participants)")
        if "ci95_s" in e and (e["ci95_s"][1] - e["ci95_s"][0]) / 2 > cfg["max_ci_halfwidth_s"]:
            warnings.append(f"noise {noise}: correction is uncertain (95% CI {e['ci95_s'][0]:+.3f} to "
                            f"{e['ci95_s'][1]:+.3f} s)")
    if set(est["per_noise"]) != {str(float(n)) for n in S.NOISE_LEVELS}:
        warnings.append(f"noise levels in the data: {sorted(est['per_noise'])}")
    return warnings


def estimate_for_participant(pid, root=None, n_boot=None):
    """Main-method estimate, the v1-method values for comparison, warnings and source files."""
    df = load_crossmodal(pid, root)
    if df.empty:
        raise FileNotFoundError(f"No completed cross-modal session for {pid}. Run the cross-modal task first.")
    est = estimate(df, n_boot=n_boot)
    est["warnings"] = quality_warnings(est)
    v1 = estimate(df, method="linear_v1", n_boot=0)
    est["comparison_linear_v1"] = {k: v["correction_s"] for k, v in v1["per_noise"].items()}
    est["source_sessions"] = sorted(df["session_file"].unique())
    return est


def save(pid, est, confirmed_by, root=None, solid_root=None):
    """Save a confirmed correction; for real participants also copy it to data_solid/."""
    folder = Path(root or storage.DATA_ROOT) / pid / "bias"
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(est, participant=pid, confirmed_by=confirmed_by,
                   confirmed_at=dt.datetime.now().isoformat(timespec="seconds"))
    path = folder / f"{pid}_bias_{storage.timestamp()}.json"
    storage.write_json(path, payload)
    if storage.is_real_root(root) and solid_root is not False:
        storage.copy_to_solid([path], pid, "bias", solid_root)
    return path


def fixed(values, note):
    """A bias payload with given corrections instead of a fit (autotests only)."""
    return dict(method="fixed", n_trials=0, source_sessions=[], converged=True, at_bound=[],
                per_noise={str(float(n)): dict(correction_s=float(v)) for n, v in values.items()},
                warnings=[note])


def correction_for(bias, noise):
    """Correction (s) for one noise level from a saved bias file."""
    return float(bias["per_noise"][str(float(noise))]["correction_s"])


def summary_text(est):
    lines = [f"Method: {est['method']}   ({est['n_trials']} trials; sessions: {', '.join(est['source_sessions'])})"]
    for noise, e in sorted(est["per_noise"].items(), key=lambda kv: float(kv[0])):
        ci = f"  95% CI {e['ci95_s'][0]:+.3f} to {e['ci95_s'][1]:+.3f}" if "ci95_s" in e else ""
        v1 = est.get("comparison_linear_v1", {}).get(noise)
        lines.append(f"Noise {noise}: visual correction {e['correction_s']:+.3f} s{ci}"
                     + (f"   (v1 method: {v1:+.3f} s)" if v1 is not None else ""))
    lines += [f"WARNING: {w}" for w in est.get("warnings", [])] or ["No quality warnings."]
    return "\n".join(lines)
