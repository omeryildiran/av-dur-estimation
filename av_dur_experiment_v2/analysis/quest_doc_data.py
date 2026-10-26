"""Collect the numbers the QUEST+ documentation page shows (from quest_simulation_results/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "quest_simulation_results"


def equivalent_trials(s, scenario, col, method="quest"):
    """QUEST+ trials needed to match the staircases' error at 154 trials (log-linear interpolation)."""
    st = s[(s.scenario == scenario) & (s.method == "staircase") & (s.budget == 154)][col].iloc[0]
    q = s[(s.scenario == scenario) & (s.method == method)].sort_values("budget")
    b, v = q.budget.to_numpy(float), q[col].to_numpy(float)
    if v[0] <= st:
        return float(b[0]), True, st
    for i in range(1, len(b)):
        if v[i] <= st:
            f = (np.log(v[i - 1]) - np.log(st)) / (np.log(v[i - 1]) - np.log(v[i]))
            return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1])))), False, st
    return float("inf"), False, st


def build():
    s = pd.concat([pd.read_csv(R / "summary.csv"), pd.read_csv(R / "summary_mu.csv")])
    raw = pd.concat([pd.read_csv(R / "raw.csv"), pd.read_csv(R / "raw_mu.csv")])
    curves = {}
    for sc, g in s.groupby("scenario"):
        curves[sc] = {m: {c: gg.sort_values("budget")[c].round(4).tolist()
                          for c in ("budget", "pse_rmse_ms", "pse_mae_ms", "sigma_rmse_ln", "pse_bias_ms", "sigma_median_ratio",
                                    "mu_rmse_lapse_known", "sigma_rmse_ln_lapse_known", "share_near_pse", "sigma_fail")}
                      for m, gg in g.groupby("method")}
    equiv = []
    for sc in sorted(s.scenario.unique()):
        row = dict(scenario=sc)
        for method in ("quest", "quest_mu"):
            for col, key in (("pse_mae_ms", "pse"), ("sigma_rmse_ln", "sigma")):
                n, below, ref = equivalent_trials(s, sc, col, method)
                row[f"{method}_{key}"] = None if not np.isfinite(n) else round(n)
                row[f"{method}_{key}_below"] = below
                row[f"{key}_ref"] = round(ref, 3)
        equiv.append(row)
    # pooled over the main grid (all sigma and lapse levels)
    grid = raw[raw.scenario.str.startswith("grid")]
    pooled = {}
    for m, g in grid.groupby("method"):
        e = g.assign(e_ms=500 * (np.exp(g.mu_hat) - np.exp(g.true_mu)), e_ls=np.log(g.sigma_hat / g.true_sigma))
        pooled[m] = e.groupby("budget").agg(pse=("e_ms", lambda v: float(np.median(np.abs(v)))),
                                              pse_rmse=("e_ms", lambda v: float(np.sqrt(np.mean(v ** 2)))),
                                              sig=("e_ls", lambda v: float(np.sqrt(np.mean(v ** 2))))).round(4).reset_index().to_dict("list")
    return dict(curves=curves, equiv=equiv, pooled=pooled,
                example=json.loads((R / "example_run.json").read_text()),
                n_obs=int(raw[["scenario", "rep"]].drop_duplicates().shape[0]))


if __name__ == "__main__":
    d = build()
    (R / "doc_data.json").write_text(json.dumps(d))
    for e in d["equiv"]:
        print(e)
    print(json.dumps(d["pooled"]))
