"""Reproduce PSE-vs-conflict statistics used in the manuscript.

This script uses the same free-mu/sigma/lambda psychometric fits as
plotConflictvsPSE_clean.ipynb. The manuscript inference is based on
participant-level slopes, not on correlations over group-averaged points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, t, ttest_1samp, ttest_rel, wilcoxon

from free_psychometric_refits import aggregate_free_fits, collect_free_fits


PIDS = ["as", "dt", "hh", "ip", "ln2", "mh", "ml", "mt", "oy", "qs", "sx"]
LOW_NOISE = 0.1
HIGH_NOISE = 1.2


def _fmt_p(p_value: float) -> str:
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def load_pse_fits() -> pd.DataFrame:
    data_pp, _model_pp, missing = collect_free_fits(
        PIDS,
        model_names=[],
        pf_mode="lognormal",
        force=False,
    )
    if missing:
        raise RuntimeError(f"Unexpected missing fits: {missing}")
    return data_pp


def aggregate_correlations(data_pp: pd.DataFrame) -> pd.DataFrame:
    """Correlations over group-mean PSE points; useful descriptively only."""
    data_agg = aggregate_free_fits(data_pp, value_col="mu_shift_s")
    rows = []
    for noise in [LOW_NOISE, HIGH_NOISE]:
        sub = data_agg[np.isclose(data_agg["audioNoise"], noise)].sort_values("conflict")
        x_ms = sub["conflict"].to_numpy(float) * 1000
        y_ms = sub["mean"].to_numpy(float) * 1000
        slope, intercept, r_value, p_value, stderr = linregress(x_ms, y_ms)
        pearson_r, pearson_p = pearsonr(x_ms, y_ms)
        rows.append(
            dict(
                audioNoise=noise,
                slope_ms_per_ms=slope,
                intercept_ms=intercept,
                linregress_r=r_value,
                linregress_p=p_value,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                slope_stderr=stderr,
                n_points=len(sub),
            )
        )
    return pd.DataFrame(rows)


def participant_slopes(data_pp: pd.DataFrame) -> pd.DataFrame:
    """Fit PSE shift ~ signed conflict separately for each participant/noise."""
    rows = []
    for (pid, noise), sub in data_pp.groupby(["pid", "audioNoise"]):
        sub = sub.sort_values("conflict")
        x_ms = sub["conflict"].to_numpy(float) * 1000
        y_ms = sub["mu_shift_s"].to_numpy(float) * 1000
        slope, intercept, r_value, p_value, stderr = linregress(x_ms, y_ms)
        rows.append(
            dict(
                pid=pid,
                audioNoise=float(noise),
                slope_ms_per_ms=slope,
                intercept_ms=intercept,
                r=r_value,
                p=p_value,
                slope_stderr=stderr,
                n_points=len(sub),
            )
        )
    return pd.DataFrame(rows)


def slope_tests(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for noise in [LOW_NOISE, HIGH_NOISE]:
        vals = slopes[np.isclose(slopes["audioNoise"], noise)]["slope_ms_per_ms"].to_numpy()
        t_stat, p_value = ttest_1samp(vals, 0.0)
        ci_lo, ci_hi = t.interval(
            0.95,
            df=len(vals) - 1,
            loc=float(np.mean(vals)),
            scale=float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
        )
        w_stat, w_p = wilcoxon(vals, alternative="two-sided")
        rows.append(
            dict(
                comparison="slope > 0",
                audioNoise=noise,
                mean=float(np.mean(vals)),
                sem=float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                ci_lo=float(ci_lo),
                ci_hi=float(ci_hi),
                t=float(t_stat),
                df=len(vals) - 1,
                p=float(p_value),
                wilcoxon_w=float(w_stat),
                wilcoxon_p=float(w_p),
                n=len(vals),
            )
        )

    wide = slopes.pivot(index="pid", columns="audioNoise", values="slope_ms_per_ms")
    diff = wide[HIGH_NOISE].to_numpy() - wide[LOW_NOISE].to_numpy()
    t_stat, p_value = ttest_rel(wide[HIGH_NOISE], wide[LOW_NOISE])
    ci_lo, ci_hi = t.interval(
        0.95,
        df=len(diff) - 1,
        loc=float(np.mean(diff)),
        scale=float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
    )
    w_stat, w_p = wilcoxon(diff, alternative="two-sided")
    rows.append(
        dict(
            comparison="high - low slope",
            audioNoise=np.nan,
            mean=float(np.mean(diff)),
            sem=float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
            ci_lo=float(ci_lo),
            ci_hi=float(ci_hi),
            t=float(t_stat),
            df=len(diff) - 1,
            p=float(p_value),
            wilcoxon_w=float(w_stat),
            wilcoxon_p=float(w_p),
            n=len(diff),
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    data_pp = load_pse_fits()
    agg = aggregate_correlations(data_pp)
    slopes = participant_slopes(data_pp)
    tests = slope_tests(slopes)

    print("\nAggregate correlations over group-mean points (descriptive):")
    print(
        agg[
            [
                "audioNoise",
                "slope_ms_per_ms",
                "pearson_r",
                "pearson_p",
                "n_points",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nParticipant-level slopes:")
    print(
        slopes[["pid", "audioNoise", "slope_ms_per_ms", "r", "p"]]
        .sort_values(["audioNoise", "pid"])
        .to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nParticipant-level slope tests:")
    print(
        tests[
            [
                "comparison",
                "audioNoise",
                "mean",
                "sem",
                "ci_lo",
                "ci_hi",
                "t",
                "df",
                "p",
                "wilcoxon_p",
                "n",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    low = tests[np.isclose(tests["audioNoise"], LOW_NOISE, equal_nan=False)].iloc[0]
    high = tests[np.isclose(tests["audioNoise"], HIGH_NOISE, equal_nan=False)].iloc[0]
    diff = tests[tests["comparison"].eq("high - low slope")].iloc[0]
    print("\nManuscript-ready values:")
    print(
        f"Low noise: mean slope = {low['mean']:.3f} ms/ms, SEM = {low['sem']:.3f}, "
        f"t({int(low['df'])}) = {low['t']:.2f}, {_fmt_p(low['p'])}"
    )
    print(
        f"High noise: mean slope = {high['mean']:.3f} ms/ms, SEM = {high['sem']:.3f}, "
        f"t({int(high['df'])}) = {high['t']:.2f}, {_fmt_p(high['p'])}"
    )
    print(
        f"High-low difference: mean = {diff['mean']:.3f} ms/ms, SEM = {diff['sem']:.3f}, "
        f"paired t({int(diff['df'])}) = {diff['t']:.2f}, {_fmt_p(diff['p'])}"
    )


if __name__ == "__main__":
    main()
