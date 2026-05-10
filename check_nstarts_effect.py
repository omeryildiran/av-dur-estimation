#!/usr/bin/env python3
"""Tiny same-data check for the effect of optimizer restarts.

This is intentionally much smaller than the full model-recovery grid.  It
generates a few fixed simulated datasets, then fits the same candidate models
with different nStarts values so the log-likelihood differences are directly
comparable.
"""

import argparse
import builtins
import contextlib
import functools
import io
import json
import time
from pathlib import Path

import numpy as np

import monteCarloClass
import run_model_recovery_grid as grid
import run_param_recovery_favorable as favo

print = functools.partial(builtins.print, flush=True)


def _fit_once(sim_data, fit_model, n_simul, n_starts, seed):
    np.random.seed(seed)
    mc_fit = monteCarloClass.OmerMonteCarlo(sim_data)
    mc_fit.modelName = fit_model
    mc_fit.freeP_c = False
    mc_fit.sharedLambda = True
    mc_fit.nSimul = n_simul
    mc_fit.nStart = n_starts
    mc_fit.optimizationMethod = "bads"

    quiet = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        fitted_params = mc_fit.fitCausalInferenceMonteCarlo(mc_fit.groupedData)
        nll = mc_fit.nLLMonteCarloCausal(fitted_params, mc_fit.groupedData)

    ll = -float(nll)
    k = len(fitted_params)
    return {
        "logLikelihood": ll,
        "AIC": float(2 * k - 2 * ll),
        "fittedParams": fitted_params.tolist(),
        "seconds": time.time() - start,
        "boundary_clips": favo.count_boundary_clips(fitted_params, fit_model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_models", nargs="+",
                        default=["probabilityMatchingLogNorm", "selection"])
    parser.add_argument("--fit_models", nargs="+",
                        default=["probabilityMatchingLogNorm", "selection"])
    parser.add_argument("--sigma_level", default="c",
                        choices=sorted(grid.SIGMA_LEVELS))
    parser.add_argument("--conflict_max", type=float, default=0.45)
    parser.add_argument("--n_cases", type=int, default=1)
    parser.add_argument("--nSimul", type=int, default=300)
    parser.add_argument("--nStarts", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--n_conflict_steps", type=int, default=5)
    parser.add_argument("--n_trials_per_cell", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--out", default="nstart_effect_check_results.json")
    args = parser.parse_args()

    slevel = grid.SIGMA_LEVELS[args.sigma_level]
    ranges = grid.build_ranges_for_cell(
        args.gen_models,
        slevel["sigma_a"],
        slevel["sigma_v"],
    )
    template = favo.build_synthetic_template(
        conflict_max=args.conflict_max,
        n_conflict_steps=args.n_conflict_steps,
        n_trials_per_cell=args.n_trials_per_cell,
    )

    rng = np.random.default_rng(args.seed)
    results = {
        "settings": vars(args),
        "template_rows": int(len(template)),
        "results": [],
    }

    print("nStarts same-data check")
    print(f"  gen models : {args.gen_models}")
    print(f"  fit models : {args.fit_models}")
    print(f"  n cases    : {args.n_cases} per generating model")
    print(f"  nSimul     : {args.nSimul}")
    print(f"  nStarts    : {args.nStarts}")
    print(f"  template   : {len(template)} rows")
    print()

    for gen_model in args.gen_models:
        for case_idx in range(args.n_cases):
            sampled_unique, sampled_full = favo.sample_params(gen_model, ranges, rng)

            np.random.seed(args.seed + 1000 * case_idx + len(results["results"]))
            mc_gen = monteCarloClass.OmerMonteCarlo(template)
            mc_gen.modelName = gen_model
            mc_gen.freeP_c = False
            mc_gen.sharedLambda = True
            mc_gen.nSimul = args.nSimul
            mc_gen.nStart = 1
            mc_gen.optimizationMethod = "bads"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sim_data = mc_gen.simulateMonteCarloData(sampled_full, template)

            case = {
                "generating_model": gen_model,
                "case": case_idx,
                "sampled_unique": sampled_unique.tolist(),
                "fits": {},
            }

            print(f"[{gen_model} case {case_idx}]")
            for fit_model in args.fit_models:
                case["fits"][fit_model] = {}
                base_seed = args.seed + 100_000 + 1000 * case_idx + 17 * len(results["results"])
                for n_starts in args.nStarts:
                    fit = _fit_once(sim_data, fit_model, args.nSimul, n_starts, base_seed)
                    case["fits"][fit_model][str(n_starts)] = fit
                    print(
                        f"  fit {fit_model:<28} nStarts={n_starts:<2} "
                        f"LL={fit['logLikelihood']:.3f} "
                        f"time={fit['seconds']:.1f}s"
                    )

                low, high = min(args.nStarts), max(args.nStarts)
                ll_low = case["fits"][fit_model][str(low)]["logLikelihood"]
                ll_high = case["fits"][fit_model][str(high)]["logLikelihood"]
                print(f"    delta LL ({high}-{low}) = {ll_high - ll_low:+.3f}")

            results["results"].append(case)
            print()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
