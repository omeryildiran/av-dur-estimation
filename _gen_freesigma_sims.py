"""Regenerate simulated choices from the FREE-sigma model_fits (model_fits/P0x),
matching the existing sample->PF-fit pipeline but with free sensory sigma.

Output: simulated_data_freeSigma/<pid>/<pid>_<model>_LapseFree_sharedPrior_simulated.csv
"""
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import loadData
import monteCarloClass

# lowercase data code -> anonymized free-sigma fit id
PID_TO_ANON = {'as': 'P01', 'dt': 'P02', 'hh': 'P03', 'ip': 'P04', 'ln2': 'P07',
               'mh': 'P08', 'ml': 'P09', 'mt': 'P10', 'oy': 'P11', 'qs': 'P12', 'sx': 'P13'}
PIDS = list(PID_TO_ANON.keys())
MODELS = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']

VARIANT = 'LapseFree_sharedPrior'   # free lapse per group, shared causal prior
N_PASSES = 10                       # concat passes -> density comparable to originals
NSIMUL = 2000
OUT_ROOT = 'simulated_data_freeSigma'


def simulate_one(pid, model):
    anon = PID_TO_ANON[pid]
    fit_path = f'model_fits/{anon}/{anon}_{model}_{VARIANT}_fit.json'
    if not os.path.exists(fit_path):
        return None, f'missing fit {fit_path}'
    params = json.load(open(fit_path))['fittedParams']

    with np.errstate(divide='ignore', invalid='ignore'):
        data, _ = loadData.loadData(f'{pid}_all.csv', verbose=False)
    mc = monteCarloClass.OmerMonteCarlo(data)
    mc.modelName = model
    mc.sharedLambda = False   # LapseFree -> 3 lapse params
    mc.freeP_c = False        # sharedPrior
    mc.nSimul = NSIMUL
    mc.modelFit = params

    passes = []
    for _ in range(N_PASSES):
        passes.append(mc.simulateMonteCarloData(params, data))
    sim = pd.concat(passes, ignore_index=True)

    out_dir = os.path.join(OUT_ROOT, pid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{pid}_{model}_{VARIANT}_simulated.csv')
    sim.to_csv(out_path, index=False)
    return out_path, f'{len(sim)} rows | params[{len(params)}] p_c~{params[3]:.3f}'


if __name__ == '__main__':
    t0 = time.time()
    for pid in PIDS:
        for model in MODELS:
            t = time.time()
            path, info = simulate_one(pid, model)
            print(f'[{time.time()-t0:6.1f}s] {pid:4} {model:18} -> {info} ({time.time()-t:.1f}s)',
                  flush=True)
    print(f'DONE in {time.time()-t0:.1f}s -> {OUT_ROOT}/', flush=True)
