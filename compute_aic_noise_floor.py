"""Monte-Carlo noise floor of per-participant AIC (seed variability).

Re-evaluates each saved fit's log-likelihood at its FITTED parameters under many
RNG seeds (same nSimul as fitting), measuring the AIC SD induced purely by the
Monte-Carlo likelihood. Validates each reconstruction against the saved logLik.
Writes aic_noise_floor_seeds.json (one row per participant x model x fit_source).
"""
import os, json, glob, io, contextlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SEEDS  = list(range(20))
NSIMUL = 2000
CACHE  = 'aic_noise_floor_seeds.json'

TARGET_MODELS = {
    'lognorm_sharedPrior': 'Causal inference',
    'fusionOnlyLogNorm_sharedPrior': 'Forced fusion',
    'switchingFree_sharedPrior': 'Probabilistic cue switching',
}
FIT_SOURCE_CONFIGS = {
    'free_sigma':     {'label': 'Free sensory noise',  'glob': 'model_fits/**/*.json',
                       'excl_sub': ('all','lnd1','ln1','loglinear'), 'excl_tag': ('LapseFree',)},
    'unimodal_sigma': {'label': 'Fixed sensory noise', 'glob': 'unimodalSigma_modelFits/**/*.json',
                       'excl_sub': ('all','lnd1','ln1','loglinear'), 'excl_tag': ()},
}

def load_fits(source):
    cfg = FIT_SOURCE_CONFIGS[source]
    rows = []
    for fp in sorted(glob.glob(cfg['glob'], recursive=True)):
        fn = os.path.basename(fp); fnl = fn.lower()
        if any(s in fnl for s in cfg['excl_sub']):       continue
        if cfg['excl_tag'] and any(t in fn for t in cfg['excl_tag']): continue
        parts = fn.replace('.json','').split('_')
        raw_tag = f"{parts[1]}_{parts[3]}"
        if raw_tag not in TARGET_MODELS: continue
        d = json.load(open(fp))
        rows.append({'participantID': str(d.get('participantID', parts[0])).lower(),
                     'modelType': TARGET_MODELS[raw_tag], 'fit_file': fp,
                     'fittedParams': d['fittedParams'], 'logLikelihood': d['logLikelihood'],
                     'modelName': parts[1], 'sharedLambda': ('LapseFix' in fn),
                     'freeP_c': ('contextualPrior' in fn)})
    return rows

def resolve_csv(pid):
    for f in glob.glob('data/*_all.csv'):
        if os.path.basename(f).lower() == f'{pid.lower()}_all.csv':
            return os.path.basename(f)
    return None

def get_noise_floor(seeds=SEEDS, nsimul=NSIMUL, cache=CACHE, recompute=False, verbose=True):
    """Return per-(participant x model x fit_source) MC noise-floor DataFrame.

    Loads `cache` if present (unless recompute=True); else computes and writes it.
    AIC_sd = SD of (2k - 2*logL) across `seeds` RNG seeds, evaluated at the saved
    fitted params with nSimul=`nsimul`. Each reconstruction is validated against the
    saved logLik (column `valid`).
    """
    import pandas as pd
    if (not recompute) and os.path.exists(cache):
        return pd.DataFrame(json.load(open(cache)))
    rows = _compute_rows(seeds, nsimul, verbose=verbose)
    json.dump(rows, open(cache, 'w'), indent=1)
    if verbose:
        print(f"\nwrote {cache} ({len(rows)} rows)")
    return pd.DataFrame(rows)


def _compute_rows(seeds, nsimul, verbose=True):
    import loadData, monteCarloClass
    mc_by_csv = {}
    out = []
    for source in FIT_SOURCE_CONFIGS:
        label = FIT_SOURCE_CONFIGS[source]['label']
        fits = load_fits(source)
        if verbose:
            print(f"\n[{source}] {len(fits)} fits")
        for r in fits:
            pid = r['participantID']; csv = resolve_csv(pid)
            if csv is None:
                print(f"  ! no csv for {pid}; skip"); continue
            if csv not in mc_by_csv:
                with contextlib.redirect_stdout(io.StringIO()):
                    data, _ = loadData.loadData(csv, verbose=False)
                    mc = monteCarloClass.OmerMonteCarlo(data); mc.nSimul = nsimul
                mc_by_csv[csv] = mc
            mc = mc_by_csv[csv]
            mc.modelName = r['modelName']; mc.sharedLambda = r['sharedLambda']; mc.freeP_c = r['freeP_c']
            params = np.asarray(r['fittedParams'], float); k = len(params)
            lls = []
            for s in seeds:
                np.random.seed(s)
                with contextlib.redirect_stdout(io.StringIO()):
                    lls.append(-mc.nLLMonteCarloCausal(params, mc.groupedData))
            lls = np.asarray(lls); aics = 2*k - 2*lls
            diff = float(lls.mean() - r['logLikelihood'])
            valid = bool(abs(diff) <= max(5.0, 4*lls.std(ddof=1)))
            out.append({'fit_source': label, 'participantID': pid, 'modelType': r['modelType'],
                        'k': k, 'saved_logLik': r['logLikelihood'],
                        'recomp_logLik_mean': float(lls.mean()), 'logLik_sd': float(lls.std(ddof=1)),
                        'AIC_sd': float(aics.std(ddof=1)), 'AIC_mean': float(aics.mean()),
                        'mean_minus_saved': diff, 'valid': valid})
            if verbose:
                print(f"  {pid:>4} {r['modelType']:<28} k={k} AIC_sd={aics.std(ddof=1):4.2f} "
                      f"diff={diff:+5.2f} {'ok' if valid else 'CHECK'}")
    return out


if __name__ == '__main__':
    get_noise_floor(recompute=True)
