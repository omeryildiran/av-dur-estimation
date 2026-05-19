"""
Run model fitting separately for each audNoise level.

Usage:
    python runFitting_byNoise.py <files> <model> <nSimul> <optim> <nStarts> <freeP_c> <integrationMethod> <nCores> <noiseLevel>

    noiseLevel ∈ {low, high, both}
        low  → keep only trials with audNoise == 0.1
        high → keep only trials with audNoise == 1.2
        both → run both levels (each file expands to two parallel jobs)

Examples:
    python runFitting_byNoise.py "as_all.csv,oy_all.csv,dt_all.csv,hh_all.csv,ip_all.csv,ln2_all.csv,mh_all.csv,ml_all.csv,mt_all.csv,qs_all.csv,sx_all.csv" "lognorm" 1000 "bads" 10 False "analytical" 11 both
    python runFitting_byNoise.py "mt_all.csv" "lognorm" 200 "bads" 2 False "analytical" 1 low
"""

NOISE_LEVEL_MAP = {
    "low": 0.1,
    "high": 1.2,
}


def filter_by_noise(data, noiseLabel):
    """Return rows matching the requested audNoise level."""
    target = NOISE_LEVEL_MAP[noiseLabel]
    filtered = data[data["audNoise"] == target].copy()
    if filtered.empty:
        raise ValueError(
            f"No trials with audNoise == {target} (label='{noiseLabel}'). "
            f"Unique audNoise values: {sorted(data['audNoise'].unique())}"
        )
    return filtered


def process_single_file(args):
    """
    Fit one (file, noiseLabel) job. Runs in a worker process.
    """
    dataFile, modelName, nSimul, optimMethod, nStarts, freeP_c, integrationMethod, noiseLabel = args
    print(
        f"Starting {dataFile} [{noiseLabel}] | model={modelName} "
        f"nSimul={nSimul} optim={optimMethod} nStarts={nStarts} freeP_c={freeP_c}"
    )

    import time
    import loadData
    import monteCarloClass
    import fitSaver

    try:
        print("\n" + 60 * "=" + "\n")
        print(f"=== Processing {dataFile} [noise={noiseLabel}] ===")

        data, dataName = loadData.loadData(dataFile)
        n_before = len(data)
        data = filter_by_noise(data, noiseLabel)
        print(f"Filtered audNoise={NOISE_LEVEL_MAP[noiseLabel]} : {n_before} → {len(data)} trials")

        mc_fitter = monteCarloClass.OmerMonteCarlo(data)
        mc_fitter.nSimul = nSimul
        mc_fitter.optimizationMethod = optimMethod
        mc_fitter.nStart = nStarts
        mc_fitter.modelName = modelName
        mc_fitter.freeP_c = freeP_c
        print(f"Model: {mc_fitter.modelName} | sharedLambda={mc_fitter.sharedLambda} | freeP_c={mc_fitter.freeP_c}")

        timeStart = time.time()
        print(f"\nFitting {dataName} [{noiseLabel}] over {len(mc_fitter.groupedData)} conditions")
        fittedParams = mc_fitter.fitCausalInferenceMonteCarlo(mc_fitter.groupedData)

        if fittedParams is None:
            raise RuntimeError(f"Fitting failed for {dataName} [{noiseLabel}] — returned None")

        print(f"Fitted params [{noiseLabel}]: {fittedParams}")
        print(f"Time: {time.time() - timeStart:.2f}s")

        mc_fitter.modelFit = fittedParams
        mc_fitter.logLikelihood = -mc_fitter.nLLMonteCarloCausal(fittedParams, mc_fitter.groupedData)

        fitSaver.saveFitResultsSingle(mc_fitter, fittedParams, dataName, noiseLabel=noiseLabel)
        fitSaver.saveSimulatedData(mc_fitter, dataName, noiseLabel=noiseLabel)

        print(f"=== Done: {dataFile} [{noiseLabel}] ===\n")
        return (dataFile, noiseLabel, True, None)

    except Exception as e:
        import traceback
        error_msg = f"Error processing {dataFile} [{noiseLabel}]: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return (dataFile, noiseLabel, False, error_msg)


if __name__ == "__main__":
    import sys
    import time
    from multiprocessing import Pool, cpu_count

    dataFiles = sys.argv[1].split(',') if len(sys.argv) > 1 else ["mt_all.csv"]
    dataFiles = [f.strip() for f in dataFiles]

    modelName = sys.argv[2] if len(sys.argv) > 2 else "lognorm"
    nSimul = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    optimMethod = sys.argv[4] if len(sys.argv) > 4 else "bads"
    nStarts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    freeP_c = sys.argv[6].lower() in ['true', '1', 'yes'] if len(sys.argv) > 6 else False
    integrationMethod = sys.argv[7] if len(sys.argv) > 7 else "analytical"

    # noiseLevel comes BEFORE n_cores so that the common case (just adding it
    # to an existing command) keeps a sensible default for cores. Accept it at
    # either position 8 or 9 — whichever is a known label.
    raw_remaining = sys.argv[8:]
    noiseLevel = None
    n_cores = None
    for arg in raw_remaining:
        if arg.lower() in {"low", "high", "both"}:
            noiseLevel = arg.lower()
        else:
            try:
                n_cores = int(arg)
            except ValueError:
                raise ValueError(f"Unrecognized argument: {arg!r}")

    if noiseLevel is None:
        raise ValueError("Missing required noiseLevel arg (low | high | both)")

    # Expand 'both' into two jobs per file
    if noiseLevel == "both":
        jobs = [(f, lab) for f in dataFiles for lab in ("low", "high")]
    else:
        jobs = [(f, noiseLevel) for f in dataFiles]

    if n_cores is None:
        n_cores = min(cpu_count(), len(jobs))

    print(f"Data files       : {dataFiles}")
    print(f"Model            : {modelName}")
    print(f"nSimul           : {nSimul}")
    print(f"Optimizer        : {optimMethod}")
    print(f"nStarts          : {nStarts}")
    print(f"freeP_c          : {freeP_c}")
    print(f"Integration      : {integrationMethod}")
    print(f"Noise level      : {noiseLevel}  (jobs total = {len(jobs)})")
    print(f"Cores            : {n_cores}  (available = {cpu_count()})")

    args_list = [
        (dataFile, modelName, nSimul, optimMethod, nStarts, freeP_c, integrationMethod, noiseLabel)
        for (dataFile, noiseLabel) in jobs
    ]

    overall_start = time.time()

    if n_cores > 1 and len(args_list) > 1:
        with Pool(processes=n_cores) as pool:
            results = pool.map(process_single_file, args_list)
    else:
        print("Running sequentially (single core or single job)...")
        results = [process_single_file(a) for a in args_list]

    overall_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time         : {overall_time:.2f}s")
    print(f"Avg time per job   : {overall_time / max(len(args_list), 1):.2f}s")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    print(f"\nSuccessful jobs    : {len(successful)}/{len(args_list)}")
    for dataFile, noiseLabel, _, _ in successful:
        print(f"  ✅ {dataFile} [{noiseLabel}]")

    if failed:
        print(f"\nFailed jobs        : {len(failed)}")
        for dataFile, noiseLabel, _, error_msg in failed:
            print(f"  ❌ {dataFile} [{noiseLabel}]")
            if error_msg:
                first_line = error_msg.splitlines()[0] if error_msg else ""
                print(f"     {first_line}")
