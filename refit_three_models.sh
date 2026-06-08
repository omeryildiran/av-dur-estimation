#!/usr/bin/env bash
# Refit the 3 main models with BADS (free sigma, fixed lapse, shared prior).
#   nSimul=3000, optim=bads, nStarts=5, freeP_c=False  -> *_LapseFix_sharedPrior_fit.json
# Outputs (lowercase participant codes):
#   fits -> model_fits/<pid>/<pid>_<model>_LapseFix_sharedPrior_fit.json
#   sims -> simulated_data/<pid>/<pid>_<model>_LapseFix_sharedPrior_simulated.csv
set -u
cd "$(dirname "$0")"

# 11 participants, no ln1, NO stray spaces in the comma list
FILES="as_all.csv,oy_all.csv,dt_all.csv,hh_all.csv,ip_all.csv,ln2_all.csv,mh_all.csv,ml_all.csv,mt_all.csv,qs_all.csv,sx_all.csv"
NSIMUL=3000
OPTIM=bads
NSTARTS=5
FREEPC=False
MODELS=(lognorm fusionOnlyLogNorm switchingFree)

PY=/opt/miniconda3/envs/mathmod/bin/python
mkdir -p _refit_logs

overall_start=$(date +%s)
for model in "${MODELS[@]}"; do
    echo "================================================================"
    echo "=== fitting ${model}  ($(date '+%H:%M:%S')) ==="
    echo "================================================================"
    t0=$(date +%s)
    "$PY" -u runFitting.py "$FILES" "$model" "$NSIMUL" "$OPTIM" "$NSTARTS" "$FREEPC" \
        2>&1 | tee "_refit_logs/refit_${model}.log"
    rc=${PIPESTATUS[0]}
    echo "=== done ${model}: exit ${rc}, $(( $(date +%s) - t0 ))s ==="
done
echo "ALL DONE in $(( $(date +%s) - overall_start ))s"
