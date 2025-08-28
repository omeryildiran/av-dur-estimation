#!/bin/bash
# save as run_all_models.sh

DATA_FILES="$1"
N_SIMUL="${2:-2000}"
OPTIM_METHOD="${3:-bads}"
N_STARTS="${4:-5}"

# Run all three models in parallel
python runFitting.py "$DATA_FILES" "lognorm" "$N_SIMUL" "$OPTIM_METHOD" "$N_STARTS" &
python runFitting.py "$DATA_FILES" "logLinearMismatch" "$N_SIMUL" "$OPTIM_METHOD" "$N_STARTS" &
python runFitting.py "$DATA_FILES" "gaussian" "$N_SIMUL" "$OPTIM_METHOD" "$N_STARTS" &

# Wait for all background processes to complete
wait

echo "All models completed!"