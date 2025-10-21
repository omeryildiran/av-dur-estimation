"""
Batch Psychometric Fitting Script for SIMULATED DATA
Fits psychometric functions to simulated data (lognorm_LapseFree_sharedPrior) and saves results
"""
from psychometricFitSaver_simulated import batch_fit_simulated

# List of all participant IDs
PARTICIPANT_IDS = [
    "as", "oy", "dt", "HH", "ip", "ln", "LN01", 
    "mh", "ml", "mt", "qs", "sx"
]

# Model type to fit (simulated data)
MODEL_TYPE = "lognorm_LapseFree_sharedPrior"

if __name__ == "__main__":
    print("="*70)
    print("BATCH PSYCHOMETRIC FITTING - SIMULATED DATA")
    print("="*70)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Processing {len(PARTICIPANT_IDS)} participants")
    print(f"Participants: {', '.join(PARTICIPANT_IDS)}")
    print("="*70)
    
    # Run batch fitting with 1 starting point (simulated data is cleaner)
    results = batch_fit_simulated(
        participant_ids=PARTICIPANT_IDS,
        model_type=MODEL_TYPE,
        nStart=1,
        save_dir="psychometric_fits_simulated",
        sim_base_dir="simulated_data"
    )
    
    print("\n\n" + "="*70)
    print("FITTING COMPLETE!")
    print("="*70)
    print(f"Results saved in: psychometric_fits_simulated/")
    print(f"✅ Successfully fitted: {len(results)} participants")
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY OF FITS")
        print("="*70)
        for pid, result in results.items():
            print(f"{pid}: AIC={result['AIC']:.2f}, BIC={result['BIC']:.2f}, "
                  f"Params={result['n_params']}, Conditions={result['n_conditions']}")
