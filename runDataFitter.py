"""
Batch Psychometric Fitting Script
Fits psychometric functions to all participant data and saves results
"""
from psychometricFitSaver import batch_fit_participants

# List of all participant data files
DATA_FILES = [
    "as_all.csv",
    "oy_all.csv",
    "dt_all.csv",
    "HH_all.csv",
    "ip_all.csv",
    "ln_all.csv",
    "LN01_all.csv",
    "mh_all.csv",
    "ml_all.csv",
    "mt_all.csv",
    "qs_all.csv",
    "sx_all.csv"
]

if __name__ == "__main__":
    print("="*60)
    print("BATCH PSYCHOMETRIC FITTING")
    print("="*60)
    print(f"Processing {len(DATA_FILES)} participants")
    print(f"Files: {', '.join([f.split('_')[0] for f in DATA_FILES])}")
    print("="*60)
    
    # Run batch fitting with 3 starting points
    results = batch_fit_participants(
        dataNames=DATA_FILES,
        nStart=1,
        save_dir="psychometric_fits_data"
    )
    
    print("\n\n" + "="*60)
    print("FITTING COMPLETE!")
    print("="*60)
    print(f"Results saved in: psychometric_fits/")
    print(f" ✅ Successfully fitted: {len(results)} participants")
