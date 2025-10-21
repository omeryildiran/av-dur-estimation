# Psychometric Function Fitting System

## Overview

This system provides fast, efficient psychometric function fitting and loading for all participant data. Instead of re-fitting models every time you run an analysis, you can:

1. **Fit once** - Run the batch fitter to fit all participants
2. **Save results** - Fitted parameters and model metrics saved to JSON
3. **Load instantly** - Load pre-fitted parameters in notebooks/scripts

---

## Quick Start

### 1. Fit All Participants (One-Time Setup)

```bash
python runDataFitter.py
```

This will:
- Fit psychometric functions to all 12 participants
- Use 3 random starting points for robust optimization
- Save results to `psychometric_fits/` directory
- Take ~5-15 minutes depending on your machine

### 2. Load Results in Your Analysis

```python
import psychometricFitLoader as pfl

# Load single participant
fit = pfl.load_psychometric_fit('as')
params = fit['parameters']
aic = fit['AIC']

# Load all participants
all_fits = pfl.load_all_psychometric_fits()

# Get summary table
summary = pfl.get_fit_summary()
print(summary)

# Get specific parameters
params_mt = pfl.get_parameters('mt')
```

---

## File Structure

```
av-dur-estimation/
├── runDataFitter.py              # Main script to fit all participants
├── psychometricFitSaver.py       # Fitting and saving functions
├── psychometricFitLoader.py      # Loading and analysis functions
├── psychometric_fits/            # Saved fit results (created after first run)
│   ├── as/
│   │   └── as_psychometric_fit.json
│   ├── oy/
│   │   └── oy_psychometric_fit.json
│   └── ...
└── data/                         # Original participant CSV files
    ├── as_all.csv
    ├── oy_all.csv
    └── ...
```

---

## Detailed Usage

### Running Batch Fits

```python
from psychometricFitSaver import batch_fit_participants

data_files = [
    "as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv",
    "ip_all.csv", "ln_all.csv", "LN01_all.csv", "mh_all.csv",
    "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"
]

# Fit with 3 starting points (recommended)
results = batch_fit_participants(data_files, nStart=3)

# Or fit with more starting points for better convergence
results = batch_fit_participants(data_files, nStart=5)
```

### Loading and Analyzing Results

```python
import psychometricFitLoader as pfl

# 1. Load single participant
fit = pfl.load_psychometric_fit('mt')
print(f"AIC: {fit['AIC']:.2f}")
print(f"BIC: {fit['BIC']:.2f}")
print(f"Parameters: {fit['parameters']}")

# 2. Get summary of all fits
summary_df = pfl.get_fit_summary()
print(summary_df[['participantID', 'AIC', 'BIC', 'n_params']])

# 3. Model comparison
comparison = pfl.get_model_comparison()
print("Best to worst fits:")
print(comparison)

# 4. Load all fits for custom analysis
all_fits = pfl.load_all_psychometric_fits()
for pid, fit in all_fits.items():
    print(f"{pid}: {fit['n_params']} params, AIC={fit['AIC']:.2f}")
```

---

## Saved Data Format

Each participant's fit is saved as JSON with this structure:

```json
{
    "participantID": "as",
    "dataName": "as_all.csv",
    "n_params": 42,
    "n_conditions": 120,
    "n_trials": 2400,
    "parameters": [0.02, 0.15, 0.12, ...],
    "log_likelihood": -1234.56,
    "AIC": 2553.12,
    "BIC": 2789.45,
    "success": true,
    "message": "Optimization terminated successfully",
    "uniqueSensory": [0.1, 1.2],
    "uniqueStandard": [0.5],
    "uniqueConflict": [-0.25, -0.15, ..., 0.25],
    "nStart": 3
}
```

---

## Key Functions Reference

### psychometricFitSaver.py

- `fit_and_save_psychometric(dataName, nStart=3)` - Fit single participant
- `batch_fit_participants(dataNames, nStart=3)` - Fit multiple participants

### psychometricFitLoader.py

- `load_psychometric_fit(participantID)` - Load single participant's fit
- `load_all_psychometric_fits()` - Load all saved fits
- `get_fit_summary()` - Get DataFrame summary of all fits
- `get_parameters(participantID)` - Extract just the parameters
- `get_model_comparison()` - Get AIC/BIC comparison table

---

## Benefits

✅ **Speed**: Load fits instantly instead of re-fitting (seconds vs minutes)  
✅ **Consistency**: Same fits across all analyses  
✅ **Reproducibility**: Exact parameters saved with metadata  
✅ **Traceability**: Know exactly which data and settings produced each fit  
✅ **Efficiency**: Fit once, analyze many times  

---

## Participants Included

The system processes these 12 participants:
- as, oy, dt, HH, ip, ln, LN01, mh, ml, mt, qs, sx

---

## Model Information

### Psychometric Function
- **Type**: Cumulative Gaussian with lapse rate
- **Parameters per condition**: Lambda (lapse), Mu (PSE), Sigma (slope)
- **Optimization**: Multiple random starts with bounded BFGS
- **Model Selection**: AIC and BIC calculated for comparison

### Fit Quality Metrics
- **AIC**: Akaike Information Criterion (lower is better)
- **BIC**: Bayesian Information Criterion (lower is better, penalizes complexity more)
- **Log-likelihood**: Goodness of fit measure

---

## Example Notebook Usage

See the top of `modelAnalysisOverall.ipynb` for example code showing:
1. How to load single participant fits
2. How to generate summary tables
3. How to extract parameters for analysis
4. How to run batch fitting (commented out by default)

---

## Troubleshooting

### "File not found" error
Run `python runDataFitter.py` first to generate the fits.

### Need to re-fit a participant
Delete their folder from `psychometric_fits/` and re-run the fitter.

### Want different optimization settings
Edit `nStart` parameter in `runDataFitter.py` (line with `nStart=3`)

### Need to check if fits converged
Load the fit and check `fit['success']` and `fit['message']`

---

## Advanced: Integration with Existing Code

Your existing Monte Carlo fitter can use these pre-fitted psychometric parameters:

```python
import psychometricFitLoader as pfl

# Load pre-fitted psychometric parameters
psychometric_fit = pfl.load_psychometric_fit('mt')
psychometric_params = psychometric_fit['parameters']

# Use as starting point or comparison for Monte Carlo model
mc_fitter = monteCarloClass.OmerMonteCarlo(data)
mc_fitter.dataFit = psychometric_params  # Use as baseline comparison
```

---

## Performance Notes

- **Fitting time**: ~30 seconds per participant (with nStart=3)
- **Total batch time**: ~6-10 minutes for all 12 participants
- **Loading time**: <0.1 seconds per participant
- **Storage**: ~5-10 KB per participant (JSON files)

---

## Future Enhancements

Possible additions:
- Bootstrap confidence intervals for parameters
- Cross-validation metrics
- Parameter recovery simulations
- Automated model comparison reports

---

## Questions?

Check the code comments in:
- `psychometricFitSaver.py` for fitting details
- `psychometricFitLoader.py` for loading details
- `runDataFitter.py` for the batch processing script
