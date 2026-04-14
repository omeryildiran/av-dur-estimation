# Bayesian Causal Inference in Audio-Visual Duration Estimation

Code and data for the paper:

> **Bayesian Causal Inference Accounts for Audio-Visual Duration Estimation**  
> Omer Faruk Yildiran, Michael S. Landy  
> *[Journal / bioRxiv]*, 2025

## Overview

Observers estimated the relative duration of audio-visual stimuli in a 2AFC paradigm. On each trial, an auditory and a visual stimulus were presented together as the standard, then an auditory-only test interval. The visual stimulus could differ from the auditory standard by a small conflict duration (±83, ±167, or ±250 ms). We fitted several models of multisensory integration to the data, including Bayesian causal inference, forced fusion, probability matching, sensory selection, and a switching model.

## Repository Structure

```
├── data/
│   ├── all_main.csv                    # Pooled behavioral data (all participants)
│   ├── all_auditory.csv                # Unimodal auditory experiment
│   ├── all_visual.csv                  # Unimodal visual experiment
│   ├── all_woBiasedParticipants.csv    # Subset excluding biased participants
│   ├── all_wo_ln1.csv                  # Subset excluding P06 (LN1)
│   └── combineCSVs.py                  # Utility to combine per-session CSV files
│
├── model_fits/                         # Fitted model parameters (per participant)
│   └── P01/ ... P13/                   # One folder per participant (P01–P13)
│
├── bootstrapped_params/                # Bootstrap confidence intervals on parameters
├── model_recovery_results/             # Model recovery analysis (JSON per participant)
│
├── mainExpAvDurEstimate/               # Main experiment (audio-visual 2AFC)
├── unimodalAudDurEst/                  # Unimodal auditory experiment
├── unimodalVisualDurEst/               # Unimodal visual experiment
│
├── monteCarloClass.py                  # Main model class (Monte Carlo causal inference)
├── fitMainClass.py                     # Base psychometric fitting class
├── psychometric_fitter.py              # Psychometric curve fitting utilities
├── loadData.py                         # Data loading and preprocessing
├── loadResults.py                      # Load saved model fits
├── fitSaver.py                         # Save model fits to JSON
├── runFitting.py                       # Run model fitting pipeline
├── runBootstrapper.py                  # Run bootstrap parameter estimation
├── bootstrapperSaveLoad.py             # Save/load bootstrap results
├── bootstrap_pooled_models.py          # Fit pooled (group-level) models
├── generate_model_bootstraps.py        # Generate bootstrap samples
├── simDataGenerator.py                 # Simulate data from model parameters
├── audio_cue_generator.py             # Generate auditory stimuli
├── plot_style.py                       # Shared matplotlib style settings
│
├── interactive_model_predictions.ipynb # Interactive model exploration (start here)
├── cleanAnalysis.ipynb                 # Main analysis notebook
├── plotAllModelFits.ipynb              # Plot all model fits
├── plotConflictvsPSE_clean.ipynb       # PSE vs conflict figure
├── modelComparisonTable.ipynb          # AIC/BIC model comparison table
├── bootstrapModels.ipynb               # Bootstrap analysis and CIs
├── plotPosteriorCommons.ipynb          # Posterior P(common cause) plots
├── plotPSEConflict_bootstrapCI.ipynb   # PSE with bootstrap confidence intervals
├── weber_fractions_jnds_crossmodal.ipynb # JNDs and Weber fractions
│
├── fitted_parameters_all_models.csv    # All fitted parameters across models
├── best_models_by_participant.csv      # Best-fitting model per participant
├── model_comparison_all_results.csv    # Full AIC/BIC comparison table
├── model_comparison_summary.json       # Summary of model comparison
└── all_crossmodal.csv                  # Crossmodal psychometric data
```

## Models

| Model | Description | Parameters |
|---|---|---|
| `lognorm` | Bayesian causal inference in log space | λ, σa1, σv, pc, σa2 |
| `fusionOnlyLogNorm` | Forced fusion in log space (pc = 1) | λ, σa1, σv, σa2 |
| `probabilityMatchingLogNorm` | Probability matching (stochastic causal inference) | λ, σa1, σv, pc, σa2 |
| `selection` | Auditory-only selection (ignores visual) | λ, σa1, σv, pc, σa2 |
| `switchingFree` | Switching between modalities with free switch probabilities | λ, σa1, σv, p_sw1, σa2, p_sw2 |
| `logLinearMismatch` | Log-normal measurements with linear-space inference | λ, σa1, σv, pc, σa2 |

Parameters: λ = lapse rate, σa1/σa2 = auditory noise (low/high), σv = visual noise, pc = prior probability of common cause.

## Getting Started

### Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn tqdm ipywidgets pybads
```

The `pybads` package is optional (used for Bayesian Adaptive Direct Search optimization). The code falls back to `scipy.optimize` if not installed.

### Interactive Exploration

Open `interactive_model_predictions.ipynb` to explore model predictions with parameter sliders — no fitting required.

### Reproducing the Analysis

1. **Load data and fit models:**
   ```python
   from loadData import loadData
   from monteCarloClass import OmerMonteCarlo

   data, dataName = loadData("all_main.csv")
   model = OmerMonteCarlo(data, dataName=dataName)
   model.modelName = "lognorm"
   ```

2. **Load pre-fitted parameters:**
   ```python
   from loadResults import loadResults
   results = loadResults("P01", "lognorm_LapseFix_sharedPrior")
   params = results["fittedParams"]
   ```

3. **Run the analysis notebooks** in this order:
   - `cleanAnalysis.ipynb` — preprocessing and basic psychometrics
   - `plotAllModelFits.ipynb` — visualize model fits
   - `modelComparisonTable.ipynb` — AIC/BIC comparison
   - `plotConflictvsPSE_clean.ipynb` — main figure (PSE vs conflict)

## Data

Participant IDs have been anonymized (P01–P13). The mapping between anonymous IDs and internal codes is retained by the authors.

Raw trial-by-trial data is available upon request. The aggregated files in `data/` contain all information needed to reproduce the analysis.

## Citation

```bibtex
@article{yildiran2025,
  title   = {Bayesian Causal Inference Accounts for Audio-Visual Duration Estimation},
  author  = {Yildiran, Omer Faruk and Landy, Michael S.},
  journal = {},
  year    = {2025}
}
```
