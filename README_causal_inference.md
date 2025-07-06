# Causal Inference Model for Audiovisual Duration Discrimination

This implementation integrates the causal inference model from `timeCausalInference.ipynb` into the psychometric fitting framework in `fitMain.py`.

## Overview

The causal inference model extends standard psychometric functions to account for how observers decide whether auditory and visual signals come from the same source (common cause) or different sources (independent causes). This is particularly relevant for audiovisual experiments where conflicts between modalities occur.

## Key Features

### 1. Causal Inference Mechanism
- **Common Cause (C=1)**: Observer assumes signals come from same source → reliability-weighted fusion
- **Independent Causes (C=2)**: Observer assumes signals come from different sources → modality-specific estimates
- **Model Averaging**: Final estimate combines both scenarios weighted by posterior probability

### 2. Model Parameters
- `λ` (lambda): Lapse rate
- `σ_a` (sigma_a): Auditory noise standard deviation in AV condition
- `σ_v` (sigma_v): Visual noise standard deviation in AV condition
- `P(common)`: Prior probability of common cause

**Note**: Unlike standard psychometric models, there is no μ (PSE) parameter because the causal inference process naturally creates biases based on visual conflicts. Adding μ would be redundant and theoretically incorrect.

### 3. Model Comparison
- Automatic comparison with standard psychometric models
- AIC/BIC model selection criteria
- Bootstrap parameter confidence intervals

## Usage

### Quick Start
```python
from fitMain import *

# Load and analyze data with causal inference model
results = run_causal_inference_analysis(
    data_file="your_data.csv",
    shared_sigma=True,
    all_independent=False
)
```

### Manual Fitting
```python
# Load your data
data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(
    "your_data.csv", isShared=True, isAllIndependent=False
)

# Group data for fitting
grouped_data = groupByChooseTest(data)

# Fit causal inference model (4 parameters)
initial_guesses = [0.05, 0.2, 0.2, 0.5]  # [λ, σ_a, σ_v, P(common)]
ci_fit = fit_causal_inference_4param(grouped_data, initial_guesses)

# Plot results
plot_causal_inference_4param(data, ci_fit.x)
```

### Model Comparison
```python
# Compare standard vs causal inference models (4 parameters)
results = compare_4param_vs_standard(data, n_starts=3)

# Plot side-by-side comparison
plot_model_comparison(data, results)

# Check which model is preferred
if results["delta_aic"] < 0:
    print("Causal inference model preferred")
else:
    print("Standard model preferred")
```

## Data Requirements

Your CSV file should contain these columns:
- `delta_dur_percents`: Duration difference between test and standard (%)
- `audNoise`: Auditory noise level
- `standardDur`: Standard duration value
- `conflictDur`: Visual conflict level (difference between visual and auditory)
- `responses`: Participant responses
- `chose_test`: Binary indicator (1 if chose test, 0 if chose standard)

## Mathematical Foundation

### Fusion Under Common Cause
When assuming common cause, the model uses reliability-weighted averaging:

```
ŝ_fused = (J_a * m_a + J_v * m_v) / (J_a + J_v)
```

Where:
- `J_a = 1/σ_a²` (auditory precision)  
- `J_v = 1/σ_v²` (visual precision)
- `m_a`, `m_v` are noisy measurements

### Posterior Common Cause Probability
```
P(C=1|m_a,m_v) = L(m_a,m_v|C=1) * P(C=1) / [L(m_a,m_v|C=1) * P(C=1) + L(m_a,m_v|C=2) * P(C=2)]
```

### Final Estimate
```
ŝ_final = ŝ_fused * P(C=1|m_a,m_v) + m_a * (1 - P(C=1|m_a,m_v))
```

## Interpretation Guidelines

### Prior Probability P(common)
- **P(common) > 0.7**: Strong tendency to integrate audiovisual signals
- **P(common) = 0.5**: No bias toward integration vs segregation  
- **P(common) < 0.3**: Strong tendency to process modalities independently

### Noise Parameters
- **σ_a vs σ_v**: Relative reliability of auditory vs visual modalities
- Lower σ indicates higher reliability
- The model will weight more reliable modality more heavily during fusion

### Model Selection
- **ΔAIC < -2**: Strong evidence for causal inference model
- **-2 ≤ ΔAIC ≤ 2**: Weak evidence
- **ΔAIC > 2**: Standard model preferred

## Example Output

```
MODEL COMPARISON RESULTS
==================================================
Standard Model:
  Parameters: 8
  NLL: 245.67
  AIC: 507.34
  BIC: 539.12

4-Parameter Causal Inference Model:  
  Parameters: 4
  λ (lapse): 0.034
  σ_a (aud noise): 0.187
  σ_v (vis noise): 0.245
  P(common): 0.732
  NLL: 239.45
  AIC: 486.90
  BIC: 503.34

Model Comparison:
  ΔAIC: -20.44 (negative favors CI model)
  ΔBIC: -35.78 (negative favors CI model)
  → 4-Parameter Causal Inference model preferred by AIC
  → 4-Parameter Causal Inference model preferred by BIC
```

## Advanced Usage

### Bootstrap Confidence Intervals
```python
# Get parameter confidence intervals
boot_params = paramBootstrap(ci_fit.x, nBoots=100)
ci_lower = np.percentile(boot_params, 2.5, axis=0)
ci_upper = np.percentile(boot_params, 97.5, axis=0)
```

### Custom Parameter Bounds
```python
# Modify bounds in fit_causal_inference_model function
bounds = [
    (0, 0.1),      # λ: stricter lapse rate
    (-1, 1),       # μ: PSE range
    (0.05, 0.5),   # σ_a: auditory noise range
    (0.05, 0.5),   # σ_v: visual noise range  
    (0.1, 0.9)     # P(common): prior range
]
```

## Troubleshooting

### Common Issues
1. **Fit fails to converge**: Try different initial guesses or looser bounds
2. **Unrealistic parameters**: Check data preprocessing and column names
3. **Model comparison fails**: Ensure both models fit successfully

### Parameter Validation
- λ should be close to empirical lapse rate (typically < 0.1)
- σ values should reflect measurement precision
- P(common) should be between 0 and 1
- μ should be near observed PSE

## Files

- `fitMain.py`: Main fitting framework with causal inference implementation
- `example_causal_inference.py`: Example usage script
- `timeCausalInference.ipynb`: Original model development notebook
- `README_causal_inference.md`: This documentation

## References

Based on causal inference models for multisensory perception:
- Körding et al. (2007). Causal inference in multisensory perception. PLOS ONE.
- Rohe & Noppeney (2015). Cortical hierarchies perform Bayesian causal inference in multisensory perception. PLOS Biology.
