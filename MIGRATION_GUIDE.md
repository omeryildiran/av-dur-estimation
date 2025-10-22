# Migration Guide: From Script to Class

This guide shows how to migrate from the old `fitNonSharedwErrorBars_logNormal.py` script to the new `PsychometricFitter` class.

## Quick Start

### Old Way (Script)
```python
python fitNonSharedwErrorBars_logNormal.py --data my_data.csv --no-error-bars
```

### New Way (Class)
```python
from psychometric_fitter import PsychometricFitter

fitter = PsychometricFitter(data_path='my_data.csv', fix_mu=False)
fitter.fit(n_start=3)
fitter.plot_fitted_psychometric(show_error_bars=True)
```

## Common Use Cases

### 1. Basic Fitting and Plotting

**Old:**
```python
# Run from command line
python fitNonSharedwErrorBars_logNormal.py --data my_data.csv
```

**New:**
```python
from psychometric_fitter import PsychometricFitter

# Load and fit
fitter = PsychometricFitter(data_path='my_data.csv')
fitter.fit(n_start=1)  # n_start=1 for single starting point

# Plot
fitter.plot_fitted_psychometric(show_error_bars=True)
```

### 2. Fix Mu (Bias) to Zero

**Old:**
```python
# Set global variable in script
fixedMu = True
```

**New:**
```python
fitter = PsychometricFitter(data_path='my_data.csv', fix_mu=True)
fitter.fit()
```

### 3. Use Pre-loaded DataFrame

**Old:**
```python
# Had to modify loadData() function
data = pd.read_csv('my_data.csv')
# ... manual preprocessing
```

**New:**
```python
import pandas as pd
from psychometric_fitter import PsychometricFitter

data = pd.read_csv('data/my_data.csv')
fitter = PsychometricFitter(data=data)
fitter.fit()
```

### 4. Access Fitted Parameters

**Old:**
```python
# Parameters were in fit.x
lambda_ = fit.x[0]
# Had to manually call getParams() for each condition
```

**New:**
```python
# Get parameters for specific condition
params = fitter.get_condition_params(audio_noise=0.1, conflict=0)
print(f"Lambda: {params['lambda']:.3f}")
print(f"Mu: {params['mu']:.3f}")
print(f"Sigma: {params['sigma']:.3f}")

# Or access all parameters
all_params = fitter.fitted_params
```

### 5. Make Predictions

**Old:**
```python
# Had to manually call psychometric_function with extracted parameters
lambda_, mu, sigma = getParams(fit.x, conflict, audio_noise, nLambda, nSigma)
p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
```

**New:**
```python
# Simple prediction interface
p = fitter.predict(test_dur=0.6, standard_dur=0.5, 
                   audio_noise=0.1, conflict=0)
```

### 6. Calculate PSE Statistics

**Old:**
```python
# Call static function
pse_stats = calculate_pse_stats(mu, sigma, lambda_, standard_dur)
```

**New:**
```python
# Still available as static method
pse_stats = PsychometricFitter.calculate_pse_stats(
    mu, sigma, lambda_, standard_dur
)

# Or automatically during plotting (printed to console)
fitter.plot_fitted_psychometric()
```

## Advanced Usage

### Integration with Other Scripts

```python
from psychometric_fitter import PsychometricFitter
import pandas as pd

# Example: Batch processing multiple datasets
data_files = ['participant1.csv', 'participant2.csv', 'participant3.csv']

results = []
for file in data_files:
    fitter = PsychometricFitter(data_path=file)
    fitter.fit(n_start=3)
    
    # Extract results
    params = fitter.get_condition_params()
    results.append({
        'file': file,
        'lambda': params['lambda'],
        'mu': params['mu'],
        'sigma': params['sigma'],
        'nll': fitter.fit_result.fun
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('batch_results.csv', index=False)
```

### Accessing Grouping Methods

```python
# Group data for custom analysis
fitter = PsychometricFitter(data_path='my_data.csv')

# Get aggregated data (like the old groupByChooseTest)
grouped = fitter.group_by_choose_test()

# Get participant-level data for error bars
participant_grouped = fitter.group_by_choose_test_with_participants()
```

### Using Static Methods Without Instance

```python
from psychometric_fitter import PsychometricFitter

# Calculate psychometric function
p = PsychometricFitter.psychometric_function(
    test_dur=0.6, standard_dur=0.5, 
    lambda_=0.05, mu=0.1, sigma=0.2, fix_mu=False
)

# Convert between mu and PSE shift
pse_shift = PsychometricFitter.mu_to_pse_shift(mu=0.1, standard_dur=0.5)
mu = PsychometricFitter.pse_shift_to_mu(pse_shift=0.05, standard_dur=0.5)
```

## Key Differences

| Aspect | Old Script | New Class |
|--------|-----------|-----------|
| **Interface** | Command-line only | Python API + can create scripts |
| **Data Loading** | Global variables | Instance attributes |
| **Parameters** | In `fit.x` array | Accessible via methods |
| **Reusability** | Single run per execution | Reusable object |
| **State** | No persistence | Stores fitted parameters |
| **Integration** | Difficult | Easy to import and use |
| **Flexibility** | Fixed workflow | Mix and match methods |

## Benefits of Class Approach

1. **Reusability**: Import once, use in multiple scripts
2. **State Management**: Fitted parameters stored in object
3. **Clean API**: Clear method names instead of global functions
4. **Extensibility**: Easy to subclass for custom behavior
5. **Testing**: Easier to write unit tests
6. **Documentation**: Built-in docstrings accessible via `help()`

## Backward Compatibility

The original script still works! If you have existing workflows using the command-line script, they will continue to function. The class is an additional interface for more flexible usage.

```bash
# Old script still works
python fitNonSharedwErrorBars_logNormal.py --data my_data.csv
```
