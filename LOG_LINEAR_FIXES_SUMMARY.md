# Summary of Monte Carlo Class Fixes

## Overview
This document summarizes the comprehensive fixes applied to the Monte Carlo causal inference class to address issues with parameter bounds, initial parameters, model accuracies, and log-linear conversions.

## Key Issues Fixed

### 1. ✅ t_min=0 Validation Issue
**Problem**: Code was incorrectly rejecting t_min=0 when t_min=0 should be valid for duration bounds.

**Fix**: Updated validation logic in `nLLMonteCarloCausal` method:
```python
# OLD (incorrect):
if t_min <= 0:
    return 1e10

# NEW (correct):
# Allow t_min = 0 for duration bounds, but ensure t_max > t_min    
if t_max <= 0:
    return 1e10
```

### 2. ✅ Log-Linear Space Parameter Bounds
**Problem**: Parameter bounds were not properly handling log-space vs linear-space for different models.

**Fix**: Updated all parameter bounds sections to handle log-space models correctly:

#### For Fusion-Only Models:
```python
if self.modelName == "fusionOnlyLogNorm":
    # For log-space models, t_min and t_max should be in log space
    log_data_t_min = np.log(max(self.data_t_min, 0.001))  # Avoid log(0)
    log_data_t_max = np.log(self.data_t_max)
    bounds = np.array([
        (0.001, 0.4),    # λ
        (0.05, 2.0),     # σa1
        (0.05, 2.0),     # σv
        (0.05, 2.5),     # σa2
        (log_data_t_min - 0.5, log_data_t_min + 0.1),  # t_min in log space
        (log_data_t_max - 0.1, log_data_t_max + 0.5),  # t_max in log space
    ])
else:
    # For linear-space fusion models
    bounds = np.array([
        (0.001, 0.4),    # λ
        (0.05, 2.0),     # σa1
        (0.05, 2.0),     # σv
        (0.05, 2.5),     # σa2
        (0.0, max(self.data_t_min * 0.9, 0.4)),  # t_min - allow 0
        (max(self.data_t_max * 1.1, 0.6), 10.0),  # t_max
    ])
```

#### For Standard Causal Inference Models:
Similar logic applied for `lognorm`, `logLinearMismatch`, and `probabilityMatchingLogNorm` models.

### 3. ✅ Parameter Extraction (getParamsCausal)
**Problem**: Need to handle log-space parameter storage correctly.

**Fix**: Updated parameter extraction to NOT convert log-space bounds back to linear space because:
- Log-space models store bounds in log space in the parameter array
- Likelihood functions expect bounds in the same space as measurements
- Converting back would break the model calculations

```python
# Note: For log-space models (lognorm, logLinearMismatch, probabilityMatchingLogNorm),
# t_min and t_max are stored in log space in the parameter array and are NOT converted 
# back to linear space here because the likelihood functions need them in the same space
# as the measurements (which are in log space for these models).
```

### 4. ✅ Bounds Validation for Different Spaces
**Problem**: Validation logic didn't account for log-space vs linear-space bounds.

**Fix**: Updated validation to handle both spaces correctly:
```python
if self.modelName in ["lognorm", "logLinearMismatch", "probabilityMatchingLogNorm", "fusionOnlyLogNorm"]:
    # For log-space models, t_min and t_max are in log space
    if t_min >= t_max:
        return 1e10
    if (t_max - t_min) < 0.01:  # Minimum meaningful range in log space
        return 1e10
    if t_max > 10:  # exp(10) ≈ 22000, reasonable upper bound
        return 1e10
else:
    # For linear-space models
    if t_min >= t_max:
        return 1e10
    if t_max <= 0:  # Allow t_min = 0
        return 1e10
    if (t_max - t_min) < 0.01:  # Minimum meaningful range
        return 1e10
```

## Model-Specific Behavior

### Linear-Space Models
- **Models**: `gaussian`, `fusionOnly`, `probabilityMatching`, `logLinearMismatch`
- **Parameter Storage**: t_min and t_max stored in linear space
- **Bounds**: Allow t_min=0, require t_max > 0
- **Validation**: Standard linear space checks

### Log-Space Models  
- **Models**: `lognorm`, `fusionOnlyLogNorm`, `probabilityMatchingLogNorm`
- **Parameter Storage**: t_min and t_max stored in log space
- **Bounds**: log(data_bounds) with appropriate margins
- **Validation**: Log space checks with overflow protection

### Special Case: logLinearMismatch
- **Measurements**: Generated in log space but converted to linear (log-normal)
- **Observer Model**: Assumes linear Gaussian noise (incorrect assumption)
- **Parameters**: t_min and t_max stored in log space but used with linear measurements

## Testing Results

All tests pass successfully:
- ✅ Parameter extraction working correctly for all 7 models
- ✅ t_min=0 handling correct for linear-space models
- ✅ Log-space bounds calculation correct
- ✅ Parameter validation working for both spaces
- ✅ All model configurations supported

## Configuration Matrix

| Model | Space | t_min=0 | Bounds Space | Measurements |
|-------|-------|---------|--------------|--------------|
| gaussian | Linear | ✅ | Linear | Linear Gaussian |
| fusionOnly | Linear | ✅ | Linear | Linear Gaussian |
| probabilityMatching | Linear | ✅ | Linear | Linear Gaussian |
| lognorm | Log | ❌ | Log | Log Gaussian |
| fusionOnlyLogNorm | Log | ❌ | Log | Log Gaussian |
| probabilityMatchingLogNorm | Log | ❌ | Log | Log Gaussian |
| logLinearMismatch | Mixed | ❌ | Log | Log-normal → Linear |

## Key Design Decisions

1. **No Conversion in getParamsCausal**: Parameters stay in their storage space to match model expectations
2. **Bounds Match Model Space**: Log models get log bounds, linear models get linear bounds  
3. **Validation Adapts to Space**: Different validation logic for different parameter spaces
4. **t_min=0 Only for Linear**: Log models can't have t_min=0 (would be log(0) = -∞)

## Files Modified
- `monteCarloClass.py`: Main fixes in parameter bounds, validation, and extraction
- `test_simple_log_linear.py`: Comprehensive test suite to validate fixes

All fixes maintain backward compatibility while properly handling the edge cases and log-linear conversions.