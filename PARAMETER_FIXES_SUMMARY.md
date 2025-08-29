## Parameter Retrieval Fixes for getParamsCausal Function

### Issues Found and Fixed:

#### 1. ❌ Missing `self.freeP_c` Attribute
**Problem**: The `getParamsCausal` function referenced `self.freeP_c` but this attribute was not defined in the `__init__` method.
**Solution**: Added `self.freeP_c = False` to the `__init__` method.

#### 2. ❌ Incorrect Parameter Indexing for `freeP_c=True`
**Problem**: When `freeP_c=True` and `sharedLambda=True`, the function was using the wrong index for the second `p_c` parameter.
**Solution**: Fixed the indexing logic to correctly handle different configuration combinations:
- `sharedLambda=True, freeP_c=True`: Use `params[5]` for SNR=1.2
- `sharedLambda=False, freeP_c=True`: Use `params[7]` for SNR=1.2

#### 3. ❌ No Parameter Length Validation
**Problem**: The function didn't validate that the parameter array had the correct length for the given configuration.
**Solution**: Added comprehensive parameter length validation with clear error messages.

#### 4. ❌ Missing Documentation
**Problem**: Complex parameter indexing logic was not documented.
**Solution**: Added detailed docstring explaining parameter layouts for all configurations.

### Parameter Array Layouts:

```python
# Configuration-dependent parameter layouts:
sharedLambda=False, freeP_c=False: [λ, σa1, σv, pc, σa2, λ2, λ3] (7 params)
sharedLambda=True,  freeP_c=False: [λ, σa1, σv, pc, σa2] (5 params)
sharedLambda=False, freeP_c=True:  [λ, σa1, σv, pc1, σa2, λ2, λ3, pc2] (8 params)
sharedLambda=True,  freeP_c=True:  [λ, σa1, σv, pc1, σa2, pc2] (6 params)
```

### Key Changes Made:

1. **Added missing attribute in `__init__`:**
```python
self.freeP_c = False  # Fix: Add missing attribute for parameter configuration
```

2. **Fixed parameter indexing logic:**
```python
if self.freeP_c:
    if np.isclose(SNR, 0.1):
        p_c=params[3] 
    elif np.isclose(SNR,1.2):
        if self.sharedLambda:
            p_c=params[5]  # Fixed: Different index when sharedLambda=True
        else:
            p_c=params[7]  # Fixed: Different index when sharedLambda=False
```

3. **Added parameter length validation:**
```python
expected_lengths = {
    (True, True): 6,    # sharedLambda=True, freeP_c=True
    (True, False): 5,   # sharedLambda=True, freeP_c=False
    (False, True): 8,   # sharedLambda=False, freeP_c=True
    (False, False): 7   # sharedLambda=False, freeP_c=False
}

if len(params) != expected_length:
    raise ValueError(f"Parameter array length {len(params)} doesn't match expected length {expected_length}")
```

4. **Added comprehensive documentation:**
   - Parameter layout explanation for each configuration
   - Clear return value description
   - Input parameter validation details

### Test Results:
✅ All parameter configurations working correctly
✅ Parameter length validation working
✅ Error handling for invalid SNR values
✅ Correct indexing for all `freeP_c` and `sharedLambda` combinations

### Impact:
These fixes ensure that the `getParamsCausal` function:
- No longer crashes due to missing attributes
- Correctly extracts parameters for all model configurations
- Provides clear error messages for invalid inputs
- Has proper documentation for maintainability
