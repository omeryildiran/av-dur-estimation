# Decision Noise in Causal Inference Psychometric Function

## The Problem

In the causal inference psychometric function, we need to determine the correct formula for `sigma_decision` - the noise in the decision variable when comparing two intervals.

## Current Implementation

```python
sigma_decision = np.sqrt(sigmaAV_A**2 + sigmaAV_V**2) / 2  # Current approximate
```

## Theoretical Analysis

### 1. What is the Decision Variable?

The decision variable is:
```
decision_diff = hat_S_test - hat_S_standard
```

Where `hat_S_test` and `hat_S_standard` are causal inference estimates.

### 2. Variance of Causal Inference Estimates

Each causal inference estimate `hat_S` has variance that depends on:
- Whether the trial has a common cause (C=1) or separate causes (C=2)
- The posterior probability of common cause
- The fusion process

For a single trial with causal inference:
```
Var(hat_S) = p_posterior_c1 * Var(hat_S | C=1) + p_posterior_c2 * Var(hat_S | C=2)
           + p_posterior_c1 * p_posterior_c2 * (E[hat_S | C=1] - E[hat_S | C=2])^2
```

### 3. Variance Components

**Common cause (C=1) - Optimal fusion:**
```
Var(hat_S | C=1) = 1 / (1/σ_a² + 1/σ_v²)  # Optimal fusion variance
```

**Separate causes (C=2) - Auditory only:**
```
Var(hat_S | C=2) = σ_a²  # Auditory variance only
```

### 4. Decision Variable Variance

For the difference of two independent causal inference estimates:
```
Var(decision_diff) = Var(hat_S_test) + Var(hat_S_standard)
```

If both intervals have similar statistics (reasonable assumption):
```
Var(decision_diff) ≈ 2 * Var(hat_S)
```

Therefore:
```
sigma_decision = sqrt(2 * Var(hat_S))
```

## Proposed Corrections

### Option 1: Trial-by-Trial Calculation (Most Accurate)

Calculate the actual variance for each trial based on the posterior:

```python
def calculate_decision_noise_exact(sigmaAV_A, sigmaAV_V, p_posterior_c1):
    """Calculate exact decision noise based on causal inference theory."""
    # Variance under common cause (optimal fusion)
    var_c1 = 1 / (1/sigmaAV_A**2 + 1/sigmaAV_V**2)
    
    # Variance under separate causes (auditory only)
    var_c2 = sigmaAV_A**2
    
    # Expected variance of causal inference estimate
    var_estimate = p_posterior_c1 * var_c1 + (1 - p_posterior_c1) * var_c2
    
    # Decision noise for difference of two estimates
    sigma_decision = np.sqrt(2 * var_estimate)
    
    return sigma_decision
```

### Option 2: Average Approximation (Computationally Efficient)

Use an average over typical posterior values:

```python
def calculate_decision_noise_approx(sigmaAV_A, sigmaAV_V, p_common, conflict):
    """Approximate decision noise using expected posterior."""
    # Estimate typical posterior (could be refined)
    p_posterior_c1_typical = p_common  # Simplified assumption
    
    # Variance components
    var_c1 = 1 / (1/sigmaAV_A**2 + 1/sigmaAV_V**2)
    var_c2 = sigmaAV_A**2
    
    # Expected variance
    var_estimate = p_posterior_c1_typical * var_c1 + (1 - p_posterior_c1_typical) * var_c2
    
    # Decision noise
    sigma_decision = np.sqrt(2 * var_estimate)
    
    return sigma_decision
```

### Option 3: Conservative Bound

Use the maximum possible variance:

```python
def calculate_decision_noise_conservative(sigmaAV_A, sigmaAV_V):
    """Conservative upper bound on decision noise."""
    # Worst case is separate causes (auditory only)
    sigma_decision = np.sqrt(2) * sigmaAV_A
    return sigma_decision
```

## Recommendation

**Use Option 2 (Average Approximation)** for the following reasons:

1. **Theoretically grounded**: Based on actual causal inference variance
2. **Computationally efficient**: No need for trial-by-trial calculation
3. **Captures key effects**: Incorporates both modality reliabilities and integration tendency
4. **Realistic**: More accurate than current ad-hoc formula

## Current vs. Proposed

**Current (incorrect):**
```python
sigma_decision = np.sqrt(sigmaAV_A**2 + sigmaAV_V**2) / 2
```

**Proposed (theoretically correct):**
```python
var_c1 = 1 / (1/sigmaAV_A**2 + 1/sigmaAV_V**2)  # Fusion variance
var_c2 = sigmaAV_A**2  # Auditory-only variance
var_estimate = p_c * var_c1 + (1 - p_c) * var_c2  # Expected variance
sigma_decision = np.sqrt(2 * var_estimate)  # Decision noise
```

The proposed formula correctly accounts for:
- Optimal fusion when modalities are integrated
- Auditory-only noise when modalities are segregated
- The probability of integration (p_common)
- The factor of √2 for comparing two independent estimates
