# Causal Inference Implementation Fix

## The Problem You Identified

You were absolutely correct to question the implementation! The original `causalInference` function had a fundamental conceptual error:

```python
# WRONG APPROACH
m_a, p_m_a = unimodalLikelihood(S_a, sigmaAV_A)  # Returns arrays!
m_v, p_m_v = unimodalLikelihood(S_v, sigmaAV_V)  # Returns arrays!
likelihood_c1 = likelihood_C1(m_a, m_v, sigmaAV_A, sigmaAV_V)  # Can't work with arrays
```

### What was wrong:
- `unimodalLikelihood()` returns **arrays** of possible measurement values and their probabilities
- `likelihood_C1()` expects **single measurement values**, not arrays
- You can't compute a meaningful likelihood for Bayes' rule using probability distributions as inputs

## The Corrected Implementation

```python
# CORRECT APPROACH
def causalInference(sigmaAV_A, sigmaAV_V, S_a, p_c, visualConflict):
    S_v = S_a + visualConflict

    # Generate actual noisy measurements (single values)
    m_a = np.random.normal(S_a, sigmaAV_A)  # Single measurement
    m_v = np.random.normal(S_v, sigmaAV_V)  # Single measurement
    
    # Compute likelihoods under each causal structure
    likelihood_c1 = likelihood_C1(m_a, m_v, sigmaAV_A, sigmaAV_V)
    likelihood_c2 = likelihood_C2(m_a, m_v, S_a, visualConflict, sigmaAV_A, sigmaAV_V)

    # Bayes' rule for posterior probability
    posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
    
    # Model averaging
    fused_estimate, _ = fusionAV(sigmaAV_A, sigmaAV_V, S_a, visualConflict)
    auditory_only = m_a
    
    final_estimate = posterior_c1 * fused_estimate + (1 - posterior_c1) * auditory_only
    
    return final_estimate, posterior_c1, m_a, m_v
```

## Key Conceptual Points

### 1. **Measurements vs Likelihood Functions**
- **Measurements**: Single noisy values that an observer actually receives
- **Likelihood functions**: Mathematical descriptions of how likely different measurements are

### 2. **What Causal Inference Does**
- Observer receives specific noisy measurements `m_a` and `m_v`
- Asks: "Given these specific values, what's P(common cause)?"
- Uses Bayes' rule with the actual measurements as evidence

### 3. **Why Arrays Don't Work**
- Causal inference is about **specific evidence** on each trial
- Observer doesn't have access to entire probability distributions
- The decision depends on how similar/different the actual measurements are

## Behavioral Implications

### With Single Measurements (Correct):
- P(common cause) varies trial-by-trial based on measurement similarity
- When `m_a ≈ m_v`: Higher P(common cause) → more fusion
- When `m_a ≠ m_v`: Lower P(common cause) → more segregation
- Creates realistic behavioral patterns

### With Arrays (Incorrect):
- Cannot compute meaningful likelihoods
- No trial-by-trial variation in causal inference
- Mathematically inconsistent
- Doesn't reflect actual perceptual process

## Usage Examples

### For Single Trial Simulation:
```python
estimate, p_common, m_a, m_v = causalInference(
    sigmaAV_A=0.2, sigmaAV_V=0.3, S_a=0.5, p_c=0.7, visualConflict=0.2
)
print(f"Measurements: m_a={m_a:.3f}, m_v={m_v:.3f}")
print(f"P(common cause) = {p_common:.3f}")
print(f"Final estimate = {estimate:.3f}")
```

### For Psychometric Function:
```python
def causal_inference_psychometric(delta_dur, params, conflict):
    lambda_, mu, sigma_a, sigma_v, p_c = params
    
    # Simulate many trials and average
    estimates = []
    for _ in range(100):  # Average over trials to reduce noise
        S_test = baseline + delta_dur + mu
        estimate, _, _, _ = causalInference(sigma_a, sigma_v, S_test, p_c, conflict)
        estimates.append(estimate)
    
    # Convert to choice probability
    # ... (rest of psychometric function logic)
```

## Files Updated

1. **`fitCausalInference.py`**: Fixed the `causalInference()` function
2. **`test_causal_inference_fix.py`**: Demonstration script showing the difference
3. This documentation file

## Next Steps

1. **Test the corrected implementation** with your data
2. **Compare psychometric fits** between standard and causal inference models
3. **Interpret the P(common cause) parameter** - this tells you about audiovisual integration strength
4. **Consider trial averaging** for smoother psychometric functions if needed

The corrected implementation now properly reflects the computational principles of causal inference in multisensory perception!
