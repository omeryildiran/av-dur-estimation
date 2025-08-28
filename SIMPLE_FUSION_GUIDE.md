# Simple Fusion Model - Quick Start Guide

## 🎯 Overview

The `SimpleFusionModel` is an easy-to-understand audiovisual fusion model that uses clear **if-else statements** to decide how to combine audio and visual information based on:

- **Conflict level** between audio and visual signals
- **Reliability** of each modality (inverse of noise)
- **Fusion weight** parameter (learned from data)

## 🧠 Decision Logic

The model follows this simple decision tree:

```
1. IF conflict < threshold:
   → Use optimal fusion (reliability-weighted average)

2. ELIF audio much more reliable:
   → Use audio only

3. ELIF visual much more reliable:
   → Use visual only

4. ELSE (similar reliability + high conflict):
   → Use weighted combination of fusion and segregation
```

## 📋 Quick Usage

```python
from simpleFusionModel import SimpleFusionModel

# 1. Load your data (DataFrame with required columns)
model = SimpleFusionModel(data, dataName="my_experiment")

# 2. Fit the model
result = model.fitSimpleFusion(nStart=5)

# 3. Visualize results
model.plot_fusion_strategy(result.x)
model.compare_with_data(result.x)
```

## 📊 Required Data Columns

Your DataFrame must have these columns:
- `deltaDurS`: Duration difference (test - standard)
- `audNoise`: Audio noise level / SNR condition
- `standardDur`: Standard stimulus duration
- `conflictDur`: Conflict between audio and visual
- `testDurS`: Test stimulus duration
- `unbiasedVisualStandardDur`: Visual standard duration
- `unbiasedVisualTestDur`: Visual test duration
- `chose_test`: Binary response (1 = chose test, 0 = chose standard)
- `chose_standard`: Binary response (1 = chose standard, 0 = chose test)

## 🔧 Model Parameters

The model fits 6 parameters:

1. **`lambda`** (0-0.3): Lapse rate - random response probability
2. **`sigma_a1`** (0.05-1.0): Audio noise for low SNR condition
3. **`sigma_v1`** (0.05-1.0): Visual noise for low SNR condition  
4. **`fusion_weight`** (0-1): **Key parameter** - fusion vs segregation
   - 0 = Always segregate (use single modality)
   - 1 = Always fuse (combine modalities)
   - 0.5 = Balanced strategy
5. **`sigma_a2`** (0.05-1.5): Audio noise for high SNR condition
6. **`sigma_v2`** (0.05-1.0): Visual noise for high SNR condition

## 🎛️ Adjustable Thresholds

You can modify the decision thresholds:

```python
model.conflict_threshold = 0.05  # Lower = fuse more often
model.reliability_threshold = 3.0  # Higher = require stronger dominance
```

## 📈 Interpretation Guide

### Fusion Weight Parameter:
- **High (>0.7)**: "Integrator" - prefers to combine audio/visual
- **Medium (0.3-0.7)**: "Flexible" - adapts based on conflict
- **Low (<0.3)**: "Segregator" - prefers single modality

### Noise Parameters:
- **Lower values** = More precise/reliable modality
- **Higher values** = More noisy/unreliable modality
- **Ratio matters**: `sigma_a/sigma_v` determines dominance

## 🔍 Advantages vs Complex Models

| Simple Fusion Model | Complex Causal Inference |
|---------------------|---------------------------|
| ✅ Clear if-else logic | ❓ Complex probability calculations |
| ✅ Easy to debug | ❓ Hard to troubleshoot |
| ✅ Fast fitting | ❓ Slow Monte Carlo |
| ✅ Interpretable parameters | ❓ Many interdependent parameters |
| ✅ Good for exploration | ✅ Theoretically grounded |

## 🚀 Example Results

After fitting, you can interpret:

```python
# Example fitted parameters
params = [0.05, 0.15, 0.20, 0.65, 0.30, 0.20]

print(f"Lapse rate: {params[0]:.3f}")           # 0.050 = 5% random responses
print(f"Fusion weight: {params[3]:.3f}")        # 0.650 = prefers fusion
print(f"Audio reliability ratio: {(1/params[1]**2)/(1/params[4]**2):.1f}")  # How much better low vs high noise
```

## ⚡ Quick Demo

Run the demo to see it in action:

```bash
python demo_simple_fusion.py
```

This will:
- Generate synthetic data
- Fit the model
- Show decision boundaries
- Compare strategies
- Plot psychometric curves

Perfect for understanding how the model works before using your real data!
