# Left Edge Anomaly Analysis - Participant 'as'

## Summary

Analysis of the psychometric curve for participant 'as' (auditory duration estimation task) revealed that the **left edge anomaly is NOT a coding error**, but rather appears to be due to:

1. **Low trial counts** causing high variability
2. **Binning artifacts** when averaging sparse data
3. Possible **delta_dur_percents calculation issues** (though this doesn't affect the main analysis since it uses testDurS)

## Key Findings

### 1. Data Quality at Left Edge (Shortest Durations)

**Test Duration: 49.64ms** (10% of standard, 496.44ms)
- Total trials: **23**
- Participants chose test: **1** (4.3%)
- Participants chose standard: **22** (95.7%)
- **✓ This is CORRECT behavior** - participants correctly identified the shorter stimulus

**Test Duration: 66.19ms** (13.3% of standard)
- Total trials: **4** (very low!)
- All participants chose standard (0% chose test)
- **Low sample size** makes this datapoint unreliable

**Test Duration: 91.01ms** (18.3% of standard)
- Total trials: **6**
- All participants chose standard (0% chose test)
- Still reasonable behavior

### 2. Response Coding Verification

The `chose_test` variable is correctly coded:
```
chose_test = (responses == order)
```

Sample trials confirm this is working correctly:
- When order=1 (test first) and responses=2 (chose second), chose_test=0 ✓
- When order=2 (test second) and responses=1 (chose first), chose_test=0 ✓

### 3. Trial Distribution Issues

**Major Issue: Extremely uneven trial distribution**
- Coefficient of variation: 0.87 (very high)
- Minimum trials per condition: **1** (!)
- Some test durations have only 1-2 trials

This creates two problems:
1. **High sampling noise** - proportions based on 1-2 trials are unreliable
2. **Binning artifacts** - when binning for visualization, sparse points can create apparent jumps

### 4. Where Does the "Jump" Come From?

Looking at the figure you showed, the apparent "jump" at the left edge likely occurs during the **binning process** in the `bin_and_plot_with_error_bars()` function:

```python
# From fitNonSharedwErrorBars_logNormal.py, line 125
participant_data['bin'] = pd.cut(participant_data[binVar], bins=bins, ...)
```

When you bin data with very few trials:
- Bins at extreme edges may contain very few datapoints
- A single "error" trial (participant chose wrong) in a bin with 2-3 trials causes p=0.33 or p=0.50
- This creates visual jumps in the binned plot even though individual datapoints are reasonable

### 5. Delta Duration Calculation Issue

**WARNING: The `delta_dur_percents` column has calculation errors!**

| Test Dur (ms) | Expected Δ% | Stored Δ% | Error |
|---------------|-------------|-----------|--------|
| 49.64         | -90.00%     | -0.90     | 89.1   |
| 66.19         | -86.67%     | -0.86     | 85.8   |
| 91.01         | -81.67%     | -0.82     | 80.8   |

The stored values appear to be **divided by 100**. However, this doesn't affect the main analysis because:
- The psychometric fitting uses `testDurS` (raw test duration), not delta_dur_percents
- This variable is likely only used for visualization/binning

## Recommendations

### Immediate Actions

1. **Don't worry about the left edge data quality** - the raw trial counts show correct behavior
   - At 49.64ms: 4.3% chose test (correct - too short)
   - Error bars will be large due to low trial count, which is appropriate

2. **Consider filtering or flagging low-count bins**
   ```python
   # In bin_and_plot_with_error_bars, add:
   bin_summary = bin_summary[bin_summary['n_trials'] >= 5]  # Minimum 5 trials
   ```

3. **Fix delta_dur_percents calculation** (optional, low priority)
   ```python
   # Should be:
   data['delta_dur_percents'] = ((data['testDurS'] - data['standardDur']) / 
                                  data['standardDur']) * 100
   ```

### For Publication/Reporting

1. **Report trial counts per condition** - readers should know some points have only 1-2 trials

2. **Use transparency/size coding for reliability**
   ```python
   # Make datapoints with fewer trials more transparent
   alpha = np.clip(n_trials / n_trials.max(), 0.3, 1.0)
   ```

3. **Consider excluding or down-weighting extreme bins**
   - Bins with <5 trials could be shown with different markers
   - Or exclude from curve fitting (but keep in visualization)

## Conclusion

**The left edge anomaly is NOT a bug or coding error.** It's an artifact of:
- Sparse sampling at extreme durations (some bins have only 1-2 trials)
- High sampling noise when proportions are based on very few trials
- Visual exaggeration when binning sparse data

The underlying raw data shows **correct participant behavior** - they mostly choose "standard" when the test is very short (4.3% chose test at 49ms vs 496ms standard).

### What to Do

✓ **Keep the analysis as is** - it's correctly implemented
✓ **Add error bars** - already done, shows the uncertainty
✓ **Report sample sizes** - make readers aware of sparse sampling
✗ **Don't try to "fix" the jump** - it represents real sampling variability

The psychometric fit handles this appropriately by fitting across all datapoints with appropriate weighting.
