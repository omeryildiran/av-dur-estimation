# Error Bars Across Participants - Usage Guide

## Overview

I've enhanced your `fitNonShared.py` script to plot error bars that represent variability across participants (participantID). This gives you a better understanding of individual differences in psychometric performance.

## What's New

### 1. New Functions Added

- **`groupByChooseTestWithParticipants(data)`**: Groups data by participant AND experimental conditions
- **`bin_and_plot_with_error_bars(data, ...)`**: Creates plots with error bars across participants
- **Enhanced `bin_and_plot(data, ..., add_error_bars=True)`**: Automatically detects participant data and adds error bars

### 2. Error Bar Calculation

The error bars represent the **standard error of the mean (SEM)** across participants:
- For each intensity bin, the script calculates each participant's proportion of "chose test" responses
- Then computes the mean and SEM across all participants for that bin
- Error bars show: mean ± SEM

## How to Use

### Option 1: Automatic Error Bars (Recommended)

Your existing code will automatically show error bars if your data contains a `participantID` column:

```python
# Your existing code - now with error bars!
from fitNonShared import *

# Load data (must contain 'participantID' column)
data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData("all_auditory.csv")

# Fit model
fit = fitMultipleStartingPoints(data, nStart=1)

# Plot with automatic error bars
plot_fitted_psychometric(data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict, standardVar, sensoryVar, conflictVar, 'delta_dur_percents')
```

### Option 2: Manual Error Bar Control

```python
# Filter your data for specific conditions
filtered_data = data[
    (data['audNoise'] == 0.1) & 
    (data['conflictDur'] == 0.0)
]

# Plot with explicit error bars
bin_summary = bin_and_plot_with_error_bars(
    filtered_data, 
    bin_method='cut', 
    bins=10, 
    plot=True, 
    color='red'
)

# The bin_summary contains useful statistics:
print(bin_summary[['x_mean', 'y_mean', 'y_sem', 'n_participants']])
```

### Option 3: Disable Error Bars

```python
# If you want the old behavior (no error bars)
bin_and_plot(data, add_error_bars=False)
```

## What the Error Bars Show

- **Error bar length**: Standard error of the mean across participants
- **Marker position**: Mean proportion across all participants  
- **Text annotation**: Shows number of participants contributing to each condition
- **Larger error bars**: More variability between participants
- **Smaller error bars**: More consistent performance across participants

## Example Output

The enhanced plots will show:
1. **Fitted psychometric curves** (as before)
2. **Data points with error bars** showing participant variability
3. **Participant count annotation** (e.g., "n = 8 participants")
4. **Error bars** representing SEM across participants

## Running the Demo

To see the new functionality in action:

```bash
python demo_error_bars.py
```

This will show:
- Comparison between error bar plots and traditional plots
- Individual participant data
- Summary statistics

## Data Requirements

Your CSV data must contain:
- `participantID` column (string or numeric)
- Standard psychometric experiment columns (`delta_dur_percents`, `responses`, etc.)

## Benefits

1. **Better visualization of individual differences**: See how much participants vary
2. **Statistical rigor**: Error bars show uncertainty in your measurements  
3. **Publication quality**: Standard scientific visualization practice
4. **Easy interpretation**: Smaller error bars = more reliable/consistent results

## Technical Details

- **Error calculation**: SEM = std(participants) / sqrt(n_participants)
- **Binning**: First bins by intensity, then calculates participant statistics within each bin
- **Fallback**: If no `participantID` found, reverts to original plotting behavior
- **Performance**: Optimized to handle large datasets efficiently

## Troubleshooting

**Q: I don't see error bars**
- Check that your data contains a `participantID` column
- Ensure you have multiple participants per condition
- Try: `print(data['participantID'].unique())` to verify participant data

**Q: Error bars are too large/small**
- Large bars = high inter-participant variability (this is informative!)
- Small bars = consistent performance across participants
- This is real data, not a problem to fix

**Q: Some bins have no error bars**
- This happens when only one participant contributed data to that bin
- Consider using fewer bins or collecting more data

## Example Analysis Workflow

```python
# 1. Load and examine data
data, *_ = loadData("your_data.csv")
print(f"Participants: {data['participantID'].nunique()}")

# 2. Fit model 
fit = fitMultipleStartingPoints(data, nStart=1)

# 3. Plot with error bars (automatic)
plot_fitted_psychometric(data, fit, ...)

# 4. Analyze participant variability
participant_data = groupByChooseTestWithParticipants(data)
print("Per-participant summary:")
print(participant_data.groupby('participantID')['p_choose_test'].describe())
```

The error bars will help you understand not just the average psychometric performance, but also how reliable and consistent that performance is across your participant sample.
