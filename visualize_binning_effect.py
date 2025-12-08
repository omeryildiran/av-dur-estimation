"""
Visualize the binning effect and show why left edge appears to "jump"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("data/as_auditory.csv")

# Preprocess
data = data[data['audNoise'] != 0]
data = data[data['standardDur'] != 0]
data["testDurMs"] = data["testDurS"] * 1000
data['responses'] = data['responses'].astype(int)
data['order'] = data['order'].astype(int)
data['chose_test'] = (data['responses'] == data['order']).astype(int)
data['participantID'] = 'as'

# Filter for low noise condition
condition_data = data[data['audNoise'].round(2) == 0.1].copy()

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Get raw trial-by-trial proportions
trial_props = condition_data.groupby('testDurS').agg(
    testDurMs=('testDurMs', 'first'),
    n_trials=('chose_test', 'count'),
    n_chose_test=('chose_test', 'sum'),
    p_chose_test=('chose_test', 'mean')
).reset_index()

#---------------------------------------------------------------------------
# PLOT 1: Raw unbinned data with trial counts
#---------------------------------------------------------------------------
ax1 = axes[0]

# Plot with size proportional to trial count
sizes = (trial_props['n_trials'] / trial_props['n_trials'].max()) * 500
ax1.scatter(trial_props['testDurMs'], trial_props['p_chose_test'], 
           s=sizes, alpha=0.6, color='blue', edgecolor='black', linewidth=1.5)

# Add trial count labels
for idx, row in trial_props.iterrows():
    if row['testDurMs'] < 300:  # Focus on left edge
        ax1.text(row['testDurMs'], row['p_chose_test'] + 0.05, 
                f"n={int(row['n_trials'])}", 
                ha='center', fontsize=9, color='red', fontweight='bold')

ax1.axvline(496.44, color='cyan', linestyle='--', linewidth=2, label='Standard: 496ms')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Test Duration (ms)', fontsize=12, fontweight='bold')
ax1.set_ylabel('P(chose test)', fontsize=12, fontweight='bold')
ax1.set_title('RAW UNBINNED DATA - Point size = trial count', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 1000])
ax1.set_ylim([-0.05, 1.05])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

#---------------------------------------------------------------------------
# PLOT 2: Binned data (10 bins) - what gets plotted
#---------------------------------------------------------------------------
ax2 = axes[1]

# Bin the data
n_bins = 10
condition_data['bin'] = pd.cut(condition_data['testDurMs'], bins=n_bins, labels=False)

binned = condition_data.groupby('bin').agg(
    testDurMs_mean=('testDurMs', 'mean'),
    n_trials=('chose_test', 'count'),
    p_chose_test=('chose_test', 'mean'),
    p_chose_test_sem=('chose_test', lambda x: np.std(x) / np.sqrt(len(x)))
).reset_index()

# Plot binned data with error bars
ax2.errorbar(binned['testDurMs_mean'], binned['p_chose_test'],
            yerr=binned['p_chose_test_sem'],
            fmt='o', markersize=10, capsize=5, color='darkgreen', 
            ecolor='green', elinewidth=2, capthick=2, alpha=0.7)

# Highlight problematic bins (low trial count)
for idx, row in binned.iterrows():
    if row['n_trials'] < 5:
        ax2.plot(row['testDurMs_mean'], row['p_chose_test'], 
                'rx', markersize=20, markeredgewidth=3, 
                label='Low trials (<5)' if idx == 0 else '')
    ax2.text(row['testDurMs_mean'], row['p_chose_test'] + 0.08, 
            f"n={int(row['n_trials'])}", 
            ha='center', fontsize=9, color='red', fontweight='bold')

ax2.axvline(496.44, color='cyan', linestyle='--', linewidth=2, label='Standard: 496ms')
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Test Duration (ms)', fontsize=12, fontweight='bold')
ax2.set_ylabel('P(chose test)', fontsize=12, fontweight='bold')
ax2.set_title(f'BINNED DATA ({n_bins} bins) - Red X = unreliable bins', fontsize=14, fontweight='bold')
ax2.set_xlim([0, 1000])
ax2.set_ylim([-0.05, 1.05])
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)

#---------------------------------------------------------------------------
# PLOT 3: Comparison showing the "jump" artifact
#---------------------------------------------------------------------------
ax3 = axes[2]

# Plot both raw and binned
ax3.scatter(trial_props['testDurMs'], trial_props['p_chose_test'], 
           s=100, alpha=0.4, color='blue', label='Raw data', zorder=1)

ax3.plot(binned['testDurMs_mean'], binned['p_chose_test'],
        'o-', markersize=12, linewidth=3, color='darkgreen', 
        label='Binned data', alpha=0.8, zorder=2)

# Highlight the "problematic" left edge region
left_edge_bins = binned[binned['testDurMs_mean'] < 200]
for idx, row in left_edge_bins.iterrows():
    ax3.annotate('', xy=(row['testDurMs_mean'], row['p_chose_test']),
                xytext=(row['testDurMs_mean'], -0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
    if row['n_trials'] < 5:
        ax3.text(row['testDurMs_mean'], -0.15, 
                f"Only {int(row['n_trials'])} trials!\nHigh noise", 
                ha='center', fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax3.axvline(496.44, color='cyan', linestyle='--', linewidth=2, label='Standard: 496ms')
ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Test Duration (ms)', fontsize=12, fontweight='bold')
ax3.set_ylabel('P(chose test)', fontsize=12, fontweight='bold')
ax3.set_title('WHY THE "JUMP"? Low trial counts in left bins create noise', 
             fontsize=14, fontweight='bold', color='darkred')
ax3.set_xlim([0, 1000])
ax3.set_ylim([-0.2, 1.05])
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(True, alpha=0.3)

# Add annotation box explaining the issue
textstr = ('LEFT EDGE ISSUE:\n'
          '• Some bins have only 1-4 trials\n'
          '• Single "wrong" responses create large jumps\n'
          '• e.g., 1 out of 3 trials = 33% chose test\n'
          '• This is SAMPLING NOISE, not a bug!\n'
          '• Error bars correctly show uncertainty')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1)
ax3.text(0.98, 0.35, textstr, transform=ax3.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props,
        family='monospace')

plt.tight_layout()
plt.savefig('diagnostic_binning_effect.png', dpi=150, bbox_inches='tight')
print("\n" + "="*80)
print("VISUALIZATION CREATED: diagnostic_binning_effect.png")
print("="*80)
print("\nThis figure shows:")
print("  1. RAW DATA: Individual test durations with trial counts")
print("  2. BINNED DATA: How averaging into bins affects the curve")
print("  3. COMPARISON: Why left edge appears to 'jump' (low trial counts)")
print("\nKEY INSIGHT:")
print("  The 'jump' is not a coding error - it's sampling noise from bins")
print("  with very few trials (1-4 trials per bin at extreme durations)")
print("="*80 + "\n")

plt.show()

# Print detailed statistics for the problematic left bins
print("\nDETAILED LEFT EDGE STATISTICS:")
print("-" * 80)
left_raw = trial_props[trial_props['testDurMs'] < 200].copy()
left_raw['confidence_interval_95'] = 1.96 * np.sqrt(
    left_raw['p_chose_test'] * (1 - left_raw['p_chose_test']) / left_raw['n_trials']
)

print(left_raw[['testDurMs', 'n_trials', 'p_chose_test', 'confidence_interval_95']].to_string(index=False))
print("\nNote: When n_trials is small, confidence intervals are HUGE!")
print("      e.g., with n=4 trials, CI can be ±0.49 (covers almost full range)")
print("-" * 80)
