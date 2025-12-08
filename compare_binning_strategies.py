"""
Test if better binning strategy reduces visual artifacts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("data/as_auditory.csv")
data = data[data['audNoise'] != 0]
data = data[data['standardDur'] != 0]
data["testDurMs"] = data["testDurS"] * 1000
data['responses'] = data['responses'].astype(int)
data['order'] = data['order'].astype(int)
data['chose_test'] = (data['responses'] == data['order']).astype(int)
data['participantID'] = 'as'

# Filter for low noise condition
condition_data = data[data['audNoise'].round(2) == 0.1].copy()

print("="*80)
print("COMPARING BINNING STRATEGIES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#---------------------------------------------------------------------------
# Strategy 1: Equal-width bins (current approach)
#---------------------------------------------------------------------------
ax1 = axes[0, 0]
n_bins = 10
condition_data['bin_equal'] = pd.cut(condition_data['testDurMs'], bins=n_bins, labels=False)
binned_equal = condition_data.groupby('bin_equal').agg(
    testDurMs_mean=('testDurMs', 'mean'),
    n_trials=('chose_test', 'count'),
    p_chose_test=('chose_test', 'mean')
).reset_index()

ax1.errorbar(binned_equal['testDurMs_mean'], binned_equal['p_chose_test'],
            fmt='o-', markersize=8, color='blue', linewidth=2, alpha=0.7)
for idx, row in binned_equal.iterrows():
    ax1.text(row['testDurMs_mean'], row['p_chose_test'] + 0.06, 
            f"n={int(row['n_trials'])}", ha='center', fontsize=8, color='red')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax1.set_xlabel('Test Duration (ms)')
ax1.set_ylabel('P(chose test)')
ax1.set_title('Strategy 1: Equal-Width Bins (CURRENT)\n→ Poor: uneven trial distribution')
ax1.set_ylim([-0.1, 1.1])
ax1.grid(True, alpha=0.3)

#---------------------------------------------------------------------------
# Strategy 2: Equal-count bins (quantile-based)
#---------------------------------------------------------------------------
ax2 = axes[0, 1]
condition_data['bin_quantile'] = pd.qcut(condition_data['testDurMs'], q=n_bins, labels=False, duplicates='drop')
binned_quantile = condition_data.groupby('bin_quantile').agg(
    testDurMs_mean=('testDurMs', 'mean'),
    n_trials=('chose_test', 'count'),
    p_chose_test=('chose_test', 'mean')
).reset_index()

ax2.errorbar(binned_quantile['testDurMs_mean'], binned_quantile['p_chose_test'],
            fmt='o-', markersize=8, color='green', linewidth=2, alpha=0.7)
for idx, row in binned_quantile.iterrows():
    ax2.text(row['testDurMs_mean'], row['p_chose_test'] + 0.06, 
            f"n={int(row['n_trials'])}", ha='center', fontsize=8, color='red')
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax2.set_xlabel('Test Duration (ms)')
ax2.set_ylabel('P(chose test)')
ax2.set_title('Strategy 2: Equal-Count Bins (BETTER)\n→ Good: balanced trial distribution')
ax2.set_ylim([-0.1, 1.1])
ax2.grid(True, alpha=0.3)

#---------------------------------------------------------------------------
# Strategy 3: Adaptive bins (more bins where data is dense)
#---------------------------------------------------------------------------
ax3 = axes[1, 0]
# Create custom bin edges with more bins at extremes
test_durs = condition_data['testDurMs'].values
bin_edges = np.percentile(test_durs, np.linspace(0, 100, n_bins+1))
condition_data['bin_adaptive'] = pd.cut(condition_data['testDurMs'], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
binned_adaptive = condition_data.groupby('bin_adaptive').agg(
    testDurMs_mean=('testDurMs', 'mean'),
    n_trials=('chose_test', 'count'),
    p_chose_test=('chose_test', 'mean')
).reset_index()

ax3.errorbar(binned_adaptive['testDurMs_mean'], binned_adaptive['p_chose_test'],
            fmt='o-', markersize=8, color='purple', linewidth=2, alpha=0.7)
for idx, row in binned_adaptive.iterrows():
    ax3.text(row['testDurMs_mean'], row['p_chose_test'] + 0.06, 
            f"n={int(row['n_trials'])}", ha='center', fontsize=8, color='red')
ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax3.set_xlabel('Test Duration (ms)')
ax3.set_ylabel('P(chose test)')
ax3.set_title('Strategy 3: Adaptive Bins (Percentile)\n→ Good: balanced and adaptive')
ax3.set_ylim([-0.1, 1.1])
ax3.grid(True, alpha=0.3)

#---------------------------------------------------------------------------
# Strategy 4: No binning - plot raw grouped data
#---------------------------------------------------------------------------
ax4 = axes[1, 1]
raw_grouped = condition_data.groupby('testDurS').agg(
    testDurMs=('testDurMs', 'first'),
    n_trials=('chose_test', 'count'),
    p_chose_test=('chose_test', 'mean')
).reset_index()

# Size points by trial count
sizes = (raw_grouped['n_trials'] / raw_grouped['n_trials'].max()) * 300
ax4.scatter(raw_grouped['testDurMs'], raw_grouped['p_chose_test'],
           s=sizes, alpha=0.6, color='orange', edgecolor='black', linewidth=1)
ax4.plot(raw_grouped['testDurMs'], raw_grouped['p_chose_test'],
        '-', alpha=0.3, color='orange', linewidth=1)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax4.set_xlabel('Test Duration (ms)')
ax4.set_ylabel('P(chose test)')
ax4.set_title('Strategy 4: No Binning (RAW DATA)\n→ Best: shows actual data, size=trial count')
ax4.set_ylim([-0.1, 1.1])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compare_binning_strategies.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: compare_binning_strategies.png")
plt.show()

# Print comparison
print("\n" + "="*80)
print("BINNING STRATEGY COMPARISON")
print("="*80)

print("\n1. EQUAL-WIDTH BINS (current):")
print(f"   Min trials per bin: {binned_equal['n_trials'].min()}")
print(f"   Max trials per bin: {binned_equal['n_trials'].max()}")
print(f"   CV (variability): {binned_equal['n_trials'].std() / binned_equal['n_trials'].mean():.2f}")

print("\n2. EQUAL-COUNT BINS (quantile):")
print(f"   Min trials per bin: {binned_quantile['n_trials'].min()}")
print(f"   Max trials per bin: {binned_quantile['n_trials'].max()}")
print(f"   CV (variability): {binned_quantile['n_trials'].std() / binned_quantile['n_trials'].mean():.2f}")

print("\n3. ADAPTIVE BINS:")
print(f"   Min trials per bin: {binned_adaptive['n_trials'].min()}")
print(f"   Max trials per bin: {binned_adaptive['n_trials'].max()}")
print(f"   CV (variability): {binned_adaptive['n_trials'].std() / binned_adaptive['n_trials'].mean():.2f}")

print("\n4. NO BINNING:")
print(f"   Total unique test durations: {len(raw_grouped)}")
print(f"   Min trials per duration: {raw_grouped['n_trials'].min()}")
print(f"   Max trials per duration: {raw_grouped['n_trials'].max()}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("Use Strategy 2 (quantile/qcut) or Strategy 4 (no binning)")
print("These avoid the visual artifacts from unbalanced bins")
print("="*80)
