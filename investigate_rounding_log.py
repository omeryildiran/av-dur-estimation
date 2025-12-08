"""
Investigate potential issues with rounding and log transformations
that could cause anomalies in the psychometric data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data/as_auditory.csv")

# Minimal preprocessing
data = data[data['audNoise'] != 0]
data = data[data['standardDur'] != 0]
data["testDurMs"] = data["testDurS"] * 1000
data['responses'] = data['responses'].astype(int)
data['order'] = data['order'].astype(int)
data['chose_test'] = (data['responses'] == data['order']).astype(int)

print("="*80)
print("INVESTIGATING ROUNDING AND LOG TRANSFORMATION ISSUES")
print("="*80)

# Filter for low noise condition
condition_data = data[data['audNoise'].round(2) == 0.1].copy()

# Focus on testDurS < 0.2s as you mentioned
short_dur_data = condition_data[condition_data['testDurS'] < 0.2].copy()

print(f"\n1. DATA WITH testDurS < 0.2s:")
print(f"   Total trials: {len(short_dur_data)}")
print(f"   Chose test: {short_dur_data['chose_test'].sum()} ({short_dur_data['chose_test'].mean()*100:.1f}%)")
print(f"   Chose standard: {(1-short_dur_data['chose_test']).sum()} ({(1-short_dur_data['chose_test']).mean()*100:.1f}%)")

# Check for rounding issues
print("\n" + "="*80)
print("2. ROUNDING INVESTIGATION")
print("="*80)

# Look at raw values before any rounding
print("\nRaw testDurS values (first 20):")
print(short_dur_data['testDurS'].head(20).values)

print("\nRaw standardDur values (unique):")
print(condition_data['standardDur'].unique())

# Check if rounding is causing test and standard to become equal
short_dur_data['testDur_rounded2'] = short_dur_data['testDurS'].round(2)
short_dur_data['standardDur_rounded2'] = short_dur_data['standardDur'].round(2)
short_dur_data['appears_equal'] = (short_dur_data['testDur_rounded2'] == short_dur_data['standardDur_rounded2'])

print(f"\nAfter rounding to 2 decimals:")
print(f"  Cases where test ≈ standard: {short_dur_data['appears_equal'].sum()}")

if short_dur_data['appears_equal'].sum() > 0:
    print("\n  ⚠️ WARNING: Rounding makes some test durations equal to standard!")
    equal_cases = short_dur_data[short_dur_data['appears_equal']]
    print(f"  These {len(equal_cases)} trials might be problematic:")
    print(equal_cases[['testDurS', 'testDur_rounded2', 'standardDur', 'standardDur_rounded2', 'chose_test']].head(10))

# Check for log transformation issues
print("\n" + "="*80)
print("3. LOG TRANSFORMATION INVESTIGATION")
print("="*80)

# Calculate log ratios
short_dur_data['log_test'] = np.log(short_dur_data['testDurS'])
short_dur_data['log_standard'] = np.log(short_dur_data['standardDur'])
short_dur_data['log_ratio'] = np.log(short_dur_data['testDurS'] / short_dur_data['standardDur'])

print("\nLog transformation values:")
print(f"  log(test) range: {short_dur_data['log_test'].min():.3f} to {short_dur_data['log_test'].max():.3f}")
print(f"  log(standard) range: {short_dur_data['log_standard'].min():.3f} to {short_dur_data['log_standard'].max():.3f}")
print(f"  log(test/standard) range: {short_dur_data['log_ratio'].min():.3f} to {short_dur_data['log_ratio'].max():.3f}")

# Check for any NaN or inf values after log
print(f"\n  NaN values in log_ratio: {short_dur_data['log_ratio'].isna().sum()}")
print(f"  Inf values in log_ratio: {np.isinf(short_dur_data['log_ratio']).sum()}")

# Check for potential numerical precision issues
print("\n" + "="*80)
print("4. NUMERICAL PRECISION ISSUES")
print("="*80)

# Very small durations might have precision issues
very_small = short_dur_data[short_dur_data['testDurS'] < 0.05]
print(f"\nVery small durations (< 0.05s):")
print(f"  Count: {len(very_small)}")
print(f"  Chose test: {very_small['chose_test'].sum()} ({very_small['chose_test'].mean()*100:.1f}%)")

print("\nDetailed view of very small duration trials:")
print(very_small[['testDurS', 'standardDur', 'order', 'responses', 'chose_test']].head(15))

# Check if there's a pattern with 'order' (presentation order)
print("\n" + "="*80)
print("5. ORDER EFFECT INVESTIGATION")
print("="*80)

order_effect = short_dur_data.groupby('order').agg({
    'chose_test': ['count', 'sum', 'mean']
}).round(3)
print("\nResponses by presentation order (1=test first, 2=test second):")
print(order_effect)

# Check if definition of chose_test might be inverted
print("\n" + "="*80)
print("6. RESPONSE CODING VERIFICATION")
print("="*80)

print("\nLogic check: chose_test = (responses == order)")
print("  When order=1 (test first):")
print("    - If chose first interval (responses=1) → chose_test=1 ✓")
print("    - If chose second interval (responses=2) → chose_test=0 ✓")
print("  When order=2 (test second):")
print("    - If chose first interval (responses=1) → chose_test=0 ✓")
print("    - If chose second interval (responses=2) → chose_test=1 ✓")

# But let's verify with actual data
print("\nActual samples from data:")
sample = short_dur_data[['testDurS', 'standardDur', 'order', 'responses', 'chose_test']].head(20)
print(sample.to_string())

# Critical check: at very short durations, we should have MOSTLY chose_test=0
print("\n" + "="*80)
print("7. CRITICAL SANITY CHECK")
print("="*80)

# At the SHORTEST durations, participants should nearly always choose standard
shortest_20 = short_dur_data.nsmallest(20, 'testDurS')
print(f"\nShortest 20 test durations:")
print(f"  Mean testDurS: {shortest_20['testDurS'].mean():.4f}s ({shortest_20['testDurS'].mean()*1000:.1f}ms)")
print(f"  Mean standardDur: {shortest_20['standardDur'].mean():.4f}s ({shortest_20['standardDur'].mean()*1000:.1f}ms)")
print(f"  Proportion chose test: {shortest_20['chose_test'].mean():.3f}")
print(f"  Expected: Should be close to 0 (chose standard)")

if shortest_20['chose_test'].mean() > 0.3:
    print("\n  ⚠️  ANOMALY: More than 30% chose test at shortest durations!")
    print("  This suggests a potential issue with the response coding or data.")

# Check for specific problematic trials
print("\n" + "="*80)
print("8. IDENTIFY PROBLEMATIC TRIALS")
print("="*80)

# Find trials where test << standard but participant chose test
problematic = short_dur_data[
    (short_dur_data['testDurS'] < 0.1) & 
    (short_dur_data['chose_test'] == 1)
]

print(f"\nTrials where test < 0.1s BUT participant chose test:")
print(f"  Count: {len(problematic)}")
if len(problematic) > 0:
    print("\nDetails of these problematic trials:")
    print(problematic[['testDurS', 'testDurMs', 'standardDur', 'order', 'responses', 'chose_test', 'response_rts']].to_string())
    
    # Check if these have something in common
    print(f"\n  Mean response time: {problematic['response_rts'].mean():.3f}s")
    print(f"  Order distribution: {problematic['order'].value_counts().to_dict()}")

# Visual comparison
print("\n" + "="*80)
print("9. CREATING DIAGNOSTIC PLOT")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw test durations vs chose_test
ax1 = axes[0, 0]
ax1.scatter(short_dur_data['testDurMs'], short_dur_data['chose_test'], 
           alpha=0.3, s=30, c='blue')
ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax1.axvline(short_dur_data['standardDur'].iloc[0]*1000, color='cyan', linestyle='--', alpha=0.5)
ax1.set_xlabel('Test Duration (ms)')
ax1.set_ylabel('Chose Test (0 or 1)')
ax1.set_title('Raw Binary Responses')
ax1.grid(True, alpha=0.3)

# Plot 2: Log-space view
ax2 = axes[0, 1]
ax2.scatter(short_dur_data['log_ratio'], short_dur_data['chose_test'], 
           alpha=0.3, s=30, c='green')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax2.axvline(0, color='cyan', linestyle='--', alpha=0.5, label='log(test/std)=0')
ax2.set_xlabel('log(test/standard)')
ax2.set_ylabel('Chose Test (0 or 1)')
ax2.set_title('Log-Space Representation')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Binned proportions
ax3 = axes[1, 0]
bins = np.linspace(0, 200, 11)
short_dur_data['bin'] = pd.cut(short_dur_data['testDurMs'], bins=bins, labels=False)
binned = short_dur_data.groupby('bin').agg({
    'testDurMs': 'mean',
    'chose_test': ['mean', 'count']
}).dropna()
binned.columns = ['testDurMs_mean', 'prop_chose_test', 'n_trials']

ax3.plot(binned['testDurMs_mean'], binned['prop_chose_test'], 'o-', markersize=8, linewidth=2)
for idx, row in binned.iterrows():
    ax3.text(row['testDurMs_mean'], row['prop_chose_test'] + 0.05, 
            f"n={int(row['n_trials'])}", ha='center', fontsize=8, color='red')
ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('Test Duration (ms)')
ax3.set_ylabel('Proportion Chose Test')
ax3.set_title('Binned Proportions with Trial Counts')
ax3.set_ylim([-0.1, 1.1])
ax3.grid(True, alpha=0.3)

# Plot 4: Response distribution by order
ax4 = axes[1, 1]
for order_val in [1, 2]:
    order_data = short_dur_data[short_dur_data['order'] == order_val]
    binned_order = order_data.groupby(pd.cut(order_data['testDurMs'], bins=bins, labels=False)).agg({
        'testDurMs': 'mean',
        'chose_test': 'mean'
    }).dropna()
    ax4.plot(binned_order['testDurMs'], binned_order['chose_test'], 
            'o-', label=f'Order={order_val}', markersize=6, linewidth=2, alpha=0.7)
ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('Test Duration (ms)')
ax4.set_ylabel('Proportion Chose Test')
ax4.set_title('By Presentation Order (potential bias?)')
ax4.legend()
ax4.set_ylim([-0.1, 1.1])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_rounding_log_issues.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: diagnostic_rounding_log_issues.png")
plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf you're seeing 80% correct at very short durations, the code is likely")
print("working correctly. The 'jump' in the binned plot may be due to:")
print("  1. Sparse sampling creating noisy bins")
print("  2. A few 'error' trials being amplified by binning")
print("  3. Order effects or response biases")
print("\nCheck the plots above to see if there are any systematic patterns.")
print("="*80)
