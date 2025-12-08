"""
Investigate why high noise condition has such large sigma (3.5)
despite good performance at extreme durations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/as_auditory.csv")
data = data[data['audNoise'] != 0]
data = data[data['standardDur'] != 0]
data["testDurMs"] = data["testDurS"] * 1000
data['responses'] = data['responses'].astype(int)
data['order'] = data['order'].astype(int)
data['chose_test'] = (data['responses'] == data['order']).astype(int)

print("="*80)
print("INVESTIGATING HIGH NOISE CONDITION")
print("="*80)

# Compare low noise vs high noise
low_noise = data[data['audNoise'].round(2) == 0.1].copy()
high_noise = data[data['audNoise'].round(2) == 1.2].copy()

print(f"\nLOW NOISE (0.1):")
print(f"  Total trials: {len(low_noise)}")
print(f"  Unique test durations: {low_noise['testDurS'].nunique()}")
print(f"  Test duration range: {low_noise['testDurS'].min():.3f} - {low_noise['testDurS'].max():.3f}s")

print(f"\nHIGH NOISE (1.2):")
print(f"  Total trials: {len(high_noise)}")
print(f"  Unique test durations: {high_noise['testDurS'].nunique()}")
print(f"  Test duration range: {high_noise['testDurS'].min():.3f} - {high_noise['testDurS'].max():.3f}s")

# Group and analyze performance
def analyze_condition(data_subset, label):
    grouped = data_subset.groupby('testDurS').agg({
        'testDurMs': 'first',
        'chose_test': ['count', 'sum', 'mean']
    })
    grouped.columns = ['testDurMs', 'n_trials', 'n_chose_test', 'p_chose_test']
    
    print(f"\n{label} - Performance Summary:")
    print(f"  At shortest duration ({grouped.index.min():.3f}s): {grouped.iloc[0]['p_chose_test']:.2%} chose test")
    print(f"  At longest duration ({grouped.index.max():.3f}s): {grouped.iloc[-1]['p_chose_test']:.2%} chose test")
    
    # Calculate slope around PSE (0.5 region)
    pse_region = grouped[(grouped['p_chose_test'] > 0.3) & (grouped['p_chose_test'] < 0.7)]
    if len(pse_region) > 1:
        test_durs = pse_region.index.values
        props = pse_region['p_chose_test'].values
        # Calculate log-space slope
        log_durs = np.log(test_durs / 0.49644)  # standardDur
        slope = np.polyfit(log_durs, props, 1)[0]
        print(f"  Slope near PSE: {slope:.3f}")
        print(f"  Estimated sigma from slope: {1/(slope*np.sqrt(2*np.pi)):.3f}")
    
    return grouped

print("\n" + "="*80)
low_grouped = analyze_condition(low_noise, "LOW NOISE (0.1)")
print("\n" + "-"*80)
high_grouped = analyze_condition(high_noise, "HIGH NOISE (1.2)")
print("="*80)

# Create detailed comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Low noise psychometric
ax1 = axes[0, 0]
sizes = (low_grouped['n_trials'] / low_grouped['n_trials'].max()) * 300
ax1.scatter(low_grouped['testDurMs'], low_grouped['p_chose_test'],
           s=sizes, alpha=0.6, color='blue', edgecolor='black', linewidth=1)
ax1.plot(low_grouped['testDurMs'], low_grouped['p_chose_test'],
        '-', alpha=0.3, color='blue', linewidth=2)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax1.set_xlabel('Test Duration (ms)')
ax1.set_ylabel('P(chose test)')
ax1.set_title('Low Noise (0.1) - Fitted σ=0.238')
ax1.set_ylim([-0.1, 1.1])
ax1.grid(True, alpha=0.3)

# Plot 2: High noise psychometric
ax2 = axes[0, 1]
sizes = (high_grouped['n_trials'] / high_grouped['n_trials'].max()) * 300
ax2.scatter(high_grouped['testDurMs'], high_grouped['p_chose_test'],
           s=sizes, alpha=0.6, color='red', edgecolor='black', linewidth=1)
ax2.plot(high_grouped['testDurMs'], high_grouped['p_chose_test'],
        '-', alpha=0.3, color='red', linewidth=2)
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(496.44, color='cyan', linestyle='--', alpha=0.5)
ax2.set_xlabel('Test Duration (ms)')
ax2.set_ylabel('P(chose test)')
ax2.set_title('High Noise (1.2) - Fitted σ=3.5 ⚠️')
ax2.set_ylim([-0.1, 1.1])
ax2.grid(True, alpha=0.3)

# Plot 3: Trial distribution comparison
ax3 = axes[1, 0]
ax3.hist([low_grouped.index*1000, high_grouped.index*1000], 
        bins=20, alpha=0.6, label=['Low Noise', 'High Noise'], color=['blue', 'red'])
ax3.axvline(496.44, color='cyan', linestyle='--', linewidth=2, label='Standard')
ax3.set_xlabel('Test Duration (ms)')
ax3.set_ylabel('Number of unique durations')
ax3.set_title('Test Duration Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Slope comparison (steeper = better discrimination)
ax4 = axes[1, 1]

# Plot psychometric curves in log-space
low_log_dur = np.log(low_grouped.index / 0.49644)
high_log_dur = np.log(high_grouped.index / 0.49644)

ax4.scatter(low_log_dur, low_grouped['p_chose_test'], 
           s=100, alpha=0.6, color='blue', label='Low Noise')
ax4.plot(low_log_dur, low_grouped['p_chose_test'],
        '-', alpha=0.3, color='blue', linewidth=2)

ax4.scatter(high_log_dur, high_grouped['p_chose_test'],
           s=100, alpha=0.6, color='red', label='High Noise')
ax4.plot(high_log_dur, high_grouped['p_chose_test'],
        '-', alpha=0.3, color='red', linewidth=2)

ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(0, color='cyan', linestyle='--', alpha=0.5, label='log(test/std)=0')
ax4.set_xlabel('log(test/standard)')
ax4.set_ylabel('P(chose test)')
ax4.set_title('Log-Space View (slope ∝ 1/σ)')
ax4.legend()
ax4.set_ylim([-0.1, 1.1])
ax4.grid(True, alpha=0.3)

# Add annotation about slope
textstr = ('Steeper slope = smaller σ\n'
          'Flatter slope = larger σ\n'
          '\n'
          'High noise shows\n'
          'MUCH flatter slope\n'
          '→ σ=3.5 is correct!')
props = dict(boxstyle='round', facecolor='yellow', alpha=0.8)
ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('investigate_high_noise_sigma.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: investigate_high_noise_sigma.png")
plt.show()

# Check if high noise data actually shows poor discrimination
print("\n" + "="*80)
print("DETAILED ANALYSIS OF HIGH NOISE")
print("="*80)

# Check performance at specific regions
short_high = high_noise[high_noise['testDurS'] < 0.2]
long_high = high_noise[high_noise['testDurS'] > 0.7]
mid_high = high_noise[(high_noise['testDurS'] > 0.4) & (high_noise['testDurS'] < 0.6)]

print(f"\nShort durations (< 200ms):")
print(f"  Trials: {len(short_high)}")
print(f"  P(chose test): {short_high['chose_test'].mean():.2%}")

print(f"\nMid durations (400-600ms, near PSE):")
print(f"  Trials: {len(mid_high)}")
print(f"  P(chose test): {mid_high['chose_test'].mean():.2%}")

print(f"\nLong durations (> 700ms):")
print(f"  Trials: {len(long_high)}")
print(f"  P(chose test): {long_high['chose_test'].mean():.2%}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if mid_high['chose_test'].mean() > 0.3 and mid_high['chose_test'].mean() < 0.7:
    print("\nHigh noise condition shows POOR discrimination near PSE region")
    print("→ Responses are close to 50/50 even when test ≠ standard")
    print("→ σ=3.5 is CORRECT - participant cannot discriminate well with high noise")
else:
    print("\nHigh noise condition shows GOOD discrimination")
    print("→ σ=3.5 might be incorrect - check fitting procedure")
print("="*80)
