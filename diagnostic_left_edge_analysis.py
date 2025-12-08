"""
Diagnostic script to analyze the anomaly at the left edge of psychometric curves
where p_choose_test jumps higher for shorter test durations (counterintuitive).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(__file__))

# Import necessary modules
exec(open('data/fitNonSharedwErrorBars_logNormal.py').read())

def analyze_left_edge_anomaly(participantID, audioNoise=0.1, standardDur=0.496):
    """
    Analyze left edge data for a specific participant to identify potential issues.
    
    Parameters:
    - participantID: str, participant identifier
    - audioNoise: float, audio noise level to analyze
    - standardDur: float, standard duration in seconds
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC ANALYSIS FOR PARTICIPANT: {participantID}")
    print(f"Audio Noise: {audioNoise}, Standard Duration: {standardDur}s ({standardDur*1000}ms)")
    print(f"{'='*80}\n")
    
    # Try different file patterns
    possible_files = [
        f"data/{participantID}_auditory.csv",
        f"data/{participantID}_all.csv"
    ]
    
    dataFile = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            dataFile = file_path
            break
    
    if dataFile is None:
        print(f"ERROR: No data file found for participant {participantID}")
        print(f"Tried: {possible_files}")
        return None, None
    
    data = pd.read_csv(dataFile)
    
    # Add participantID if it doesn't exist (for single-participant files)
    if 'participantID' not in data.columns:
        data['participantID'] = participantID
    
    # Apply same preprocessing as in the main script
    data = data[data['audNoise'] != 0]
    data = data[data['standardDur'] != 0]
    data["testDurMs"] = data["testDurS"] * 1000
    data["standardDurMs"] = data["standardDur"] * 1000
    data = data.round({'audNoise': 2, 'conflictDur': 2, 'delta_dur_percents': 2})
    
    # The key fix: responses and order are floats, need to be ints for comparison
    data['responses'] = data['responses'].astype(int)
    data['order'] = data['order'].astype(int)
    
    data['chose_test'] = (data['responses'] == data['order']).astype(int)
    data['chose_standard'] = (data['responses'] != data['order']).astype(int)
    
    # Round standard duration to match filtering
    standardDur_rounded = round(standardDur, 2)
    
    # Filter for the specific condition
    condition_data = data[
        (data['audNoise'].round(2) == audioNoise) & 
        (data['standardDur'].round(2) == standardDur_rounded)
    ].copy()
    
    print(f"Total trials in this condition: {len(condition_data)}")
    
    if len(condition_data) == 0:
        print(f"WARNING: No data found for this condition!")
        return
    
    # 1. Check raw trial distribution
    print(f"\n{'-'*60}")
    print("1. RAW TRIAL DISTRIBUTION BY TEST DURATION")
    print(f"{'-'*60}")
    
    trial_counts = condition_data.groupby('testDurS').agg(
        n_trials=('chose_test', 'count'),
        n_chose_test=('chose_test', 'sum'),
        n_chose_standard=('chose_standard', 'sum'),
        p_chose_test=('chose_test', 'mean'),
        testDurMs=('testDurMs', 'first')
    ).sort_index()
    
    print(trial_counts.to_string())
    
    # Identify the "left edge" (shortest durations)
    left_edge_durs = trial_counts.index[:3]  # Shortest 3 durations
    print(f"\nLeft edge durations: {left_edge_durs.tolist()}")
    
    # 2. Check if there's bias in response order at left edge
    print(f"\n{'-'*60}")
    print("2. RESPONSE ORDER ANALYSIS AT LEFT EDGE")
    print(f"{'-'*60}")
    
    for dur in left_edge_durs:
        dur_data = condition_data[condition_data['testDurS'] == dur]
        order_counts = dur_data.groupby('order').agg(
            n_trials=('chose_test', 'count'),
            n_chose_test=('chose_test', 'sum'),
            p_chose_test=('chose_test', 'mean')
        )
        print(f"\nTest Duration: {dur}s ({dur*1000:.1f}ms)")
        print(order_counts.to_string())
        
        # Check for order bias
        if len(order_counts) > 1:
            p_values = order_counts['p_chose_test'].values
            if len(p_values) == 2 and abs(p_values[0] - p_values[1]) > 0.3:
                print(f"  ⚠️  WARNING: Large order effect detected! Difference: {abs(p_values[0] - p_values[1]):.2f}")
    
    # 3. Check for coding/labeling errors
    print(f"\n{'-'*60}")
    print("3. RESPONSE CODING VERIFICATION")
    print(f"{'-'*60}")
    
    # Check if chose_test definition makes sense
    sample_trials = condition_data[condition_data['testDurS'].isin(left_edge_durs)].head(10)
    print("\nSample trials at left edge:")
    print(sample_trials[['testDurS', 'testDurMs', 'standardDur', 'order', 'responses', 'chose_test']].to_string())
    
    # Verify: when testDur << standardDur, participant should mostly choose standard (chose_test should be low)
    shortest_dur_data = condition_data[condition_data['testDurS'] == condition_data['testDurS'].min()]
    p_chose_test_shortest = shortest_dur_data['chose_test'].mean()
    print(f"\nAt shortest duration ({condition_data['testDurS'].min()}s):")
    print(f"  p(chose test) = {p_chose_test_shortest:.3f}")
    if p_chose_test_shortest > 0.5:
        print(f"  ⚠️  ANOMALY: p(chose test) > 0.5 at shortest duration!")
        print(f"     This is counterintuitive - participants should prefer the longer standard.")
    
    # 4. Check delta_dur_percents calculation
    print(f"\n{'-'*60}")
    print("4. DELTA DURATION VERIFICATION")
    print(f"{'-'*60}")
    
    delta_check = condition_data.groupby('testDurS').agg(
        delta_dur_percents_mean=('delta_dur_percents', 'mean'),
        testDurMs=('testDurMs', 'mean'),
        standardDurMs=('standardDurMs', 'mean')
    )
    delta_check['expected_delta_pct'] = (delta_check['testDurMs'] - delta_check['standardDurMs']) / delta_check['standardDurMs'] * 100
    delta_check['delta_error'] = delta_check['delta_dur_percents_mean'] - delta_check['expected_delta_pct']
    
    print(delta_check.to_string())
    
    if (np.abs(delta_check['delta_error']) > 1).any():
        print(f"\n  ⚠️  WARNING: delta_dur_percents may be incorrectly calculated!")
    
    # 5. Visualize the psychometric curve with trial counts
    print(f"\n{'-'*60}")
    print("5. PLOTTING PSYCHOMETRIC CURVE")
    print(f"{'-'*60}")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Psychometric curve with trial counts
    ax1 = axes[0]
    ax1.errorbar(trial_counts.index * 1000, 
                 trial_counts['p_chose_test'],
                 yerr=np.sqrt(trial_counts['p_chose_test'] * (1 - trial_counts['p_chose_test']) / trial_counts['n_trials']),
                 fmt='o', markersize=8, capsize=5, color='blue', alpha=0.7)
    
    # Add trial count labels
    for idx, row in trial_counts.iterrows():
        ax1.text(idx * 1000, row['p_chose_test'] + 0.05, 
                f"n={int(row['n_trials'])}", 
                ha='center', fontsize=8, color='red')
    
    ax1.axvline(standardDur * 1000, color='cyan', linestyle='--', linewidth=2, label=f'Standard: {standardDur*1000:.0f}ms')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Test Duration (ms)', fontsize=12)
    ax1.set_ylabel('P(chose test)', fontsize=12)
    ax1.set_title(f'Psychometric Curve - {participantID}\nAudio Noise: {audioNoise}, Standard: {standardDur*1000:.0f}ms', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    
    # Highlight left edge anomaly
    for dur in left_edge_durs:
        if dur in trial_counts.index:
            y_val = trial_counts.loc[dur, 'p_chose_test']
            ax1.plot(dur * 1000, y_val, 'ro', markersize=15, fillstyle='none', linewidth=3)
    
    # Plot 2: Trial counts by test duration
    ax2 = axes[1]
    ax2.bar(trial_counts.index * 1000, trial_counts['n_trials'], width=10, alpha=0.6, color='green')
    ax2.axvline(standardDur * 1000, color='cyan', linestyle='--', linewidth=2, label=f'Standard: {standardDur*1000:.0f}ms')
    ax2.set_xlabel('Test Duration (ms)', fontsize=12)
    ax2.set_ylabel('Number of Trials', fontsize=12)
    ax2.set_title('Trial Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'diagnostic_left_edge_{participantID}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: diagnostic_left_edge_{participantID}.png")
    plt.show()
    
    # 6. Summary and recommendations
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    issues_found = []
    
    # Check for low trial counts
    min_trials = trial_counts['n_trials'].min()
    if min_trials < 5:
        issues_found.append(f"Very low trial counts (min={min_trials}) - may cause noise")
    
    # Check for anomalous responses at left edge
    if p_chose_test_shortest > 0.5:
        issues_found.append(f"Counterintuitive responses at shortest duration (p={p_chose_test_shortest:.2f})")
    
    # Check for uneven trial distribution
    trial_std = trial_counts['n_trials'].std()
    trial_mean = trial_counts['n_trials'].mean()
    if trial_std / trial_mean > 0.5:
        issues_found.append(f"Uneven trial distribution (CV={trial_std/trial_mean:.2f})")
    
    if len(issues_found) > 0:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No obvious data quality issues detected.")
    
    print(f"\nRECOMMENDATIONS:")
    print("  1. Check if 'chose_test' is correctly coded (responses==order)")
    print("  2. Verify that 'order' variable correctly indicates test stimulus position")
    print("  3. Consider excluding or down-weighting data points with very few trials")
    print("  4. Check if there's a response bias or misunderstanding at extreme durations")
    print(f"\n{'='*80}\n")
    
    return condition_data, trial_counts


if __name__ == "__main__":
    # The figure shows as_auditory.csv data based on the script
    # This is a single participant dataset
    
    print("Analyzing 'as' participant auditory data (as shown in the figure)")
    print("="*80)
    
    # Check what the actual standard duration is in this dataset
    dataFile = "data/as_auditory.csv"
    if os.path.exists(dataFile):
        temp_data = pd.read_csv(dataFile)
        temp_data = temp_data[temp_data['standardDur'] != 0]
        actual_standards = temp_data['standardDur'].unique()
        print(f"Actual standard durations in dataset: {sorted(actual_standards)}")
        
        # Use the actual standard duration
        standard_to_use = float(sorted(actual_standards)[0])  # Use first standard
        
        # Run analysis for low noise condition (the blue curve in the figure)
        condition_data, trial_counts = analyze_left_edge_anomaly(
            participantID="as",
            audioNoise=0.1,  # Low noise condition (blue curve)
            standardDur=standard_to_use
        )
    else:
        print(f"Data file not found: {dataFile}")
        print("\nTrying _all.csv format...")
        
        dataFile = "data/as_all.csv"
        if os.path.exists(dataFile):
            temp_data = pd.read_csv(dataFile)
            temp_data = temp_data[temp_data['standardDur'] != 0]
            actual_standards = temp_data['standardDur'].unique()
            print(f"Actual standard durations in dataset: {sorted(actual_standards)}")
            
            standard_to_use = float(sorted(actual_standards)[0])
            
            condition_data, trial_counts = analyze_left_edge_anomaly(
                participantID="as",
                audioNoise=0.1,
                standardDur=standard_to_use
            )
        else:
            print(f"Data file not found: {dataFile}")
