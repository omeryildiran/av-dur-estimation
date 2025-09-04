#!/usr/bin/env python3
"""
Demo script showing how to plot psychometric data with error bars across participants.

This demonstrates the enhanced bin_and_plot function that calculates error bars
based on variability across participants.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fitNonShared import (
    loadData, groupByChooseTest, groupByChooseTestWithParticipants, 
    bin_and_plot_with_error_bars, fitMultipleStartingPoints, 
    plot_fitted_psychometric
)

def demo_error_bars():
    """
    Demonstrate plotting with error bars calculated across participants
    """
    print("Loading data...")
    
    # Load your data
    dataName = "all_auditory.csv"
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName)
    
    print(f"Loaded data with {len(data)} trials from {data['participantID'].nunique()} participants")
    print(f"Participants: {sorted(data['participantID'].unique())}")
    
    # Create a simple plot to demonstrate error bars
    plt.figure(figsize=(12, 8))
    
    # Filter data for a specific condition (first noise level, first conflict level)
    first_noise = uniqueSensory[0]
    first_conflict = uniqueConflict[0] 
    first_standard = uniqueStandard[0]
    
    print(f"Plotting for: Noise={first_noise}, Conflict={first_conflict}, Standard={first_standard}")
    
    # Filter the data
    filtered_data = data[
        (data[sensoryVar] == first_noise) & 
        (data[conflictVar] == first_conflict) & 
        (np.abs(data[standardVar] - first_standard) < 0.01)  # Account for floating point precision
    ].copy()
    
    print(f"Filtered data: {len(filtered_data)} trials from {filtered_data['participantID'].nunique()} participants")
    
    if len(filtered_data) == 0:
        print("No data found for the selected condition. Trying with all data...")
        filtered_data = data.copy()
    
    # Group the data to get p_choose_test values
    grouped_data = groupByChooseTest(filtered_data)
    
    # Plot with error bars across participants
    plt.subplot(2, 2, 1)
    bin_summary = bin_and_plot_with_error_bars(
        filtered_data, 
        bin_method='cut', 
        bins=8, 
        plot=True, 
        color='red'
    )
    plt.title('Data with Error Bars Across Participants')
    plt.xlabel('Delta Duration Percent')
    plt.ylabel('P(Choose Test)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Compare with traditional plotting (no error bars)
    plt.subplot(2, 2, 2)
    grouped_data['bin'] = pd.cut(grouped_data['delta_dur_percents'], bins=8, labels=False, include_lowest=True)
    grouped_by_bin = grouped_data.groupby('bin').agg(
        x_mean=('delta_dur_percents', 'mean'),
        y_mean=('p_choose_test', 'mean'),
        total_resp=('total_responses', 'sum')
    )
    
    plt.scatter(grouped_by_bin['x_mean'], grouped_by_bin['y_mean'], 
                s=grouped_by_bin['total_resp']/grouped_data['total_responses'].sum()*500, 
                color='blue', alpha=0.7)
    plt.title('Traditional Plot (No Error Bars)')
    plt.xlabel('Delta Duration Percent') 
    plt.ylabel('P(Choose Test)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Show participant-level data
    plt.subplot(2, 2, 3)
    participant_data = groupByChooseTestWithParticipants(filtered_data)
    
    # Plot individual participant curves
    colors = plt.cm.Set3(np.linspace(0, 1, len(participant_data['participantID'].unique())))
    for i, participant in enumerate(participant_data['participantID'].unique()):
        p_data = participant_data[participant_data['participantID'] == participant]
        plt.plot(p_data['delta_dur_percents'], p_data['p_choose_test'], 
                'o-', alpha=0.6, color=colors[i], label=f'P{participant}', markersize=4)
    
    plt.title('Individual Participant Data')
    plt.xlabel('Delta Duration Percent')
    plt.ylabel('P(Choose Test)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    if len(bin_summary) > 0:
        plt.bar(range(len(bin_summary)), bin_summary['n_participants'], 
                color='lightblue', alpha=0.7)
        plt.xlabel('Bin Number')
        plt.ylabel('Number of Participants')
        plt.title('Participants per Bin')
        plt.grid(True, alpha=0.3)
        
        # Add text with summary stats
        mean_participants = bin_summary['n_participants'].mean()
        std_participants = bin_summary['n_participants'].std()
        plt.text(0.5, 0.8, f'Mean participants per bin: {mean_participants:.1f}±{std_participants:.1f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\\nSummary of error bar calculation:")
    if len(bin_summary) > 0:
        print(bin_summary[['x_mean', 'y_mean', 'y_sem', 'y_std', 'n_participants']].round(3))
    else:
        print("No binned data available")

def demonstrate_full_analysis():
    """
    Demonstrate the full analysis workflow with error bars
    """
    print("\\n" + "="*50)
    print("FULL ANALYSIS WITH ERROR BARS")
    print("="*50)
    
    # Load data
    dataName = "all_auditory.csv"
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName)
    
    # Fit the model
    print("Fitting psychometric model...")
    fit = fitMultipleStartingPoints(data, nStart=1)
    print(f"Fitted parameters: {fit.x}")
    
    # Plot with error bars - this will use the modified plot_fitted_psychometric function
    # The error bars will automatically be added since we have participantID in the data
    plot_fitted_psychometric(
        data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, 'delta_dur_percents'
    )

if __name__ == "__main__":
    # Run the demos
    demo_error_bars()
    demonstrate_full_analysis()
