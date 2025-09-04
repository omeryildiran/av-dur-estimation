#!/usr/bin/env python3
"""
Simple demonstration showing the difference between plotting with and without error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitNonShared import (
    loadData, groupByChooseTest, bin_and_plot, 
    bin_and_plot_with_error_bars
)

def compare_plots():
    """Compare plots with and without error bars side by side"""
    
    # Load data
    dataName = "all_auditory.csv"
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName)
    
    # Filter for one condition to make comparison clear
    filtered_data = data[
        (data[sensoryVar] == uniqueSensory[0]) & 
        (data[conflictVar] == uniqueConflict[0]) & 
        (np.abs(data[standardVar] - uniqueStandard[0]) < 0.01)
    ].copy()
    
    print(f"Using {len(filtered_data)} trials from {filtered_data['participantID'].nunique()} participants")
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Traditional method (no error bars)
    plt.sca(ax1)
    grouped_data = groupByChooseTest(filtered_data)
    bin_and_plot(grouped_data, bin_method='cut', bins=8, plot=True, color='blue', add_error_bars=False)
    plt.title('Traditional Plot\n(No Error Bars)', fontsize=14)
    plt.xlabel('Delta Duration Percent')
    plt.ylabel('P(Choose Test)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: New method with error bars across participants
    plt.sca(ax2)
    bin_and_plot_with_error_bars(filtered_data, bin_method='cut', bins=8, plot=True, color='red')
    plt.title('Enhanced Plot\n(With Error Bars Across Participants)', fontsize=14)
    plt.xlabel('Delta Duration Percent')
    plt.ylabel('P(Choose Test)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("LEFT PLOT (Traditional):")
    print("- Shows average performance across all trials")
    print("- Marker size indicates number of trials")
    print("- No information about participant variability")
    print("\\nRIGHT PLOT (Enhanced):")
    print("- Shows average performance ± standard error across participants")
    print("- Error bars show reliability/consistency of the effect")
    print("- Smaller error bars = more consistent across participants")
    print("- Larger error bars = more individual differences")

if __name__ == "__main__":
    compare_plots()
