"""
Quick script to identify which participant has data and what conditions exist
"""
import pandas as pd
import glob
import os

csv_files = glob.glob("data/*_all.csv")
csv_files = [f for f in csv_files if 'all_data' not in f and f != 'data/all_all.csv']

print(f"Found {len(csv_files)} participant files\n")

for csv_file in sorted(csv_files)[:5]:  # Check first 5
    participant_id = os.path.basename(csv_file).replace('_all.csv', '')
    print(f"\n{'='*80}")
    print(f"Participant: {participant_id}")
    print(f"{'='*80}")
    
    try:
        data = pd.read_csv(csv_file)
        print(f"Total trials: {len(data)}")
        
        # Check for required columns
        if 'audNoise' in data.columns and 'standardDur' in data.columns:
            data = data[data['audNoise'] != 0]
            data = data[data['standardDur'] != 0]
            
            print(f"\nUnique audio noise levels: {sorted(data['audNoise'].unique())}")
            print(f"Unique standard durations: {sorted(data['standardDur'].unique())}")
            
            # Check the condition matching the figure (standard ~496ms, low noise ~0.1)
            for noise in [0.1, 0.12]:
                for std in [0.496, 0.49644]:
                    condition = data[(data['audNoise'].round(2) == noise) & 
                                    (data['standardDur'].round(3) == std)]
                    if len(condition) > 0:
                        print(f"\n  Condition: noise={noise}, standard={std}s:")
                        print(f"    Trials: {len(condition)}")
                        if 'testDurS' in condition.columns:
                            print(f"    Test duration range: {condition['testDurS'].min():.3f} - {condition['testDurS'].max():.3f}s")
        else:
            print("Missing required columns (audNoise or standardDur)")
            print(f"Available columns: {data.columns.tolist()}")
            
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

print("\n" + "="*80)
