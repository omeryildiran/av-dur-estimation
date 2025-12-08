import pandas as pd
import os
from pathlib import Path

def reduce_csv_file_size(input_path, output_path=None):
    """
    Reduce CSV file size by optimizing data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, overwrites input)
    """
    # Read CSV
    df = pd.read_csv(input_path)
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize object columns to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Save without compression
    if output_path is None:
        output_path = input_path
    
    df.to_csv(output_path, index=False)
    print(f"Processed: {input_path}")

def process_all_files():
    """Process all CSV files in simulated_data/all/ directory."""
    input_dir = Path('simulated_data/all')
    
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist!")
        return
    
    csv_files = list(input_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            # Overwrite original file with optimized version
            reduce_csv_file_size(csv_file)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    process_all_files()
