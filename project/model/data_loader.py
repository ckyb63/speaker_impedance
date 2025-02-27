import pandas as pd
import os
import re
import random

def load_data(folder_path, max_files=None, sample_rate=1.0):
    """
    Load data from CSV files and add metadata columns
    
    Parameters:
    - folder_path: Path to the directory containing the data
    - max_files: Maximum number of files to load (None for all)
    - sample_rate: Fraction of rows to keep from each file (1.0 = all rows)
    
    Returns:
    - DataFrame with all data and metadata
    """
    data_frames = []
    file_paths = []
    
    # Regular expression to extract speaker and length from file path
    # Example: A/A_5/A_5_Run1.csv -> Speaker: A, Length: 5
    pattern = r'([A-D])/\1_([^/]+)/\1_\2_Run\d+\.csv'
    
    # First, collect all file paths
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    
    # Shuffle and limit the number of files if specified
    random.shuffle(file_paths)
    if max_files is not None and max_files > 0:
        file_paths = file_paths[:max_files]
        print(f"Limited to {max_files} files")
    
    # Process the selected files
    for file_path in file_paths:
        # Extract speaker and length information from the file path
        relative_path = os.path.relpath(file_path, folder_path)
        match = re.search(pattern, relative_path.replace('\\', '/'))
        
        if match:
            speaker = match.group(1)
            length = match.group(2)
            filename = os.path.basename(file_path)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Sample rows if sample_rate < 1.0
            if sample_rate < 1.0:
                df = df.sample(frac=sample_rate, random_state=42)
            
            # Add metadata columns
            df['Speaker'] = speaker
            df['Length'] = length
            df['Filename'] = filename
            
            data_frames.append(df)
        else:
            print(f"Warning: Could not extract metadata from {relative_path}")
    
    if not data_frames:
        raise ValueError("No CSV files found in the specified directory.")
    
    # Combine all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    # Print summary of loaded data
    print(f"Loaded {len(data_frames)} files with {len(combined_df)} total rows")
    print(f"Speakers found: {combined_df['Speaker'].unique()}")
    print(f"Lengths found: {combined_df['Length'].unique()}")
    
    return combined_df 