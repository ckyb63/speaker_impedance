import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.data_loader import load_data

def analyze_dataset_structure(df):
    """
    Analyze the structure of the dataset
    """
    print("\n===== DATASET STRUCTURE ANALYSIS =====")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics for numerical columns:")
    print(df.describe())
    
    # Categorical columns analysis
    if 'Speaker' in df.columns:
        print("\nSpeaker distribution:")
        print(df['Speaker'].value_counts())
    
    if 'Length' in df.columns:
        print("\nLength distribution:")
        print(df['Length'].value_counts().sort_index())

def analyze_impedance_length_relationship(df):
    """
    Analyze the relationship between impedance values and length categories
    """
    if 'Length' not in df.columns or 'Trace |Z| (Ohm)' not in df.columns:
        print("Required columns not found in DataFrame")
        return
    
    print("\n===== IMPEDANCE-LENGTH RELATIONSHIP ANALYSIS =====")
    
    # Group by length and calculate statistics for impedance values
    length_stats = df.groupby('Length')['Trace |Z| (Ohm)'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).reset_index()
    
    # Sort by median impedance for better visualization
    length_stats = length_stats.sort_values('median')
    
    print("\nImpedance statistics by length category:")
    print(length_stats)
    
    # Create a boxplot to visualize the distribution
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Length', y='Trace |Z| (Ohm)', data=df, order=length_stats['Length'])
    plt.title('Impedance Distribution by Length Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/impedance_by_length_boxplot.png')
    
    # Create a violin plot for more detailed distribution
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Length', y='Trace |Z| (Ohm)', data=df, order=length_stats['Length'])
    plt.title('Impedance Distribution by Length Category (Violin Plot)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/impedance_by_length_violin.png')
    
    # Calculate overlap between categories
    print("\nAnalyzing overlap between length categories...")
    overlap_matrix = {}
    
    for i, row1 in length_stats.iterrows():
        length1 = row1['Length']
        min1 = row1['min']
        max1 = row1['max']
        
        overlaps = []
        for j, row2 in length_stats.iterrows():
            if i != j:
                length2 = row2['Length']
                min2 = row2['min']
                max2 = row2['max']
                
                # Check for overlap
                if (min1 <= max2 and max1 >= min2):
                    overlap_percent = min(max1, max2) - max(min1, min2)
                    overlap_percent = overlap_percent / (max1 - min1) * 100
                    overlaps.append((length2, overlap_percent))
        
        if overlaps:
            overlap_matrix[length1] = sorted(overlaps, key=lambda x: x[1], reverse=True)
    
    # Print overlap information
    print("\nOverlap between length categories:")
    for length, overlaps in overlap_matrix.items():
        if overlaps:
            print(f"Length {length} overlaps with: {', '.join([f'{l} ({p:.1f}%)' for l, p in overlaps if p > 5])}")
    
    # Create a heatmap of median impedance values
    plt.figure(figsize=(10, 8))
    length_pivot = pd.pivot_table(length_stats, values='median', index=['Length'])
    sns.heatmap(length_pivot, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Median Impedance Values by Length Category')
    plt.tight_layout()
    plt.savefig('outputs/impedance_by_length_heatmap.png')
    
    return length_stats

def suggest_thresholds(length_stats):
    """
    Suggest thresholds for impedance values based on median values
    """
    print("\n===== SUGGESTED THRESHOLDS FOR LENGTH PREDICTION =====")
    
    # Convert to list of (median, length) tuples and sort by median
    median_length_pairs = [(row['median'], row['Length']) for _, row in length_stats.iterrows()]
    median_length_pairs.sort()
    
    # Create thresholds between consecutive medians
    thresholds = []
    for i in range(len(median_length_pairs) - 1):
        current_median, current_length = median_length_pairs[i]
        next_median, next_length = median_length_pairs[i + 1]
        threshold = (current_median + next_median) / 2
        thresholds.append((threshold, current_length, next_length))
    
    # Print the thresholds
    print("\nSuggested thresholds for length prediction:")
    for threshold, current_length, next_length in thresholds:
        print(f"Threshold {threshold:.2f} Ohm: Below = {current_length}, Above = {next_length}")
    
    # Create a more detailed mapping function
    print("\nDetailed mapping function:")
    print("def predict_length(impedance_value):")
    for i, (threshold, current_length, _) in enumerate(thresholds):
        if i == 0:
            print(f"    if impedance_value < {threshold:.2f}:")
            print(f"        return '{current_length}'")
        else:
            print(f"    elif impedance_value < {threshold:.2f}:")
            print(f"        return '{current_length}'")
    
    # Add the last length category
    _, last_length = median_length_pairs[-1]
    print(f"    else:")
    print(f"        return '{last_length}'")

def main():
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    folder_path = r'/home/max/speaker_stuff_wslver/speaker_impedance/Collected_Data_Sep16'
    print(f"Loading data from {folder_path}...")
    
    # Load a subset of files for analysis
    max_files = 1000  # Adjust as needed
    sample_rate = 1.0  # Use all rows
    
    df = load_data(folder_path, max_files=max_files, sample_rate=sample_rate)
    
    # Analyze dataset structure
    analyze_dataset_structure(df)
    
    # Analyze impedance-length relationship
    length_stats = analyze_impedance_length_relationship(df)
    
    # Suggest thresholds for length prediction
    suggest_thresholds(length_stats)
    
    print("\nAnalysis complete. Results saved to output files.")

if __name__ == "__main__":
    main() 