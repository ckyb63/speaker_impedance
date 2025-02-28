import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from model.data_loader import load_data
from model.data_preprocessor import preprocess_data
from model.model_trainer import train_model, evaluate_model, verify_model
from analyze_results import plot_confusion_matrix

# Configure GPU memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except:
            print(f"Could not set memory growth for {device}")
    
    # REMOVED: Do not enable mixed precision globally
    # This was causing conflicts with the model-specific settings
else:
    print("No GPU found. Using CPU for training.")
    # Set thread count for CPU training
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

def check_gpu_availability():
    """
    Perform detailed checks on GPU availability and configuration
    """
    print("\n===== GPU AVAILABILITY CHECK =====")
    
    # Check if TensorFlow can see any GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU found by TensorFlow")
        print("   This means TensorFlow will use CPU for computations")
    else:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    
    # Check if CUDA is available
    cuda_available = tf.test.is_built_with_cuda()
    if cuda_available:
        print("✅ TensorFlow is built with CUDA")
    else:
        print("❌ TensorFlow is NOT built with CUDA")
        print("   This means GPU acceleration is not available")
    
    # Check if TensorFlow can access the GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
        print("✅ Successfully executed operations on GPU")
        print(f"   Result: {c.numpy()}")
    except:
        print("❌ Failed to execute operations on GPU")
        print("   This indicates a problem with GPU configuration")
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Print CUDA version if available
    try:
        # Try to get CUDA version from nvidia-smi
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].strip()
                print(f"CUDA version: {cuda_version}")
                break
    except:
        print("Could not determine CUDA version")
    
    # Print GPU device information if available
    if gpus:
        try:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU memory available: {gpu_info['available'] / 1024**3:.2f} GB")
            print(f"GPU memory total: {gpu_info['total'] / 1024**3:.2f} GB")
        except:
            print("Could not retrieve detailed GPU memory information")
    
    print("==================================\n")

def analyze_impedance_length_relationship(df):
    """
    Analyze the relationship between impedance values and length categories
    to create a more accurate mapping function.
    
    Parameters:
    - df: DataFrame containing the data with 'Trace |Z| (Ohm)' and 'Length' columns
    
    Returns:
    - Dictionary mapping length categories to their median impedance values
    - Dictionary mapping length categories to their impedance ranges (min, max)
    """
    if 'Length' not in df.columns or 'Trace |Z| (Ohm)' not in df.columns:
        print("Required columns not found in DataFrame")
        return {}, {}
    
    # Group by length and calculate statistics for impedance values
    length_stats = df.groupby('Length')['Trace |Z| (Ohm)'].agg(['median', 'min', 'max']).reset_index()
    
    # Create dictionaries for median values and ranges
    length_to_median = {}
    length_to_range = {}
    
    for _, row in length_stats.iterrows():
        length = row['Length']
        median_z = row['median']
        min_z = row['min']
        max_z = row['max']
        
        length_to_median[length] = median_z
        length_to_range[length] = (min_z, max_z)
    
    # Print the statistics for reference
    print("\nImpedance statistics by length category:")
    print(length_stats.sort_values('median'))
    
    return length_to_median, length_to_range

def create_impedance_thresholds(length_to_median):
    """
    Create thresholds for impedance values based on median values
    
    Parameters:
    - length_to_median: Dictionary mapping length categories to their median impedance values
    
    Returns:
    - List of (threshold, length) tuples sorted by threshold
    """
    # Convert to list of (median, length) tuples and sort by median
    median_length_pairs = [(median, length) for length, median in length_to_median.items()]
    median_length_pairs.sort()
    
    # Create thresholds between consecutive medians
    thresholds = []
    for i in range(len(median_length_pairs) - 1):
        current_median, current_length = median_length_pairs[i]
        next_median, _ = median_length_pairs[i + 1]
        threshold = (current_median + next_median) / 2
        thresholds.append((threshold, current_length))
    
    # Add the last length category with a very high threshold
    _, last_length = median_length_pairs[-1]
    thresholds.append((float('inf'), last_length))
    
    return thresholds

# Define the length prediction function
def predict_length(impedance_value, thresholds):
    """
    Map impedance values to length categories based on learned thresholds
    
    Parameters:
    - impedance_value: The impedance value to map to a length category
    - thresholds: List of (threshold, length) tuples sorted by threshold
    
    Returns:
    - Length category as a string
    """
    for threshold, length in thresholds:
        if impedance_value < threshold:
            return length
    
    # This should never happen due to the inf threshold, but just in case
    return thresholds[-1][1]

def main():
    # Check GPU availability
    check_gpu_availability()
    
    # Create output directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load data with sampling to reduce dataset size
    folder_path = r'/home/max/speaker_stuff_wslver/speaker_impedance/Collected_Data_Sep16'  # Update this path
    
    # Load only a subset of files and sample rows to speed up training
    # Adjust these parameters based on your available memory and desired training time
    max_files = 2500   # Increased from 2000 to 2500 files for better learning
    sample_rate = 1.0  # Use all rows from each file for better accuracy
    
    print(f"Loading data with max_files={max_files}, sample_rate={sample_rate}...")
    df = load_data(folder_path, max_files=max_files, sample_rate=sample_rate)
    
    # Print data information
    print("Dataset information:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nData sample:")
    print(df.head())
    
    # Analyze the relationship between impedance values and length categories
    if 'Length' in df.columns:
        length_to_median, length_to_range = analyze_impedance_length_relationship(df)
        impedance_thresholds = create_impedance_thresholds(length_to_median)
        print("\nImpedance thresholds for length prediction:")
        for threshold, length in impedance_thresholds:
            if threshold != float('inf'):
                print(f"Length {length}: < {threshold:.2f} Ohm")
            else:
                print(f"Length {length}: >= {impedance_thresholds[-2][0]:.2f} Ohm")
    
    # Data visualization before preprocessing - only if dataset is small enough
    if len(df) < 100000:  # Skip visualization for large datasets
        plt.figure(figsize=(15, 10))
        
        # Plot distributions of features
        plt.subplot(2, 2, 1)
        sns.histplot(df['Trace θ (deg)'].sample(min(10000, len(df))), kde=True)
        plt.title('Distribution of Trace θ (deg)')
        
        plt.subplot(2, 2, 2)
        sns.histplot(df['Trace Rs (Ohm)'].sample(min(10000, len(df))), kde=True)
        plt.title('Distribution of Trace Rs (Ohm)')
        
        plt.subplot(2, 2, 3)
        sns.histplot(df['Trace Xs (Ohm)'].sample(min(10000, len(df))), kde=True)
        plt.title('Distribution of Trace Xs (Ohm)')
        
        plt.subplot(2, 2, 4)
        sns.histplot(df['Trace |Z| (Ohm)'].sample(min(10000, len(df))), kde=True)
        plt.title('Distribution of Trace |Z| (Ohm) (Target)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    else:
        print("Skipping visualization due to large dataset size")
    
    # Create a copy of the dataframe to preserve the Length column for later use
    df_with_length = df.copy()
    
    # Preprocess data - now using Deg, Rs, and Xs
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, test_indices = preprocess_data(df, return_indices=True)
    
    # Indicate that training is starting
    print("\nStarting model training...")
    
    # Train model with improved architecture and reduced epochs
    epochs = 10
    batch_size = 128  # Increased from 64 to 128 for faster training
    
    # Enable mixed precision if GPU is available, but default to False for now due to compatibility issues
    mixed_precision = False  # Disabled by default to avoid type mismatch errors
    print(f"Mixed precision training: {'Enabled' if mixed_precision else 'Disabled'}")
    model = train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, mixed_precision=mixed_precision, output_dir=output_dir)
    
    # Save the model in the newer Keras format
    model.save(os.path.join(output_dir, 'trained_model.keras'))  # Save as Keras format
    
    # Evaluate model with enhanced metrics
    print("\nEvaluating model...")
    mse = evaluate_model(model, X_test, y_test, output_dir=output_dir)
    
    # Verify model with detailed analysis
    print("\nVerifying model...")
    verify_model(model, X_test, y_test, output_dir=output_dir)
    
    # Example prediction using Deg, Rs, and Xs
    # Extract a sample from the test data
    sample_idx = np.random.randint(0, len(X_test))
    sample_features = X_test[sample_idx:sample_idx+1]
    sample_actual = y_test[sample_idx]
    
    # Make prediction
    sample_pred = model.predict(sample_features)[0][0]
    
    print("\nSample prediction:")
    print(f"Actual |Z|: {sample_actual:.4f} Ohm")
    print(f"Predicted |Z|: {sample_pred:.4f} Ohm")
    print(f"Absolute Error: {abs(sample_actual - sample_pred):.4f} Ohm")
    
    # Generate predictions for all test data
    print("\nGenerating predictions for all test data...")
    all_predictions = model.predict(X_test)
    
    # If you have length labels in your dataset, you can create a confusion matrix
    if 'Length' in df_with_length.columns:
        print("\nCreating confusion matrix for length predictions...")
        
        # Get true length categories using the test indices
        y_true_lengths = df_with_length.iloc[test_indices]['Length'].values
        print(f"Number of test samples with length data: {len(y_true_lengths)}")
        
        # Apply the mapping to get predicted length categories using the learned thresholds
        y_pred_lengths = np.array([predict_length(val, impedance_thresholds) for val in all_predictions.flatten()])
        print(f"Number of predicted length values: {len(y_pred_lengths)}")
        
        # Ensure the arrays have the same length
        if len(y_true_lengths) == len(y_pred_lengths):
            # Plot confusion matrix
            plot_confusion_matrix(y_true_lengths, y_pred_lengths, output_dir=output_dir)
            
            # Calculate and print accuracy
            accuracy = accuracy_score(y_true_lengths, y_pred_lengths)
            print(f"Length prediction accuracy: {accuracy:.4f}")
            
            # Save the predictions to a CSV file for further analysis
            predictions_df = pd.DataFrame({
                'True_Length': y_true_lengths,
                'Predicted_Length': y_pred_lengths,
                'Impedance': all_predictions.flatten()
            })
            predictions_df.to_csv(os.path.join(output_dir, 'length_predictions.csv'), index=False)
            print(f"Predictions saved to {os.path.join(output_dir, 'length_predictions.csv')}")
        else:
            print(f"Error: Length mismatch between true ({len(y_true_lengths)}) and predicted ({len(y_pred_lengths)}) values")
            print("Skipping confusion matrix generation")
    else:
        # If length information is not available, just calculate binned accuracy
        # This is a simplified metric - you should adapt it to your needs
        bins = 10
        y_test_binned = np.floor(y_test * bins).astype(int)
        y_pred_binned = np.floor(all_predictions.flatten() * bins).astype(int)
        binned_accuracy = np.mean(y_test_binned == y_pred_binned)
        print(f"\nBinned Accuracy: {binned_accuracy:.4f}")
    
    # Calculate and print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print("\nTraining and evaluation complete. Results saved to output files.")

if __name__ == "__main__":
    main() 