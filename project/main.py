import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
    
    # Enable mixed precision for TensorFlow 2.4+
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except:
        print("Could not enable mixed precision")
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

from model.data_loader import load_data
from model.data_preprocessor import preprocess_data
from model.model_trainer import train_model, evaluate_model, verify_model
from model.predictor import predict_length

def main():
    # Check GPU availability
    check_gpu_availability()
    
    # Start timing
    start_time = time.time()
    
    # Load data with sampling to reduce dataset size
    folder_path = r'C:\Users\maxch\Documents\Purdue Files\Audio Research\Github\speaker_impedance\Collected_Data_Sep16'  # Update this path
    
    # Load only a subset of files and sample rows to speed up training
    # Adjust these parameters based on your available memory and desired training time
    max_files = 100  # Limit to 100 files
    sample_rate = 1  # Use only 20% of rows from each file
    
    print(f"Loading data with max_files={max_files}, sample_rate={sample_rate}...")
    df = load_data(folder_path, max_files=max_files, sample_rate=sample_rate)
    
    # Print data information
    print("Dataset information:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nData sample:")
    print(df.head())
    
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
        plt.savefig('feature_distributions.png')
    else:
        print("Skipping visualization due to large dataset size")
    
    # Preprocess data - now using Deg, Rs, and Xs
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Indicate that training is starting
    print("\nStarting model training...")
    
    # Train model with improved architecture and reduced epochs
    epochs = 20  # Reduced from 100
    batch_size = 128  # Increased from 32
    
    # Enable mixed precision if GPU is available
    mixed_precision = len(physical_devices) > 0
    model = train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, mixed_precision=mixed_precision)
    
    # Save the model in the newer Keras format
    model.save('trained_model.keras')  # Save as Keras format
    
    # Evaluate model with enhanced metrics
    print("\nEvaluating model...")
    mse = evaluate_model(model, X_test, y_test)
    
    # Verify model with detailed analysis
    print("\nVerifying model...")
    verify_model(model, X_test, y_test)
    
    # Example prediction using Deg, Rs, and Xs
    # Extract a sample from the test data
    sample_idx = np.random.randint(0, len(X_test))
    sample_features = X_test[sample_idx].reshape(1, X_test.shape[1], X_test.shape[2])
    actual_value = y_test[sample_idx]
    
    # Make prediction
    predicted_value = model.predict(sample_features)[0][0]
    print(f"\nSample prediction:")
    print(f"Actual |Z|: {actual_value:.4f} Ohm")
    print(f"Predicted |Z|: {predicted_value:.4f} Ohm")
    print(f"Absolute Error: {abs(actual_value - predicted_value):.4f} Ohm")
    
    # Generate predictions for all test data
    print("\nGenerating predictions for all test data...")
    y_pred = model.predict(X_test, batch_size=256).flatten()
    
    # Create regression plot
    plt.figure(figsize=(10, 6))
    # Sample points if there are too many
    if len(y_test) > 1000:
        indices = np.random.choice(len(y_test), 1000, replace=False)
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_test_sample = y_test
        y_pred_sample = y_pred
        
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
    plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], 'r--')
    plt.xlabel('Actual |Z| (Ohm)')
    plt.ylabel('Predicted |Z| (Ohm)')
    plt.title('Actual vs Predicted |Z| Values')
    plt.savefig('regression_results.png')
    
    # For classification-like analysis, bin the continuous values
    # Define bins for classification
    bins = np.linspace(min(y_test), max(y_test), num=10)  # Create 10 bins
    y_test_binned = np.digitize(y_test, bins)  # Binning the true values
    y_pred_binned = np.digitize(y_pred, bins)  # Binning the predicted values
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Create bin labels for display
    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    
    # Fix the error by creating a custom ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    
    # Manually set the tick labels to match the number of ticks
    ax = plt.gca()
    n_classes = cm.shape[0]
    tick_marks = np.arange(n_classes)
    
    # Set x and y ticks and labels
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # Only set labels if they match the number of ticks
    if len(bin_labels) == n_classes:
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.set_yticklabels(bin_labels)
    else:
        # If there's a mismatch, use simple numeric labels
        ax.set_xticklabels(np.arange(n_classes))
        ax.set_yticklabels(np.arange(n_classes))
        print(f"Warning: Mismatch between bin labels ({len(bin_labels)}) and confusion matrix size ({n_classes})")
    
    plt.title('Confusion Matrix (Binned Values)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Calculate accuracy of binned predictions
    accuracy = accuracy_score(y_test_binned, y_pred_binned)
    print(f'\nBinned Accuracy: {accuracy:.4f}')
    
    # Calculate and print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print("\nTraining and evaluation complete. Results saved to output files.")

if __name__ == '__main__':
    main() 