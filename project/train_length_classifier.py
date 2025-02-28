"""
This script is used to train the length classification model.

So it predicts the length of the speaker based on the impedance data across the frequency range.
"""


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from model.data_loader import load_data
from model.length_data_preprocessor import preprocess_data_for_length_classification
from model.length_model_trainer import train_length_classifier, evaluate_length_classifier

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
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    print("==================================\n")

def main():
    # Check GPU availability
    check_gpu_availability()
    
    # Create output directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load data
    folder_path = r'/home/max/speaker_stuff_wslver/speaker_impedance/Collected_Data_Sep16'
    
    # Load data with sampling to reduce dataset size
    max_files = 2500   # Use more files for better learning
    sample_rate = 1.0  # Use all rows for better accuracy
    
    print(f"Loading data with max_files={max_files}, sample_rate={sample_rate}...")
    df = load_data(folder_path, max_files=max_files, sample_rate=sample_rate)
    
    # Print data information
    print("Dataset information:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if Length column exists
    if 'Length' not in df.columns:
        print("Error: 'Length' column not found in the dataset")
        return
    
    # Check if Frequency column exists
    if 'Frequency (Hz)' not in df.columns:
        print("Error: 'Frequency (Hz)' column not found in the dataset")
        return
    
    # Print length distribution
    print("\nLength distribution:")
    length_counts = df['Length'].value_counts().sort_index()
    print(length_counts)
    
    # Print frequency range
    print("\nFrequency range:")
    freq_min = df['Frequency (Hz)'].min()
    freq_max = df['Frequency (Hz)'].max()
    freq_count = df.groupby(['Speaker', 'Length', 'Filename'])['Frequency (Hz)'].count().mean()
    print(f"Min frequency: {freq_min} Hz")
    print(f"Max frequency: {freq_max} Hz")
    print(f"Average frequency points per sample: {freq_count:.1f}")
    
    # Set number of frequency points to use in each sequence
    # This should be less than or equal to the average number of frequency points
    num_freq_points = min(50, int(freq_count))
    print(f"Using {num_freq_points} frequency points per sequence")
    
    # Preprocess data for length classification
    print("\nPreprocessing data for length classification...")
    X_train, X_test, y_train, y_test, label_encoder, class_weights = preprocess_data_for_length_classification(
        df, augment_factor=0.3, test_size=0.25, num_freq_points=num_freq_points
    )
    
    # Print class information
    print("\nLength categories:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"  {i}: {category}")
    
    # Print class weights
    print("\nClass weights:")
    for i, weight in class_weights.items():
        print(f"  Class {i} ({label_encoder.classes_[i]}): {weight:.4f}")
    
    # Print input shape details
    print(f"\nInput shape: {X_train.shape}")
    print(f"  - Number of training samples: {X_train.shape[0]}")
    print(f"  - Frequency points per sample: {X_train.shape[1]}")
    print(f"  - Features per frequency point: {X_train.shape[2]}")
    
    # Train the length classification model
    print("\nTraining length classification model...")
    num_classes = len(label_encoder.classes_)
    epochs = 20
    batch_size = 64  # Smaller batch size for more complex model
    
    model = train_length_classifier(
        X_train, y_train, 
        num_classes=num_classes, 
        epochs=epochs, 
        batch_size=batch_size, 
        class_weights=class_weights,
        output_dir=output_dir
    )
    
    # Save the model
    model.save(os.path.join(output_dir, 'length_classifier_model.keras'))
    print(f"Model saved to {os.path.join(output_dir, 'length_classifier_model.keras')}")
    
    # Evaluate the model
    print("\nEvaluating length classification model...")
    accuracy = evaluate_length_classifier(
        model, X_test, y_test, label_encoder, output_dir=output_dir
    )
    
    # Calculate and print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print("\nTraining and evaluation complete. Results saved to output files.")

if __name__ == "__main__":
    main() 