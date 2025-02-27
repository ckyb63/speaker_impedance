"""
Test script to verify TensorFlow GPU support in Docker container.
Run this script inside the TensorFlow Docker container to verify GPU support.
"""

import os
import time
import numpy as np
import tensorflow as tf

def print_separator():
    print("=" * 80)

def test_gpu_availability():
    print_separator()
    print("TensorFlow version:", tf.__version__)
    print("CUDA built with TensorFlow:", tf.test.is_built_with_cuda())
    
    # List physical devices
    physical_devices = tf.config.list_physical_devices()
    print("\nAll physical devices:", physical_devices)
    
    # List GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print("\nGPUs available:", gpus)
    
    if not gpus:
        print("\n❌ No GPUs detected by TensorFlow!")
        print("This could be due to:")
        print("  - Missing NVIDIA drivers in the Docker container")
        print("  - Missing NVIDIA Container Toolkit")
        print("  - Incompatible GPU")
        print("  - Docker not configured correctly for GPU passthrough")
        return False
    
    print("\n✅ GPU(s) detected by TensorFlow!")
    
    # Enable memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
        except:
            print(f"Could not enable memory growth for {gpu}")
    
    return True

def benchmark_performance():
    print_separator()
    print("Running performance benchmark...")
    
    # Matrix sizes for multiplication
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\nMatrix multiplication benchmark (size: {size}x{size})")
        
        # Create random matrices
        A = tf.random.normal((size, size))
        B = tf.random.normal((size, size))
        
        # CPU benchmark
        with tf.device('/CPU:0'):
            start_time = time.time()
            C_cpu = tf.matmul(A, B)
            # Force execution
            C_cpu_result = C_cpu.numpy()
            cpu_time = time.time() - start_time
            print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU benchmark (if available)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                start_time = time.time()
                C_gpu = tf.matmul(A, B)
                # Force execution
                C_gpu_result = C_gpu.numpy()
                gpu_time = time.time() - start_time
                print(f"GPU time: {gpu_time:.4f} seconds")
                
                if cpu_time > 0:
                    speedup = cpu_time / gpu_time
                    print(f"GPU speedup: {speedup:.2f}x")
        else:
            print("GPU benchmark skipped - no GPU available")

def train_small_model():
    print_separator()
    print("Training a small model to verify GPU usage...")
    
    # Generate synthetic data
    x_train = np.random.random((1000, 28, 28, 1))
    y_train = np.random.randint(0, 10, (1000,))
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")

def main():
    print_separator()
    print("TensorFlow Docker GPU Test")
    print_separator()
    
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv')
    print(f"Running in Docker container: {in_docker}")
    
    if not in_docker:
        print("⚠️ This script is designed to run inside a Docker container.")
        print("Please run the Docker container first using run_tensorflow_docker.bat")
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    # Run benchmarks if GPU is available
    if gpu_available:
        benchmark_performance()
        train_small_model()
    else:
        print("\nSkipping benchmarks due to no GPU availability.")
    
    print_separator()
    print("Test completed!")
    print_separator()

if __name__ == "__main__":
    main() 