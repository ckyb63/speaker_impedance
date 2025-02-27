import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def test_gpu_availability():
    """Test if TensorFlow can detect and use GPU"""
    print("\n===== GPU AVAILABILITY TEST =====")
    
    # Check if TensorFlow can see any GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU found by TensorFlow")
        print("   This means TensorFlow will use CPU for computations")
        return False
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
        return False
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    return True

def test_gpu_performance():
    """Test GPU performance with matrix multiplication"""
    print("\n===== GPU PERFORMANCE TEST =====")
    
    # Matrix sizes to test
    sizes = [1000, 2000, 4000, 6000, 8000]
    cpu_times = []
    gpu_times = []
    
    for size in sizes:
        print(f"\nTesting matrix multiplication with size {size}x{size}...")
        
        # Create random matrices
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Test on CPU
        print("Running on CPU...")
        with tf.device('/CPU:0'):
            cpu_start = time.time()
            c_cpu = tf.matmul(a, b)
            # Force execution
            _ = c_cpu.numpy()
            cpu_end = time.time()
        cpu_time = cpu_end - cpu_start
        cpu_times.append(cpu_time)
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # Test on GPU if available
        if tf.config.list_physical_devices('GPU'):
            print("Running on GPU...")
            with tf.device('/GPU:0'):
                gpu_start = time.time()
                c_gpu = tf.matmul(a, b)
                # Force execution
                _ = c_gpu.numpy()
                gpu_end = time.time()
            gpu_time = gpu_end - gpu_start
            gpu_times.append(gpu_time)
            print(f"GPU time: {gpu_time:.4f} seconds")
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print(f"GPU is {speedup:.2f}x faster than CPU")
        else:
            gpu_times.append(0)
            print("GPU not available for testing")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'b-o', label='CPU')
    if any(gpu_times):
        plt.plot(sizes, gpu_times, 'r-o', label='GPU')
    plt.title('Matrix Multiplication Performance: CPU vs GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('gpu_performance_test.png')
    plt.close()
    
    print("\nPerformance test complete. Results saved to 'gpu_performance_test.png'")

def test_model_performance():
    """Test GPU performance with a small CNN model"""
    print("\n===== MODEL PERFORMANCE TEST =====")
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Create random data
    x_train = np.random.random((1000, 32, 32, 3))
    y_train = np.random.randint(0, 10, (1000,))
    
    # Train on CPU
    print("\nTraining on CPU...")
    with tf.device('/CPU:0'):
        cpu_start = time.time()
        model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)
        cpu_end = time.time()
    cpu_time = cpu_end - cpu_start
    print(f"CPU training time: {cpu_time:.4f} seconds")
    
    # Reset model and train on GPU if available
    if tf.config.list_physical_devices('GPU'):
        # Recreate model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        print("\nTraining on GPU...")
        with tf.device('/GPU:0'):
            gpu_start = time.time()
            model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)
            gpu_end = time.time()
        gpu_time = gpu_end - gpu_start
        print(f"GPU training time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU is {speedup:.2f}x faster than CPU for model training")
    else:
        print("GPU not available for testing")

if __name__ == "__main__":
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu}")
            except:
                print(f"Could not set memory growth for {gpu}")
    
    # Run tests
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        test_gpu_performance()
        test_model_performance()
    else:
        print("\nSkipping performance tests since GPU is not available")
    
    print("\nAll tests completed") 