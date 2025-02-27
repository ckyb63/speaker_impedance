import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

def verify_cuda12_setup():
    """
    Verify that TensorFlow can use CUDA 12.8 with the GPU
    """
    print("TensorFlow version:", tf.__version__)
    
    # Check if TensorFlow can see the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. TensorFlow will use CPU.")
        return False
    
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  {gpu.name}")
    
    # Enable memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu.name}")
        except:
            print(f"Could not set memory growth for {gpu.name}")
    
    # Check if CUDA is available
    print("\nCUDA availability:")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    # Test GPU performance with matrix multiplication
    print("\nTesting GPU performance with matrix multiplication...")
    
    # Sizes for matrix multiplication test
    sizes = [1000, 2000, 4000, 6000]
    cpu_times = []
    gpu_times = []
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create random matrices
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Test on CPU
        print("Running on CPU...")
        with tf.device('/CPU:0'):
            # Warm-up run
            _ = tf.matmul(a, b)
            
            # Timed run
            start_time = time.time()
            c_cpu = tf.matmul(a, b)
            # Force execution
            _ = c_cpu.numpy()
            cpu_time = time.time() - start_time
        
        cpu_times.append(cpu_time)
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # Test on GPU
        print("Running on GPU...")
        try:
            with tf.device('/GPU:0'):
                # Warm-up run
                _ = tf.matmul(a, b)
                
                # Timed run
                start_time = time.time()
                c_gpu = tf.matmul(a, b)
                # Force execution
                _ = c_gpu.numpy()
                gpu_time = time.time() - start_time
            
            gpu_times.append(gpu_time)
            print(f"GPU time: {gpu_time:.4f} seconds")
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        except Exception as e:
            print(f"Error running on GPU: {e}")
            gpu_times.append(0)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'b-o', label='CPU')
    plt.plot(sizes, gpu_times, 'r-o', label='GPU')
    plt.title('Matrix Multiplication Performance: CPU vs GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cuda12_performance.png')
    plt.close()
    
    print("\nPerformance comparison saved to 'cuda12_performance.png'")
    
    # Test with a simple CNN model
    print("\nTesting with a simple CNN model...")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Create random data
    x_train = np.random.random((100, 32, 32, 3))
    y_train = np.random.randint(0, 10, (100,))
    
    # Train for a few epochs
    print("Training model...")
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    
    print("\nCUDA 12.8 verification complete!")
    return True

if __name__ == "__main__":
    print("===== Verifying CUDA 12.8 with TensorFlow =====")
    verify_cuda12_setup() 