import tensorflow as tf
import os
import numpy as np

print("=" * 50)
print("DETAILED GPU CHECK")
print("=" * 50)

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check environment variables
print("\nEnvironment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}")

# Check if TensorFlow can see any GPUs
print("\nGPU detection:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
    
    # Try to enable memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth enabled for {gpu.name}")
        except Exception as e:
            print(f"❌ Could not set memory growth for {gpu.name}: {e}")
else:
    print("❌ No GPUs detected by TensorFlow")

# Check if CUDA is available
print("\nCUDA availability:")
print(f"CUDA built with TensorFlow: {'✅ Yes' if tf.test.is_built_with_cuda() else '❌ No'}")

# Try a simple operation on GPU if available
print("\nGPU operation test:")
try:
    if gpus:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = a + b
            print("✅ Successfully executed addition on GPU")
            print(f"   Result: {c.numpy()}")
            
            # More complex operation - matrix multiplication
            start = tf.timestamp()
            matrix1 = tf.random.normal([1000, 1000])
            matrix2 = tf.random.normal([1000, 1000])
            result = tf.matmul(matrix1, matrix2)
            end = tf.timestamp()
            print(f"✅ Successfully executed matrix multiplication on GPU")
            print(f"   Time taken: {(end - start) * 1000:.2f} ms")
    else:
        print("❌ Skipping GPU operations as no GPU was detected")
except Exception as e:
    print(f"❌ Error during GPU operations: {e}")

# Check CUDA version
print("\nCUDA version check:")
try:
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    print(f"✅ CUDA version: {cuda_version}")
except Exception as e:
    print(f"❌ Could not determine CUDA version: {e}")

# Check if GPU is actually being used
print("\nVerifying GPU usage:")
try:
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Generate some random data
    x = tf.random.normal([1000, 1024])
    y = tf.random.uniform([1000], minval=0, maxval=10, dtype=tf.int32)
    
    # Check where the model is placed
    print(f"Model placement: {model.layers[0].weights[0].device}")
    
    # Train for just one step to see if GPU is used
    start = tf.timestamp()
    model.fit(x, y, epochs=1, verbose=0)
    end = tf.timestamp()
    print(f"✅ Model training completed in {(end - start) * 1000:.2f} ms")
    
except Exception as e:
    print(f"❌ Error during model training: {e}")

print("\nNVIDIA-SMI equivalent check:")
try:
    # This is a Python equivalent to check GPU utilization
    # It's not as comprehensive as nvidia-smi but gives some info
    if gpus:
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"✅ GPU memory info available")
        print(f"   Current: {mem_info['current'] / 1024**2:.2f} MB")
        print(f"   Peak: {mem_info['peak'] / 1024**2:.2f} MB")
    else:
        print("❌ No GPU memory info available")
except Exception as e:
    print(f"❌ Could not get GPU memory info: {e}")

print("=" * 50) 