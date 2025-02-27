import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)
print("CUDA_PATH:", os.environ.get("CUDA_PATH", "Not set"))

# Check if TensorFlow can see any GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow detected {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Try to enable memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu.name}")
        except:
            print(f"Could not set memory growth for {gpu.name}")
else:
    print("No GPUs detected by TensorFlow")

# Check if CUDA is available
print("\nCUDA built with TensorFlow:", tf.test.is_built_with_cuda())

# Try a simple operation on GPU if available
if gpus:
    print("\nAttempting matrix multiplication on GPU...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("Matrix multiplication result shape:", c.shape)
        print("Operation completed successfully on GPU")
else:
    print("\nSkipping GPU operations as no GPU was detected") 