import os
import sys
import subprocess
import platform
import tensorflow as tf
import pkg_resources

def check_tf_version():
    """Check the installed TensorFlow version"""
    tf_version = tf.__version__
    print(f"Installed TensorFlow version: {tf_version}")
    return tf_version

def check_cuda_version():
    """Check the installed CUDA version"""
    try:
        # Try to get CUDA version from nvidia-smi
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].strip()
                print(f"Installed CUDA version (from nvidia-smi): {cuda_version}")
                return cuda_version
    except:
        print("Could not determine CUDA version from nvidia-smi")
    
    # Try alternative method
    try:
        if platform.system() == 'Windows':
            nvcc_path = os.path.join(os.environ.get('CUDA_PATH', ''), 'bin', 'nvcc.exe')
            if os.path.exists(nvcc_path):
                result = subprocess.run([nvcc_path, '--version'], stdout=subprocess.PIPE, text=True)
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[1].split(',')[0].strip()
                        print(f"Installed CUDA version (from nvcc): {cuda_version}")
                        return cuda_version
        else:
            result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, text=True)
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    print(f"Installed CUDA version (from nvcc): {cuda_version}")
                    return cuda_version
    except:
        print("Could not determine CUDA version from nvcc")
    
    return "Unknown"

def check_cudnn_version():
    """Try to determine cuDNN version"""
    print("cuDNN version cannot be determined programmatically with certainty.")
    print("You reported having cuDNN 9.7.1 installed.")
    return "9.7.1 (user reported)"

def check_compatibility():
    """Check compatibility between TensorFlow, CUDA, and cuDNN"""
    tf_version = check_tf_version()
    cuda_version = check_cuda_version()
    cudnn_version = check_cudnn_version()
    
    print("\n===== Compatibility Analysis =====")
    
    # TensorFlow 2.9.0 officially supports CUDA 11.2 and cuDNN 8.1.0
    # TensorFlow 2.10.0+ has better support for newer CUDA versions
    
    if tf_version.startswith("2.9"):
        print("TensorFlow 2.9.x officially supports CUDA 11.2 and cuDNN 8.1.0")
        print("Your setup:")
        print(f"- TensorFlow: {tf_version}")
        print(f"- CUDA: {cuda_version}")
        print(f"- cuDNN: {cudnn_version}")
        
        print("\nRecommendations:")
        print("1. Upgrade TensorFlow to a newer version that better supports CUDA 12.x:")
        print("   pip install tensorflow>=2.12.0")
        print("2. Or downgrade CUDA to 11.2 and cuDNN to 8.1.0 to match TensorFlow 2.9.0 requirements")
        
    elif tf_version.startswith("2.10") or tf_version.startswith("2.11"):
        print("TensorFlow 2.10-2.11 officially supports CUDA 11.2 and cuDNN 8.1.0")
        print("Your setup may work but is not officially supported")
        
    elif tf_version.startswith("2.12") or tf_version.startswith("2.13"):
        print("TensorFlow 2.12-2.13 officially supports CUDA 11.8 and cuDNN 8.6")
        print("Your CUDA 12.8 may work but is not officially supported")
        
    elif tf_version.startswith("2.14") or tf_version.startswith("2.15"):
        print("TensorFlow 2.14-2.15 officially supports CUDA 11.8 and cuDNN 8.7")
        print("Your CUDA 12.8 may work but is not officially supported")
        
    elif tf_version.startswith("2.16") or tf_version.startswith("2.17"):
        print("TensorFlow 2.16-2.17 officially supports CUDA 12.3 and cuDNN 8.9")
        print("Your CUDA 12.8 should work with minimal issues")
        
    else:
        print(f"TensorFlow {tf_version} compatibility with CUDA {cuda_version} is unknown")
    
    print("\n===== Environment Variables =====")
    # Check environment variables
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    path = os.environ.get('PATH', 'Not set')
    
    print(f"CUDA_PATH: {cuda_path}")
    if "CUDA" in path:
        cuda_in_path = [p for p in path.split(os.pathsep) if "CUDA" in p]
        print("CUDA in PATH:")
        for p in cuda_in_path:
            print(f"  - {p}")
    else:
        print("CUDA not found in PATH")

def setup_environment_variables():
    """Set up environment variables for CUDA 12.8 and cuDNN 9.7.1"""
    if platform.system() == 'Windows':
        print("\n===== Setting up Environment Variables for Windows =====")
        print("Add the following to your system environment variables:")
        print("CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8")
        print("\nAdd these to your PATH:")
        print("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin")
        print("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\libnvvp")
        print("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\extras\\CUPTI\\lib64")
        print("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include")
        print("Path to your cuDNN installation")
    else:
        print("\n===== Setting up Environment Variables for Linux/macOS =====")
        print("Add the following to your ~/.bashrc or ~/.zshrc:")
        print("export CUDA_HOME=/usr/local/cuda-12.8")
        print("export PATH=$CUDA_HOME/bin:$PATH")
        print("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")

def recommend_tf_version():
    """Recommend TensorFlow version based on CUDA 12.8"""
    print("\n===== TensorFlow Version Recommendation =====")
    print("For CUDA 12.8 and cuDNN 9.7.1, we recommend:")
    print("TensorFlow 2.16.0 or 2.17.0")
    print("\nTo install:")
    print("pip install tensorflow==2.17.0")
    print("\nNote: You may need to uninstall the current TensorFlow first:")
    print("pip uninstall tensorflow tensorflow-gpu")

def test_gpu():
    """Test if TensorFlow can access the GPU"""
    print("\n===== Testing GPU Access =====")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Try a simple operation on GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
            print("✅ Successfully executed operations on GPU")
            print(f"   Result: {c.numpy()}")
        except:
            print("❌ Failed to execute operations on GPU")
    else:
        print("❌ No GPU found by TensorFlow")

if __name__ == "__main__":
    print("===== CUDA Compatibility Checker =====")
    check_compatibility()
    setup_environment_variables()
    recommend_tf_version()
    test_gpu()
    
    print("\n===== Next Steps =====")
    print("1. If you're using TensorFlow 2.9.0, consider upgrading to TensorFlow 2.17.0")
    print("2. Make sure your environment variables are set correctly")
    print("3. Run 'python test_gpu.py' to verify GPU functionality")
    print("4. If issues persist, consider using a TensorFlow Docker container")
    print("   docker pull tensorflow/tensorflow:latest-gpu") 