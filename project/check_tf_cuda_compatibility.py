import os
import sys
import subprocess
import platform
import importlib.util

def print_header(title):
    """Print a header with the given title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def run_command(command):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, shell=True)
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            return package.__version__
        return "Unknown version"
    except ImportError:
        return None

def check_tensorflow():
    """Check TensorFlow installation and CUDA compatibility"""
    print_header("TensorFlow and CUDA Compatibility Check")
    
    # Check if TensorFlow is installed
    tf_version = check_package_version("tensorflow")
    if tf_version is None:
        print("❌ TensorFlow is not installed")
        print("   Install with: pip install tensorflow")
        return False
    
    print(f"✅ TensorFlow {tf_version} is installed")
    
    # Import TensorFlow
    try:
        import tensorflow as tf
        
        # Check if TensorFlow can see the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow can see {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("❌ TensorFlow cannot see any GPUs")
        
        # Check if TensorFlow is built with CUDA
        cuda_built = tf.test.is_built_with_cuda()
        if cuda_built:
            print("✅ TensorFlow is built with CUDA support")
        else:
            print("❌ TensorFlow is NOT built with CUDA support")
            print("   This might be because you installed the CPU-only version")
            print("   Try reinstalling: pip install tensorflow")
        
        # Check CUDA version that TensorFlow was built with
        try:
            cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
            cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
            print(f"ℹ️ TensorFlow was built with CUDA {cuda_version} and cuDNN {cudnn_version}")
        except:
            print("ℹ️ Could not determine CUDA/cuDNN versions TensorFlow was built with")
        
        # Check if GPU is available for TensorFlow
        gpu_available = tf.test.is_gpu_available()
        if gpu_available:
            print("✅ GPU is available for TensorFlow")
        else:
            print("❌ GPU is NOT available for TensorFlow")
        
        # Check TensorFlow device placement
        print("\nDevice Placement Test:")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ Matrix multiplication result: {c.numpy()}")
                print("   This operation was executed on GPU")
        except:
            print("❌ Could not execute operations on GPU")
            print("   This indicates a problem with GPU configuration")
        
        return True
    except ImportError:
        print("❌ Failed to import TensorFlow")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {str(e)}")
        return False

def check_cuda():
    """Check CUDA installation"""
    print_header("CUDA Installation Check")
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        print(f"✅ CUDA_PATH is set to: {cuda_path}")
        
        # Check if the directory exists
        if os.path.exists(cuda_path):
            print(f"✅ CUDA directory exists: {cuda_path}")
        else:
            print(f"❌ CUDA directory does not exist: {cuda_path}")
    else:
        print("❌ CUDA_PATH environment variable is not set")
    
    # Check for nvcc
    nvcc_output = run_command("nvcc --version")
    if "release" in nvcc_output:
        print(f"✅ CUDA compiler (nvcc) is installed:")
        for line in nvcc_output.split('\n'):
            if "release" in line:
                print(f"   {line.strip()}")
    else:
        print("❌ CUDA compiler (nvcc) is not installed or not in PATH")
    
    # Check nvidia-smi
    nvidia_smi_output = run_command("nvidia-smi")
    if "NVIDIA-SMI" in nvidia_smi_output:
        print("✅ NVIDIA driver is installed and working")
        
        # Extract CUDA version from nvidia-smi
        for line in nvidia_smi_output.split('\n'):
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version:")[1].strip()
                print(f"ℹ️ CUDA Version from driver: {cuda_version}")
                break
    else:
        print("❌ NVIDIA driver is not installed or not working")
    
    return True

def check_cudnn():
    """Check cuDNN installation"""
    print_header("cuDNN Installation Check")
    
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        print("❌ CUDA_PATH environment variable is not set")
        return False
    
    # Check for cuDNN header file
    cudnn_h_path = os.path.join(cuda_path, "include", "cudnn.h")
    if os.path.exists(cudnn_h_path):
        print(f"✅ cuDNN header file found: {cudnn_h_path}")
        
        # Try to determine cuDNN version from header file
        try:
            with open(cudnn_h_path, 'r') as f:
                content = f.read()
                major = None
                minor = None
                patch = None
                
                for line in content.split('\n'):
                    if "CUDNN_MAJOR" in line and "#define" in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            major = parts[2]
                    elif "CUDNN_MINOR" in line and "#define" in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            minor = parts[2]
                    elif "CUDNN_PATCHLEVEL" in line and "#define" in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            patch = parts[2]
                
                if major and minor and patch:
                    print(f"ℹ️ cuDNN version: {major}.{minor}.{patch}")
        except:
            print("⚠️ Could not determine cuDNN version from header file")
    else:
        print(f"❌ cuDNN header file not found: {cudnn_h_path}")
    
    # Check for cuDNN library files
    cudnn_lib_path = os.path.join(cuda_path, "lib", "x64", "cudnn.lib")
    if not os.path.exists(cudnn_lib_path):
        cudnn_lib_path = os.path.join(cuda_path, "lib64", "cudnn.lib")
    
    if os.path.exists(cudnn_lib_path):
        print(f"✅ cuDNN library file found: {cudnn_lib_path}")
    else:
        print(f"❌ cuDNN library file not found")
    
    # Check for cuDNN DLL files
    cudnn_dll_found = False
    for i in range(0, 10):
        cudnn_dll_path = os.path.join(cuda_path, "bin", f"cudnn64_{i}.dll")
        if os.path.exists(cudnn_dll_path):
            print(f"✅ cuDNN DLL file found: {cudnn_dll_path}")
            cudnn_dll_found = True
    
    if not cudnn_dll_found:
        print("❌ No cuDNN DLL files found")
    
    return True

def check_compatibility():
    """Check compatibility between TensorFlow, CUDA, and cuDNN"""
    print_header("Compatibility Analysis")
    
    # Get TensorFlow version
    tf_version = check_package_version("tensorflow")
    if tf_version is None:
        print("❌ TensorFlow is not installed")
        return
    
    # Get CUDA version from nvidia-smi
    nvidia_smi_output = run_command("nvidia-smi")
    cuda_version = None
    for line in nvidia_smi_output.split('\n'):
        if "CUDA Version" in line:
            cuda_version = line.split("CUDA Version:")[1].strip()
            break
    
    if cuda_version is None:
        print("❌ Could not determine CUDA version")
        return
    
    # Compatibility matrix
    compatibility = {
        "2.9": {"cuda": "11.2", "cudnn": "8.1"},
        "2.10": {"cuda": "11.2", "cudnn": "8.1"},
        "2.11": {"cuda": "11.2", "cudnn": "8.1"},
        "2.12": {"cuda": "11.8", "cudnn": "8.6"},
        "2.13": {"cuda": "11.8", "cudnn": "8.6"},
        "2.14": {"cuda": "11.8", "cudnn": "8.7"},
        "2.15": {"cuda": "11.8", "cudnn": "8.7"},
        "2.16": {"cuda": "12.3", "cudnn": "8.9"},
        "2.17": {"cuda": "12.3", "cudnn": "8.9"},
    }
    
    # Find the closest TensorFlow version
    tf_major_minor = ".".join(tf_version.split(".")[:2])
    if tf_major_minor in compatibility:
        recommended_cuda = compatibility[tf_major_minor]["cuda"]
        recommended_cudnn = compatibility[tf_major_minor]["cudnn"]
        
        print(f"TensorFlow {tf_version} officially supports:")
        print(f"- CUDA {recommended_cuda}")
        print(f"- cuDNN {recommended_cudnn}")
        print(f"\nYour system has:")
        print(f"- CUDA {cuda_version}")
        
        # Compare versions
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])
        if cuda_major_minor == recommended_cuda:
            print("\n✅ Your CUDA version is compatible with your TensorFlow version")
        else:
            print("\n⚠️ Your CUDA version differs from the recommended version for your TensorFlow")
            print("   This might still work, but could cause compatibility issues")
            
            # Suggest TensorFlow version for CUDA 12.8
            if cuda_version.startswith("12.8"):
                print("\nFor CUDA 12.8, we recommend:")
                print("- TensorFlow 2.16.0 or 2.17.0")
                print("\nTo install:")
                print("pip install tensorflow==2.17.0")
    else:
        print(f"⚠️ No compatibility information available for TensorFlow {tf_version}")

def main():
    """Main function"""
    print_header("TensorFlow-CUDA Compatibility Checker")
    
    check_tensorflow()
    check_cuda()
    check_cudnn()
    check_compatibility()
    
    print_header("Recommendations")
    
    # Check if TensorFlow can see GPUs
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ TensorFlow cannot see any GPUs. Try the following:")
            print("1. Make sure you have the correct versions of CUDA and cuDNN installed")
            print("2. Run 'python fix_gpu_detection.py' to diagnose and fix issues")
            print("3. Download and install cuDNN from NVIDIA website")
            print("4. Run 'install_cudnn_for_cuda12.bat' to install cuDNN files")
            print("5. Restart your computer and try again")
        else:
            print("✅ TensorFlow can see your GPU(s)")
            print("   You're good to go! Run 'python test_gpu.py' to verify performance")
    except:
        print("❌ Could not check GPU availability")
        print("   Run 'python fix_gpu_detection.py' to diagnose and fix issues")

if __name__ == "__main__":
    main() 