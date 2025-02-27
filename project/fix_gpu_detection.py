import os
import sys
import subprocess
import platform
import shutil
import ctypes
from pathlib import Path
import time

def is_admin():
    """Check if the script is running with administrator privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def print_section(title):
    """Print a section title"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def run_command(command, show_output=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if show_output:
            print(f"Command: {command}")
            print(f"Output:\n{result.stdout}")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
        return result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command '{command}': {e}")
        return "", str(e)

def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working"""
    print_section("NVIDIA Driver Check")
    
    # Check if nvidia-smi is available
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        print("❌ nvidia-smi not found. NVIDIA driver might not be installed.")
        print("Please download and install the latest NVIDIA driver from:")
        print("https://www.nvidia.com/Download/index.aspx")
        return False
    
    # Run nvidia-smi to check driver
    stdout, stderr = run_command("nvidia-smi")
    
    if "NVIDIA-SMI has failed" in stdout or "NVIDIA-SMI has failed" in stderr:
        print("❌ NVIDIA driver is installed but not working properly.")
        print("Try reinstalling the NVIDIA driver or rebooting your system.")
        return False
    
    # Extract driver version
    driver_version = None
    for line in stdout.split('\n'):
        if "Driver Version" in line:
            driver_version = line.split("Driver Version:")[1].split()[0]
            break
    
    if driver_version:
        print(f"✅ NVIDIA driver version {driver_version} is installed and working.")
        
        # Check if driver is recent enough
        major, minor = map(int, driver_version.split('.')[:2])
        if major < 470:  # Minimum recommended for CUDA 12.x
            print("⚠️ Your NVIDIA driver is older than recommended for CUDA 12.x.")
            print("Consider updating to driver version 470 or newer.")
    else:
        print("⚠️ Could not determine NVIDIA driver version.")
    
    return True

def check_cuda_installation():
    """Check CUDA installation"""
    print_section("CUDA Installation Check")
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        print("❌ CUDA_PATH environment variable is not set.")
        print("Please set it to your CUDA installation directory.")
        return False
    
    print(f"CUDA_PATH: {cuda_path}")
    
    # Check if CUDA_PATH exists
    if not os.path.exists(cuda_path):
        print(f"❌ CUDA installation directory {cuda_path} does not exist.")
        return False
    
    # Check for nvcc
    nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
    if not os.path.exists(nvcc_path):
        print(f"❌ CUDA compiler (nvcc) not found at {nvcc_path}")
        return False
    
    # Run nvcc to get version
    stdout, stderr = run_command(f'"{nvcc_path}" --version')
    
    cuda_version = None
    for line in stdout.split('\n'):
        if "release" in line:
            cuda_version = line.split("release")[1].split(",")[0].strip()
            break
    
    if cuda_version:
        print(f"✅ CUDA {cuda_version} is installed.")
    else:
        print("⚠️ Could not determine CUDA version.")
    
    # Check PATH for CUDA directories
    path = os.environ.get("PATH", "")
    cuda_in_path = False
    
    for p in path.split(os.pathsep):
        if "CUDA" in p and os.path.exists(p):
            cuda_in_path = True
            print(f"Found CUDA directory in PATH: {p}")
    
    if not cuda_in_path:
        print("❌ CUDA directories are not in PATH.")
        print("Please add the following directories to your PATH:")
        print(f"{os.path.join(cuda_path, 'bin')}")
        print(f"{os.path.join(cuda_path, 'libnvvp')}")
        return False
    
    return True

def check_cudnn_installation():
    """Check cuDNN installation"""
    print_section("cuDNN Installation Check")
    
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        print("❌ CUDA_PATH environment variable is not set.")
        return False
    
    # Check for cuDNN files
    cudnn_header = os.path.join(cuda_path, "include", "cudnn.h")
    cudnn_lib = os.path.join(cuda_path, "lib", "x64", "cudnn.lib")
    cudnn_dll = os.path.join(cuda_path, "bin", "cudnn64_8.dll")  # For cuDNN 8.x
    
    # Also check for newer versions of cuDNN DLL
    cudnn_dll_alt = None
    for i in range(0, 10):
        path = os.path.join(cuda_path, "bin", f"cudnn64_{i}.dll")
        if os.path.exists(path):
            cudnn_dll_alt = path
            break
    
    if not os.path.exists(cudnn_header):
        print(f"❌ cuDNN header not found at {cudnn_header}")
        cudnn_installed = False
    else:
        print(f"✅ cuDNN header found at {cudnn_header}")
        cudnn_installed = True
    
    if not os.path.exists(os.path.join(cuda_path, "lib", "x64")):
        # Check alternative location
        cudnn_lib = os.path.join(cuda_path, "lib64", "cudnn.lib")
    
    if not os.path.exists(cudnn_lib):
        print(f"❌ cuDNN library not found at {cudnn_lib}")
        cudnn_installed = False
    else:
        print(f"✅ cuDNN library found at {cudnn_lib}")
    
    if not os.path.exists(cudnn_dll) and not cudnn_dll_alt:
        print(f"❌ cuDNN DLL not found at {cudnn_dll}")
        cudnn_installed = False
    else:
        if os.path.exists(cudnn_dll):
            print(f"✅ cuDNN DLL found at {cudnn_dll}")
        elif cudnn_dll_alt:
            print(f"✅ cuDNN DLL found at {cudnn_dll_alt}")
    
    if not cudnn_installed:
        print("\nTo install cuDNN:")
        print("1. Download cuDNN v9.x from NVIDIA website (requires NVIDIA account):")
        print("   https://developer.nvidia.com/cudnn-downloads")
        print("2. Extract the files to your CUDA installation directory")
        print("3. Make sure the following files are in place:")
        print(f"   - {cudnn_header}")
        print(f"   - {cudnn_lib}")
        print(f"   - {cudnn_dll}")
        return False
    
    # Try to determine cuDNN version from header file
    if os.path.exists(cudnn_header):
        try:
            with open(cudnn_header, 'r') as f:
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
                    print(f"✅ cuDNN version {major}.{minor}.{patch} is installed.")
        except:
            print("⚠️ Could not determine cuDNN version from header file.")
    
    return cudnn_installed

def check_tensorflow_installation():
    """Check TensorFlow installation"""
    print_section("TensorFlow Installation Check")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is installed.")
        
        # Check if TensorFlow is built with CUDA
        cuda_available = tf.test.is_built_with_cuda()
        if cuda_available:
            print("✅ TensorFlow is built with CUDA support.")
        else:
            print("❌ TensorFlow is NOT built with CUDA support.")
            print("This might be because you installed the CPU-only version of TensorFlow.")
            print("Try reinstalling TensorFlow:")
            print("pip uninstall tensorflow")
            print("pip install tensorflow==2.17.0")
            return False
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("❌ TensorFlow did not detect any GPUs.")
            print("This might be due to a configuration issue.")
        
        return True
    except ImportError:
        print("❌ TensorFlow is not installed.")
        print("Please install TensorFlow:")
        print("pip install tensorflow==2.17.0")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False

def fix_environment_variables():
    """Fix environment variables for CUDA and cuDNN"""
    print_section("Fixing Environment Variables")
    
    if not is_admin():
        print("⚠️ This script is not running with administrator privileges.")
        print("Some environment variable changes may not be permanent.")
        print("Consider running this script as administrator.")
    
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        # Try to find CUDA installation
        potential_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                cuda_path = path
                print(f"Found CUDA installation at {cuda_path}")
                break
        
        if not cuda_path:
            print("❌ Could not find CUDA installation.")
            print("Please enter your CUDA installation path manually.")
            return False
    
    # Set CUDA_PATH
    os.environ["CUDA_PATH"] = cuda_path
    run_command(f'setx CUDA_PATH "{cuda_path}"', show_output=False)
    print(f"✅ Set CUDA_PATH to {cuda_path}")
    
    # Add CUDA directories to PATH
    cuda_bin = os.path.join(cuda_path, "bin")
    cuda_libnvvp = os.path.join(cuda_path, "libnvvp")
    cuda_include = os.path.join(cuda_path, "include")
    
    path = os.environ.get("PATH", "")
    path_parts = path.split(os.pathsep)
    
    # Add directories if they're not already in PATH
    if cuda_bin not in path_parts and os.path.exists(cuda_bin):
        path = cuda_bin + os.pathsep + path
        os.environ["PATH"] = path
        run_command(f'setx PATH "{path}"', show_output=False)
        print(f"✅ Added {cuda_bin} to PATH")
    
    if cuda_libnvvp not in path_parts and os.path.exists(cuda_libnvvp):
        path = cuda_libnvvp + os.pathsep + path
        os.environ["PATH"] = path
        run_command(f'setx PATH "{path}"', show_output=False)
        print(f"✅ Added {cuda_libnvvp} to PATH")
    
    if cuda_include not in path_parts and os.path.exists(cuda_include):
        path = cuda_include + os.pathsep + path
        os.environ["PATH"] = path
        run_command(f'setx PATH "{path}"', show_output=False)
        print(f"✅ Added {cuda_include} to PATH")
    
    print("\n⚠️ Environment variables have been updated.")
    print("You may need to restart your command prompt or IDE for changes to take effect.")
    
    return True

def copy_cudnn_files():
    """Copy cuDNN files to CUDA directory if they exist in the current directory"""
    print_section("Checking for cuDNN Files")
    
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        print("❌ CUDA_PATH environment variable is not set.")
        return False
    
    # Check for cuDNN files in current directory
    current_dir = os.getcwd()
    cudnn_files = {
        "include/cudnn.h": os.path.join(cuda_path, "include", "cudnn.h"),
        "lib/cudnn.lib": os.path.join(cuda_path, "lib", "x64", "cudnn.lib"),
        "bin/cudnn64_8.dll": os.path.join(cuda_path, "bin", "cudnn64_8.dll"),
        # Check for newer versions too
        "bin/cudnn64_9.dll": os.path.join(cuda_path, "bin", "cudnn64_9.dll"),
    }
    
    found_files = False
    
    for src_rel, dst in cudnn_files.items():
        src = os.path.join(current_dir, src_rel)
        if os.path.exists(src):
            found_files = True
            print(f"Found cuDNN file: {src}")
            
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            
            # Copy file
            try:
                shutil.copy2(src, dst)
                print(f"✅ Copied {src} to {dst}")
            except Exception as e:
                print(f"❌ Error copying {src} to {dst}: {e}")
    
    if not found_files:
        print("No cuDNN files found in current directory.")
        print("If you have downloaded cuDNN, extract the files and run this script from that directory.")
    
    return found_files

def test_gpu_operation():
    """Test if TensorFlow can perform operations on GPU"""
    print_section("Testing GPU Operation")
    
    try:
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU found by TensorFlow.")
            return False
        
        # Try to enable memory growth
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for {gpu.name}")
            except:
                print(f"⚠️ Could not set memory growth for {gpu.name}")
        
        # Try a simple operation on GPU
        print("Attempting matrix multiplication on GPU...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Warm-up run
            _ = tf.matmul(a, b)
            
            # Timed run
            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy()
            gpu_time = time.time() - start_time
        
        print(f"✅ Successfully executed matrix multiplication on GPU in {gpu_time:.4f} seconds")
        
        # Compare with CPU
        print("Attempting same operation on CPU for comparison...")
        with tf.device('/CPU:0'):
            # Warm-up run
            _ = tf.matmul(a, b)
            
            # Timed run
            start_time = time.time()
            c_cpu = tf.matmul(a, b)
            # Force execution
            _ = c_cpu.numpy()
            cpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.4f} seconds")
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        return True
    except ImportError:
        print("❌ TensorFlow is not installed.")
        return False
    except Exception as e:
        print(f"❌ Error testing GPU operation: {e}")
        return False

def fix_gpu_detection():
    """Main function to fix GPU detection issues"""
    print_section("GPU Detection Fix Tool")
    print("This tool will help diagnose and fix issues with GPU detection in TensorFlow.")
    
    # Check NVIDIA driver
    driver_ok = check_nvidia_driver()
    if not driver_ok:
        print("\n⚠️ NVIDIA driver issues must be fixed before proceeding.")
        print("Please install or update your NVIDIA driver and run this script again.")
        return False
    
    # Check CUDA installation
    cuda_ok = check_cuda_installation()
    
    # Check cuDNN installation
    cudnn_ok = check_cudnn_installation()
    
    # Check TensorFlow installation
    tf_ok = check_tensorflow_installation()
    
    # Fix environment variables
    fix_environment_variables()
    
    # Copy cuDNN files if available
    copy_cudnn_files()
    
    # Test GPU operation
    if tf_ok:
        test_gpu_operation()
    
    print_section("Summary and Recommendations")
    
    if not cuda_ok:
        print("❌ CUDA installation issues detected.")
        print("Please make sure CUDA 12.x is properly installed.")
        print("Download CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    
    if not cudnn_ok:
        print("❌ cuDNN installation issues detected.")
        print("Please download and install cuDNN 9.x from NVIDIA website.")
        print("https://developer.nvidia.com/cudnn-downloads")
    
    if not tf_ok:
        print("❌ TensorFlow installation issues detected.")
        print("Try reinstalling TensorFlow:")
        print("pip uninstall tensorflow tensorflow-gpu")
        print("pip install tensorflow==2.17.0")
    
    print("\nNext steps:")
    print("1. Restart your command prompt or IDE for environment variable changes to take effect")
    print("2. Run 'python test_gpu.py' to verify GPU functionality")
    print("3. If issues persist, try rebooting your system")
    
    return True

if __name__ == "__main__":
    fix_gpu_detection() 