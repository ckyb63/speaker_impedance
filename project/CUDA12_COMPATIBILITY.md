# CUDA 12.8 and cuDNN 9.7.1 Compatibility Guide

This guide will help you configure your environment to use CUDA 12.8 and cuDNN 9.7.1 with TensorFlow for the speaker impedance prediction project.

## Overview

TensorFlow has specific version requirements for CUDA and cuDNN. Your system has:
- CUDA 12.8
- cuDNN 9.7.1

These versions are newer than what's typically recommended for older TensorFlow versions, but we can make them work by:
1. Upgrading TensorFlow to a compatible version
2. Configuring the environment correctly
3. Testing the setup

## Step 1: Upgrade TensorFlow

TensorFlow 2.16.0 or 2.17.0 is recommended for CUDA 12.8 compatibility:

```bash
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow==2.17.0
```

## Step 2: Verify Your Setup

Run the compatibility checker script:

```bash
python setup_cuda_compatibility.py
```

This will:
- Check your TensorFlow version
- Verify CUDA and cuDNN detection
- Recommend environment variable settings
- Test basic GPU operations

## Step 3: Test GPU Performance

Run the verification script to test GPU performance:

```bash
python verify_cuda12_setup.py
```

This will:
- Run matrix multiplication tests on CPU and GPU
- Train a small CNN model
- Generate performance comparison graphs

## Step 4: Configure Environment Variables

### Windows

Add these to your system environment variables:

```
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

Add these to your PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
```

### Linux/macOS

Add to your ~/.bashrc or ~/.zshrc:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Step 5: Run Your Model

After configuring everything, run the main script:

```bash
python main.py
```

The script will automatically:
- Check for GPU availability
- Enable memory growth to avoid memory allocation issues
- Enable mixed precision training for compatible GPUs
- Use the GPU for model training

## Troubleshooting

If you encounter issues:

1. Check the GPU troubleshooting guide:
   ```
   GPU_TROUBLESHOOTING.md
   ```

2. Verify GPU detection:
   ```bash
   nvidia-smi
   ```

3. Check TensorFlow GPU detection:
   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   ```

4. Try running with a smaller batch size:
   ```
   Edit main.py and reduce batch_size from 128 to 64 or 32
   ```

## Performance Expectations

With CUDA 12.8 and a compatible GPU:
- Matrix operations should be 10-50x faster on GPU vs CPU
- Model training should be 5-20x faster on GPU vs CPU
- The exact speedup depends on your specific GPU model and the size of the data

## Additional Resources

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html) 