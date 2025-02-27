# Using NVIDIA Docker for TensorFlow GPU Support

This guide explains how to use NVIDIA Docker to run TensorFlow with GPU support for the speaker impedance project.

## Advantages of Using Docker

Using Docker for TensorFlow with GPU support offers several advantages:

1. **Eliminates compatibility issues** - Docker containers package TensorFlow with the exact CUDA and cuDNN versions it needs
2. **No need to modify your system** - Your existing CUDA installation won't interfere with the container
3. **Reproducible environment** - Everyone on the team can use the exact same environment
4. **Easy to switch versions** - Try different TensorFlow versions without reinstalling

## Prerequisites

Before you begin, make sure you have:

1. **NVIDIA GPU** with the latest drivers installed
2. **Docker Desktop** for Windows
3. **NVIDIA Container Toolkit** (nvidia-docker)

## Installation Steps

### 1. Install Docker Desktop

1. Download Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow the instructions
3. Make sure to enable WSL 2 integration during installation if prompted
4. Restart your computer after installation

### 2. Install NVIDIA Container Toolkit

1. Download the NVIDIA Container Toolkit from [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
2. Follow the Windows installation instructions
3. Restart Docker Desktop after installation

### 3. Verify Installation

1. Open a command prompt and run:
   ```
   docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```
2. If successful, you should see your GPU information displayed

## Running TensorFlow with GPU Support

We've created two files to help you run TensorFlow with GPU support:

1. `run_tensorflow_docker.bat` - Script to start the TensorFlow Docker container
2. `test_tensorflow_docker.py` - Python script to test GPU support inside the container

### Using the Docker Container

1. Open a command prompt in the project directory
2. Run the Docker script:
   ```
   run_tensorflow_docker.bat
   ```
3. This will:
   - Pull the TensorFlow Docker image with GPU support
   - Mount your project directory to /workspace in the container
   - Start a bash shell in the container

4. Inside the container, you can run:
   ```
   python test_tensorflow_docker.py
   ```
   to verify GPU support

### Running Your Project in Docker

Once inside the Docker container, you can run your project as usual:

```bash
# Train the model
python main.py

# Or run any other Python script
python verify_cuda12_setup.py
```

All files in your project directory are available in the container at `/workspace`.

## Troubleshooting

### No GPU Detected in Container

If TensorFlow doesn't detect your GPU inside the container:

1. Make sure your NVIDIA drivers are up to date
2. Verify that the NVIDIA Container Toolkit is installed correctly
3. Check if your GPU is supported by the TensorFlow Docker image
4. Try running with the `--gpus all` flag explicitly

### Docker Container Exits Immediately

If the Docker container exits immediately:

1. Make sure you're using the `-it` flag to run in interactive mode
2. Check if there are any error messages in the Docker logs
3. Try running a simpler container first to verify Docker is working:
   ```
   docker run -it ubuntu bash
   ```

### Performance Issues

If you experience performance issues:

1. Make sure you're mounting your project directory correctly
2. Check if other processes are using your GPU
3. Monitor GPU usage with `nvidia-smi` in another terminal

## Advanced Usage

### Using a Specific TensorFlow Version

To use a specific TensorFlow version, modify the Docker image tag in `run_tensorflow_docker.bat`:

```
tensorflow/tensorflow:2.14.0-gpu
```

### Persistent Storage

By default, any changes made outside the mounted directory will be lost when the container exits. To create persistent storage:

1. Create a Docker volume:
   ```
   docker volume create tf-data
   ```

2. Mount it in the container by adding to the `docker run` command:
   ```
   -v tf-data:/data
   ```

3. Use the `/data` directory in the container for persistent storage

## Additional Resources

- [TensorFlow Docker Documentation](https://www.tensorflow.org/install/docker)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker Documentation](https://docs.docker.com/) 