# Use the official TensorFlow GPU image as base
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /workspace

# Install additional Python packages
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command
CMD ["bash"] 