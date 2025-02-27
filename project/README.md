# Speaker Impedance Prediction

This project uses machine learning to predict speaker impedance magnitude (|Z|) based on phase angle (θ), series resistance (Rs), and series reactance (Xs) measurements.

## Features

- Data loading and preprocessing from CSV files
- Advanced CNN model architecture with regularization techniques
- Comprehensive model evaluation and visualization
- Prediction capabilities for new impedance data
- Optimized for faster training with GPU support
- Support for CUDA 12.8 and cuDNN 9.7.1

## Project Structure

```
project/
├── main.py                    # Main script to run the entire pipeline
├── requirements.txt           # Dependencies
├── test_gpu.py                # Script to test GPU availability and performance
├── setup_cuda_compatibility.py # Script to check CUDA compatibility
├── verify_cuda12_setup.py     # Script to verify CUDA 12.8 setup
├── setup_cuda12_env.bat       # Batch script to set up CUDA 12.8 environment variables
├── GPU_TROUBLESHOOTING.md     # Guide for troubleshooting GPU issues
├── CUDA12_COMPATIBILITY.md    # Guide for CUDA 12.8 compatibility
├── model/
│   ├── data_loader.py         # Functions to load data from CSV files
│   ├── data_preprocessor.py   # Data preprocessing functions
│   ├── cnn_model.py           # CNN model architecture
│   ├── model_trainer.py       # Model training and evaluation
│   └── predictor.py           # Functions for making predictions
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For GPU support (recommended for faster training):
   - Install CUDA and cuDNN compatible with your TensorFlow version
   - Verify GPU is detected by running:
     ```python
     import tensorflow as tf; print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
     ```

### CUDA 12.8 and cuDNN 9.7.1 Support

If you have CUDA 12.8 and cuDNN 9.7.1 installed:

1. Run the setup script to configure environment variables (Windows):
   ```
   setup_cuda12_env.bat
   ```

2. Verify your CUDA 12.8 setup:
   ```
   python setup_cuda_compatibility.py
   python verify_cuda12_setup.py
   ```

3. For detailed instructions, see:
   ```
   CUDA12_COMPATIBILITY.md
   ```

## Usage

1. Update the data path in `main.py` to point to your CSV files
2. Adjust training parameters in `main.py`:
   - `max_files`: Maximum number of files to load (reduce for faster training)
   - `sample_rate`: Fraction of rows to use from each file (reduce for faster training)
   - `epochs`: Number of training epochs
   - `batch_size`: Batch size for training
3. Run the main script:
   ```
   python main.py
   ```

## Performance Optimization

The code includes several optimizations for faster training:

1. **Data Sampling**: Only loads a subset of files and samples a fraction of rows
2. **Mixed Precision Training**: Uses FP16 computation when available on compatible GPUs
3. **Smaller Model**: Reduced model size with fewer parameters
4. **Batch Size Optimization**: Larger batch sizes for faster training
5. **Early Stopping**: Prevents unnecessary training epochs
6. **GPU Memory Growth**: Configures GPU memory allocation for better performance
7. **GPU Monitoring**: Tracks GPU memory usage during training

## Model Architecture

The model uses a CNN architecture with the following features:
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Dense layers for final prediction
- Adam optimizer with learning rate scheduling

## Data Format

The model expects CSV files with the following columns:
- `Frequency (Hz)`: Frequency in Hertz
- `Trace θ (deg)`: Phase angle in degrees
- `Trace |Z| (Ohm)`: Impedance magnitude in Ohms (target variable)
- `Trace Rs (Ohm)`: Series resistance in Ohms
- `Trace Xs (Ohm)`: Series reactance in Ohms

## Output

The model generates several output files:
- `trained_model.keras`: The trained model
- `checkpoints/best_model.keras`: The best model during training
- `training_history.png`: Training and validation metrics
- `actual_vs_predicted.png`: Scatter plot of actual vs predicted values
- `error_distribution.png`: Distribution of prediction errors
- `confusion_matrix.png`: Confusion matrix of binned predictions
- `feature_distributions.png`: Distributions of input features
- `gpu_usage.png`: GPU memory usage during training (if GPU is used)

## GPU Troubleshooting

If you encounter issues with GPU acceleration, refer to:
```
GPU_TROUBLESHOOTING.md
```

For CUDA 12.8 specific issues, see:
```
CUDA12_COMPATIBILITY.md
``` 