from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .cnn_model import create_cnn_model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
import os
import tensorflow as tf
import time
import threading

class GPUMonitor(Callback):
    """
    Callback to monitor GPU usage during training
    """
    def __init__(self, interval=5, output_dir='outputs'):
        super(GPUMonitor, self).__init__()
        self.interval = interval  # Interval in seconds
        self.stop_monitoring = False
        self.gpu_usage_history = []
        self.time_points = []
        self.start_time = None
        self.output_dir = output_dir
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(target=self._monitor_gpu)
        self.monitor_thread.start()
        
    def on_train_end(self, logs=None):
        # Stop monitoring
        self.stop_monitoring = True
        self.monitor_thread.join()
        
        # Plot GPU usage if data was collected
        if self.gpu_usage_history:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_points, self.gpu_usage_history)
            plt.title('GPU Memory Usage During Training')
            plt.xlabel('Time (seconds)')
            plt.ylabel('GPU Memory Usage (MB)')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'gpu_usage.png'))
            plt.close()
            
            # Print summary
            avg_usage = np.mean(self.gpu_usage_history)
            max_usage = np.max(self.gpu_usage_history)
            print(f"\nGPU Usage Summary:")
            print(f"Average GPU memory usage: {avg_usage:.2f} MB")
            print(f"Maximum GPU memory usage: {max_usage:.2f} MB")
            print(f"GPU usage graph saved to '{os.path.join(self.output_dir, 'gpu_usage.png')}'")
        
    def _monitor_gpu(self):
        """Monitor GPU memory usage in a separate thread"""
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU available for monitoring")
            return
            
        while not self.stop_monitoring:
            try:
                # Try to get GPU memory info
                memory_info = None
                try:
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    # Calculate used memory in MB
                    used_memory = (memory_info['total'] - memory_info['available']) / (1024 * 1024)
                    self.gpu_usage_history.append(used_memory)
                    self.time_points.append(time.time() - self.start_time)
                except Exception as e:
                    # Silently handle the error without printing to console
                    pass
            except Exception as e:
                # Silently handle any other errors without printing to console
                pass
                
            # Sleep for the specified interval
            time.sleep(self.interval)

def train_model(X_train, y_train, epochs=20, batch_size=64, mixed_precision=True, output_dir='outputs'):
    """
    Train the model with optimized parameters for faster training
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - epochs: Maximum number of epochs to train
    - batch_size: Batch size for training
    - mixed_precision: Whether to use mixed precision training
    - output_dir: Directory to save output files
    
    Returns:
    - Trained model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (features, 1)
    model = create_cnn_model(input_shape, mixed_precision=mixed_precision)

    # Print model summary
    model.summary()
    
    # Print the shape of the training data
    print(f"Training data shape: {X_train.shape}")

    # Create directories for model checkpoints and logs if they don't exist
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Define callbacks - optimized for balanced training
    callbacks = [
        # Early stopping with balanced patience
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Increased from 7 to 10 to allow more training time
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001  # Keep at 0.0001 for finer improvement detection
        ),
        # Model checkpoint
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Changed from 0.2 to 0.3 for less aggressive reduction
            patience=4,  # Increased from 3 to 4
            min_lr=1e-6,
            verbose=1
        ),
        # GPU monitoring
        GPUMonitor(interval=10, output_dir=output_dir)  # Check GPU usage every 10 seconds
    ]

    # Train the model with increased validation split
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.25,  # Decreased from 0.3 to 0.25 to allow more training data
        verbose=1,
        callbacks=callbacks,
        shuffle=True  # Ensure data is shuffled each epoch
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot MAE if available
    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    return model

def evaluate_model(model, X_test, y_test, output_dir='outputs'):
    """
    Evaluate the model on test data
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test targets
    - output_dir: Directory to save output files
    
    Returns:
    - Mean squared error
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a larger batch size for prediction to speed up evaluation
    predictions = model.predict(X_test, batch_size=256)
    
    # Calculate multiple metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    # Plot actual vs predicted values - use a sample if dataset is large
    if len(y_test) > 1000:
        # Sample 1000 points for the plot
        indices = np.random.choice(len(y_test), 1000, replace=False)
        y_test_sample = y_test[indices]
        predictions_sample = predictions[indices]
    else:
        y_test_sample = y_test
        predictions_sample = predictions
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_sample, predictions_sample, alpha=0.5)
    plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    
    return mse

def verify_model(model, X_test, y_test, output_dir='outputs'):
    """
    Verify model performance with detailed error analysis
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test targets
    - output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a larger batch size for prediction
    predictions = model.predict(X_test, batch_size=256)
    
    # Calculate error statistics
    errors = y_test - predictions.flatten()
    abs_errors = np.abs(errors)
    
    print(f'Verification Mean Squared Error: {mean_squared_error(y_test, predictions):.4f}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}')
    print(f'Max Absolute Error: {np.max(abs_errors):.4f}')
    print(f'Min Absolute Error: {np.min(abs_errors):.4f}')
    print(f'Standard Deviation of Errors: {np.std(errors):.4f}')
    
    # Plot error distribution - sample if dataset is large
    if len(errors) > 1000:
        errors_sample = np.random.choice(errors, 1000, replace=False)
    else:
        errors_sample = errors
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors_sample, bins=30)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(os.path.join(output_dir, 'error_distribution.png')) 