import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, mixed_precision=True):
    """
    Create a CNN model for impedance prediction
    
    Parameters:
    - input_shape: Shape of input data (features, channels)
    - mixed_precision: Whether to use mixed precision training
    
    Returns:
    - Compiled Keras model
    """
    # Enable mixed precision if requested (speeds up training on compatible GPUs)
    if mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        except:
            print("Mixed precision not available")
    
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))  # Input layer
    
    # First convolutional block - smaller and more efficient
    model.add(layers.Conv1D(32, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    # Second convolutional block
    model.add(layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, padding='same'))
    model.add(layers.Dropout(0.3))
    
    # Flatten and dense layers - reduced size
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output layer for regression
    
    # Compile with Adam optimizer and MSE loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # If using mixed precision, use loss scaling to prevent underflow
    if mixed_precision:
        try:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        except:
            pass
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model 