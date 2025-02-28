import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_cnn_model(input_shape, mixed_precision=False):
    """
    Create a CNN model for impedance prediction with balanced regularization
    to prevent both underfitting and overfitting
    
    Parameters:
    - input_shape: Shape of input data (features, channels)
    - mixed_precision: Parameter kept for backward compatibility but not used
    
    Returns:
    - Compiled Keras model
    """
    # Force float32 precision for all operations
    tf.keras.backend.clear_session()
    try:
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
    except Exception as e:
        print(f"Warning: Could not set precision policy: {e}")
    
    # Use Sequential API for a cleaner model architecture
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape, dtype='float32'),
        
        # First convolutional block - increased filters to 32
        layers.Conv1D(32, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005),  # Reduced L2 from 0.001 to 0.0005
                     dtype='float32'),
        layers.BatchNormalization(dtype='float32'),
        layers.Activation('relu'),
        layers.Dropout(0.2),  # Reduced dropout from 0.3 to 0.2
        
        # Second convolutional block - increased filters to 64
        layers.Conv1D(64, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005),  # Reduced L2 from 0.001 to 0.0005
                     dtype='float32'),
        layers.BatchNormalization(dtype='float32'),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.Dropout(0.2),  # Reduced dropout from 0.3 to 0.2
        
        # Third convolutional block - added for more capacity
        layers.Conv1D(128, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005),  # Reduced L2 from 0.001 to 0.0005
                     dtype='float32'),
        layers.BatchNormalization(dtype='float32'),
        layers.Activation('relu'),
        layers.Dropout(0.2),  # Reduced dropout from 0.3 to 0.2
        
        # Global pooling
        layers.GlobalAveragePooling1D(),
        
        # Dense layers - added more capacity
        layers.Dense(64, 
                    kernel_regularizer=regularizers.l2(0.0005),  # Reduced L2 from 0.001 to 0.0005
                    dtype='float32'),
        layers.BatchNormalization(dtype='float32'),
        layers.Activation('relu'),
        layers.Dropout(0.3),  # Reduced dropout from 0.4 to 0.3
        
        # Second dense layer
        layers.Dense(32, 
                    kernel_regularizer=regularizers.l2(0.0005),  # Reduced L2 from 0.001 to 0.0005
                    dtype='float32'),
        layers.BatchNormalization(dtype='float32'),
        layers.Activation('relu'),
        layers.Dropout(0.2),  # Reduced dropout from 0.3 to 0.2
        
        # Output layer
        layers.Dense(1, dtype='float32')
    ])
    
    # Compile with Adam optimizer and MSE loss
    # Increased learning rate slightly
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model 