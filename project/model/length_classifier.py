import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_length_classifier(input_shape, num_classes):
    """
    Create a CNN model for length classification that leverages frequency-based sequences
    
    Parameters:
    - input_shape: Shape of input data (frequency_points, features)
    - num_classes: Number of length categories to predict
    
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
    
    # Input layer
    inputs = layers.Input(shape=input_shape, dtype='float32')
    
    # Extract frequency as a separate feature
    frequency = layers.Lambda(lambda x: x[:, :, 0:1])(inputs)
    
    # Extract impedance features
    impedance_features = layers.Lambda(lambda x: x[:, :, 1:])(inputs)
    
    # Process impedance features with 1D convolutions
    x = layers.Conv1D(64, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005))(impedance_features)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block
    x = layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    # Third convolutional block with dilation to capture wider frequency patterns
    x = layers.Conv1D(256, kernel_size=3, padding='same', dilation_rate=2,
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Process frequency information separately
    freq_x = layers.Conv1D(32, kernel_size=3, padding='same')(frequency)
    freq_x = layers.BatchNormalization()(freq_x)
    freq_x = layers.Activation('relu')(freq_x)
    
    # Use Global Average Pooling to reduce frequency dimension
    freq_x = layers.GlobalAveragePooling1D()(freq_x)
    freq_x = layers.Reshape((1, 32))(freq_x)  # Reshape to match the impedance features
    
    # Combine impedance features with frequency information
    combined = layers.Concatenate()([x, freq_x])
    
    # Apply attention mechanism to focus on important frequency regions
    attention = layers.Dense(1, activation='tanh')(combined)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    attention_weights = layers.Reshape((attention_weights.shape[1], 1))(attention_weights)
    
    # Apply attention weights
    context_vector = layers.Multiply()([combined, attention_weights])
    
    # Global pooling with both max and average pooling for better feature extraction
    max_pool = layers.GlobalMaxPooling1D()(context_vector)
    avg_pool = layers.GlobalAveragePooling1D()(context_vector)
    pooled = layers.Concatenate()([max_pool, avg_pool])
    
    # Dense layers
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005))(pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Second dense layer
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer - softmax for multi-class classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer and categorical crossentropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 