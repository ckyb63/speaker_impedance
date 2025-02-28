import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model.cnn_model import create_cnn_model

# Reset TensorFlow state
tf.keras.backend.clear_session()

# Make sure mixed precision is disabled globally
try:
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Set global policy to float32")
except Exception as e:
    print(f"Could not set global policy: {e}")

def test_model():
    """Test if the model works with a small synthetic dataset"""
    print("\nTesting model with synthetic data...")
    
    # Create a simple input shape (3 features, 1 channel)
    input_shape = (3, 1)
    
    # Create model with mixed precision explicitly disabled
    model = create_cnn_model(input_shape, mixed_precision=False)
    
    # Create synthetic dataset
    n_samples = 1000
    X = np.random.random((n_samples, 3, 1)).astype(np.float32)
    y = np.random.random((n_samples, 1)).astype(np.float32)
    
    # Split into train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train for a few epochs
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    preds = model.predict(X_test[:5])
    for i in range(5):
        print(f"True: {y_test[i][0]:.4f}, Predicted: {preds[i][0]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('test_training_history.png')
    print("Training history saved to test_training_history.png")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except:
                print(f"Could not set memory growth for {device}")
    else:
        print("No GPU found. Using CPU for training.")
    
    # Run the test
    test_model() 