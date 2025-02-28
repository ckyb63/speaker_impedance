import os
import numpy as np
import tensorflow as tf
from model.cnn_model import create_cnn_model

def test_model_creation():
    """Test if the model can be created and run without errors"""
    print("Testing model creation...")
    
    # Create a simple input shape
    input_shape = (3, 1)  # 3 features, 1 channel
    
    # Create model without mixed precision
    model = create_cnn_model(input_shape, mixed_precision=False)
    
    # Create some dummy data
    X = np.random.random((10, 3, 1)).astype(np.float32)
    y = np.random.random((10, 1)).astype(np.float32)
    
    # Try to predict
    print("Testing prediction...")
    pred = model.predict(X[:1])
    print(f"Prediction shape: {pred.shape}, value: {pred[0][0]}")
    
    # Try to train for 1 batch
    print("Testing training for 1 batch...")
    history = model.fit(X, y, epochs=1, batch_size=5, verbose=1)
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {len(physical_devices)} GPU(s)")
    
    # Run the test
    test_model_creation() 