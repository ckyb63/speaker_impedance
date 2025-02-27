import numpy as np

def predict_length(model, scaler, new_data):
    """
    Make predictions using the trained model
    
    Parameters:
    - model: Trained Keras model
    - scaler: Fitted StandardScaler
    - new_data: Array of [Trace θ (deg), Trace Rs (Ohm), Trace Xs (Ohm)]
    
    Returns:
    - Predicted impedance magnitude
    """
    # Ensure data is in the right shape
    if len(new_data.shape) == 1:
        # Reshape for a single sample
        new_data = np.array(new_data).reshape(1, len(new_data))
    
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Reshape for CNN (samples, features, 1)
    new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    
    return prediction 