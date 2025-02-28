from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import tensorflow as tf

def add_noise(X, noise_factor=0.05):
    """Add random noise to the data for augmentation (restored to original level)"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def time_warp(X, max_shift=2):
    """Apply time warping to the data (restored to original level)"""
    result = np.zeros_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift+1)
        if shift > 0:
            result[i, shift:, :] = X[i, :-shift, :]
            result[i, :shift, :] = X[i, 0, :]
        elif shift < 0:
            result[i, :shift, :] = X[i, -shift:, :]
            result[i, shift:, :] = X[i, -1, :]
        else:
            result[i] = X[i]
    return result

def scale_features(X, scale_factor=0.1):
    """Scale features by a random factor (increased from 0.05 to 0.1)"""
    scales = np.random.uniform(1-scale_factor, 1+scale_factor, size=(X.shape[0], 1, X.shape[2]))
    return X * scales

def augment_data(X_train, y_train, augment_factor=0.2):
    """
    Augment the training data with multiple techniques
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - augment_factor: Factor for data augmentation (0.2 = 20% more data)
    
    Returns:
    - Augmented training data
    """
    if augment_factor <= 0:
        return X_train, y_train
    
    # Calculate how many augmented samples to create
    n_augment = int(X_train.shape[0] * augment_factor)
    
    # Create indices for augmentation (random selection with replacement)
    indices = np.random.choice(X_train.shape[0], n_augment, replace=True)
    
    # Create augmented data with multiple techniques
    X_noise = add_noise(X_train[indices])
    X_warp = time_warp(X_train[indices])
    X_scale = scale_features(X_train[indices])
    
    # Combine original and augmented data
    X_augmented = np.vstack([X_train, X_noise, X_warp, X_scale])
    y_augmented = np.concatenate([y_train, y_train[indices], y_train[indices], y_train[indices]])
    
    # Shuffle the augmented data
    shuffle_idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]
    
    return X_augmented, y_augmented

def preprocess_data(df, augment_factor=0.2, test_size=0.25, return_indices=False):
    """
    Preprocess data for model training with improved normalization
    
    Parameters:
    - df: DataFrame containing the data
    - augment_factor: Factor for data augmentation (0.2 = 20% more data)
    - test_size: Proportion of data to use for testing
    - return_indices: Whether to return the indices of the test set
    
    Returns:
    - X_train, X_test, y_train, y_test, scaler, (test_indices if return_indices=True)
    """
    # Select relevant columns - now using Deg (θ), Rs, and Xs as requested
    X = df[['Trace θ (deg)', 'Trace Rs (Ohm)', 'Trace Xs (Ohm)']].values  # Features
    y = df['Trace |Z| (Ohm)'].values  # Target variable

    # Reshape X for CNN (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add a channel dimension

    # Create indices array
    indices = np.arange(len(df))

    # Split the data into training and testing sets with increased test size
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=test_size, random_state=42, shuffle=True
    )

    # Use RobustScaler instead of StandardScaler for better handling of outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)
    
    # Apply data augmentation to training set with reduced factor
    if augment_factor > 0:
        print(f"Applying data augmentation with factor {augment_factor}...")
        X_train, y_train = augment_data(X_train, y_train, augment_factor)
        print(f"Training data size after augmentation: {X_train.shape[0]} samples")

    if return_indices:
        return X_train, X_test, y_train, y_test, scaler, test_indices
    else:
        return X_train, X_test, y_train, y_test, scaler 