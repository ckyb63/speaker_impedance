from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import numpy as np
import tensorflow as tf
import pandas as pd

def add_noise(X, noise_factor=0.05):
    """Add random noise to the data for augmentation"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def time_warp(X, max_shift=2):
    """Apply time warping to the data"""
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
    """Scale features by a random factor"""
    scales = np.random.uniform(1-scale_factor, 1+scale_factor, size=(X.shape[0], 1, X.shape[2]))
    return X * scales

def augment_data(X_train, y_train, augment_factor=0.3):
    """
    Augment the training data with multiple techniques
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets (one-hot encoded)
    - augment_factor: Factor for data augmentation (0.3 = 30% more data)
    
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
    y_augmented = np.vstack([y_train, y_train[indices], y_train[indices], y_train[indices]])
    
    # Shuffle the augmented data
    shuffle_idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]
    
    return X_augmented, y_augmented

def group_by_frequency_and_metadata(df, num_freq_points=50):
    """
    Group data by metadata (Speaker, Length, Filename) and create frequency-based sequences
    
    Parameters:
    - df: DataFrame containing the data
    - num_freq_points: Number of frequency points to include in each sequence
    
    Returns:
    - X_sequences: Array of shape (n_samples, n_freq_points, n_features)
    - y_labels: Array of length categories
    - metadata: DataFrame with metadata for each sequence
    """
    # Check required columns
    required_cols = ['Frequency (Hz)', 'Trace θ (deg)', 'Trace Rs (Ohm)', 'Trace Xs (Ohm)', 'Trace |Z| (Ohm)', 'Speaker', 'Length', 'Filename']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Sort by metadata and frequency
    df_sorted = df.sort_values(['Speaker', 'Length', 'Filename', 'Frequency (Hz)'])
    
    # Get unique combinations of metadata
    metadata_groups = df_sorted.groupby(['Speaker', 'Length', 'Filename'])
    
    X_sequences = []
    y_labels = []
    metadata_records = []
    
    for (speaker, length, filename), group in metadata_groups:
        # Skip groups with too few frequency points
        if len(group) < num_freq_points:
            print(f"Skipping group with only {len(group)} frequency points: {speaker}, {length}, {filename}")
            continue
        
        # Sample or interpolate to get exactly num_freq_points
        if len(group) > num_freq_points:
            # Sample evenly spaced frequency points
            indices = np.linspace(0, len(group) - 1, num_freq_points, dtype=int)
            group_sampled = group.iloc[indices]
        else:
            group_sampled = group
        
        # Extract features
        features = group_sampled[['Frequency (Hz)', 'Trace θ (deg)', 'Trace Rs (Ohm)', 'Trace Xs (Ohm)', 'Trace |Z| (Ohm)']].values
        
        # Add to sequences
        X_sequences.append(features)
        y_labels.append(length)
        metadata_records.append({'Speaker': speaker, 'Length': length, 'Filename': filename})
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    
    # Create metadata DataFrame
    metadata = pd.DataFrame(metadata_records)
    
    return X_sequences, y_labels, metadata

def preprocess_data_for_length_classification(df, augment_factor=0.3, test_size=0.25, num_freq_points=50):
    """
    Preprocess data for length classification, incorporating frequency information
    
    Parameters:
    - df: DataFrame containing the data
    - augment_factor: Factor for data augmentation (0.3 = 30% more data)
    - test_size: Proportion of data to use for testing
    - num_freq_points: Number of frequency points to include in each sequence
    
    Returns:
    - X_train, X_test, y_train, y_test, label_encoder, class_weights
    """
    # Check if Length column exists
    if 'Length' not in df.columns:
        raise ValueError("DataFrame must contain a 'Length' column")
    
    # Group by frequency and metadata to create sequences
    print(f"Creating frequency-based sequences with {num_freq_points} points per sequence...")
    X_sequences, y_labels, metadata = group_by_frequency_and_metadata(df, num_freq_points)
    
    print(f"Created {len(X_sequences)} sequences from {len(df)} original samples")
    print(f"Sequence shape: {X_sequences.shape}")
    
    # Encode the Length labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    
    # One-hot encode the labels for multi-class classification
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X_sequences, y, metadata, test_size=test_size, random_state=42, stratify=y
    )
    
    # Use RobustScaler for better handling of outliers
    # Scale each feature independently
    scaler = RobustScaler()
    
    # Reshape to 2D for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[2])
    
    # Scale the data
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train = X_train_2d.reshape(X_train.shape)
    X_test = X_test_2d.reshape(X_test.shape)
    
    # Calculate class weights to handle imbalanced data
    class_counts = np.sum(y_train, axis=0)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    class_weights = {}
    for i in range(n_classes):
        # Compute balanced weight: total_samples / (n_classes * samples_in_class)
        class_weights[i] = n_samples / (n_classes * class_counts[i])
    
    # Apply data augmentation to training set
    if augment_factor > 0:
        print(f"Applying data augmentation with factor {augment_factor}...")
        X_train, y_train = augment_data(X_train, y_train, augment_factor)
        print(f"Training data size after augmentation: {X_train.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, label_encoder, class_weights 