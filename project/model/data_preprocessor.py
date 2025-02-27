from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df):
    # Select relevant columns - now using Deg (θ), Rs, and Xs as requested
    X = df[['Trace θ (deg)', 'Trace Rs (Ohm)', 'Trace Xs (Ohm)']].values  # Features
    y = df['Trace |Z| (Ohm)'].values  # Target variable

    # Reshape X for CNN (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add a channel dimension

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test, scaler 