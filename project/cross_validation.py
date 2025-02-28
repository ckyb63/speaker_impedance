import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
from model.data_loader import load_data
from model.data_preprocessor import preprocess_data
from model.cnn_model import create_cnn_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_kfold_cv(df, n_splits=5, epochs=8, batch_size=64, output_dir='outputs'):
    """
    Run k-fold cross-validation to better evaluate model performance
    
    Parameters:
    - df: DataFrame containing the data
    - n_splits: Number of folds for cross-validation
    - epochs: Maximum number of epochs to train
    - batch_size: Batch size for training
    - output_dir: Directory to save output files
    
    Returns:
    - Mean and std of MSE, MAE, and R² across folds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    cv_dir = os.path.join(output_dir, 'cv_results')
    os.makedirs(cv_dir, exist_ok=True)
    
    # Extract features and target
    X = df[['Trace θ (deg)', 'Trace Rs (Ohm)', 'Trace Xs (Ohm)']].values
    y = df['Trace |Z| (Ohm)'].values
    
    # Reshape X for CNN (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    mse_scores = []
    mae_scores = []
    r2_scores = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Preprocess data (without augmentation for validation)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[1])).reshape(X_val.shape)
        
        # Create and compile model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_cnn_model(input_shape, mixed_precision=False)
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.0001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Evaluate model
        y_pred = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Store metrics
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        # Print metrics for this fold
        print(f"Fold {fold+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Plot training history for this fold
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Fold {fold+1} - Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title(f'Fold {fold+1} - Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cv_dir, f'fold_{fold+1}_history.png'))
        plt.close()
        
        # Plot actual vs predicted for this fold
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Fold {fold+1} - Actual vs Predicted')
        plt.savefig(os.path.join(cv_dir, f'fold_{fold+1}_predictions.png'))
        plt.close()
    
    # Calculate mean and std of metrics
    mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    
    # Print summary
    print("\n--- Cross-Validation Results ---")
    print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"R²: {r2_mean:.4f} ± {r2_std:.4f}")
    
    # Plot metrics across folds
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(1, n_splits+1), mse_scores)
    plt.axhline(y=mse_mean, color='r', linestyle='--')
    plt.title('MSE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    
    plt.subplot(1, 3, 2)
    plt.bar(range(1, n_splits+1), mae_scores)
    plt.axhline(y=mae_mean, color='r', linestyle='--')
    plt.title('MAE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    
    plt.subplot(1, 3, 3)
    plt.bar(range(1, n_splits+1), r2_scores)
    plt.axhline(y=r2_mean, color='r', linestyle='--')
    plt.title('R² by Fold')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, 'cv_metrics.png'))
    plt.close()
    
    return mse_mean, mse_std, mae_mean, mae_std, r2_mean, r2_std

def main():
    # Load data with sampling to reduce dataset size
    folder_path = r'/home/max/speaker_stuff_wslver/speaker_impedance/Collected_Data_Sep16'
    
    # Use more files for cross-validation
    max_files = 2000  # Increased from 1500 to 2000
    sample_rate = 1.0  # Use all rows for better accuracy
    
    print(f"Loading data with max_files={max_files}, sample_rate={sample_rate}...")
    df = load_data(folder_path, max_files=max_files, sample_rate=sample_rate)
    
    # Run k-fold cross-validation
    output_dir = 'outputs'
    n_splits = 5
    epochs = 15  # Increased from 10 to 15
    batch_size = 128  # Keep batch size at 128
    
    print(f"\nRunning {n_splits}-fold cross-validation...")
    mse_mean, mse_std, mae_mean, mae_std, r2_mean, r2_std = run_kfold_cv(
        df, n_splits=n_splits, epochs=epochs, batch_size=batch_size, output_dir=output_dir
    )
    
    # Save results to file
    with open(os.path.join(output_dir, 'cv_results.txt'), 'w') as f:
        f.write("Cross-Validation Results\n")
        f.write("=======================\n\n")
        f.write(f"Number of folds: {n_splits}\n")
        f.write(f"Max files: {max_files}\n")
        f.write(f"Sample rate: {sample_rate}\n\n")
        f.write(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}\n")
        f.write(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}\n")
        f.write(f"R²: {r2_mean:.4f} ± {r2_std:.4f}\n")
    
    print(f"\nCross-validation complete. Results saved to {os.path.join(output_dir, 'cv_results.txt')}")

if __name__ == "__main__":
    main() 