from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .length_classifier import create_length_classifier
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import tensorflow as tf
import pandas as pd
import seaborn as sns

def train_length_classifier(X_train, y_train, num_classes, epochs=20, batch_size=64, class_weights=None, output_dir='outputs'):
    """
    Train the length classification model
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets (one-hot encoded)
    - num_classes: Number of length categories
    - epochs: Maximum number of epochs to train
    - batch_size: Batch size for training
    - class_weights: Optional weights for imbalanced classes
    - output_dir: Directory to save output files
    
    Returns:
    - Trained model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (features, 1)
    model = create_length_classifier(input_shape, num_classes)

    # Print model summary
    model.summary()
    
    # Print the shape of the training data
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Create directories for model checkpoints
    checkpoint_dir = os.path.join(output_dir, 'length_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define callbacks
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        # Model checkpoint
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_length_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1,
            mode='max'
        ),
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode='max'
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
        class_weight=class_weights
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_training_history.png'))
    
    return model

def evaluate_length_classifier(model, X_test, y_test, label_encoder, output_dir='outputs'):
    """
    Evaluate the length classification model
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test targets (one-hot encoded)
    - label_encoder: LabelEncoder used to encode the length categories
    - output_dir: Directory to save output files
    
    Returns:
    - Accuracy score
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save classification report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'length_classification_report.csv'))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_confusion_matrix.png'))
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(14, 12))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_confusion_matrix_normalized.png'))
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        per_class_accuracy[class_name] = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    
    print("\nPer-class accuracy:")
    for class_name, acc in per_class_accuracy.items():
        print(f"  {class_name}: {acc:.4f}")
    
    # Save per-class accuracy to file
    with open(os.path.join(output_dir, 'length_per_class_accuracy.txt'), 'w') as f:
        f.write("Per-class accuracy:\n")
        for class_name, acc in per_class_accuracy.items():
            f.write(f"{class_name}: {acc:.4f}\n")
    
    return accuracy 