import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np

def analyze_training_history(image_path):
    """
    Analyze the training history plot to check for signs of overfitting
    """
    print(f"Analyzing training history from: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    print("Training history plot exists. Please examine it manually for:")
    print("1. Gap between training and validation loss")
    print("2. Validation loss increasing while training loss decreases")
    print("3. Early stopping point")
    print("4. Learning rate reductions")

def analyze_predictions(image_path):
    """
    Analyze the actual vs predicted plot to check for model performance
    """
    print(f"\nAnalyzing predictions from: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    print("Actual vs Predicted plot exists. Please examine it manually for:")
    print("1. Points clustering around the diagonal line")
    print("2. Outliers or systematic errors")
    print("3. Regions of poor prediction")

def analyze_error_distribution(image_path):
    """
    Analyze the error distribution to check for bias
    """
    print(f"\nAnalyzing error distribution from: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    print("Error distribution plot exists. Please examine it manually for:")
    print("1. Symmetry around zero")
    print("2. Presence of outliers")
    print("3. Shape of distribution (should be approximately normal)")

def analyze_confusion_matrix(image_path):
    """
    Analyze the confusion matrix to check for classification performance
    """
    print(f"\nAnalyzing confusion matrix from: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    print("Confusion matrix plot exists. Please examine it manually for:")
    print("1. Concentration of values along the diagonal")
    print("2. Off-diagonal patterns indicating systematic misclassification")
    print("3. Overall accuracy")

# Define the exact length categories from the LEN array
LENGTH_CATEGORIES = ["5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"]

def plot_confusion_matrix(y_true, y_pred, output_dir='outputs', normalize=True):
    """
    Plot confusion matrix for classification results with improved visualization
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - output_dir: Directory to save output files
    - normalize: Whether to normalize the confusion matrix by row (true labels)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique labels from both true and predicted values
    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    
    # Use the exact length categories if they match the unique labels
    if set(unique_labels).issubset(set(LENGTH_CATEGORIES)):
        labels = [label for label in LENGTH_CATEGORIES if label in unique_labels]
    else:
        labels = unique_labels
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create a normalized version for better visualization
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
        disp.plot(cmap='Blues', values_format='.2f')
        plt.title('Normalized Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    
    # Plot raw counts confusion matrix
    plt.figure(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Classification Accuracy: {accuracy:.4f}')
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, label in enumerate(labels):
        if cm[i].sum() > 0:  # Avoid division by zero
            per_class_accuracy[label] = cm[i, i] / cm[i].sum()
        else:
            per_class_accuracy[label] = 0
    
    print("\nPer-class accuracy:")
    for label, acc in per_class_accuracy.items():
        print(f"  {label}: {acc:.4f}")
    
    # Save per-class accuracy to file
    with open(os.path.join(output_dir, 'per_class_accuracy.txt'), 'w') as f:
        f.write("Per-class accuracy:\n")
        for label, acc in per_class_accuracy.items():
            f.write(f"{label}: {acc:.4f}\n")
    
    return accuracy

def main():
    output_dir = "outputs"
    
    # Analyze training history
    training_history_path = os.path.join(output_dir, "training_history.png")
    analyze_training_history(training_history_path)
    
    # Analyze predictions
    predictions_path = os.path.join(output_dir, "actual_vs_predicted.png")
    analyze_predictions(predictions_path)
    
    # Analyze error distribution
    error_path = os.path.join(output_dir, "error_distribution.png")
    analyze_error_distribution(error_path)
    
    # Analyze confusion matrix
    confusion_path = os.path.join(output_dir, "confusion_matrix.png")
    analyze_confusion_matrix(confusion_path)
    
    print("\nSummary:")
    print("To determine if the model is overfitting, look for:")
    print("1. Large gap between training and validation loss")
    print("2. Validation loss increasing while training loss continues to decrease")
    print("3. Poor generalization on test data (scattered points in actual vs predicted)")
    print("4. Biased or skewed error distribution")
    print("5. Poor performance on certain classes in the confusion matrix")
    
    print("\nPossible solutions if overfitting is detected:")
    print("1. Further reduce model complexity")
    print("2. Increase dropout rates")
    print("3. Add more L2 regularization")
    print("4. Collect more training data")
    print("5. Implement k-fold cross-validation")
    print("6. Reduce training epochs")

if __name__ == "__main__":
    main() 