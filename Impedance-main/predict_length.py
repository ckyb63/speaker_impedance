import os
import sys
import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def min_max(col):
    """Normalize column using min-max scaling"""
    return (col - min(col)) / (max(col) - min(col))

def preprocess_csv(file_path, columns=None, start=0, end=500):
    """
    Preprocess a CSV file for prediction
    
    Parameters:
    - file_path: Path to the CSV file
    - columns: List of columns to use (options: "FREQ", "PH", "MAG", "RS", "XS", "REC")
    - start: Starting index for data range
    - end: Ending index for data range
    
    Returns:
    - Preprocessed data ready for model input
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Default columns if not specified
    if columns is None:
        columns = ["PH", "MAG", "RS", "XS", "REC"]
    
    # Read CSV file
    data = pd.read_csv(file_path, header=0)
    
    # Extract and preprocess columns
    processed_cols = []
    for col in columns:
        col = col.upper()
        if col == "FREQ":
            freq = min_max(data.iloc[:, 0][start:end+1])
            processed_cols.append(freq)
        elif col == "PH":
            phase = min_max(data.iloc[:, 1][start:end+1])
            processed_cols.append(phase)
        elif col == "MAG":
            mag = min_max(data.iloc[:, 2][start:end+1])
            processed_cols.append(mag)
        elif col == "RS":
            rs = min_max(data.iloc[:, 3][start:end+1])
            processed_cols.append(rs)
        elif col == "XS":
            xs = min_max(data.iloc[:, 4][start:end+1])
            processed_cols.append(xs)
        elif col == "REC":
            rs = min_max(data.iloc[:, 3][start:end+1])
            xs = min_max(data.iloc[:, 4][start:end+1])
            rec = rs + 1j * xs
            processed_cols.append(rec)
    
    # Reshape for model input
    input_data = np.array([processed_cols])
    
    return input_data

def create_label_encoder(speakers=None, lengths=None, speaker_differentiation=False):
    """
    Create and fit a label encoder for the model output
    
    Parameters:
    - speakers: List of speakers (options: "A", "B", "C", "D")
    - lengths: List of lengths
    - speaker_differentiation: Whether to include speaker in the label
    
    Returns:
    - Fitted LabelEncoder
    """
    # Default values if not specified
    if speakers is None:
        speakers = ["A", "B", "C", "D"]
    if lengths is None:
        lengths = ["5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"]
    
    # Create labels
    labels = []
    for speaker in speakers:
        for length in lengths:
            # Format the length to match the model's output format
            if length in ["5", "8", "9"]:
                length = "0" + length
            if length == "Blocked":
                length = "00" + length
                
            # Create the label
            if speaker_differentiation:
                label = f"{speaker}_{length}"
            else:
                label = length
                
            labels.append(label)
    
    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    return label_encoder

def predict_length(model_path, csv_file, model_type="DNet", columns=None, start=0, end=500, 
                  speakers=None, lengths=None, speaker_differentiation=False):
    """
    Predict the length using the trained model
    
    Parameters:
    - model_path: Path to the trained model file
    - csv_file: Path to the CSV file to predict
    - model_type: Type of model ("DNet" or "CNet")
    - columns: List of columns to use
    - start: Starting index for data range
    - end: Ending index for data range
    - speakers: List of speakers
    - lengths: List of lengths
    - speaker_differentiation: Whether to include speaker in the label
    
    Returns:
    - Predicted length and confidence
    """
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    
    # Preprocess the CSV file
    input_data = preprocess_csv(csv_file, columns, start, end)
    
    # Reshape for CNet if needed
    if model_type == "CNet":
        input_data = input_data[:, np.newaxis, :]
    
    # Get the label encoder
    label_encoder = create_label_encoder(speakers, lengths, speaker_differentiation)
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Get prediction probability
    prediction_probability = np.max(prediction, axis=1)[0]
    
    return predicted_class, prediction_probability

def main():
    parser = argparse.ArgumentParser(description='Predict length from a CSV file using a trained model')
    parser.add_argument('--model', type=str, default='best_model.keras', 
                        help='Path to the trained model file')
    parser.add_argument('--csv', type=str, required=True, 
                        help='Path to the CSV file to predict')
    parser.add_argument('--model_type', type=str, choices=['DNet', 'CNet'], default='DNet',
                        help='Type of model (DNet or CNet)')
    parser.add_argument('--speaker_diff', action='store_true',
                        help='Enable speaker differentiation in labels')
    
    args = parser.parse_args()
    
    try:
        # Predict the length
        predicted_class, confidence = predict_length(
            args.model, 
            args.csv, 
            model_type=args.model_type,
            speaker_differentiation=args.speaker_diff
        )
        
        # Print the results
        print(f"\nPrediction Results:")
        print(f"CSV File: {args.csv}")
        print(f"Predicted Length: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 