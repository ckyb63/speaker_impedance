import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

def create_dataset(config):
    input_data, output_data = get_data(config)

    label_encoder = LabelEncoder()
    label_encoder.fit(output_data)
    num_classes = len(label_encoder.classes_)

    output_data = label_encoder.transform(output_data)
    output_data = to_categorical(output_data, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=config["test"], random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(config["val"]/(1-config["test"])), random_state=1) 
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), label_encoder

def get_data(config):
    input_data = []
    output_data = []

    for speaker in config["speakers"]:
        speaker_path = os.path.join(config["data_path"], speaker.upper())
        for length in config["len"]:
            label = speaker+"_"+length
            len_path = os.path.join(speaker_path, label)
            for file in os.listdir(len_path):
                path = os.path.join(len_path, file)
                data = pd.read_csv(path, header=0)

                cols = select_column(data, config)

                if (length in ["5", "8", "9"]):
                    length = "0" + length
                if (length == "Blocked"):
                    length = "00" + length
                if config["speaker_dif"]:
                    out = speaker+"_"+length
                else:
                    out = length

                input_data.append(cols)
                output_data.append(out)
    
    return np.array(input_data), np.array(output_data)

def select_column(data, config):
    cols = []
    for col in config["column"]:
        col = col.upper()
        if col == "FREQ":
            freq = min_max(data.iloc[:, 0][config["start"]:config["end"]+1])
            cols.append(freq)
        elif col == "PH":
            phase = min_max(data.iloc[:, 1][config["start"]:config["end"]+1])
            cols.append(phase)
        elif col == "MAG":
            mag = min_max(data.iloc[:, 2][config["start"]:config["end"]+1])
            cols.append(mag)
        elif col == "RS":
            rs = min_max(data.iloc[:, 3][config["start"]:config["end"]+1])
            cols.append(rs)
        elif col == "XS":
            xs = min_max(data.iloc[:, 4][config["start"]:config["end"]+1])
            cols.append(xs)
        elif col == "REC":
            rs = min_max(data.iloc[:, 3][config["start"]:config["end"]+1])
            xs = min_max(data.iloc[:, 4][config["start"]:config["end"]+1])
            rec = rs + 1j * xs
            cols.append(rec)
    return cols

def min_max(col):
    return (col - min(col)) / (max(col) - min(col))