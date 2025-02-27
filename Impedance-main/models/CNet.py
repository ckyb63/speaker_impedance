from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D

class CNet:
    def __init__(self, input_shape, num_classes, drop_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', data_format="channels_first"),
            Dropout(self.drop_rate),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(self.drop_rate),
            Flatten(),
            Dense(self.num_classes, activation='softmax'),
        ])
        return model

    def get_model(self):
        return self.model