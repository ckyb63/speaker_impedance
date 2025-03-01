from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
import math

class DNet:
    def __init__(self, input_shape, num_classes, drop_rate=0.3, weight_decay=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            # First block
            Dense(128, input_shape=self.input_shape, activation='relu', 
                  kernel_regularizer=l2(self.weight_decay)),
            BatchNormalization(),
            Dropout(self.drop_rate),
            
            # Second block - wider
            Dense(256, activation='relu', kernel_regularizer=l2(self.weight_decay)),
            BatchNormalization(),
            Dropout(self.drop_rate),
            
            # Third block
            Dense(256, activation='relu', kernel_regularizer=l2(self.weight_decay)),
            BatchNormalization(),
            Dropout(self.drop_rate),
            
            # Fourth block
            Dense(128, activation='relu', kernel_regularizer=l2(self.weight_decay)),
            BatchNormalization(),
            Dropout(self.drop_rate/2),  # Reduced dropout before final layer
            
            # Flatten the output to match target shape
            Flatten(),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def get_model(self):
        return self.model
        
    def get_callbacks(self):
        """Returns useful callbacks for training"""
        # Learning rate scheduler
        def lr_schedule(epoch):
            initial_lr = 0.001
            if epoch > 75:
                return initial_lr * 0.01
            elif epoch > 50:
                return initial_lr * 0.1
            elif epoch > 25:
                return initial_lr * 0.5
            return initial_lr
            
        callbacks = [
            LearningRateScheduler(lr_schedule)
        ]
        return callbacks