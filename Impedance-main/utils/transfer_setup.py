from keras.models import Sequential
from keras.layers import Dense

from keras.models import load_model

def get_finetune_model(path, output_shape):
    print(f"loading model from {path}")
    pretrained_model = load_model(path)

    pretrained_model.pop()

    model = Sequential([
        pretrained_model,
        Dense(output_shape, activation='softmax'),
    ])

    model.layers[0].trainable = False

    return model