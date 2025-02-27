import sys
import numpy as np
import uuid

from config import data_config, train_config
from models.DNet import DNet
from models.CNet import CNet

from utils.data_setup import create_dataset
from utils.eval import draw_heatmap

def run(config):
    (X_train, X_val, X_test, y_train, y_val, y_test), label_encoder = create_dataset(config)

    chosen_model = DNet(np.shape(X_train[0]), len(y_train[0]), config["drop_rate"])

    if (config["model"] == "CNet"):
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]

        chosen_model = CNet(np.shape(X_train[0]), len(y_train[0]), config["drop_rate"])

    model = chosen_model.get_model()
    model.summary()

    print("Training...")
    model.compile(loss=config["loss_fn"], optimizer=config["optimizer"], metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config["epoch"], batch_size=config["batch_size"])

    path = config["output_path"] + "models/" + uuid.uuid4().hex + ".h5"
    print(f"saving model to {path}")
    model.save(path)

    print("Test/evaluation...")
    # model.evaluate(X_test, y_test, batch_size=config["test_batch"])
    draw_heatmap(X_test, y_test, model, label_encoder)

    return 

def main(args):
    config = dict()
    config.update(data_config.get_config())
    config.update(train_config.get_config())

    return run(config)

if __name__ == "__main__":
    main(sys.argv[1:])