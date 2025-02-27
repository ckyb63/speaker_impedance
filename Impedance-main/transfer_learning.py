import sys
import numpy as np
import uuid

from config import data_config, train_config
from models import DNet, CNet

from utils.data_setup import create_dataset
from utils.eval import draw_heatmap
from utils.transfer_setup import get_finetune_model

def run(config):
    # pretraining

    (X_train, X_val, X_test, y_train, y_val, y_test), label_encoder = create_dataset(config)
    input_shape = np.shape(X_train[0])
    output_shape = len(y_train[0])

    chosen_model = DNet(input_shape, output_shape, config["drop_rate"])

    if (config["model"] == "CNet"):
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]

        chosen_model = CNet(input_shape, output_shape, config["drop_rate"])

    model = chosen_model.get_model()
    model.summary()

    print("Training...")
    model.compile(loss=config["loss_fn"], optimizer=config["optimizer"], metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config["epoch"], batch_size=config["batch_size"])

    pretrain_path = config["output_path"] + uuid.uuid4().hex + ".h5"
    print(f"saving model to {pretrain_path}")
    model.save(pretrain_path)

    # finetuning

    print(f"loading model from {pretrain_path}")
    model = get_finetune_model(pretrain_path, output_shape)
    model.summary()

    config["dataset"] = config["finetune_set"]
    (X_train, X_val, X_test, y_train, y_val, y_test), label_encoder = create_dataset(config)

    print("Training...")
    model.compile(loss=config["loss_fn"], optimizer=config["optimizer"], metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config["epoch"], batch_size=config["batch_size"])

    finetune_path = config["output_path"] + uuid.uuid4().hex + ".h5"
    print(f"saving model to {finetune_path}")
    model.save(finetune_path)

    print("Test/evaluation...")
    model.evaluate(X_test, y_test, batch_size=config["test_batch"])
    draw_heatmap(X_test, y_test, model, label_encoder, config)

    return 

def main(args):
    config = dict()
    config.update(data_config.get_config())
    config.update(train_config.get_config())

    return run(config)

if __name__ == "__main__":
    main(sys.argv[1:])