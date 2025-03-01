MODEL = "DNet" # later to "CNet"
EPOCH = 200
BATCH_SIZE = 32
LOSS_FN = "categorical_crossentropy"
OPTIMIZER = "adam" # "sgd" to "adam"
TEST_BATCH = 128
DROP_RATE = 0.2
OUTPUT_PATH = "Impedance-main/outputs/"

def get_config():
    config = {
        "model": MODEL,
        "epoch": EPOCH,
        "batch_size": BATCH_SIZE,
        "loss_fn": LOSS_FN,
        "optimizer": OPTIMIZER,
        "test_batch": TEST_BATCH,
        "drop_rate": DROP_RATE,
        "output_path": OUTPUT_PATH,
    }

    if (config["model"] not in ["DNet", "CNet"]):
        raise ValueError(f"{config['model']} is not in the option")
    
    return config