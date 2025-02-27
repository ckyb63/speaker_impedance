MODEL = "DNet" # later to "TNet"
EPOCH = 200
BATCH_SIZE = 32
LOSS_FN = "categorical_crossentropy"
OPTIMIZER = "sgd" # later to "adam"
TEST_BATCH = 128
DROP_RATE = 0.5 # later to 0.2
OUTPUT_PATH = "./outputs/"

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