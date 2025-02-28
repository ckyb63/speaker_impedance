import os

DATA_PATH = r"Collected_Data_Sep16" # path to datafile (new: Collected_Data, old: Complete Data)
SPEAKERS = ["A", "B", "C", "D"] # options: "A", "B", "C", "D"
SPEAKER_DIF = False # whether to differentiate speakers
COLUMN = ["PH", "MAG", "RS", "XS", "REC"] # options: "FREQ", "PH", "MAG", "RS", "XS", "REC"
VAL = 0.2 # validation split ratio
TEST = 0.2 # test split ratio
START = 0 # starting index
END = 500 # ending index
LEN = ["5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"] # options "5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"

FINETUNE_SET = ["B"] # options: "A", "B", "C", "D"

def get_config():
    config = {
        "data_path": DATA_PATH,
        "speakers": SPEAKERS,
        "speaker_dif": SPEAKER_DIF,
        "column": COLUMN,
        "val": VAL,
        "test": TEST,
        "start": START,
        "end": END,
        "len": LEN,

        "finetune_set": FINETUNE_SET,
    }

    if not os.path.isdir(config["data_path"]):
        raise FileNotFoundError(f"Directory '{config['data_path']}' does not exist.")
    if any(speaker.upper() not in ["A", "B", "C", "D"] for speaker in config["speakers"]):
        raise ValueError(f"{config['speakers']} is not in the speaker option")
    if any(col.upper() not in ["FREQ", "PH", "MAG", "RS", "XS", "REC"] for col in config["column"]):
        raise ValueError(f"{config['column']} is not in the column option")
    if (config["val"]<0) or (config["test"]<0) or (config["test"]+config["val"]>=1):
        raise ValueError("invalid split ratio")
    if (config["start"]<0) or (config["start"]>501) or (config["end"]<0) or (config["end"]>501) or (config["start"]>config["end"]):
        raise ValueError("invalid data range")
    if any(length not in ["5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"] for length in config["len"]):
        raise ValueError(f"{config['dataset']} is not in the dataset option")

    if any(speaker.upper() not in ["A", "B", "C", "D"] for speaker in config["finetune_set"]):
        raise ValueError(f"{config['finetune_set']} is not in the option")

    return config