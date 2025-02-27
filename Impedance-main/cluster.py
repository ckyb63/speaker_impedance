import sys
from config import data_config, train_config

def run(config):

    return 

def main(args):
    config = dict()
    config.update(data_config.get_config())
    config.update(train_config.get_config())

    return run(config)

if __name__ == "__main__":
    main(sys.argv[1:])