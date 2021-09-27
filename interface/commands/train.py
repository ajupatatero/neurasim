import argparse
from neurasim import *

def main():
    parser = TrainParser()
    config = parser.parse()
    launch_train(config)


if __name__ == '__main__':
    main()
