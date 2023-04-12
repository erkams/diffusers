import yaml
from addict import Dict
import torch
import shutil
import os
import random
import sys

def main():
    # argument: path to config file
    try:
        filename = sys.argv[1]
    except IndexError as error:
        print("provide yaml file as command-line argument!")
        exit()
    
    # load config file
    with open(filename, 'r') as f:
        config = Dict(yaml.safe_load(f))

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.copyfile(filename, os.path.join(config.log_dir, 'args.yaml'))
    
    pass


if __name__ == '__main__':
    main()
