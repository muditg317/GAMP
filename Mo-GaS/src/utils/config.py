import os
import sys
from yaml import safe_load

if __name__ == '__main__':
  print('This script is not meant to be run directly')
  sys.exit(1)

from src.utils.strict_run import *

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['RAW_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']
INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
VALID_ACTIONS = config['VALID_ACTIONS']
STACK_SIZE = config['STACK_SIZE']
CMP_FMT = config['CMP_FMT']
OVERWRITE_INTERIM_GAZE = config['OVERWRITE_INTERIM_GAZE']
GAMES_FOR_TRAINING = config['GAMES_FOR_TRAINING']

with open(os.path.join(RAW_DATA_DIR, 'action_enums.txt'), 'r') as f:
    ACTIONS_ENUM = f.read()
    ACTIONS_ENUM = ACTIONS_ENUM.split('\n')
    ACTIONS_ENUM = {key.strip(): value.strip() for line in ACTIONS_ENUM if line != '' and not line.startswith('#') for key,value in line.split('=')}
    print(ACTIONS_ENUM)