
from src.utils.strict_run import *
import os
import sys
from yaml import safe_load

ASSERT_NOT_RUN(__name__, __file__)

with open('src/config.yaml', 'r') as f:
  config_data = safe_load(f.read())

RAW_DATA_DIR = config_data['RAW_DATA_DIR']
PROC_DATA_DIR = config_data['PROC_DATA_DIR']
INTERIM_DATA_DIR = config_data['INTERIM_DATA_DIR']
MODEL_SAVE_DIR = config_data['MODEL_SAVE_DIR']
RUNS_DIR = config_data['RUNS_DIR']
VALID_ACTIONS = config_data['VALID_ACTIONS']
STACK_SIZE = config_data['STACK_SIZE']
CMP_FMT = config_data['CMP_FMT']
OVERWRITE_INTERIM_GAZE = config_data['OVERWRITE_INTERIM_GAZE']
BATCH_SIZE = config_data['BATCH_SIZE']
DATA_TYPES = config_data['DATA_TYPES']
GAMES_FOR_TRAINING = config_data['GAMES_FOR_TRAINING']

ACTIONS_ENUM = {}
with open(os.path.join(RAW_DATA_DIR, 'action_enums.txt'), 'r') as f:
  ACTIONS_ENUM = f.read()
  ACTIONS_ENUM = ACTIONS_ENUM.split('\n')
  ACTIONS_ENUM = [line.split('=') for line in ACTIONS_ENUM if line != '' and not line.startswith('#')]
  ACTIONS_ENUM = {key.strip(): int(value.strip()) for key,value in ACTIONS_ENUM}


# Create required directories
DIRS_REQD = [RAW_DATA_DIR, PROC_DATA_DIR, INTERIM_DATA_DIR, MODEL_SAVE_DIR]
for entry in DIRS_REQD:
  if not os.path.exists(entry):
    os.makedirs(entry)


GYM_ENV_MAP = {
  'phoenix': 'Phoenix-v0',
  'asterix': 'Asterix-v0',
  'breakout': 'Breakout-v0',
  'freeway': 'Freeway-v0',
  'name_this_game': 'NameThisGame-v0',
  'space_invaders': 'SpaceInvaders-v0',
  'demon_attack': 'DemonAttack-v0',
  'seaquest': 'Seaquest-v0',
  'ms_pacman': 'MsPacman-v0',
  'centipede': 'Centipede-v0',
}