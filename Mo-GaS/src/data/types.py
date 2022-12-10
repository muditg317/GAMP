from typing import Literal
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__)

# The game to train/test on (see src/config.yaml)
# game_t = Literal['phoenix'] | Literal['asterix'] | Literal['breakout'] | Literal['freeway'] | Literal['name_this_game'] | Literal['space_invaders'] | Literal['demon_attack'] | Literal['seaquest'] | Literal['ms_pacman'] | Literal['centipede']
game_t = str

# A specific game run within a game (usually of the form ###_XX_#######_Mmm-DD-HH-MM-SS)
# game_run_t = Literal['combined'] | str
game_run_t = str
# The types of data to generate/load from a game run (see src/data/preprocess.py)
# datatype_t = Literal['images'] | Literal['actions'] | Literal['gazes'] | Literal['fused_gazes'] | Literal['gazes_fused_noop']
datatype_t = str

# The memort format for data loading (see src/data/loaders.py)
# dataset_load_type_t = Literal['memory'] | Literal['disk'] | Literal['live'] | Literal['chunked']
dataset_load_type_t = str

