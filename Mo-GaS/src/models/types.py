from typing import Literal
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__)

# The game to train/test on
run_mode_t = Literal['train'] | Literal['eval']