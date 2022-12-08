from src.utils.config import *
from src.data.loaders import load_data_iter

def test_loader():
  dl = load_data_iter(game=GAMES_FOR_TRAINING[0], batch_size=1, load_type='chunked')

  for x in dl:
      print(x.keys())
