from src.utils.config import *
ASSERT_BEING_RUN(__name__, __file__, "This file should not be imported. It runs src/models/cnn_gaze_net.py")
from src.data.types import *
from src.data.loaders import load_hdf_data
from src.models.types import run_mode_t
from src.models.cnn_gaze_net import CNN_GazeNet
from src.features.feat_utils import draw_figs_

import torch
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--game',required=True)

args = parser.parse_args()
game: game_t = args.game

MODE: run_mode_t = 'train'

train_datasets: list[game_run_t] = ['527_RZ_4153166_Jul-26-10-00-12']  #game_run
train_dataset = train_datasets[0]
val_datasets: list[game_run_t] = ['564_RZ_4602455_Jul-31-14-48-16']
val_dataset = val_datasets[0]

device: torch.device = torch.device('cuda')

data_types: list[datatype_t] = ['images', 'gazes']

gaze_net = CNN_GazeNet(game=game,
                       data_types=data_types,
                       dataset_train=train_dataset,
                       dataset_val=val_dataset,
                       dataset_train_load_type='chunked',
                       dataset_val_load_type='chunked',
                       device=device,
                       mode=MODE).to(device=device)

optimizer = torch.optim.Adadelta(gaze_net.parameters(), lr=5e-1, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#   optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda e:0.8)

# lr_scheduler = None
loss_ = torch.nn.KLDivLoss(reduction='batchmean')

if MODE=='eval':
  curr_group_data = load_hdf_data(game=game,
                                  datasets=val_datasets,
                                  data_types=['images','gazes'],
                                  )
  x, y = curr_group_data.values()
  x = x[0]
  y = y[0]
  print(x.shape)
  print(y.shape)
  imag = np.random.randint(0,5000,16)
  print(imag)
  for i in imag:
    image_ = x[i,:,:,:]
    print(image_.shape)
    gaze_ =  y[i,:,:]

    for cpt in tqdm(range(14, 15, 1)):
      print(cpt)
      gaze_net.epoch = cpt
      gaze_net.load_model_fn(cpt)
      smax = gaze_net.infer(
      torch.Tensor(image_).to(device=device).unsqueeze(0)).squeeze().cpu().data.numpy()

      # gaze_max = np.array(gaze_max)/84.0
      # smax = gaze_pdf([g_max])
      # gaze_ = gaze_pdf([gaze_max])
      pile = np.percentile(smax,90)
      smax = np.clip(smax,pile,1)
      smax = (smax-np.min(smax))/(np.max(smax)-np.min(smax))
      smax = smax/np.sum(smax)

      # draw_figs(x_var=smax, gazes=gaze_*255)
      # draw_figs(x_var=image_[-1], gazes=gaze_*255)
      draw_figs_(x_var=smax, x_var2 = image_[-1], gazes=gaze_*255)
      # draw_figs(x_var=image_[-1], gazes=gaze_*255)
else:
  gaze_net.train_loop(optimizer, lr_scheduler, loss_)
