from __future__ import annotations
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
parser.add_argument('--mode',required=True)

args = parser.parse_args()
game: game_t = args.game

MODE: run_mode_t = args.mode
completed_epochs = 65

train_datasets: list[game_run_t] = ['143_JAW_3272885_Dec-14-11-35-58',
 '144_JAW_3273946_Dec-14-11-54-16',
 '150_KM_3357098_Dec-15-10-59-11',
 '152_KM_3359364_Dec-15-11-36-50',
 '204_RZ_4136256_Dec-06-16-48-50',
 '206_RZ_4478127_Dec-10-15-44-57',
 '210_RZ_6966080_Jan-08-10-58-48',
 '244_RZ_594187_Feb-19-10-37-42',
 '286_RZ_5620664_Apr-18-15-56-48',
 '450_RZ_3221959_Jul-15-15-19-59']
# train_dataset = train_datasets[0]
val_datasets: list[game_run_t] = [
 '481_RZ_3471184_Jul-18-12-35-05',
 '501_RZ_3566169_Jul-19-14-57-05',
 '53_RZ_2398139_Aug-10-15-50-15',
 '60_RZ_2724114_Aug-14-10-26-09',
 '69_RZ_2831643_Aug-15-16-16-35',
 '78_RZ_3068875_Aug-18-10-10-05',
 '80_RZ_3084132_Aug-18-14-23-21',
 '88_RZ_3437559_Aug-22-16-33-34',
 '94_RZ_3508931_Aug-23-12-23-06',
 '97_RZ_3586578_Aug-24-09-59-20']
# val_dataset = val_datasets[0]

device: torch.device = torch.device('cuda')
# print('Using device:', device, torch.cuda.get_device_name(device))

data_types: list[datatype_t] = ['images', 'gazes']

gaze_net = CNN_GazeNet(game=game,
                       data_types=data_types,
                       dataset_train=train_datasets,
                       dataset_val=val_datasets,
                       dataset_train_load_type='chunked' if MODE != 'eval' else None,
                       dataset_val_load_type=None,
                       device=device,
                       mode=MODE,
                       load_model=completed_epochs > 0,
                       epoch=completed_epochs,
                       ).to(device=device)

optimizer = torch.optim.Adadelta(gaze_net.parameters(), lr=5e-1, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#   optimizer, lr_lambda=lambda x: x*0.95)
# lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda e:0.8)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose = True, patience = 5,factor = 0.1,threshold=0.01)

# lr_scheduler = None
loss_ = torch.nn.KLDivLoss(reduction='batchmean')

if MODE=='eval':
  gaze_net.load_model_at_epoch(completed_epochs)
  curr_group_data = load_hdf_data(game=game,
                                  datasets=val_datasets[:1],
                                  data_types=['images','gazes'],
                                  device=device,
                                  )
  x, y = curr_group_data.values()
  x = x[0]
  y = y[0]
  # print(x.shape)
  # print(y.shape)
  imag = np.random.randint(0,5000,16)
  print(imag)

  for i in imag:
    image_ = x[i,:,:,:]
    # print(image_.shape)
    gaze_ =  y[i,:,:]

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
    draw_figs_(x_var=smax, x_var2 = image_[-1].cpu(), gazes=(gaze_*255).cpu())
    # draw_figs(x_var=image_[-1], gazes=gaze_*255)

else:
  gaze_net.train_loop(optimizer, lr_scheduler, loss_, epochs_to_train_for=100)
