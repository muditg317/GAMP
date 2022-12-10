from __future__ import annotations
from src.utils.config import *
ASSERT_BEING_RUN(__name__, __file__, "This file should not be imported. It runs src/models/selective_motion_action_net.py")
from src.data.types import *
from src.data.loaders import load_hdf_data
from src.models.types import run_mode_t
from src.models.sma_with_cl import SelectiveMotion_CL_ActionNet
from src.models.utils import dataset_to_list_and_str

import torch
import argparse
# pylint: disable=all

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--game',required=True)
args = parser.parse_args()
game:game_t = args.game

MODE:run_mode_t = 'train'
completed_epochs = {
  'breakout': {
    '527_RZ_4153166_Jul-26-10-00-12': 0,
  },
  'centipede': {
    '150_KM_3357098_Dec-15-10-59-11': 0,
  },
}
completed_epochs = completed_epochs[game] if game in completed_epochs else {}

train_datasets = val_datasets = ['combined']

if game == 'phoenix':
  train_datasets = ['214_RZ_7226016_Jan-11-11-04-01']
  val_datasets = ['214_RZ_7226016_Jan-11-11-04-01',
                  '606_RZ_5215078_Aug-07-16-59-46', '600_RZ_5203429_Aug-07-13-44-39',
                  '598_RZ_5120717_Aug-06-14-53-30', '574_RZ_4682055_Aug-01-12-54-58',
                  '565_RZ_4604537_Jul-31-15-22-57', '550_RZ_4513408_Jul-30-14-04-09',
                  '540_RZ_4425986_Jul-29-13-47-10']

elif game == 'asterix':
  val_datasets = ['543_RZ_4430054_Jul-29-14-54-56',
                  '246_RZ_721092_Feb-20-21-52-26','260_RZ_1456515_Mar-01-10-10-36',
                  '534_RZ_4166872_Jul-26-13-49-43','553_RZ_4519853_Jul-30-15-51-37']

elif game == 'breakout':
  train_datasets = ['527_RZ_4153166_Jul-26-10-00-12']
  val_datasets = [ '564_RZ_4602455_Jul-31-14-48-16',
                   '527_RZ_4153166_Jul-26-10-00-12' ]

elif game == 'freeway':
  val_datasets = ['151_JAW_3358283_Dec-15-11-19-24','157_KM_6307437_Jan-18-14-31-43',
                  '149_JAW_3355334_Dec-15-10-31-51','79_RZ_3074177_Aug-18-11-46-29',
                  '156_KM_6306308_Jan-18-14-13-55']

elif game == 'name_this_game':
  val_datasets = ['267_RZ_2956617_Mar-18-19-52-47','576_RZ_4685615_Aug-01-13-54-21']

elif game == 'space_invaders':
  val_datasets = ['554_RZ_4520643_Jul-30-16-08-32','511_RZ_3988011_Jul-24-12-07-48',
                  '512_RZ_3991738_Jul-24-13-12-15','514_RZ_3993948_Jul-24-13-47-04',
                  '541_RZ_4427259_Jul-29-14-08-29','587_RZ_4775423_Aug-02-14-51-06',
                  '596_RZ_5117737_Aug-06-13-56-16'
                  ]
    
elif game == 'demon_attack':
  val_datasets = ['618_RZ_5375788_Aug-09-13-37-48']

# elif game == 'seaquest':

# elif game =='ms_pacman':

elif game == 'centipede':
  train_datasets = ['150_KM_3357098_Dec-15-10-59-11']
  val_datasets = ['150_KM_3357098_Dec-15-10-59-11']

# val_datasets = ['']
device = torch.device('cuda')

data_types = ['images', 'actions']

train_dataset_list, train_dataset_str = dataset_to_list_and_str(train_datasets)
val_dataset_list, val_dataset_str = dataset_to_list_and_str(val_datasets)
completed_epochs = completed_epochs[train_dataset_str] if train_dataset_str in completed_epochs else 0

action_net = SelectiveMotion_CL_ActionNet(game=game,
                                          data_types=data_types,
                                          dataset_train=train_datasets,
                                          dataset_train_load_type='chunked',
                                          dataset_val=val_datasets,
                                          dataset_val_load_type='chunked',
                                          device=device,
                                          mode=MODE,
                                          load_model=True,
                                          epoch=completed_epochs,
                                          ).to(device=device)
optimizer = torch.optim.Adadelta(action_net.parameters(), lr=5e-1, rho=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda e:0.8)

# lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss().to(device=device)


action_net.train_loop(opt=optimizer,
                      lr_scheduler=lr_scheduler,
                      loss_function=loss_)
