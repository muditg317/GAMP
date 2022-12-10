import os
import sys
sys.path.append(os.getcwd())
from src.data.data_loaders import load_hdf_data
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.selective_gaze_only import SGAZED_ACTION_SL
from src.data.data_loaders import load_action_data, load_gaze_data
from src.models.cnn_gaze import CNN_GAZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--game',required=True)
parser.add_argument('--gaze_net_cpt',required=True)
args = parser.parse_args()
game = args.game
gaze_net_cpt = args.gaze_net_cpt

MODE = 'train'
BATCH_SIZE = config_data['BATCH_SIZE']
# GAZE_TYPE = ["PRED","REAL"]
GAZE_TYPE = "PRED"

dataset_train = dataset_val = 'combined'

if game == 'phoenix':
    dataset_val = ['606_RZ_5215078_Aug-07-16-59-46', '600_RZ_5203429_Aug-07-13-44-39',
                   '598_RZ_5120717_Aug-06-14-53-30', '574_RZ_4682055_Aug-01-12-54-58',
                   '565_RZ_4604537_Jul-31-15-22-57', '550_RZ_4513408_Jul-30-14-04-09',
                   '540_RZ_4425986_Jul-29-13-47-10']
    env_name = 'Phoenix-v0'

elif game == 'asterix':
    dataset_val = ['543_RZ_4430054_Jul-29-14-54-56',
                   '246_RZ_721092_Feb-20-21-52-26','260_RZ_1456515_Mar-01-10-10-36',
                   '534_RZ_4166872_Jul-26-13-49-43','553_RZ_4519853_Jul-30-15-51-37']
    env_name = 'Asterix-v0'

elif game == 'breakout':
    dataset_val = [ '564_RZ_4602455_Jul-31-14-48-16',
                    '527_RZ_4153166_Jul-26-10-00-12']
    env_name = 'Breakout-v0'

elif game == 'freeway':
    dataset_val = ['151_JAW_3358283_Dec-15-11-19-24','157_KM_6307437_Jan-18-14-31-43',
                    '149_JAW_3355334_Dec-15-10-31-51','79_RZ_3074177_Aug-18-11-46-29',
                    '156_KM_6306308_Jan-18-14-13-55']
    env_name = 'Freeway-v0'

elif game == 'name_this_game':
    dataset_val = ['267_RZ_2956617_Mar-18-19-52-47','576_RZ_4685615_Aug-01-13-54-21']
    env_name = 'NameThisGame-v0'

elif game == 'space_invaders':
    dataset_val = ['554_RZ_4520643_Jul-30-16-08-32','511_RZ_3988011_Jul-24-12-07-48',
                    '512_RZ_3991738_Jul-24-13-12-15','514_RZ_3993948_Jul-24-13-47-04',
                    '541_RZ_4427259_Jul-29-14-08-29','587_RZ_4775423_Aug-02-14-51-06',
                    '596_RZ_5117737_Aug-06-13-56-16'
                    ]
    env_name = 'SpaceInvaders-v0'
    
elif game == 'demon_attack':
    env_name = 'DemonAttack-v0'
    dataset_val = ['618_RZ_5375788_Aug-09-13-37-48']

elif game == 'seaquest':
    env_name = 'Seaquest-v0'

elif game =='ms_pacman':
    env_name = 'MsPacman-v0'

elif game == 'centipede':
    env_name = 'Centipede-v0'

# dataset_val = ['']
dataset_train=['143_JAW_3272885_Dec-14-11-35-58',
 '144_JAW_3273946_Dec-14-11-54-16',
 '150_KM_3357098_Dec-15-10-59-11',
 '152_KM_3359364_Dec-15-11-36-50',
 '204_RZ_4136256_Dec-06-16-48-50',
 '206_RZ_4478127_Dec-10-15-44-57',
 '210_RZ_6966080_Jan-08-10-58-48',
 '244_RZ_594187_Feb-19-10-37-42',
#  '245_RZ_719798_Feb-20-21-31-06',
#  '254_RZ_1239817_Feb-26-21-57-37',
#  '259_RZ_1454534_Mar-01-09-36-16',
 '286_RZ_5620664_Apr-18-15-56-48',
 '450_RZ_3221959_Jul-15-15-19-59']#,
dataset_val = ['481_RZ_3471184_Jul-18-12-35-05',
 '501_RZ_3566169_Jul-19-14-57-05',
 '53_RZ_2398139_Aug-10-15-50-15',
 '60_RZ_2724114_Aug-14-10-26-09',
#  '610_RZ_5290268_Aug-08-13-51-47',
#  '612_RZ_5293382_Aug-08-14-44-08',
#  '615_RZ_5298584_Aug-08-16-10-26',
#  '620_RZ_5632787_Aug-12-13-03-16',
 '69_RZ_2831643_Aug-15-16-16-35',
 '78_RZ_3068875_Aug-18-10-10-05',
 '80_RZ_3084132_Aug-18-14-23-21',
 '88_RZ_3437559_Aug-22-16-33-34',
 '94_RZ_3508931_Aug-23-12-23-06',
 '97_RZ_3586578_Aug-24-09-59-20']
 #'245_RZ_719798_Feb-20-21-31-06',
 #'254_RZ_1239817_Feb-26-21-57-37',
 #'259_RZ_1454534_Mar-01-09-36-16',
#  '610_RZ_5290268_Aug-08-13-51-47',
#  '612_RZ_5293382_Aug-08-14-44-08',
#  '615_RZ_5298584_Aug-08-16-10-26',
#  '620_RZ_5632787_Aug-12-13-03-16']
device = torch.device('cuda')

data_types = ['images', 'actions', 'gazes']

action_net = SGAZED_ACTION_SL(game=game,
                              data_types=data_types,
                              dataset_train=dataset_train,
                              dataset_train_load_type='chunked',
                              dataset_val=dataset_val,
                              dataset_val_load_type='chunked',
                              device=device,env_name = env_name,
                              mode=MODE,load_model=False,
                              epoch=33).to(device=device)
optimizer = torch.optim.Adadelta(action_net.parameters(), lr=5e-1, rho=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda e:0.8)

# lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss().to(device=device)

if MODE == 'eval':
    if GAZE_TYPE == "PRED":
        gaze_net = CNN_GAZE(game=game,
                            data_types=data_types,
                            dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            dataset_train_load_type='chunked',
                            dataset_val_load_type='chunked',
                            device=device,
                            mode='eval').to(device=device)

        gaze_net.epoch = gaze_net_cpt
        gaze_net.load_model_fn(gaze_net_cpt)
        curr_group_data = load_hdf_data(game=game,
                                        dataset=dataset_val,
                                        data_types=['images', 'gazes'],
                                        )

        x, y = curr_group_data.values()
        x = x[0]
        y = y[0]

        image_ = x[204]
        image_ = torch.Tensor(image_).to(device=device).unsqueeze(0)
        gaze_ = y[205]

        xg = gaze_net.infer(image_).repeat(
            1, image_.shape[1], 1, 1).to(device=device)
        acts = action_net.infer(image_, xg)
        acts = acts  # .data.item()


else:
    if GAZE_TYPE == "PRED":
        gaze_net = CNN_GAZE(game=game,
                            data_types=data_types,
                            dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            dataset_train_load_type='chunked',
                            dataset_val_load_type='chunked',
                            device=device,
                            mode='eval').to(device=device)
        gaze_net.epoch = gaze_net_cpt
        gaze_net.load_model_fn(gaze_net_cpt)

        action_net.train_loop(optimizer,
                              lr_scheduler,
                              loss_,
                              batch_size=BATCH_SIZE,
                              gaze_pred=gaze_net)

    else:
        action_net.train_loop(optimizer,
                              lr_scheduler,
                              loss_,
                              batch_size=BATCH_SIZE)
