from __future__ import annotations
from src.utils.config import *
ASSERT_BEING_RUN(__name__, __file__, "This file should not be imported. It runs src/models/selective_gaze_action_net.py and visualizes the results")
from src.data.types import *
from src.models.cnn_gaze_net import CNN_GazeNet
from src.models.selective_gaze_action_net import SelectiveGaze_ActionNet
from src.features.feat_utils import image_transforms

import torch

import gym
from gym.wrappers import FrameStack

import numpy as np

import argparse

import cv2

EVAL_MODE = 'eval'
transform_images = image_transforms()

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--game',required=True)
parser.add_argument('--action_cpt',required=True,type=int)
parser.add_argument('--gaze_net_cpt',required=True,type=int)
parser.add_argument('--episode',default=None,type=int)
args = parser.parse_args()
game:game_t = args.game
action_cpt = args.action_cpt
gaze_net_cpt = args.gaze_net_cpt
episode = args.episode


device = torch.device('cuda')

data_types = ['images', 'actions', 'gazes'] # unused
dataset_train:game_run_t = ['143_JAW_3272885_Dec-14-11-35-58',
 '144_JAW_3273946_Dec-14-11-54-16',
 '150_KM_3357098_Dec-15-10-59-11',
 '152_KM_3359364_Dec-15-11-36-50',
 '204_RZ_4136256_Dec-06-16-48-50',
 '206_RZ_4478127_Dec-10-15-44-57',
 '210_RZ_6966080_Jan-08-10-58-48',
 '244_RZ_594187_Feb-19-10-37-42',
 '286_RZ_5620664_Apr-18-15-56-48',
 '450_RZ_3221959_Jul-15-15-19-59']     # unused
dataset_val:game_run_t   = '527_RZ_4153166_Jul-26-10-00-12'     # unused

gaze_net = CNN_GazeNet(game=game,
                       data_types=data_types,
                       dataset_train=dataset_train,
                       dataset_val=dataset_val,
                       dataset_train_load_type=None,
                       dataset_val_load_type=None,
                       device=device,
                       mode=EVAL_MODE).to(device=device)
gaze_net.load_model_at_epoch(gaze_net_cpt)
gaze_net.eval()

action_net = SelectiveGaze_ActionNet(game=game,
                                     data_types=data_types,
                                     dataset_train=dataset_train,
                                     dataset_train_load_type=None,
                                     dataset_val=dataset_val,
                                     dataset_val_load_type=None,
                                     device=device,
                                     mode=EVAL_MODE,
                                    #  gaze_pred_model=gaze_net
                                     ).to(device=device)
action_net.load_model_at_epoch(action_cpt)
action_net.eval()
action_net.gaze_pred_model = gaze_net
action_net.gaze_pred_model.load_model_at_epoch(gaze_net_cpt)
action_net.gaze_pred_model.eval()

env = gym.make(GYM_ENV_MAP[game], full_action_space=True, frameskip=1)
env = FrameStack(env, 4)
# env = RecordVideo(env,env_name)
# print(env._env_info())

t_rew = 0
if episode is None:
  start_episode = 0
  end_episode = 30
else:
  start_episode = episode
  end_episode = start_episode+1
for i_episode in range(start_episode,end_episode,1):
  env.seed(i_episode)
  # env.render(mode = 'human')
  observation,info = env.reset()
  ep_rew = 0
  t = 0
  while True:
    # env.render()
    t += 1

    obs = observation
    observation = torch.stack(
      [transform_images(o.__array__()).squeeze() for o in observation]).unsqueeze(0).to(device=device)

    observation, gaze, acts, _ = action_net.run_inputs(observation)
    gaze = gaze[0]
    # print(gaze.shape)

    # gaze_ = gaze.squeeze().cpu().numpy()
    # gaze_top90 = np.percentile(gaze_,90)
    # gaze_ = np.clip(gaze_,gaze_top90,1)
    
    # gaze_ = np.array(cv2.resize(gaze_,(160,210))*255,dtype=np.uint8)
    # gaze_ = cv2.applyColorMap(gaze_,cv2.COLORMAP_INFERNO)
    # gaze_ = obs[-1]+gaze_


    # cv2.imshow("gaze_pred_normalized",gaze_)
    # cv2.waitKey(5)

    # gaze_ = gaze[0][0].cpu().numpy()
    
    # gaze_ = np.array(cv2.resize(gaze_,(160,210))*255,dtype=np.uint8)
    # gaze_ = cv2.applyColorMap(gaze_,cv2.COLORMAP_TURBO)

    obs = cv2.resize(obs[-1],(160,210))
    obs = cv2.cvtColor(obs,cv2.COLOR_RGB2BGR)
    # gaze_ = cv2.addWeighted(gaze_,0.25,obs,0.5,0)*2

    gaze_true = gaze.squeeze().cpu().numpy()
    # gaze_true = (gaze / observation).squeeze().cpu().numpy()
    gaze_true = (gaze_true - np.min(gaze_true)) / (np.max(gaze_true) - np.min(gaze_true))

    gaze_true = np.array(cv2.resize(gaze_true,(160,210))*255,dtype=np.uint8)
    gaze_true = cv2.applyColorMap(gaze_true,cv2.COLORMAP_TURBO)
    
    gaze_ = cv2.addWeighted(gaze_true,0.25,obs,0.5,0)*2

    gaze_ = cv2.resize(gaze_,(480,630))
    cv2.imshow("gaze_pred_normalized",gaze_)
    cv2.waitKey(1)
    
    action = action_net.process_activations_for_inference(acts)
    if game == 'breakout' and np.random.random() < 0.1:
      action = ACTIONS_ENUM['PLAYER_A_FIRE']
    observation, reward, done, trun, info = env.step(action)

    ep_rew += reward
    if done or trun:
      # print("Episode finished after {} timesteps".format(t + 1))
      break
  t_rew += ep_rew
  print(f"Episode {i_episode} finished...")
  print(f"\tLength: {t} timesteps")
  print(f"\tReward {ep_rew}")
  print(f"\tAvg reward {t_rew/(i_episode+1)}")


print("Mean all Episode {} reward {}".format(i_episode, t_rew / (i_episode+1)))
env.close()
