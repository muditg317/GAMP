from __future__ import annotations
from src.utils.config import *
ASSERT_BEING_RUN(__name__, __file__, "This file should not be imported. It runs src/models/selective_motion_action_net.py and visualizes the results")
from src.data.types import *
from src.models.sma_with_cl import SelectiveMotion_CL_ActionNet
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
parser.add_argument('--episode',default=None,type=int)
args = parser.parse_args()
game:game_t = args.game
action_cpt = args.action_cpt
episode = args.episode


device = torch.device('cuda')

data_types = ['images', 'actions'] # unused
if game == 'breakout':
  dataset_train:game_run_t = '527_RZ_4153166_Jul-26-10-00-12'     # unused
  dataset_val:game_run_t   = '527_RZ_4153166_Jul-26-10-00-12'     # unused
elif game == 'centipede':
  dataset_train = '150_KM_3357098_Dec-15-10-59-11'
  dataset_val = '69_RZ_2831643_Aug-15-16-16-35'

action_net = SelectiveMotion_CL_ActionNet(game=game,
                                          data_types=data_types,
                                          dataset_train=dataset_train,
                                          dataset_train_load_type=None,
                                          dataset_val=dataset_val,
                                          dataset_val_load_type=None,
                                          device=device,
                                          mode=EVAL_MODE).to(device=device)
action_net.load_model_at_epoch(action_cpt)
action_net.eval()


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

    observation, extra_in, acts, extra_out = action_net.run_inputs(observation)
    motion = extra_in[0]
    agent_attention = extra_out[0].detach()
    flat_len = agent_attention.shape[-1]
    attention_size = (int(np.sqrt(flat_len)),int(np.sqrt(flat_len)))
    agent_attention = agent_attention.view(-1,*attention_size)

    obs = cv2.resize(obs[-1],(160,210))
    obs = cv2.cvtColor(obs,cv2.COLOR_RGB2BGR)

    attn_true = agent_attention.squeeze().cpu().numpy()
    if np.max(attn_true) - np.min(attn_true) == 0:
      attn_true = np.zeros_like(attn_true)
    else:
      pass
      # attn_true = (attn_true - np.min(attn_true)) / (np.max(attn_true) - np.min(attn_true))

    attn_true = np.array(cv2.resize(attn_true,(160,210),interpolation=cv2.INTER_NEAREST)*255,dtype=np.uint8)
    attn_color = cv2.applyColorMap(attn_true,cv2.COLORMAP_TURBO)


    motion_true = motion.squeeze().cpu().numpy()
    if np.max(motion_true) - np.min(motion_true) == 0:
      motion_true = np.zeros_like(motion_true)
    else:
      pass
      # motion_true = (motion_true - np.min(motion_true)) / (np.max(motion_true) - np.min(motion_true))

    motion_true = np.array(cv2.resize(motion_true,attention_size))
    motion_true = np.array(cv2.resize(motion_true,(160,210),interpolation=cv2.INTER_NEAREST)*255,dtype=np.uint8)
    motion_color = cv2.applyColorMap(motion_true,cv2.COLORMAP_HOT)


    attn_ = cv2.addWeighted(attn_color,0.25,obs,0.5,0)*2
    motion_ = cv2.addWeighted(motion_color,0.25,obs,0.5,0)*2

    # result = cv2.addWeighted(attn_,0.5,motion_,0.5,0)
    blue_image = np.zeros_like(attn_)
    blue_image[:,:,0] = 255

    red_image = np.zeros_like(attn_)
    red_image[:,:,2] = 255
    masked_motion = np.where(motion_true[:,:,None] > 0.8, motion_true[:,:,None]*blue_image, 255)
    masked_attn = np.where(attn_true[:,:,None] > 0.8, attn_true[:,:,None]*red_image, 255)
    result = cv2.addWeighted(masked_attn,0.5,masked_motion,0.5,0)
    cv2.imshow("attn_and_motion_normalized", result)
    # cv2.imshow("motion_pred_normalized",motion_)
    # cv2.imshow("attn_pred_normalized",attn_)
    # cv2.imshow("obs",obs)
    cv2.waitKey(1)
    
    action = action_net.process_activations_for_inference(acts)
    
    if game == 'breakout':
      if action == ACTIONS_ENUM['PLAYER_A_FIRE']:
        print("Agent tried to fire")
        if np.sum(motion_true) == 0:
          print("\t While there was no motion!!")
          # action = ACTIONS_ENUM['NOOP']
      if np.random.random() < 0.1:
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
