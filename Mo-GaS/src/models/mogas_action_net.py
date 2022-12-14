from __future__ import annotations
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file is just a base class for other action selection models.")
from src.data.types import *
from src.models.mogas_net import MoGaS_Net
from src.features.feat_utils import image_transforms

from abc import ABC, abstractmethod
import random
import torch

import gym
from gym.wrappers import FrameStack

class MoGaS_ActionNet(MoGaS_Net, ABC):
  def __init__(self, *,
               data_types:list[datatype_t]                  = list(set(DATA_TYPES) - set(['gazes'])),
               num_actions                                  = len(ACTIONS_ENUM), # 18
               **kwargs):
    super(MoGaS_ActionNet, self).__init__(data_types=data_types, **kwargs)
    self.num_actions = num_actions

  @abstractmethod
  def add_extra_inputs(self, x: torch.Tensor, *other_data) -> list[torch.Tensor]:
    return x,

  @abstractmethod
  def forward(self, x: torch.Tensor, *extra_inputs) -> torch.Tensor | tuple[torch.Tensor,...]:
    pass

  @abstractmethod
  def loss_fn(self,
              loss_: torch.nn.modules.loss._Loss,
              acts: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    return loss_(acts, targets).to(device=self.device)

  def train_loop(self, *,
                 opt: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler|None,
                 loss_function: torch.nn.modules.loss._Loss,
                 GAME_PLAY_FREQ=1,
                 LR_SCHEDULER_FREQ=6,
                 epochs_to_train=300,
                 gym_episodes_per_epoch=2,
                 ):
    self.loss_ = loss_function
    self.opt = opt

    if self.load_model:
      self.load_model_at_epoch(self.epoch, load_optimizer=True)
    else:
      self.epoch = -1

    eix = 0
    start_epoch = self.epoch+1
    end_epoch = start_epoch+epochs_to_train
    for epoch in range(start_epoch,end_epoch+1):
      print(f"Training epoch {epoch}/{end_epoch}...")
      epoch_loss = {
        'total': 0.0
      }
      for i, data in enumerate(self.train_data_iter):
        self.opt.zero_grad()

        x, y, *other_data = self.get_data(data)
        x, extra_inputs, acts, extra_outputs = self.run_inputs(x, *other_data)

        batch_loss = self.loss_fn(self.loss_, acts, y, *extra_inputs, *extra_outputs)
        total_batch_loss = batch_loss
        if isinstance(batch_loss, dict):
          total_batch_loss = 0
          loss_name: str
          loss: torch.Tensor
          weight: float
          for loss_name, (loss, weight) in batch_loss.items():
            epoch_loss[loss_name] = epoch_loss.get(loss_name, 0.0) + loss.data.item()
            total_batch_loss += loss * weight
        epoch_loss['total'] += total_batch_loss.data.item()
        total_batch_loss.backward()
        self.opt.step()

        self.writer.add_scalar('Loss', total_batch_loss.data.item(), eix)
        self.writer.add_scalar('Acc', self.batch_acc(acts, y), eix)

        eix += 1

      if epoch % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': total_batch_loss,
          }, self.model_save_string.format(epoch))

        # self.writer.add_histogram("acts", y)
        # self.writer.add_histogram("preds", acts)
        print(f"Epoch {epoch} complete:")
        print(f"\tLoss: {epoch_loss}")
        print(f"\tLR: {lr_scheduler.get_last_lr()[0]}")
        print(f"\tComputing accuracy...", end='')
        accuracy = self.accuracy()
        print(f"\tAccuracy: {accuracy}")
        self.writer.add_scalar('Train Acc', accuracy, epoch)
        if lr_scheduler is not None:
          self.writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)

        # self.calc_val_metrics(epoch)

      if epoch % GAME_PLAY_FREQ == 0:
        self.game_play(epoch, episodes=gym_episodes_per_epoch)
      if lr_scheduler is not None and epoch % LR_SCHEDULER_FREQ == 0:
          lr_scheduler.step()

  def get_data(self, data: dict[str, torch.Tensor] | list[torch.Tensor]):
    """
    transforms data from the data iterator into the correct format for the model
    """
    # has_gaze = 'gazes' in self.data_types
    # has_motion = 'motion' in self.data_types
    if isinstance(data, dict):
      data = tuple(data[type_].to(device=self.device) for type_ in self.data_types if type_ in data)
    elif isinstance(data, list):
      data = tuple(data_.to(device=self.device) for data_ in data)
      # x, y, *other = data
      # x = x.to(device=self.device)
      # y = y.to(device=self.device)
      # if has_gaze:
      #   x_g = other.pop(0).to(device=self.device)
      # if has_motion:
      #   x_m = other.pop(0).to(device=self.device)
    else:
      raise ValueError("data must be a dict or a list")

    # if len(x.shape) < 4:
    #   x = x.unsqueeze(1)
    # if has_gaze and len(x_g.shape) < 4:
    #   x_g = x_g.unsqueeze(1)
    # if has_motion and len(x_m.shape) < 4:
    #   x_m = x_m.unsqueeze(1)
    
    # don't want to unsqueeze the y
    data = tuple(datum.unsqueeze(1) if len(datum.shape) < 4 and i != 1 else datum for i,datum in enumerate(data))
    
    # outputs = [x, y]
    # if has_gaze:
    #   outputs.append(x_g)
    # if has_motion:
    #   outputs.append(x_m)
    # return tuple(outputs)

    return data


  def run_inputs(self, x: torch.Tensor, *other_data):
    """
    Takes in the inputs and runs them through the model
    Returns the extra inputs, the model outputs, and the extra outputs

    :param x: the input to the model
    :param other_data: any other data that is needed to run the model
      (e.g. gaze, only supplied during training)
    
    :return: extra_inputs, acts, extra_outputs
      extra_inputs: any extra inputs that are computed at runtime based on the inputs
      acts: the model outputs
      extra_outputs: any extra outputs from the model
    """
    x = self.add_extra_inputs(x, *other_data)
    extra_inputs = []
    if isinstance(x, tuple):
      x, *extra_inputs = x

    acts = self.forward(x, *extra_inputs)
    extra_outputs = []
    if isinstance(acts, tuple):
      acts, *extra_outputs = acts
    
    return x, extra_inputs, acts, extra_outputs

  def infer(self, x_var: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      self.eval()

      _, _, acts, _ = self.run_inputs(x_var)
      result = self.process_activations_for_inference(acts)

      self.train()

    return result

  @abstractmethod
  def process_activations_for_inference(self, acts):
    return acts.argmax(dim=1)

  def batch_acc(self, acts: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
      acc = (acts.argmax(dim=1) == y).sum().data.item() / y.shape[0]
    return acc

  def calc_val_metrics(self, epoch: int):
    if self.val_data_iter is None:
      print(f"Skipping validation metrics")
      return
    prev_train_mode = self.training
    self.eval()

    acc = 0
    ix = 0
    loss = 0
    with torch.no_grad():
      for i, data in enumerate(self.val_data_iter):
        x, y, *other_data = self.get_data(data)
        other_data = [] # don't allow other data for validation
        _, extra_inputs, acts, extra_outputs = self.run_inputs(x, *other_data)

        loss_val = self.loss_fn(self.loss_, acts, y, *extra_inputs, *extra_outputs)
        loss += loss_val.data.item()

        acc += (acts.argmax(dim=1) == y).sum().data.item()
        ix += y.shape[0]

    self.train(prev_train_mode)
    self.writer.add_scalar('Val Loss', loss / ix, epoch)
    self.writer.add_scalar('Val Acc', acc / ix, epoch)

  def accuracy(self):
    acc = 0
    ix = 0
    self.eval()

    with torch.no_grad():
      for i, data in enumerate(self.train_data_iter):
        x, y, *other_data = self.get_data(data)
        _, _, acts, _ = self.run_inputs(x, *other_data)
        acts = acts.argmax(dim=1)
        acc += (acts == y).sum().data.item()
        ix += y.shape[0]

    self.train()
    return (acc / ix)
  
  def game_play(self,epoch,episodes=2):
    if episodes == 0:
      return
    print(f"Playing {episodes} episodes of {self.game} at epoch {epoch}...")

    transform_images = image_transforms()

    env = gym.make(GYM_ENV_MAP[self.game], full_action_space=True,frameskip=1, max_episode_steps=18000)
    env = FrameStack(env, STACK_SIZE)
    # env = Monitor(env,self.game,force=True)

    t_rew = 0

    for i_episode in range(episodes):
      observation,info = env.reset()
      ep_rew = 0
      t = 0
      while True:
        t += 1
        observation = torch.stack(
          [transform_images(o.__array__()).squeeze() for o in observation]).unsqueeze(0).to(device=self.device)
        action = self.infer(observation).data.item()
        if self.game == 'breakout' and random.random() < 0.1:
          action = ACTIONS_ENUM['PLAYER_A_FIRE']
        observation, reward, done, tr, info = env.step(action)

        ep_rew += reward
        if done or tr:
          # print("Episode finished after {} timesteps".format(t + 1))
          break
      t_rew += ep_rew
      print(f"\tEpisode {i_episode} finished...")
      print(f"\t\tLength: {t} timesteps")
      print(f"\t\tReward {ep_rew}")
      print(f"\t\tAvg reward {t_rew/(i_episode+1)}")
    self.writer.add_scalar("Game Reward", t_rew/(i_episode+1),epoch)
