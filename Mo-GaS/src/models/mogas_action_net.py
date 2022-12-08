from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file is just a base class for other action selection models.")
from src.data.types import *
from src.models.types import *
from src.models.mogas_gaze_net import MoGas_GazeNet

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from src.data.utils import ImbalancedDatasetSampler
from src.data.loaders import load_data_iter
from src.features.feat_utils import image_transforms
# np.random.seed(42)
# import gym
# from gym.wrappers import FrameStack, RecordVideo

class MoGaS_ActionNet(nn.Module, ABC):
  def __init__(self,
               input_shape:tuple[int,int]                   = (84, 84),
               load_model                                   = False,                   # load model from disk
               epoch                                        = 0,           # epoch to load model from
               num_actions                                  = len(ACTIONS_ENUM), # 18
               game:game_t                                  = GAMES_FOR_TRAINING[0],
               data_types:list[datatype_t]                  = list(set(DATA_TYPES) - set(['gazes'])),
               dataset_train:game_run_t                     = 'combined', # game_run | 'combined'
               dataset_train_load_type:dataset_load_type_t  = 'chunked', # 'chunked' | 'disk' | 'memory
               dataset_val:game_run_t                       = 'combined',
               dataset_val_load_type:dataset_load_type_t    = 'chunked',
               device                                       = torch.device('cuda'),
               gaze_pred_model:MoGas_GazeNet|None           = None,
               mode:run_mode_t                              = 'train',
               opt:torch.optim.Optimizer|None               = None,
              ):
    super(MoGaS_ActionNet, self).__init__()
    self.game = game
    self.data_types = data_types
    self.input_shape = input_shape
    self.num_actions = num_actions

    self.device = device

    self.load_model = load_model
    self.epoch = epoch
    self.gaze_pred_model = gaze_pred_model
    self.mode = mode
    self.opt = opt

    model_save_dir = os.path.join(MODEL_SAVE_DIR, game, dataset_train)
    if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)

    self.model_name = self.__class__.__name__
    self.model_save_string = os.path.join(
      model_save_dir, self.model_name + '_Epoch_{}.pt')

    log_dir = os.path.join(
      RUNS_DIR, game,
      '{}_{}'.format(dataset_train, self.model_name))
    self.writer = SummaryWriter(log_dir=os.path.join(
      log_dir, "run_{}".format(
        len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0)))

    self.batch_size = BATCH_SIZE
    
    if mode != 'eval':
      self.train_data_iter = load_data_iter(
        game=self.game,
        data_types=self.data_types,
        dataset=[dataset_train],
        dataset_exclude=[dataset_val],
        device=self.device,
        batch_size=self.batch_size,
        sampler=ImbalancedDatasetSampler,
        load_type=dataset_train_load_type,
      )

      self.val_data_iter = load_data_iter(
        game=self.game,
        data_types=self.data_types,
        dataset=[dataset_val],
        dataset_exclude=[dataset_train],
        device=self.device,
        batch_size=self.batch_size,
        load_type=dataset_val_load_type,
      )

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

  def train_loop(self,
                 opt: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler|None,
                 loss_function: torch.nn.modules.loss._Loss,
                 batch_size=BATCH_SIZE,
                 gaze_pred:MoGas_GazeNet=None,
                 GAME_PLAY_FREQ=1):
    self.loss_ = loss_function
    self.opt = opt
    self.gaze_pred_model = gaze_pred
    if self.gaze_pred_model is not None:
      self.gaze_pred_model.eval()

    if self.load_model:
      self.load_model_at_epoch(self.epoch, load_optimizer=True)
    else:
      self.epoch = -1
    eix = 0
    start_epoch = self.epoch+1
    end_epoch = self.epoch+300
    for epoch in range(start_epoch,end_epoch):
      print(f"Training epoch {epoch}/{end_epoch}...")
      for i, data in enumerate(self.train_data_iter):
        self.opt.zero_grad()

        x, y, *other_data = self.get_data(data)
        extra_inputs, acts, extra_outputs = self.run_inputs(x, *other_data)

        loss = self.loss_fn(self.loss_, acts, y, *extra_inputs, *extra_outputs)
        loss.backward()
        self.opt.step()

        self.writer.add_scalar('Loss', loss.data.item(), eix)
        self.writer.add_scalar('Acc', self.batch_acc(acts, y), eix)

        eix += 1
      if epoch % 1 == 0:
        # self.writer.add_histogram("acts", y)
        # self.writer.add_histogram("preds", acts)
        print(f"Epoch {epoch} complete:")
        print(f"\tLoss: {loss.data.item()}")
        print(f"\tComputing accuracy...", end='')
        accuracy = self.accuracy()
        print(f"\tAccuracy: {accuracy}")
        self.writer.add_scalar('Train Acc', accuracy, epoch)
        if lr_scheduler is not None:
          self.writer.add_scalar('Learning Rate', lr_scheduler.get_lr()[0], epoch)

        self.calc_val_metrics(epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
          }, self.model_save_string.format(epoch))
      if epoch % GAME_PLAY_FREQ == 0:
        self.game_play(epoch)
      if lr_scheduler is not None and epoch % 6 == 0:
          lr_scheduler.step()

  def get_data(self, data: dict[str, torch.Tensor] | list[torch.Tensor]):
    """
    transforms data from the data iterator into the correct format for the model
    """
    has_gaze = 'gazes' in self.data_types
    if isinstance(data, dict):
      x = data['images'].to(device=self.device)
      y = data['actions'].to(device=self.device)
      if has_gaze:
        x_g = data['gazes'].to(device=self.device)
    elif isinstance(data, list):
      x, y, *other = data
      x = x.to(device=self.device)
      y = y.to(device=self.device)
      if has_gaze:
        x_g = other[0].to(device=self.device)
    else:
      raise ValueError("data must be a dict or a list")

    if len(x.shape) < 4:
      x = x.unsqueeze(1)
    if has_gaze and len(x_g.shape) < 4:
      x_g = x_g.unsqueeze(1)
    
    if has_gaze:
      return x, y, x_g
    else:
      return x, y


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
    
    return extra_inputs, acts, extra_outputs

  def infer(self, x_var: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      self.eval()

      _, acts, _ = self.run_inputs(x_var)
      result = self.process_activations_for_inference(acts)

      self.train()

    return result

  @abstractmethod
  def process_activations_for_inference(self, acts):
    return acts.argmax(dim=1)

  def load_model_at_epoch(self, epoch: int, load_optimizer=False):
    self.epoch = epoch
    model_pickle = torch.load(self.model_save_string.format(self.epoch))
    self.load_state_dict(model_pickle['model_state_dict'])
    if load_optimizer:
      self.opt.load_state_dict(model_pickle['optimizer_state_dict'])

    self.epoch = model_pickle['epoch']
    loss_val = model_pickle['loss']
    print("Loaded {} model from saved checkpoint {} with loss {}".format(
        self.model_name, self.epoch, loss_val))

  def batch_acc(self, acts: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
      acc = (acts.argmax(dim=1) == y).sum().data.item() / y.shape[0]
    return acc

  def calc_val_metrics(self, epoch: int):
    prev_train_mode = self.training
    self.eval()

    acc = 0
    ix = 0
    loss = 0
    with torch.no_grad():
      for i, data in enumerate(self.val_data_iter):
        x, y, *other_data = self.get_data(data)
        other_data = [] # don't allow other data for validation
        extra_inputs, acts, extra_outputs = self.run_inputs(x, *other_data)

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
        _, acts, _ = self.run_inputs(x, *other_data)
        acts = acts.argmax(dim=1)
        acc += (acts == y).sum().data.item()
        ix += y.shape[0]

    self.train()
    return (acc / ix)
  
  def game_play(self,epoch,episodes=2):
    print(f"Playing {episodes} episodes of {self.game} at epoch {epoch}...")

    transform_images = image_transforms()

    env = gym.make(GYM_ENV_MAP[self.game], full_action_space=True,frameskip=1)
    env = FrameStack(env, STACK_SIZE)
    # env = Monitor(env,self.game,force=True)

    t_rew = 0

    for i_episode in range(episodes):
      observation = env.reset()
      ep_rew = 0
      t = 0
      while True:
        t += 1
        observation = torch.stack(
          [transform_images(o).squeeze() for o in observation]).unsqueeze(0).to(device=self.device)
        action = self.infer(observation).data.item()
        observation, reward, done, info = env.step(action)

        ep_rew += reward
        if done:
          # print("Episode finished after {} timesteps".format(t + 1))
          break
      t_rew += ep_rew
      print(f"Episode {i_episode} finished...")
      print(f"\tLength: {t} timesteps")
      print(f"\tReward {ep_rew}")
      print(f"\tAvg reward {t_rew/(i_episode+1)}")
    self.writer.add_scalar("Game Reward", t_rew/(i_episode+1),epoch)
