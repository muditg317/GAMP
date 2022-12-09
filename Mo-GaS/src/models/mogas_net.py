from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file is just a base class for all types of MoGaS networks.")
from src.data.utils import ImbalancedDatasetSampler
from src.data.loaders import load_data_iter
from src.data.types import *
from src.models.types import *

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

class MoGaS_Net(nn.Module, ABC):
  def __init__(self, *,
               data_types:list[datatype_t],
               input_shape:tuple[int,int]                         = (84, 84),
               load_model                                         = False,       # load model from disk
               epoch                                              = 0,           # epoch to load model from
               batch_size                                         = BATCH_SIZE,
               game:game_t                                        = GAMES_FOR_TRAINING[0],
               dataset_train:game_run_t                           = 'combined',  # game_run | 'combined'
               dataset_train_load_type:dataset_load_type_t|None   = 'chunked',   # 'chunked' | 'disk' | 'memory
               dataset_val:game_run_t                             = 'combined',
               dataset_val_load_type:dataset_load_type_t|None     = 'chunked',
               device                                             = torch.device('cuda'),
               mode:run_mode_t                                    = 'train',
               opt:torch.optim.Optimizer|None                     = None,
              ):
    super(MoGaS_Net, self).__init__()
    self.game = game
    self.data_types = data_types
    self.input_shape = input_shape
    self.batch_size = batch_size

    self.device = device

    self.load_model = load_model
    self.epoch = epoch
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

    if self.mode != 'eval':

      if dataset_train_load_type is None:
        self.train_data_iter = None
      else:
        print(f"Training data: Loading {dataset_train_load_type} data for {dataset_train}\n\tUsing sampler {ImbalancedDatasetSampler}")
        self.train_data_iter = load_data_iter(
          game=self.game,
          data_types=self.data_types,
          datasets=[dataset_train],
          dataset_exclude=[dataset_val],
          device=self.device,
          batch_size=self.batch_size,
          sampler=ImbalancedDatasetSampler,
          load_type=dataset_train_load_type,
        )

      if dataset_val_load_type is None:
        self.val_data_iter = None
      else:
        self.val_data_iter = None
        # self.val_data_iter = load_data_iter(
        #   game=self.game,
        #   data_types=self.data_types,
        #   datasets=[dataset_val],
        #   dataset_exclude=[dataset_train],
        #   device=self.device,
        #   batch_size=self.batch_size,
        #   load_type=dataset_val_load_type,
        # )

  @abstractmethod
  def forward(self, x: torch.Tensor, *extra_inputs) -> torch.Tensor | tuple[torch.Tensor,...]:
    pass

  @abstractmethod
  def loss_fn(self,
              loss_: torch.nn.modules.loss._Loss,
              acts: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  def train_loop(self, *,
                 opt: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler|None,
                 loss_function: torch.nn.modules.loss._Loss,
                 ):
    pass

  @abstractmethod
  def get_data(self, data: dict[str, torch.Tensor] | list[torch.Tensor]):
    """
    transforms data from the data iterator into the correct format for the model
    """
    pass


  @abstractmethod
  def infer(self, x_var: torch.Tensor) -> torch.Tensor:
    pass

  def load_model_at_epoch(self, epoch: int, load_optimizer=False):
    if epoch == 0:
      print(f"Epoch is 0, not loading {self.model_name} model from disk")
      return
    self.epoch = epoch
    model_pickle = torch.load(self.model_save_string.format(self.epoch))
    self.load_state_dict(model_pickle['model_state_dict'])
    if load_optimizer:
      self.opt.load_state_dict(model_pickle['optimizer_state_dict'])

    self.epoch = model_pickle['epoch']
    loss_val = model_pickle['loss']
    print("Loaded {} model from saved checkpoint {} with loss {}".format(
        self.model_name, self.epoch, loss_val))
