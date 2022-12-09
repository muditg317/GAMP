from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file is just a base class for other gaze models.")
from src.data.types import *
from src.models.mogas_net import MoGas_Net

from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
from src.data.utils import ImbalancedDatasetSampler
from src.data.loaders import HDF5TorchDataset, load_data_iter

np.random.seed(42)


class MoGas_GazeNet(nn.Module, ABC):
  def __init__(self,
               data_types:list[datatype_t]                  = list(set(DATA_TYPES) - set(['actions'])),
               **kwargs):
    super(MoGas_GazeNet, self).__init__(data_types=data_types, **kwargs)
    
    
  @abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Should output a tensor in the "log-space" for easy input into KL-div loss
    """
    pass

  # @abstractmethod
  def loss_fn(self,
              loss_: torch.nn.modules.loss._Loss,
              smax_pi: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    """
    Takes in the output of the forward pass and the targets and returns the loss

    Args:
      loss_ (torch.nn.modules.loss): loss function
      smax_pi (torch.Tensor): output of forward pass
        - in the log space with shape (batch_size, flattened)
      targets (torch.Tensor): targets
        - straight from data
    """
    targets_reshaped = targets.view(-1,
                    targets.shape[1] * targets.shape[2])

    return loss_(smax_pi, targets_reshaped)

  # if scheduler is declared, ought to use & update it , else model never trains
  def train_loop(self,
                 opt: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler|None,
                 loss_function: torch.nn.modules.loss._Loss,
                 LR_SCHEDULER_FREQ=2,
                 ):
    self.loss_ = loss_function
    self.opt = opt

    if self.load_model:
      self.load_model_at_epoch(self.epoch, load_optimizer=True)
    else:
      self.epoch = -1

    eix = 0
    start_epoch = self.epoch+1
    end_epoch = start_epoch+15
    for epoch in range(start_epoch,end_epoch):
      print(f"Training epoch {epoch}/{end_epoch}...")
      for i, data in enumerate(self.train_data_iter):
        x, y = self.get_data(data)

        opt.zero_grad()

        smax_pi = self.forward(x)

        loss = self.loss_fn(self.loss_, smax_pi, y)
        loss.backward()
        opt.step()

        self.writer.add_scalar('Loss', loss.data.item(), eix)

        eix+=1

      if epoch % 1 == 0:
        torch.save(
          {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
          }, self.model_save_string.format(epoch))

        # self.writer.add_histogram('smax', smax_pi[0])
        # self.writer.add_histogram('target', y)
        print(f"Epoch {epoch} complete:")
        print(f"\tLoss: {loss.data.item()}")
        self.writer.add_scalar('Epoch Loss', loss.data.item(), epoch)
        if lr_scheduler is not None:
          self.writer.add_scalar('Learning Rate', lr_scheduler.get_lr()[0], epoch)
        # self.writer.add_scalar('Epoch Val Loss',
        #                        self.val_loss().data.item(), epoch)

      if lr_scheduler is not None and epoch % LR_SCHEDULER_FREQ ==0 :
        lr_scheduler.step()

  def get_data(self, data: dict[str, torch.Tensor] | list[torch.Tensor]):
    if isinstance(data, dict):
      x = data['images'].to(device=self.device)
      y = data['gazes'].to(device=self.device)

    elif isinstance(data, list):
      x, y = data
      x = x.to(device=self.device)
      y = y.to(device=self.device)

    return x, y

  def val_loss(self):
    self.eval()
    val_loss = []
    with torch.no_grad():
      for i, data in enumerate(self.val_data_iter):
        x, y = self.get_data(data)
        smax_pi = self.forward(x)

        val_loss.append(self.loss_fn(self.loss_, smax_pi, y))
    self.train()
    return torch.mean(torch.Tensor(val_loss))

  def infer(self, x_var):
    self.eval()

    with torch.no_grad():

      smax_dist = self.forward(x_var).view(-1, self.input_shape[0],
                         self.input_shape[1])

    self.train()

    return smax_dist