from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines an action selection network that uses gaze for Atari gameplay, it should simply be imported elsewhere.")

from src.models.mogas_gazed_action_net import MoGaS_Gazed_ActionNet, NOOP_POOL_PARAMS
from src.models.utils import conv_group_output_shape

import torch
import numpy as np
import torch.nn as nn


class Gazed_ActionNet(MoGaS_Gazed_ActionNet):
  def __init__(self,
               **kwargs):
    super(Gazed_ActionNet, self).__init__(**kwargs)

    self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
    self.pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)

    self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))

    # self.W = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    self.lin_in_shape = conv_group_output_shape([self.conv1, self.conv2, self.conv3], self.input_shape)
    self.linear1 = nn.Linear(64 * np.prod(self.lin_in_shape), 512)
    self.linear2 = nn.Linear(512, 128)
    self.linear3 = nn.Linear(128, self.num_actions)
    self.batch_norm32 = nn.BatchNorm2d(32)
    self.batch_norm64 = nn.BatchNorm2d(64)
    self.dropout = nn.Dropout()
    self.relu = nn.ReLU()
    self.softmax = torch.nn.Softmax()

  def add_extra_inputs(self, x: torch.Tensor, x_g: torch.Tensor = None):
    if self.gaze_pred_model is None:
      assert x_g is not None, "If gaze_pred_model is None, x_g must be provided"
    else:
      with torch.no_grad():
        x_g = self.gaze_pred_model.infer(x)
        x_g = self.process_gaze(x_g)
        x_g = x_g.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        x_g = x*x_g
    return x, x_g


  def forward(self, x, x_g):
    # frame forward
    x = self.pool(self.relu(self.conv1(x)))
    # x = self.batch_norm32(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv2(x)))
    # x = self.batch_norm64(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv3(x)))
    # x = self.batch_norm64(x)
    # x = self.dropout(x)

    # gaze_overlay forward
    # x_g = (self.W * x_g)
    x_g = self.pool(self.relu(self.conv1(x_g)))
    # x_g = self.batch_norm32(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.pool(self.relu(self.conv2(x_g)))
    # x_g = self.batch_norm64(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.pool(self.relu(self.conv3(x_g)))
    # x_g = self.batch_norm64(x_g)
    # x_g = self.dropout(x_g)

    # combine gaze conv + frame conv
    x = 0.5 * (x + x_g)
    x = x.view(-1, 64 * np.prod(self.lin_in_shape)) # reshape to [batch_N, flattened]
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)

    return x

  def loss_fn(self, loss_, acts, targets, x_g=None):
    return super(Gazed_ActionNet, self).loss_fn(loss_, acts, targets)

  def process_activations_for_inference(self, acts):
    return super(Gazed_ActionNet, self).process_activations_for_inference(acts)