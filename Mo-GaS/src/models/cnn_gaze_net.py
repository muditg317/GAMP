from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines a CNN-based gaze predictor for Atari gameplay, it should simply be imported elsewhere.")
from src.models.mogas_gaze_net import MoGas_GazeNet

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# np.random.seed(42)

class CNN_GazeNet(MoGas_GazeNet):
  def __init__(self,
               **kwargs
              ):

    super(CNN_GazeNet, self).__init__(**kwargs)

    self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
    self.pool = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
    # self.pool = lambda x: x

    self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
    self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=(1, 1))
    self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=(2, 2))
    self.deconv3 = nn.ConvTranspose2d(32, 1, 8, stride=(4, 4))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))

    x = self.pool(F.relu(self.conv2(x)))

    x = self.pool(F.relu(self.conv3(x)))

    x = self.pool(F.relu(self.deconv1(x)))

    x = self.pool(F.relu(self.deconv2(x)))

    x = self.deconv3(x)

    x = x.squeeze(1)

    x = x.view(-1, x.shape[1] * x.shape[2]) # reshape to (batch_size, flattened image)

    x = F.log_softmax(x, dim=1) # use log_softmax for KL-div loss (requires input to be log probabilities)

    return x