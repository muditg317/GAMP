from __future__ import annotations
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines a selective-motion-usage action selection network for Atari gameplay, it should simply be imported elsewhere.")

from src.models.mogas_action_net import MoGaS_ActionNet
from src.models.utils import conv_group_output_shape, NOOP_POOL_PARAMS
from src.features.feat_utils import compute_coverage_loss, compute_motion

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SelectiveMotion_CL_ActionNet(MoGaS_ActionNet):
  def __init__(self, *,
               coverage_loss_weight: float = 0.1,
               **kwargs):
    super().__init__(**kwargs)
    self.coverage_loss_weight = coverage_loss_weight

    self.motion_gate = nn.GRU(3136, 1, 1, batch_first=True)

    self.conv1 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    # self.pool = lambda x: x

    self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))

    # self.conv4_cl = nn.Conv2d(64, 1, 1, stride=(1, 1))
    # self.softmax_cl = nn.Softmax(dim=0)

    self.conv21 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.pool2 = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    # self.pool = lambda x: x

    self.conv22 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.conv23 = nn.Conv2d(64, 64, 3, stride=(1, 1))

    # self.W = torch.nn.Parameter(
    #     torch.Tensor([0.0]), requires_grad=True)

    self.lin_in_shape = conv_group_output_shape([self.conv1, self.conv2, self.conv3], self.input_shape)
    self.linear1 = nn.Linear(2*64 * np.prod(self.lin_in_shape), 512)
    self.linear2 = nn.Linear(512, 128)
    self.linear3 = nn.Linear(128, self.num_actions)
    # self.batch_norm32 = nn.BatchNorm2d(32)
    self.batch_norm32_1 = nn.BatchNorm2d(32)
    self.batch_norm32_2 = nn.BatchNorm2d(32)
    # self.batch_norm64 = nn.BatchNorm2d(64)
    self.batch_norm64_1 = nn.BatchNorm2d(64)
    self.batch_norm64_2 = nn.BatchNorm2d(64)
    self.batch_norm64_3 = nn.BatchNorm2d(64)
    self.batch_norm64_4 = nn.BatchNorm2d(64)
    self.dropout = nn.Dropout(p=0.2)
    self.relu = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.softmax = torch.nn.Softmax()

    self.cl_conv = nn.Conv2d(64, 1, 1, stride=(1, 1))
    self.cl_softmax = nn.Softmax()

    self.gate_output = 0

  def add_extra_inputs(self, x: torch.Tensor, x_m: torch.Tensor = None):
    if x_m is None:
      with torch.no_grad():
        x_m = compute_motion(x).unsqueeze(1)

    x = x[:, -1].unsqueeze(1)
      
    # x_m = x * x_m ## Moved scaling to forward pass
    return x, x_m

  def forward(self, x, x_m):
    # frame forward
    x_m = x * x_m

    x = self.pool(self.relu(self.conv1(x)))
    x = self.batch_norm32_1(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv2(x)))
    x = self.batch_norm64_1(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv3(x)))
    x = self.batch_norm64_2(x)
    # x = self.dropout(x)

    # x_cl = self.softmax_cl(self.conv4_cl(x))
    # motion_overlay forward
    # x_m = self.W * x_m


    embed = torch.flatten(x, start_dim=1).unsqueeze(1).detach()

    h = (torch.ones(1, x.shape[0], 1) * -1).to(device=self.device)
    h.requires_grad = False

    out, h = self.motion_gate(embed, h)

    out = out.flatten()
    gate_output = torch.relu(torch.sign(out))

    # self.writer.add_scalar('gate_output',gate_output.data.item())
    # print(gate_output.data)
    # self.gate_output= gate_output.data.item()
    self.gate_output= gate_output.data.tolist() # changed because this is a list and not a single value

    gate_output = torch.stack([
        vote * torch.ones(x_m.shape[1:]).to(device=self.device) for vote in gate_output
    ]).to(device=self.device)

    x_m = x_m * gate_output

    x_m = self.pool2(self.relu2(self.conv21(x_m)))
    x_m = self.batch_norm32_2(x_m)
    # x_m = self.dropout(x_m)

    x_m = self.pool2(self.relu2(self.conv22(x_m)))
    x_m = self.batch_norm64_3(x_m)
    # x_m = self.dropout(x_m)

    x_m = self.pool2(self.relu2(self.conv23(x_m)))
    x_m = self.batch_norm64_4(x_m)
    # x_m = self.dropout(x_m)

    # combine motion conv + frame conv
    # x = (x + x_m)


    y = torch.cat([x, x_m], dim=1)
    y = y.flatten(start_dim=1)
    y = self.linear1(y)
    y = self.linear2(y)
    y = self.linear3(y)


    cl_out = self.cl_conv(x)
    cl_out = cl_out.squeeze(1)
    cl_out = cl_out.view(-1, cl_out.shape[1] * cl_out.shape[2]) # reshape to (batch_size, flattened image)
    cl_out = F.softmax(cl_out, dim=1)

    return y, cl_out

  def loss_fn(self, loss_, acts, targets, motion_true:torch.Tensor=None, cl_out=None):
    ce_loss = super().loss_fn(loss_, acts, targets)
    cl_loss = compute_coverage_loss(cl_out, motion_true.squeeze(1))
    return {
      'CrossEntropyLoss': (ce_loss, 1-self.coverage_loss_weight),
      'CoverageLoss': (cl_loss, self.coverage_loss_weight)
    }

  def process_activations_for_inference(self, acts):
    acts = super().process_activations_for_inference(acts)

    self.writer.add_scalars('actions_gate',{'actions':torch.sign(acts).data.item(),'gate_out':torch.as_tensor(self.gate_output)})
    if self.gate_output == 0:
        self.gate_output = -1

    return acts