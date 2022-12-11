from __future__ import annotations
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines a selective-gaze-and-motion-usage action selection network for Atari gameplay, it should simply be imported elsewhere.")

from src.models.mogas_gazed_action_net import MoGaS_Gazed_ActionNet
from src.models.utils import conv_group_output_shape, NOOP_POOL_PARAMS
from src.features.feat_utils import compute_coverage_loss, compute_motion

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SelectiveGazeAndMotion_CL_ActionNet(MoGaS_Gazed_ActionNet):
  def __init__(self, *,
               gaze_coverage_loss_weight: float = 0.0005,
               motion_coverage_loss_weight: float = 0.0005,
               **kwargs):
    super().__init__(**kwargs)
    self.gaze_coverage_loss_weight = gaze_coverage_loss_weight
    self.motion_coverage_loss_weight = motion_coverage_loss_weight

    self.gaze_gate = nn.GRU(3136, 1, 1, batch_first=True)
    self.motion_gate = nn.GRU(3136, 1, 1, batch_first=True)

    self.frame_conv1 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.frame_pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    self.frame_conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.frame_conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
    self.frame_relu = nn.ReLU()

    self.frame_bn32_1 = nn.BatchNorm2d(32)
    self.frame_bn64_1 = nn.BatchNorm2d(64)
    self.frame_bn64_2 = nn.BatchNorm2d(64)


    self.gaze_conv1 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.gaze_pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    self.gaze_conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.gaze_conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
    self.gaze_relu = nn.ReLU()

    self.gaze_bn32_1 = nn.BatchNorm2d(32)
    self.gaze_bn64_1 = nn.BatchNorm2d(64)
    self.gaze_bn64_2 = nn.BatchNorm2d(64)


    self.motion_conv1 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.motion_pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    self.motion_conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.motion_conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
    self.motion_relu = nn.ReLU()

    self.motion_bn32_1 = nn.BatchNorm2d(32)
    self.motion_bn64_1 = nn.BatchNorm2d(64)
    self.motion_bn64_2 = nn.BatchNorm2d(64)


    self.lin_in_shape = conv_group_output_shape([self.frame_conv1, self.frame_conv2, self.frame_conv3], self.input_shape)
    self.linear1 = nn.Linear(3*64 * np.prod(self.lin_in_shape), 512)
    self.linear2 = nn.Linear(512, 128)
    self.linear3 = nn.Linear(128, self.num_actions)

    self.dropout = nn.Dropout(p=0.2)
    self.softmax = torch.nn.Softmax()


    self.gaze_cl_conv = nn.Conv2d(64, 1, 1, stride=(1, 1))
    self.motion_cl_conv = nn.Conv2d(64, 1, 1, stride=(1, 1))

    self.gaze_gate_output = 0
    self.motion_gate_output = 0

  def add_extra_inputs(self, x: torch.Tensor, x_g: torch.Tensor = None, x_m: torch.Tensor = None):
    if x_m is None:
      with torch.no_grad():
        x_m = compute_motion(x).unsqueeze(1)

    if self.gaze_pred_model is None:
      assert x_g is not None, "If gaze_pred_model is None, x_g must be provided"
    else:
      with torch.no_grad():
        x_g = self.gaze_pred_model.infer(x)
        x_g = self.process_gaze(x_g).unsqueeze(1)

    x = x[:, -1].unsqueeze(1)
    
    # x_g = x * x_g ## Moved scaling to forward pass
    
    return x, x_g, x_m

  def forward(self, x, x_g, x_m):
    # frame forward
    x_g = x * x_g
    x_m = x * x_m

    x = self.frame_pool(self.frame_relu(self.frame_conv1(x)))
    x = self.frame_bn32_1(x)
    # x = self.dropout(x)

    x = self.frame_pool(self.frame_relu(self.frame_conv2(x)))
    x = self.frame_bn64_1(x)
    # x = self.dropout(x)

    x = self.frame_pool(self.frame_relu(self.frame_conv3(x)))
    x = self.frame_bn64_2(x)
    # x = self.dropout(x)

    # x_cgl = self.softmax_cgl(self.conv4_cgl(x))
    # gaze_overlay forward
    # x_g = self.W * x_g


    embed = torch.flatten(x, start_dim=1).unsqueeze(1).detach()


    # gaze gating
    h = (torch.ones(1, x.shape[0], 1) * -1).to(device=self.device)
    h.requires_grad = False

    out_g, h = self.gaze_gate(embed, h)

    out_g = out_g.flatten()
    gaze_gate_output = torch.relu(torch.sign(out_g))

    # self.writer.add_scalar('gaze_gate_output',gaze_gate_output.data.item())
    # print(gaze_gate_output.data)
    # self.gaze_gate_output= gaze_gate_output.data.item()
    self.gaze_gate_output= gaze_gate_output.data.tolist() # changed because this is a list and not a single value

    gaze_gate_output = torch.stack([
        vote * torch.ones(x_g.shape[1:]).to(device=self.device) for vote in gaze_gate_output
    ]).to(device=self.device)

    x_g = x_g * gaze_gate_output


    # gaze forward
    x_g = self.gaze_pool(self.gaze_relu(self.gaze_conv1(x_g)))
    x_g = self.gaze_bn32_1(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.gaze_pool(self.gaze_relu(self.gaze_conv2(x_g)))
    x_g = self.gaze_bn64_1(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.gaze_pool(self.gaze_relu(self.gaze_conv3(x_g)))
    x_g = self.gaze_bn64_2(x_g)
    # x_g = self.dropout(x_g)


    # motion gating
    h = (torch.ones(1, x.shape[0], 1) * -1).to(device=self.device)
    h.requires_grad = False

    out_m, h = self.motion_gate(embed, h)

    out_m = out_m.flatten()
    motion_gate_output = torch.relu(torch.sign(out_m))

    # self.writer.add_scalar('motion_gate_output',motion_gate_output.data.item())
    # print(motion_gate_output.data)
    # self.motion_gate_output= motion_gate_output.data.item()
    self.motion_gate_output = motion_gate_output.data.tolist() # changed because this is a list and not a single value

    motion_gate_output = torch.stack([
        vote * torch.ones(x_m.shape[1:]).to(device=self.device) for vote in motion_gate_output
    ]).to(device=self.device)

    x_m = x_m * motion_gate_output


    # motion forward
    x_m = self.motion_pool(self.motion_relu(self.motion_conv1(x_m)))
    x_m = self.motion_bn32_1(x_m)
    # x_m = self.dropout(x_m)

    x_m = self.motion_pool(self.motion_relu(self.motion_conv2(x_m)))
    x_m = self.motion_bn64_1(x_m)
    # x_m = self.dropout(x_m)

    x_m = self.motion_pool(self.motion_relu(self.motion_conv3(x_m)))
    x_m = self.motion_bn64_2(x_m)
    # x_m = self.dropout(x_m)


    # concat and linear
    y = torch.cat([x, x_g, x_m], dim=1)
    y = y.flatten(start_dim=1)
    y = self.linear1(y)
    y = self.linear2(y)
    y = self.linear3(y)


    gaze_cl_out = self.gaze_cl_conv(x)
    gaze_cl_out = gaze_cl_out.squeeze(1)
    gaze_cl_out = gaze_cl_out.view(-1, gaze_cl_out.shape[1] * gaze_cl_out.shape[2]) # reshape to (batch_size, flattened image)
    gaze_cl_out = F.softmax(gaze_cl_out, dim=1)

    motion_cl_out = self.motion_cl_conv(x)
    motion_cl_out = motion_cl_out.squeeze(1)
    motion_cl_out = motion_cl_out.view(-1, motion_cl_out.shape[1] * motion_cl_out.shape[2]) # reshape to (batch_size, flattened image)
    motion_cl_out = F.softmax(motion_cl_out, dim=1)

    return y, gaze_cl_out, motion_cl_out

  def loss_fn(self, loss_, acts, targets,
              gaze_true:torch.Tensor=None, motion_true:torch.Tensor=None,
              gaze_cl_out:torch.Tensor=None, motion_cl_out:torch.Tensor=None):
    ce_loss = super().loss_fn(loss_, acts, targets)
    gaze_cl_loss = compute_coverage_loss(gaze_cl_out, gaze_true.squeeze(1))
    motion_cl_loss = compute_coverage_loss(motion_cl_out, motion_true.squeeze(1))
    return {
      'CrossEntropyLoss': (ce_loss, 1-self.gaze_coverage_loss_weight-self.motion_coverage_loss_weight),
      'Gaze-CoverageLoss': (gaze_cl_loss, self.gaze_coverage_loss_weight),
      'Motion-CoverageLoss': (motion_cl_loss, self.motion_coverage_loss_weight)
    }

  def process_activations_for_inference(self, acts):
    acts = super().process_activations_for_inference(acts)

    self.writer.add_scalars('actions_gates',{
        'actions':torch.sign(acts).data.item(),
        'gaze_gate_out':torch.as_tensor(self.gaze_gate_output),
        'motion_gate_out':torch.as_tensor(self.motion_gate_output)
      })
    if self.gaze_gate_output == 0:
      self.gaze_gate_output = -1
    if self.motion_gate_output == 0:
      self.motion_gate_output = -1

    return acts