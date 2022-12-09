from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines a selective-gaze-usage action selection network for Atari gameplay, it should simply be imported elsewhere.")

from src.models.mogas_gazed_action_net import MoGaS_Gazed_ActionNet
from src.models.utils import conv_group_output_shape, NOOP_POOL_PARAMS

import torch
import numpy as np
import torch.nn as nn


class SelectiveGaze_ActionNet(MoGaS_Gazed_ActionNet):
  def __init__(self, **kwargs):
    super(SelectiveGaze_ActionNet, self).__init__(**kwargs)

    self.gaze_gate = nn.GRU(3136, 1, 1, batch_first=True)

    self.conv1 = nn.Conv2d(1, 32, 8, stride=(4, 4))
    self.pool = nn.MaxPool2d(**NOOP_POOL_PARAMS)
    # self.pool = lambda x: x

    self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
    self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))

    # self.conv4_cgl = nn.Conv2d(64, 1, 1, stride=(1, 1))
    # self.softmax_cgl = nn.Softmax(dim=0)

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

    self.gate_output = 0

  def add_extra_inputs(self, x: torch.Tensor, x_g: torch.Tensor = None):
    if self.gaze_pred_model is None:
      assert x_g is not None, "If gaze_pred_model is None, x_g must be provided"
    else:
      with torch.no_grad():
        x_g = self.gaze_pred_model.infer(x)
        x_g = self.process_gaze(x_g).unsqueeze(1)
        # x_g = x_g.unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        x = x[:, -1].unsqueeze(1)
        
        x_g = x * x_g ## TODO: for whatever reason they don't apply this during training??
    return x, x_g

  def forward(self, x, x_g):
    # frame forward

    x = self.pool(self.relu(self.conv1(x)))
    x = self.batch_norm32_1(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv2(x)))
    x = self.batch_norm64_1(x)
    # x = self.dropout(x)

    x = self.pool(self.relu(self.conv3(x)))
    x = self.batch_norm64_2(x)
    # x = self.dropout(x)

    # x_cgl = self.softmax_cgl(self.conv4_cgl(x))
    # gaze_overlay forward
    # x_g = self.W * x_g


    embed = torch.flatten(x, start_dim=1).unsqueeze(1).detach()

    h = (torch.ones(1, x.shape[0], 1) * -1).to(device=self.device)
    h.requires_grad = False

    out, h = self.gaze_gate(embed, h)

    out = out.flatten()
    gate_output = torch.relu(torch.sign(out))

    # self.writer.add_scalar('gate_output',gate_output.data.item())
    # print(gate_output.data)
    # self.gate_output= gate_output.data.item()
    self.gate_output= gate_output.data.tolist() # changed because this is a list and not a single value

    gate_output = torch.stack([
        vote * torch.ones(x_g.shape[1:]).to(device=self.device) for vote in gate_output
    ]).to(device=self.device)

    x_g = x_g * gate_output

    x_g = self.pool2(self.relu2(self.conv21(x_g)))
    x_g = self.batch_norm32_2(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.pool2(self.relu2(self.conv22(x_g)))
    x_g = self.batch_norm64_3(x_g)
    # x_g = self.dropout(x_g)

    x_g = self.pool2(self.relu2(self.conv23(x_g)))
    x_g = self.batch_norm64_4(x_g)
    # x_g = self.dropout(x_g)

    # combine gaze conv + frame conv
    # x = (x + x_g)


    x = torch.cat([x, x_g], dim=1)
    # x = x.view(-1, 64 * torch.prod(self.lin_in_shape))
    x = x.flatten(start_dim=1)
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    return x

  def loss_fn(self, loss_, acts, targets, x_g=None):
    return super(SelectiveGaze_ActionNet, self).loss_fn(loss_, acts, targets)

  def process_activations_for_inference(self, acts):
    acts = super().process_activations_for_inference(acts)

    self.writer.add_scalars('actions_gate',{'actions':torch.sign(acts).data.item(),'gate_out':torch.as_tensor(self.gate_output)})
    if self.gate_output == 0:
        self.gate_output = -1

    return acts