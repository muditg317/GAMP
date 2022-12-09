from __future__ import annotations
from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__)

from math import floor
import torch.nn as nn

NOOP_POOL_PARAMS = {
  'kernel_size': (1, 1),
  'stride': (1, 1),
  'padding': (0, 0),
  'dilation': (1, 1),
}

def conv_layer_output_shape(conv_layer: nn.Conv2d, input_shape: tuple[int,int]):
  h_in, w_in = input_shape
  h_out, w_out = floor((
    (h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] *
      (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0]) + 1), floor((
        (w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] *
        (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1]) + 1)
  return h_out, w_out

def conv_group_output_shape(conv_layers: list[nn.Conv2d], input_shape: tuple[int,int]):
  out_shape = input_shape
  for conv_layer in conv_layers:
    out_shape = conv_layer_output_shape(conv_layer, out_shape)
  return out_shape


def dataset_to_list_and_str(dataset: list[str]) -> str:
  dataset_str = dataset
  dataset_list = dataset
  if isinstance(dataset, list):
    dataset_str = '__'.join([dt[:5] for dt in dataset])
  else:
    dataset_list = [dataset_list]
  return dataset_list, dataset_str