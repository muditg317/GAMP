from src.utils.config import *
ASSERT_NOT_RUN(__name__, __file__, "This file defines a generic gaze-based action selection network for Atari gameplay, it should simply be imported elsewhere.")
from src.data.types import datatype_t
from src.models.mogas_action_net import MoGaS_ActionNet
from src.models.mogas_gaze_net import MoGas_GazeNet

from abc import ABC, abstractmethod
import torch

class MoGaS_Gazed_ActionNet(MoGaS_ActionNet, ABC):
  def __init__(self, *,
               data_types:list[datatype_t]                  = DATA_TYPES,
               gaze_pred_model:MoGas_GazeNet                = None,
               **kwargs):
    self.gaze_pred_model = gaze_pred_model
    super(MoGaS_Gazed_ActionNet, self).__init__(data_types=data_types, **kwargs)

  def process_gaze(self,gaze):
    """
    Process gaze prediction to be in the range [0, 1]
    gaze comes in as "log_softmax"
    output such that gazes scaled to make each POI equal value
    """
    gaze = torch.exp(gaze)
    gazes = []
    for g in gaze:
        g = (g - torch.min(g)) / (torch.max(g) - torch.min(g))
        # g = g / torch.sum(g) ## Don't normalize here (would undo line above)
        gazes.append(g)
    gaze = torch.stack(gazes)
    del gazes
    return gaze

  def train_loop(self, *, gaze_pred: MoGas_GazeNet = None, **kwargs):
    self.gaze_pred_model = gaze_pred
    if self.gaze_pred_model is not None:
      self.gaze_pred_model.eval()
    return super().train_loop(**kwargs)

  @abstractmethod
  def process_activations_for_inference(self, acts):
    return torch.argmax(torch.softmax(acts, dim=1))