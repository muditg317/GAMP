from src.utils.config import *
from src.data.types import *
import numpy as np
import cv2
from collections import OrderedDict
from yaml import safe_load
import os
import pandas as pd
import h5py
from src.data.utils import stack_data
from src.features.feat_utils import transform_images, fuse_gazes_noop, fuse_gazes
from torch.utils import data
import torch
from itertools import cycle
from collections import Counter

ASSERT_NOT_RUN(__name__, __file__, "You may have meant to run download.py or preprocess.py")

def load_pp_data(game:game_t='breakout', game_run:game_run_t='198_RZ_3877709_Dec-03-16-56-11'):
  """Loads interim data for the specified game and game run 
  
  Args:
  ----
   game : game to load the data from, directory of game runs
   game_run : game_run to load the data from, directory of frames and gaze data

  Returns:
  ----
   gaze_data :  gaze data for the specified game run
   game_run_dir : directory conatining game run frames
  
  """
  game = game
  game_run = game_run

  game_dir = os.path.join(INTERIM_DATA_DIR, game)
  game_run_dir = os.path.join(game_dir, game_run)
  gaze_file = os.path.join(game_run_dir, game_run + '_gaze_data.csv')

  gaze_data = pd.read_csv(gaze_file)

  game_run_frames = OrderedDict({
    int(entry.split('_')[-1].split('.png')[0]): entry
    for entry in os.listdir(game_run_dir) if entry.__contains__('.png')
  })
  if len(game_run_frames) != len(gaze_data.index):
    unks = set(gaze_data.index).symmetric_difference(
      game_run_frames.keys())
    unks_ = []
    for unk in unks:
      if unk in game_run_frames:
        del game_run_frames[unk]
      else:
        unks_.append(unk)

    gaze_data = gaze_data.drop([gaze_data.index[unk] for unk in unks_])
    assert len(game_run_frames) == len(gaze_data.index), print(
      len(game_run_frames), len(gaze_data.index))
    assert set(game_run_frames.keys()) == set(gaze_data.index)
  return gaze_data, game_run_dir


def load_gaze_data(stack=1,
                   stack_type='',
                   stacking_skip=1,
                   from_ix=0,
                   till_ix=-1,
                   *,
                   game:game_t='breakout',
                   game_run:game_run_t='198_RZ_3877709_Dec-03-16-56-11',
                   skip_images=False):
  """Loads and processes gaze data/images for the specified game and game run
  
  Args:
  ----
   stack : Number of frames to stack
   stack_type : ',
   stacking_skip : Number of frames to skip while stacking,
   from_ix :  starting index in the data, default is first, 0
   
   till_ix : last index of the the data to be considered, default is last ,-1
   
   game : game to load the data from, directory of game runs
   game_run : game_run to load the data from, directory of frames and gaze data
 
   skip_images= if True doesn't return frames for the gaem run
 
  Returns:
  ----
   images_ : None or images for the specified game run
   gazes_ :  Formatted gaze data for the specified game run
  
  """
  gaze_data, game_run_dir = load_pp_data(game=game, game_run=game_run)
  gaze_range = [160.0, 210.0]  # w,h
  gaze_data['gaze_positions'] = gaze_data['gaze_positions'].apply(
    lambda gps: [
      np.divide([float(co.strip()) for co in gp.split(',')], gaze_range)
      for gp in gps[2:-2].split('], [')
    ])
  data_ix_f = from_ix
  data_ix_t = till_ix
  images = []
  gazes = list(gaze_data['gaze_positions'])[data_ix_f:data_ix_t]
  if not skip_images:
    for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
      img_data = cv2.imread(os.path.join(game_run_dir,
                         frame_id + '.png'))
      images.append(img_data)

  images_, gazes_ = stack_data(images,
                               gazes,
                               stack=stack,
                               stack_type=stack_type,
                               stacking_skip=stacking_skip)
  return images_, gazes_


def load_action_data(stack=1,
           stack_type='',
           stacking_skip=1,
           from_ix=0,
           till_ix=10,
           *,
           game:game_t,
           game_run:game_run_t):
  """Loads and processes action data/images for the specified game and game run

  Args:
  ----
   stack : Number of frames to stack
   stack_type : ',
   stacking_skip : Number of frames to skip while stacking,
   from_ix :  starting index in the data, default is first, 0
   
   till_ix : last index of the the data to be considered, default is last ,-1
   
   game : game to load the data from, directory of game runs
   game_run : game_run to load the data from, directory of frames and gaze data
 
  Returns:
  ----
   images_ : images for the specified game run
   action_ : action data for the specified game run
  
  """
  gaze_data, game_run_dir = load_pp_data(game=game, game_run=game_run)
  data_ix_f = from_ix
  data_ix_t = till_ix
  images = []

  actions = list(gaze_data['action'])[data_ix_f:data_ix_t]
  for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
    img_data = cv2.imread(os.path.join(game_run_dir, frame_id + '.png'))
    images.append(img_data)

  images_, actions_ = stack_data(images,
                   actions,
                   stack=stack,
                   stack_type=stack_type,
                   stacking_skip=stacking_skip)
  return images_, actions_


def load_hdf_data(
  *,
  game:game_t,
  datasets:list[game_run_t],
  data_types:list[datatype_t],
  device=None,
  hdf5_file: h5py.File=None) -> dict[datatype_t, list[torch.Tensor]]:
  """ Loads data from the hdf game file 
  
  Args:
  ----
   game : game to load the data from, directory of game runs
   datasets : game_run to load the data from, a list of game runs.
        datasets=['564_RZ_4602455_Jul-31-14-48-16'].
   data_types : types of data to load from the file a list of data types.
        ['images', 'actions', 'fused_gazes']
  Returns:
  ----
   game_data : a dcit of game_data loaded from hdf5file for the specified game runs
  
  """
  if hdf5_file is None:
    game_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    game_h5_file = h5py.File(game_file, 'r')
    should_close = True
  else:
    game_h5_file = hdf5_file
    should_close = False
  game_data = []

  # print(datasets, list(game_h5_file.keys()))
  if len(datasets) == 1 and 'combined' in datasets:
    datasets = list(game_h5_file.keys())
  game_data = {k: [] for k in data_types}
  # print(f"Loading hdf5 data from [{game} - {datasets}]")
  actions = []
  for game_run in datasets:
    assert game_h5_file.__contains__(game_run), f"{game_run} doesn't exist in game {game}"
    # print(f"{game_run} found in game {game} file")
    game_run_data_h5 = game_h5_file[game_run]
    # print(game_run_data_h5.keys())
    for datum in data_types:
      # print(data_types)
      assert game_run_data_h5.__contains__(datum), f"{datum} doesn't exist in game {game} {game_run}"
      # print(f"{datum} found in game {game} {game_run} -- {game_run_data_h5[datum].shape}")
      game_data[datum].append(game_run_data_h5[datum][:])
      if device is not None:
        game_data[datum][-1] = torch.Tensor(game_data[datum][-1]).to(device=device)
    
  if should_close:
    game_h5_file.close()
  return game_data


def load_data_iter(*,
  game=None,
  data_types=['images', 'actions', 'gazes', 'gazes_fused_noop'],
  datasets:list[game_run_t]=['combined'],
  dataset_exclude=['combined'],
  device=torch.device('cpu'),
  load_type='memory',
  batch_size=32,
  sampler=None):
  """
  Creates a dataset and a data iterator to use in the train loop
  
  Args:
  ----
   game : game to load the data from, directory of game runs
   data_types : types of data to load, contains atleast on of the following
        ['images', 'actions', 'gazes',' gazes_fused_noop']

   datasets : game_run to load the data from, directory of frames and gaze data
   device : device to load the data to cpu or gpu
   load_type : data load type, different types are described below
    'memory' --  'loads everything into memoey if possible else errors
            fastest option'
    'disk' -- 'Reads from the hdf5 file the specified index every 
            iteration,slowest'
    'live'  -- 'Directly loads from the interim files and  
            preprocesss the data'
    'chunked' -- 'Splits the given dataset hdf5 file into 
            specified chunks and cycles through them in the train loop'
  'batch_size : batch size of the data to iterate over, default 32
  sampler : Type of sampler class to use when sampling the data, defualt None 

  Returns:
  ----
   data_iter : a data iterator with the specified data types that can be looped over
  
  """
  if isinstance(datasets, str):
    datasets = [datasets]

  print(f"Loading data iterator for [{game} - {datasets}]...\n\tUsing {load_type} load type\n\tFetching data types: {data_types}\n\tUsing sampler: {sampler}")

  if load_type == 'memory':
    hdf_data = load_hdf_data(game=game, datasets=datasets, data_types=data_types)
    x, y_, _, x_g = hdf_data.values()
    x = torch.Tensor(x).squeeze().to(device=device)
    y = torch.LongTensor(y_).squeeze()[:, -1].to(device=device)
    x_g = torch.Tensor(x_g).squeeze().to(device=device)
    dataset = data.TensorDataset(x, y, x_g)
    dataset.labels = y_[0][:, -1]

  elif load_type == 'disk':
    assert len(datasets) == 1, "Disk load type only supports one dataset"
    dataset = HDF5TorchDataset(game=game,
                   data_types=data_types,
                   dataset=datasets[0],
                   device=device)
  elif load_type == 'live':
    assert len(datasets) == 1, "Live load type only supports one dataset"
    print("prepping and loading data for {},{}".format(game, datasets[0]))
    images_, actions_ = load_action_data(stack=STACK_SIZE,
                       game=game,
                       till_ix=-1,
                       game_run=datasets[0])

    _, gazes = load_gaze_data(stack=STACK_SIZE,
                  game=game,
                  till_ix=-1,
                  game_run=datasets[0],
                  skip_images=True)
    images_ = transform_images(images_, type='torch')
    gazes = fuse_gazes(images_, gazes, gaze_count=1)
    x = images_.to(device=device)
    y = torch.LongTensor(actions_)[:, -1].to(device=device)
    x_g = gazes.to(device=device)
    dataset = data.TensorDataset(x, y, x_g)
    dataset.labels = np.array(actions_)[:, -1]

  elif load_type == 'chunked':
    sampler = None

    dataset = HDF5TorchChunkDataset(game=game,
                                    data_types=data_types,
                                    dataset_exclude=dataset_exclude,
                                    datasets=datasets,
                                    device=device)

  print(f"    Using sampler: {sampler}")
  if sampler is None:
    data_iter = data.DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=0)
  else:
    data_iter = data.DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler(dataset))

  print(f"    Loaded data iterator for [{game} - {datasets}]...\n\tDataset size: {len(dataset)}")

  return data_iter


class HDF5TorchChunkDataset(data.Dataset):
  def __init__(self,
               game:game_t,
               data_types:list[datatype_t]        = ['images', 'actions', 'gazes', 'gazes_fused_noop'],
               datasets:list[game_run_t]          = ['combined'],
               dataset_exclude:list[game_run_t]   = ['combined'],
               device: torch.device               = torch.device('cpu'),
               num_epochs_per_collation           = 1,
               num_groups_to_collate              = 1
              ):
    self.game = game
    self.datasets = datasets
    self.dataset_exclude = dataset_exclude
    self.data_types = data_types

    self.device = device

    self.num_epochs_per_collation = num_epochs_per_collation
    self.num_groups_to_collate = num_groups_to_collate

    self.curr_collation_epoch = 0
    self.curr_collation_data = {}

    hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(game))
    self.hdf5_file = h5py.File(hdf5_file, 'r')

    groups = set(self.hdf5_file.keys()) - set(self.dataset_exclude)
    if 'combined' not in self.datasets:
      groups = groups & set(self.datasets)
    groups = list(sorted(groups, reverse=True))
    print(f"Loading chunked data for {game} with groups: {groups}")
    
    self.groups = cycle(groups)
    self.group_lens = [
      self.hdf5_file[g]['actions'].len() for g in groups
    ]
    self.total_count = self.num_epochs_per_collation * sum(self.group_lens)

    self.curr_collation = None

    self.__reset_dataset__()

  def __load_data__(self):
    for dtype in list(self.curr_collation_data.keys()):
      del self.curr_collation_data[dtype]
    self.curr_collation_data = load_hdf_data(game=self.game,
                                             datasets=self.curr_collation,
                                             data_types=self.data_types,
                                             device=self.device,
                                             hdf5_file=self.hdf5_file)
    print(f"Loaded hdf data for [{self.game} - {self.curr_collation}]:\n\tTypes: {self.data_types}")

    for dtype in self.curr_collation_data:
      datum = self.curr_collation_data[dtype]
      datum: torch.Tensor = torch.concatenate(datum, axis=0)
      if dtype == 'actions':
        datum = datum.long().squeeze()[:, -1].to(
          device=self.device)
      else:
        datum = datum.squeeze().to(device=self.device)
      self.curr_collation_data[dtype] = datum.detach()
    group_lens = [
      self.curr_collation_data[datum].shape[0]
      for datum in self.curr_collation_data
    ]

    assert len(set(group_lens)) == 1
    self.curr_collation_len = group_lens[0]

    print(f"Concated hdf data ({self.curr_collation_data.keys()})\n\tLength: {self.curr_collation_len}")

  def __reset_dataset__(self):

    self.curr_ix = 0

    if self.curr_collation_epoch == self.num_epochs_per_collation or self.curr_collation is None:
      self.curr_collation_epoch = 0

      old_collation = self.curr_collation
      self.curr_collation = [
        next(self.groups) for _ in range(self.num_groups_to_collate)
      ]

      if old_collation is None or len(set(old_collation) ^ set(self.curr_collation)) > 0:
        print(f"Cycling chunked datasets\n\tfrom {old_collation}")
        print(f"\tto   {self.curr_collation}")
        self.__load_data__()
      # else:
      #   print(f"Skipping loading data for {self.curr_collation} (already loaded)")

      if 'actions' in self.curr_collation_data:
        labels = self.curr_collation_data['actions'].cpu().numpy(
        ).copy()
        label_to_count = Counter(labels)
        weights = torch.DoubleTensor(
          [1.0 / label_to_count[ix] for ix in labels])
        self.sample_ixs = torch.multinomial(weights,
                          self.curr_collation_len,
                          replacement=True)
      else:
        self.sample_ixs = list(range(self.curr_collation_len))

    self.curr_collation_epoch += 1

  def __len__(self):
    return self.total_count

  def __getitem__(self, ix):
    if self.curr_ix == self.curr_collation_len:
      self.__reset_dataset__()

    sample_ix = self.sample_ixs[self.curr_ix]

    tensors = {
      dtype: self.curr_collation_data[dtype][sample_ix]
      for dtype in self.data_types
    }

    self.curr_ix += 1

    return tensors


class HDF5TorchDataset(data.Dataset):
  def __init__(self,
               game:game_t,
               data_types:list[datatype_t]  =['images', 'actions', 'gazes'],
               dataset:game_run_t           ='combined',
               device:torch.device          =torch.device('cpu')):
    hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(game))
    self.hdf5_file = h5py.File(hdf5_file, 'r')
    self.data_types = data_types
    self.group = self.hdf5_file[dataset]
    self.group_counts = self.group['actions'].len()

    self.labels = self.group['actions'][:, -1]
    self.device = device

  def __len__(self):
    return self.group_counts

  def __getitem__(self, ix):
    tensors = []
    for datum in self.data_types:
      if datum == 'actions':
        tensors.append(
          torch.tensor(self.labels[ix]).to(device=self.device).detach())
      else:
        tensors.append(
          torch.Tensor(self.group[datum][ix]).to(device=self.device).detach())
    return tensors