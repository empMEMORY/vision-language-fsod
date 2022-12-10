# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/data/multi_dataset_dataloader.py (Apache-2.0 License)
import copy
import logging
import numpy as np
import operator
import torch
import torch.utils.data
import json
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, log_first_n

from detectron2.config import configurable
from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build