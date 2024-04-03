# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from collections import defaultdict
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm
import os
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.data.datasets.coco import load_coco_json

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds, get_fed_loss_inds_deterministic

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_w