# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Xingyi Zhou from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py
import copy
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager, file_lock
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_coco_instances", "register_coco_panoptic_separated"]



def register_oid_instances(name, metadata, json_file, image_root):
    """
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json_mem_efficient(
        json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization