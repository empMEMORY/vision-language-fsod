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

l