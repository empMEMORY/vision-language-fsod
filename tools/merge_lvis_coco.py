# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
import torch
import sys
import json
import numpy as np

from detectron2.structures import Boxes, pairwise_iou
COCO_PATH = 'datasets/coco/annotations/instances_train2017.json'
IMG_PATH = 'datasets/coco/train2017/'
LVIS_PATH = 'datasets/lvis/lvis