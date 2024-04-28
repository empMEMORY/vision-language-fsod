# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
import torch
import sys
import json
import numpy as np

from detectron2.structures import Boxes, pairwise_iou
COCO_PATH = 'datasets/coco/annotations/instances_train2017.json'
IMG_PATH = 'datasets/coco/train2017/'
LVIS_PATH = 'datasets/lvis/lvis_v1_train.json'
NO_SEG = False
if NO_SEG:
    SAVE_PATH = 'datasets/lvis/lvis_v1_train+coco_box.json'
else:
    SAVE_PATH = 'datasets/lvis/lvis_v1_train+coco_mask.json'
THRESH = 0.7
DEBUG = False

# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},