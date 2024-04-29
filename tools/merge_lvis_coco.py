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
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
