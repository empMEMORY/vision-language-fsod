# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .lvis_v1 import custom_register_lvis_instances

categories_seen = [
    {'id': 1, 'name': 'person'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'motorcycle'},
    {'id': 7, 'name': 'train'},
    {'id': 8, 'name': 'truck'},
    {'id': 9, 'name': 'boat'},
    {'id': 15, 'name': 'bench'},
    {'id': 16, 'name': 'bird'},
    {'id': 19, 'name': 'horse'},
    {'id