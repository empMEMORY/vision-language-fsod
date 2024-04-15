
# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import numpy as np
from torch.nn import functional as F
import os 
from detectron2.data import MetadataCatalog

def load_class_freq(
    path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=1.0, use_ann_count=False):
    cat_info = json.load(open(path, 'r'))

    if use_ann_count:
        cat_info = torch.tensor(
            [c['instance_count'] for c in sorted(cat_info, key=lambda x: x['id'])]) 
    else:
        cat_info = torch.tensor(
            [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight