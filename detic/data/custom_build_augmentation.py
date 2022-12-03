# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image


from detectron2.data import transforms as T
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

def build_custom_augmentation(cfg, is_train, scale=None, size=None, \
    min_size=None, max_size=None):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge':
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN if min_size is None else min_size
            max_size = cfg.INPUT.MAX_SIZE_TRAIN if max_size is None else max_size
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    elif cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
        if is_train:
            scale = cfg.INPUT.SCALE_RANGE 