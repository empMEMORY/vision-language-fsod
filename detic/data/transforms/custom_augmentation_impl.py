# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Part of the code is from https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/data/transforms.py 
# Modified by Xingyi Zhou
# The original code is under Apache-2.0 License
import numpy as np
import sys
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data.transforms.augmentation import Augmentation
from .custom_transform import EfficientDetResizeCropTransform

__all__ = [
    "EfficientDetResizeCrop",
]

class EfficientDetResizeCrop(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """