
# Copyright (c) Facebook, Inc. and its affiliates. 
import torch
import torch.nn.functional as F
from torch import nn
import math

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, Instances
from ..utils import load_class_freq, get_fed_loss_inds

from models.backbone import Joiner
from models.deformable_detr import DeformableDETR, SetCriterion, MLP
from models.deformable_detr import _get_clones
from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine
from models.deformable_transformer import DeformableTransformer
from models.segmentation import sigmoid_focal_loss
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor, accuracy


__all__ = ["DeformableDetr"]

class CustomSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, \
        focal_alpha=0.25, use_fed_loss=False):
        super().__init__(num_classes, matcher, weight_dict, losses, focal_alpha)
        self.use_fed_loss = use_fed_loss
        if self.use_fed_loss:
            self.register_buffer(
                'fed_loss_weight', load_class_freq(freq_weight=0.5))

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """