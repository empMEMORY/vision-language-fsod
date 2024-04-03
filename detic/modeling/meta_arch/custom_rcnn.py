# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from collections import defaultdict
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm
import os
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.data.datasets.coco import load_coco_json

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds, get_fed_loss_inds_deterministic

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        modify_neg_loss = False,
        use_zs_preds_nl = False,
        zs_preds_path_nl = None,
        zs_conf_thresh = None,
        use_gt_nl = False,
        gt_path_nl = None,

        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        self.modify_neg_loss = modify_neg_loss
        self.use_zs_preds_nl = use_zs_preds_nl
        self.zs_preds_path_nl = zs_preds_path_nl 
        self.use_gt_nl = use_gt_nl 
        self.gt_path_nl = gt_path_nl
        self.zs_conf_thresh = zs_conf_thresh

        if modify_neg_loss and use_gt_nl:
            self.gt_annos = self.get_anno_from_gt_file()
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
            self.keep_neg_cls_inds = kwargs.pop('keep_neg_cls_inds')
            self.inverse_weights = kwargs.pop('fed_inverse_weight')
            self.deterministic_fed_loss = kwargs.pop('deterministic_fed_loss')
            self.all_ann_file = kwargs.pop('all_ann_file')

            if self.deterministic_fed_loss:
                all_train_data = load