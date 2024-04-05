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
                all_train_data = load_coco_json(self.all_ann_file, '', dataset_name='_')
        
                self.img_cat_map = {}
                for idx, img_info in enumerate(all_train_data):
                    # img_id = img_info['image_id']
                    
                    all_cats = [x['category_id'] for x in img_info['annotations']]
                    img_name = os.path.basename(img_info['file_name'])
                    
                    self.img_cat_map[img_name] = all_cats


        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

    def get_anno_from_gt_file(self):
        assert self.gt_path_nl is not None, "self.gt_path_nl is None, add correct path"

        with open(self.gt_path_nl, 'r') as f:
            gt_annos = json.load(f)
        
        img_anno_map = defaultdict(list)
        for anno in gt_annos['annotations']:
            img_id = anno['image_id']
            file_name = gt_annos['images'][img_id]['file_name']
            img_anno_map[file_name].append(anno)

        return img_anno_map

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'modify_neg_loss': cfg.MODEL.NEG_LOSS.MODIFY,
            'use_zs_preds_nl': cfg.MODEL.NEG_LOSS.USE_ZS_PREDS,
            'zs_preds_path_nl': cfg.MODEL.NEG_LOSS.ZS_PREDS_PATH,
            'use_gt_nl': cfg.MODEL.NEG_LOSS.USE_GT,
            'gt_path_nl': cfg.MODEL.NEG_LOSS.GT_PATH,
            'zs_conf_thresh': cfg.MODEL.NEG_LOSS.ZS_CONF_THRESH,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
                )
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
            ret['keep_neg_cls_inds'] = cfg.MODEL.ROI_BOX_HEAD.KEEP_FED_NEG_CLS_INDS
            ret['fed_inverse_weight'] = cfg.MODEL.ROI_BOX_HEAD.INVERSE_WEIGHTS
            ret['all_ann_file'] = cfg.MODEL.ROI_BOX_HEAD.ALL_ANN_FILE
            ret['deterministic_fed_loss'] = cfg.MODEL.ROI_BOX_HEAD.DETERMINISTIC_FED_LOSS

        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None
        images = self.preprocess_image(batched_inputs)
        if 'file_name' in batched_inputs[0]:
            file_names = [x['file_name'] for x in batched_inputs]
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            features = self.backbone(images.tensor)
            proposals, _ = self.proposal_generator(images, features, gt_instance