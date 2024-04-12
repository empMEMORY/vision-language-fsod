
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import json
import numpy as np
from typing import Dict, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.lvis import load_lvis_json
from detectron2.data import MetadataCatalog
from torch.cuda.amp import autocast
from ..utils import load_class_freq, get_fed_loss_inds, get_fed_loss_inds_deterministic, get_fed_loss_inds_deterministic2, get_fed_loss_inds_prob, get_fed_loss_inds_deterministic_with_negs
from .zero_shot_classifier import ZeroShotClassifier
import os 


__all__ = ["DeticFastRCNNOutputLayers"]


class DeticFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self, 
        input_shape: ShapeSpec,
        *,
        mult_proposal_score=False,
        cls_score=None,
        sync_caption_batch = False,
        use_sigmoid_ce = False,
        use_fed_loss = False,
        ignore_zero_cats = False,
        fed_loss_num_cat = 50,
        dynamic_classifier = False,
        image_label_loss = '',
        use_zeroshot_cls = False,
        image_loss_weight = 0.1,
        with_softmax_prop = False,
        caption_weight = 1.0,
        neg_cap_weight = 1.0,
        add_image_box = False,
        debug = False,
        prior_prob = 0.01,
        cat_freq_path = '',
        fed_loss_freq_weight = 0.5,
        softmax_weak_loss = False,
        all_ann_file = None,
        deterministic_fed_loss = False,
        inverse_weights_fed_loss = False,
        use_ann_count_for_fedloss = False,
        dataset_train_name=None,
        all_gt_data_file = None,
        # num_classes=1203,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape, 
            **kwargs,
        )
        self.mult_proposal_score = mult_proposal_score
        self.sync_caption_batch = sync_caption_batch
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.dynamic_classifier = dynamic_classifier
        self.image_label_loss = image_label_loss
        self.use_zeroshot_cls = use_zeroshot_cls
        self.image_loss_weight = image_loss_weight
        self.with_softmax_prop = with_softmax_prop
        self.caption_weight = caption_weight
        self.neg_cap_weight = neg_cap_weight
        self.add_image_box = add_image_box
        self.softmax_weak_loss = softmax_weak_loss
        self.debug = debug
        self.all_ann_file = all_ann_file
        self.deterministic_fed_loss = deterministic_fed_loss
        self.inverse_weights_fed_loss = inverse_weights_fed_loss
        self.use_ann_count_for_fedloss = use_ann_count_for_fedloss
        self.dataset_train_name = dataset_train_name
        self.all_gt_data_file = all_gt_data_file
        # self.num_classes = num_classes

        if softmax_weak_loss:
            assert image_label_loss in ['max_size'] 

        if self.use_sigmoid_ce:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight, use_ann_count=use_ann_count_for_fedloss)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        if self.use_fed_loss:
            metadata = MetadataCatalog.get(self.dataset_train_name)
                # # unmap the category mapping ids for COCO
            if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):                                 # this mapping is 1-indexed for LVIS classes, i.e we map [0,1,2...336] to [13, 144, ...1230]
                reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
                reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
                id_mapping = metadata.thing_dataset_id_to_contiguous_id                           ## it maps from 1-indexed to 0-indexed
                id_mapper = lambda x: id_mapping[x]
                
            else:
                reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa
                id_mapper = lambda contiguous_id: contiguous_id

            #### IMP: If using the load_lvis_json(), and if "thing_dataset_id_to_contiguous_id" is not present in metadata, then the classes are transformed to 0-index from 1-index,
            ## i.e 1 is subtracted from the classes loaded from the annotations. 
            ### We'll also probably need to use  "thing_dataset_id_to_contiguous_id" for COCO data. Also, for LVIS v1 rare, we use this mapping. 

            if self.all_ann_file is not None:
                if 'lvis' in self.all_ann_file:
                    all_train_data = load_lvis_json(self.all_ann_file, '', dataset_name=None)
                else:
                    all_train_data = load_coco_json(self.all_ann_file, '', dataset_name='_')

                self.img_cat_map = {}
                for idx, img_info in enumerate(all_train_data):
                    img_id = img_info['image_id']
                    if 'lvis' in self.all_ann_file:

                        ### NOTE: img_info_annotations are loaded from load_lvis_json(), therefore they are 0 indexed. So we add +1 before using the mapping function.
                        all_cats = [id_mapper(x['category_id']+1) for x in img_info['annotations']]   # adding +1 since lvis annotations start from 1
                        img_name = img_info['file_name']
                    else:
                        all_cats = [x['category_id'] for x in img_info['annotations']]       # for nuimages, no such mapping is needed since we use load_coco_json() for it which does not change the indexing format
                        img_name = os.path.basename(img_info['file_name'])
                    
                    self.img_cat_map[img_name] = all_cats

            self.img_neg_cat_map = None
            if self.all_gt_data_file is not None and 'lvis' in self.all_gt_data_file:
                all_gt_data = load_lvis_json(self.all_gt_data_file, '', dataset_name=None)
                self.img_neg_cat_map = {}
                for idx, img_info in enumerate(all_gt_data):
                    img_name = img_info['file_name']
                    neg_cats = img_info['neg_category_ids']     # can be an empty list as well    # 1-indexed
                    if self.num_classes==1230 or self.num_classes==1203:       # if all classes are evaluated on
                        neg_cats_mapped = [x-1 for x in neg_cats]    # if using all classes then subtract 1 to adjust for lvis annotations starting from 1. This is because We use the negative cats as indices to array of size C
                    else:                                             # if using a subset of lvis classes like rare, then the mapping will take care of the indexing.
                        neg_cats_mapped = [id_mapper(x) for x in neg_cats if x in id_mapping]

                    self.img_neg_cat_map[img_name] = neg_cats_mapped

        if self.use_fed_loss and len(self.freq_weight) < self.num_classes:
            # assert self.num_classes == 11493
            print('Extending federated loss weight')
            self.freq_weight = torch.cat(
                [self.freq_weight, 
                self.freq_weight.new_zeros(
                    self.num_classes - len(self.freq_weight))]
            )

        assert (not self.dynamic_classifier) or (not self.use_fed_loss)
        input_size = input_shape.channels * \
            (input_shape.width or 1) * (input_shape.height or 1)
        
        if self.use_zeroshot_cls:
            del self.cls_score
            del self.bbox_pred
            assert cls_score is not None
            self.cls_score = cls_score
            self.bbox_pred = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, 4)
            )
            weight_init.c2_xavier_fill(self.bbox_pred[0])
            nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
            nn.init.constant_(self.bbox_pred[-1].bias, 0)

        if self.with_softmax_prop:
            self.prop_score = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, self.num_classes + 1),
            )
            weight_init.c2_xavier_fill(self.prop_score[0])
            nn.init.normal_(self.prop_score[-1].weight, mean=0, std=0.001)
            nn.init.constant_(self.prop_score[-1].bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'image_label_loss': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS,
            'use_zeroshot_cls': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
            'image_loss_weight': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT,
            'with_softmax_prop': cfg.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP,
            'caption_weight': cfg.MODEL.ROI_BOX_HEAD.CAPTION_WEIGHT,
            'neg_cap_weight': cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'debug': cfg.DEBUG or cfg.SAVE_DEBUG or cfg.IS_DEBUG,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
            'softmax_weak_loss': cfg.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS,
            'all_ann_file': cfg.MODEL.ROI_BOX_HEAD.ALL_ANN_FILE,
            'deterministic_fed_loss': cfg.MODEL.ROI_BOX_HEAD.DETERMINISTIC_FED_LOSS,
            'inverse_weights_fed_loss': cfg.MODEL.ROI_BOX_HEAD.INVERSE_WEIGHTS,
            'use_ann_count_for_fedloss': cfg.MODEL.ROI_BOX_HEAD.USE_ANN_COUNT_FOR_FEDLOSS,
            'dataset_train_name': cfg.DATASETS.TRAIN[0],
            'all_gt_data_file': cfg.MODEL.ROI_BOX_HEAD.ALL_GT_DATA_FILE,
            # 'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        })
        if ret['use_zeroshot_cls']:
            ret['cls_score'] = ZeroShotClassifier(cfg, input_shape)    #input_shape=1024  : zeroshotclassifier will convert it to 512 dim for clip 
        return ret

    def losses(self, predictions, proposals, \
        use_advanced_loss=True,
        classifier_info=(None,None,None),
        file_names=None,
        valmode=False):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        try:
            # img_wise_gt_classes = torch.stack([p.gt_classes for p in proposals])      # if this fails, then desired number of proposals not being generated for each image to stack into one tensor. 
            img_wise_gt_classes = [p.gt_classes for p in proposals]      # if this fails, then desired number of proposals not being generated for each image to stack into one tensor. 
        except:
            import ipdb; ipdb.set_trace()
        num_classes = self.num_classes               # =18
        if self.dynamic_classifier:

            _, cls_id_map = classifier_info[1]

            gt_classes = cls_id_map[gt_classes]          # size = len(proposals) = 512*B unless changed in config
            img_wise_gt_classes = cls_id_map[img_wise_gt_classes]

            num_classes = scores.shape[1] - 1
            assert cls_id_map[self.num_classes] == num_classes     # probably to check that id mapping is done correctly in case using dynamic classifier and also that background class is never picked in GT, 
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss2(scores, img_wise_gt_classes, file_names=file_names, valmode=valmode)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)
        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, 
                num_classes=num_classes)
        }

    def sigmoid_cross_entropy_loss2(self, pred_class_logits, gt_classes, file_names=None, valmode=False):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]             # 1024 (for bsize=2, num_proposals=1024)
        C = pred_class_logits.shape[1] - 1      # 18 for nuimgs/wc

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(torch.cat(gt_classes))), torch.cat(gt_classes)] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1