# Part of the code is from https://github.com/tensorflow/models/blob/master/research/object_detection/metrics/oid_challenge_evaluation.py
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# The original code is under Apache License, Version 2.0 (the "License");
# Part of the code is from https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py
# Copyright (c) 2019, Agrim Gupta and Ross Girshick
# Modified by Xingyi Zhou
# This script re-implement OpenImages evaluation in detectron2
# The code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/evaluation/oideval.py
# The original code is under Apache-2.0 License
# Copyright (c) Facebook, Inc. and its affiliates.
import os 
import datetime
import logging
import itertools
from collections import OrderedDict
from collections import defaultdict
import copy
import json
import numpy as np
import torch
from tabulate import tabulate

# from lvis.lvis import LVIS
# from lvis.results import LVISResults
from lvis import LVIS
from lvis import LVISResults


import pycocotools.mask as mask_utils

from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import DatasetEvaluator

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])
  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision

class OIDEval:
    def __init__(
        self, lvis_gt, lvis_dt, iou_type="bbox", expand_pred_label=False, 
        oid_hierarchy_path='./datasets/oid/annotations/challenge-2019-label500-hierarchy.json'):
        """Constructor for OIDEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            # self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt, max_dets=-1)
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        if expand_pred_label:
            oid_hierarchy = json.load(open(oid_hierarchy_path, 'r'))
            cat_info = self.lvis_gt.dataset['categories']
            freebase2id = {x['freebase_id']: x['id'] for x in cat_info}
            id2freebase = {x['id']: x['freebase_id'] for x in cat_info}
            id2name = {x['id']: x['name'] for x in cat_info}
            
            fas = defaultdict(set)
            def dfs(hierarchy, cur_id):
                all_childs = set()
                all_keyed_child = {}
                if 'Subcategory' in hierarchy:
                    for x in hierarchy['Subcategory']:
                        childs = dfs(x, freebase2id[x['LabelName']])
                        all_childs.update(childs)
                if cur_id != -1:
                    for c in all_childs:
                        fas[c].add(cur_id)
                all_childs.add(cur_id)
                return all_childs
            dfs(oid_hierarchy, -1)
            
            expanded_pred = []
            id_count = 0
            for d in self.lvis_dt.dataset['annotations']:
                cur_id = d['category_id']
                ids = [cur_id] + [x for x in fas[cur_id]]
                for cat_id in ids:
                    new_box = copy.deepcopy(d)
                    id_count = id_count + 1
                    new_box['id'] = id_count
                    new_box['category_id'] = cat_id
                    expanded_pred.append(new_box)

            print('Expanding original {} preds to {} preds'.format(
                len(self.lvis_dt.dataset['annotations']),
                len(expanded_pred)
                ))
            self.lvis_dt.dataset['annotations'] = expanded_pred
            self.lvis_dt._create_index()
        
        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = {d["id"]: d["pos_category_ids"] for d in img_data}
        # img_pl = defaultdict(set)
        for ann in gts:
            # img_pl[ann["image_id"]].add(ann["category_id"])
            assert ann["category_id"] in img_pl[ann["image_id"]]
        # print('check pos ids OK.')
        
        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info("Running per image evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id)
            for img_id in self.params.img_ids
            for cat_id in cat_ids
        }

        # loop through images, area range, max detection number
        print('Evaluating ...')
        self.eval_imgs = [
            self.evaluate_img_google(img_id, cat_id, area_rng)
            for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._gts[img_id, cat_id]
            ]
            dt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._dts[img_id, cat_id]
            ]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        # iscrowd = [int(False)] * len(gt)
        iscrowd = [int('iscrowd' in g and g['iscrowd'] > 0) for g in gt]

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img_google(self, img_id, cat_id, area_rng):
        gt, dt = self._get_gt_dt(img_id, cat_id)
        if len(gt) == 0 and len(dt) == 0:
            return None
        
        if len(dt) == 0:
            return {
                "image_id": img_id,
                "category_id": cat_id,
                "area_rng": area_rng,
                "dt_ids": [],
                "dt_matches": np.array([], dtype=np.int32).reshape(1, -1),
                "dt_scores": [],
                "dt_ignore": np.array([], dtype=np.int32).reshape(1, -1),
                'num_gt': len(gt)
            }

        no_crowd_inds = [i for i, g in enumerate(gt) \
            if ('iscrowd' not in g) or g['iscrowd'] == 0]
        crowd_inds = [i for i, g in enumerate(gt) \
            if 'iscrowd' in g and g['iscrowd'] == 1]
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")

        if len(self.ious[img_id, cat_id]) > 0:
            ious = self.ious[img_id, cat_id]
            iou = ious[:, no_crowd_inds]
            iou = iou[dt_idx]
            ioa = ious[:, crowd_inds]
            ioa = ioa[dt_idx]
        else:
            iou = np.zeros((len(dt_idx), 0))
            ioa = np.zeros((len(dt_idx), 0))
        scores = np.array([dt[i]['score'] for i in dt_idx])

        num_detected_boxes = len(dt)
        tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_group_of = np.zeros(num_detected_boxes, dtype=bool)

        def compute_match_iou(iou):
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_gt_ids[i]
                is_evaluatable = (not tp_fp_labels[i] and
                                iou[i, gt_id] >= 0.5 and
                                not is_matched_to_group_of[i])
                if is_evaluatable:
                    if not is_gt_detected[gt_id]:
                        tp_fp_labels[i] = True
                        is_gt_detected[gt_id] = True

        def compute_match_ioa(ioa):
            scores_group_of = np.zeros(ioa.shape[1], dtype=float)
            tp_fp_labels_group_of = np.ones(
                ioa.shape[1], dtype=float)
            max_overlap_group_of_gt_ids = np.argmax(ioa, axis=1)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_group_of_gt_ids[i]
                is_evaluatable = (not tp_fp_labels[i] and
                                ioa[i, gt_id] >= 0.5 and
                                not is_matched_to_group_of[i])
                if is_evaluatable:
                    is_matched_to_group_of[i] = True
                    scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])
            selector = np.where((scores_group_of > 0) & (tp_fp_labels_group_of > 0))
            scores_group_of = scores_group_of[selector]
            tp_fp_labels_group_of = tp_fp_labels_group_of[selector]

            return scores_group_of, tp_fp_labels_group_of

        if iou.shape[1] > 0:
            compute_match_iou(iou)

        scores_box_group_of = np.ndarray([0], dtype=float)
        tp_fp_labels_box_group_of = np.ndarray([0], dtype=float)

        if ioa.shape[1] > 0:
            scores_box_group_of, tp_fp_labels_box_group_of = compute_match_ioa(ioa)

        valid_entries = (~is_matched_to_group_of)

        scores =  np.concatenate(
            (scores[valid_entries], scores_box_group_of))
        tp_fps = np.concatenate(
            (tp_fp_labels[valid_entries].astype(float),
             tp_fp_labels_box_group_of))
    
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_matches": np.array([1 if x > 0 else 0 for x in tp_fps], dtype=np.int32).reshape(1, -1),
            "dt_scores": [x for x in scores],
            "dt_ignore":  np.array([0 for x in scores], dtype=np.int32).reshape(1, -1),
            'num_gt': len(gt)
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = 1
        num_recalls = 1

        num_cats = len(cat_ids)
        num_area_rngs = 1
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, num_cats, num_area_rngs)
        )
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                num_gt = sum([e['num_gt'] for e in E])
                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[
                           