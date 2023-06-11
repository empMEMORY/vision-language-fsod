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
                for cat_id