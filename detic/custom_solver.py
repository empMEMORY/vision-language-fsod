# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
import itertools
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from detectron2.config import CfgNode

from detectron2.solver.build import maybe_add_gradient_clipping

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def build_custom_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    custom_multiplier_name = cfg.SOLVER.CUSTOM_MULTIPLIER_NAME
    optimizer_type = cfg.SOLVER.OPTIMIZER
    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        if cfg.SOLVER.FREEZE_BACKBONE:
            if 'backbone' in key:
                value.requires_grad = False          # explicitly set requires grad as False to save memory, skipping this would just use more memory   
                continue                   # skip param if it is of the backbone
        
        if cfg.MODEL.RESET_CLS_TRAIN:   # probably redundant because reset_cls_train actually makes it such that zs_weight doesn't show up in named_parameters
            if 'zs' in key:
                value.requires_grad = False          # explicitly set requires grad as False to save memory, skipping this would just use more memory   
                continue
       