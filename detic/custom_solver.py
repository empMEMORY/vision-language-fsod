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
        
        if cfg.SOLVER.FINETUNE_MODEL_KEYWORDS is not None:
            finetune_flag=0
            for keyword in cfg.SOLVER.FINETUNE_MODEL_KEYWORDS:
                if keyword in key:
                    finetune_flag=1
            if finetune_flag==0:
                value.requires_grad = False
                continue
            if finetune_flag==1:
                print('Key to be finetuned', key)

        memo.add(value)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        if match_name_keywords(key, custom_multiplier_name):
            lr = lr * cfg.SOLVER.CUSTOM_MULTIPLIER
            print('Costum LR', key, lr)
        param = {"params": [value], "lr": lr}
        if optimizer_type != 'ADAMW':
            param['weight_decay'] = weight_decay
        params += [param]

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
         