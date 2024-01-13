
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import math
from os.path import join
import numpy as np
import copy
from functools import partial

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import FPN
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d
from detectron2.modeling.backbone import Backbone

from timm import create_model
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import ResNet, Bottleneck
from timm.models.resnet import default_cfgs as default_cfgs_resnet
from timm.models.convnext import ConvNeXt, default_cfgs, checkpoint_filter_fn


@register_model
def convnext_tiny_21k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    cfg = default_cfgs['convnext_tiny']
    cfg['url'] = 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth'
    model = build_model_with_cfg(
        ConvNeXt, 'convnext_tiny', pretrained,
        default_cfg=cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **model_args)
    return model

class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices')
        super().__init__(**kwargs)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        ret = [x]
        x = self.layer1(x)
        ret.append(x)
        x = self.layer2(x)
        ret.append(x)
        x = self.layer3(x)
        ret.append(x)
        x = self.layer4(x)
        ret.append(x)
        return [ret[i] for i in self.out_indices]


    def load_pretrained(self, cached_file):
        data = torch.load(cached_file, map_location='cpu')
        if 'state_dict' in data:
            self.load_state_dict(data['state_dict'])
        else:
            self.load_state_dict(data)


model_params = {
    'resnet50_in21k': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
}


def create_timm_resnet(variant, out_indices, pretrained=False, **kwargs):
    params = model_params[variant]
    default_cfgs_resnet['resnet50_in21k'] = \
        copy.deepcopy(default_cfgs_resnet['resnet50'])
    default_cfgs_resnet['resnet50_in21k']['url'] = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    default_cfgs_resnet['resnet50_in21k']['num_classes'] = 11221

    return build_model_with_cfg(
        CustomResNet, variant, pretrained,
        default_cfg=default_cfgs_resnet[variant],
        out_indices=out_indices,
        pretrained_custom_load=True,
        **params,
        **kwargs)


class LastLevelP6P7_P5(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)