# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/data/multi_dataset_dataloader.py (Apache-2.0 License)
import copy
import logging
import numpy as np
import operator
import torch
import torch.utils.data
import json
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, log_first_n

from detectron2.config import configurable
from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
from collections import defaultdict
from typing import Optional


def _custom_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if 'MultiDataset' in sampler_name:
        dataset_dicts = get_detection_dataset_dicts_with_source(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    else:
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is not None:
        pass
    elif sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "MultiDatasetSampler":
        sampler = MultiDatasetSampler(
            dataset_dicts,
            dataset_ratio = cfg.DATALOADER.DATASET_RATIO,
            use_rfs = cfg.DATALOADER.USE_RFS,
            dataset_ann = cfg.DATALOADER.DATASET_ANN,
            repeat_threshold = cfg.DATALOADER.REPEAT_THRESHOLD,
        )
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset_dicts,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        'multi_dataset_grouping': cfg.DATALOADER.MULTI_DATASET_GROUPING,
        'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
        'dataset_bs': cfg.DATALOADER.DATASET_BS,
        'num_datasets': len(cfg.DATASETS.TRAIN)
    }


@configurable(from_config=_custom_train_loader_from_config)
def build_custom_train_loader(
        dataset, *, mapper, sampler, 
        total_batch_size=16,
        aspect_ratio_grouping=True, 
        num_workers=0,
        num_datasets=1,
        multi_dataset_grouping=False,
        use_diff_bs_size=False,
        dataset_bs=[]
    ):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    if multi_dataset_grouping:
        return build_multi_dataset_batch_data_loader(
            use_diff_bs_size,
            dataset_bs,
            dataset,
            sampler,
            total_batch_size,
            num_datasets=num_datasets,
            num_workers=num_workers,
        )
    else:
        return build_batch_data_loader(
            dataset,
            sampler,
            total_batch_size,
            aspect_ratio_grouping=aspect_ratio_grouping,
            num_workers=num_workers,
        )


def build_multi_dataset_batch_data_loader(
    use_diff_bs_size, dataset_bs,
    dataset, sampler, total_batch_size, num_datasets, num_workers=0
):
    """
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    if use_diff_bs_size:
        return DIFFMDAspectRatioGroupedDataset(
            data_loader, dataset_bs, num_datasets)
    else:
        return MDAspectRatioGroupedDataset(
            data_loader, batch_size, num_datasets)


def get_detection_dataset_dicts_with_source(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    
    for source_id, (dataset_name, dicts) in \
        enumerate(zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        for d in dicts:
            d['dataset_source'] = source_