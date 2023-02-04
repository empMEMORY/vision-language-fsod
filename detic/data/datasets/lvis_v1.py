# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta

logger = logging.getLogger(__name__)

__all__ = ["custom_load_lvis_json", "custom_register_lvis_instances"]


def custom_register_lvis_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="lvis", **metadata
    )


def custom_load_lvis_json(json_file, image_root, dataset_name=None):
    '''
    Modifications:
      use `file_name`
      convert neg_category_ids
      add pos_category_ids
    '''
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    catid2contid = {x['id']: i for i, x in enumerate(
        sorted(lvis_api.dataset['categories'], key=lambda x: x['id']))}
    if len(lvis_api.dataset['categories']) == 1203:
        for x in lvis_api.dataset['categories']:
      