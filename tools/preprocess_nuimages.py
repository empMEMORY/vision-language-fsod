import json
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_ann_file", default='/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/nuimages_v1.0-train.json', help='Path to training annotation file')
parser.add_argument("--base_save_path", default='/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/no_wc/', help='base save path for modified annotations')
parser.add_argument("--wc_id", type=int, default=13, help='(Wheelchair) Class id for removing its annotations')
args = parser.parse_args()


def get_updated_annotations(orig_annos, cls_id_to_remove=13):
    new_annotations = {}
    new_annotations['images'] = orig_annos['images']
    new_annotations['categories'] = []
    new_annotations['annotations'] = []

    for ann_info in orig_annos['annotations']:
        if ann_info['category_id'] == cls_id_to_remove:      # remove annotation for cls_id_to_remove
            continue
        elif ann_info['category_id'] > cls_id_to_remove:
            new_ann_info = de