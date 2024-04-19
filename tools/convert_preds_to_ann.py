import os
import numpy as np 
import torch
import glob
import json 
from detectron2.data.datasets.coco import load_coco_json, convert_to_coco_dict_with_dset, convert_to_coco_dict_with_preds
from copy import deepcopy
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path_train", default='/home/anishmad/msr_thesis/detic-lt3d/results/nuimages/fedloss_all/basic_prompt_inference_fedloss_inverse_sample_cats_6_10_shots_s1/inference_nuimages_fsod_train_seed_1_shots_10/instances_predictions.pth')
    parser.add_ar