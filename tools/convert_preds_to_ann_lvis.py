
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
    parser.add_argument("--pred_path_val", default='/home/anishmad/msr_thesis/detic-lt3d/results/nuimages/fedloss_all/basic_prompt_inference_fedloss_inverse_sample_cats_6_10_shots_s1/inference_nuimages_fsod_val_seed_1_shots_10/instances_predictions.pth')

    parser.add_argument("--conf_thresh", default=0.2, type=float)
    parser.add_argument("--dataset_name", default="lvis_v1_train_fsod_rare")
    # parser.add_argument("--save_path", default='datasets/metadata/Objects365_names_fix.csv')
    args = parser.parse_args()

    preds = torch.load(args.pred_path_train)
    val_preds = torch.load(args.pred_path_val)
    preds.extend(val_preds)
    train_dataset_name = args.dataset_name
    save_path = os.path.join(os.path.dirname(args.pred_path_train), f'trainval_preds_coco_fmt_conf_thresh_{args.conf_thresh}.json')

    preds_coco_ann_fmt = convert_to_coco_dict_with_preds(train_dataset_name, preds, conf_thresh=float(args.conf_thresh), trainval=True)

    with open(save_path, 'w') as f:
        json.dump(preds_coco_ann_fmt,f)
