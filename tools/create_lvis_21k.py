# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', default='datasets/imagenet/annotations/imagenet-21k_image_info.json')
    parser.add_argument('--lvis_path', default='datasets/lvis/lvis_v1_train.json')
    parser.add_argument('--save_categories', default='')
    parser.add_argument('--not_save_imagenet', action='store_true')
    parse