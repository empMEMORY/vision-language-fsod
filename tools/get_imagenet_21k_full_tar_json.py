# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import numpy as np
import pickle
import io
import gzip
import sys
import time
from nltk.corpus import wordnet
from tqdm import tqdm
import operator
import torch

sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.data.tar_dataset import DiskTarDataset, _TarDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", default='datasets/imagenet/ImageNet-21k/')
    parser.add_argument("--tarfile_path", default='datasets/imagenet/metadata-22k/tar_files.npy')
    parser.add_argument("--tar_index_dir", default='datasets/imagenet/metadata-22k/tarindex_npy')
    parser.add_argument("--out_path", default='datasets/imagenet/annotations/imagenet-22k_image_info.json')
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()


    start_time = time.time()
    print('Building dataset')
    dataset = DiskTarDataset(args.tarfile_path, args.tar_i