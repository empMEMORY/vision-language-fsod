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
    parser = argpar