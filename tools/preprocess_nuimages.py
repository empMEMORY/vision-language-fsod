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
parser.add_argument("--