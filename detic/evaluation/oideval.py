# Part of the code is from https://github.com/tensorflow/models/blob/master/research/object_detection/metrics/oid_challenge_evaluation.py
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# The original code is under Apache License, Version 2.0 (the "License");
# Part of the code is from https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py
# Copyright (c) 2019, Agrim Gupta and Ross Girshick
# Modified by Xingyi Zhou
# This script re-implement OpenImages evaluation in detectron2
# The code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/evaluation/oideval.py
# The original code is under Apache-2.0 License
# Copyright (c) Facebook, Inc. and its affiliates.
import os 
import datetime
import logging
import itertools
from collections import OrderedDict
from collections import defaultdict
import copy
import json
import numpy as np
import torch
from tabulate import tabulate

# from lvis.lvis import LVIS
#