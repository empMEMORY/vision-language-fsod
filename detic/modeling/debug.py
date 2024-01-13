# Copyright (c) Facebook, Inc. and its affiliates.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os

COLORS = ((np.random.rand(1300, 3) * 0.4 + 0.6) * 255).astype(
  np.uint8).reshape(1300, 1, 1, 3)

def _get_color_image(heatmap):
  heatmap = heatmap.reshape(
    heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], 1)
  if heatmap.shape[0] == 1:
      color_map = (heatmap * np.ones((1, 1, 1, 3), np.uint8) * 255).max(
          axis=