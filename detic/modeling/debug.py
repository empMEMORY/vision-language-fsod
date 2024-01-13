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
          axis=0).astype(np.uint8) # H, W, 3
  else:
      color_map = (heatmap * COLORS[:heatmap.shape[0]]).max(axis=0).astype(np.uint8) # H, W, 3

  return color_map

def _blend_image(image, color_map, a=0.7):
  color_map = cv2.resize(color_map, (image.shape[1], image.shape[0]))
  ret = np.clip(image * (1 - a) + color_map * a, 0, 255).astype(np.uint8)
  return ret

def _blend_image_heatmaps(image, color_maps, a=0.7):
    merges = np.zeros((image.shape[0], image.shape[1], 3), np.float32)
    for color_map in color_maps:
        color_map = cv2.resize(color_map, (image.shape[1], image.shape[0]))
        merges = np.maximum(merges, col