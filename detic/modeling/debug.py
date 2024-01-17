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
        merges = np.maximum(merges, color_map)
    ret = np.clip(image * (1 - a) + merges * a, 0, 255).astype(np.uint8)
    return ret

def _decompose_level(x, shapes_per_level, N):
    '''
    x: LNHiWi x C
    '''
    x = x.view(x.shape[0], -1)
    ret = []
    st = 0
    for l in range(len(shapes_per_level)):
        ret.append([])
        h = shapes_per_level[l][0].int().item()
        w = shapes_per_level[l][1].int().item()
        for i in range(N):
            ret[l].append(x[st + h * w * i:st + h * w * (i + 1)].view(
                h, w, -1).permute(2, 0, 1))
        st += h * w * N
    return ret

def _imagelist_to_tensor(images):
    images = [x for x in images]
    image_sizes = [x.shape[-2:] for x in images]
    h = max([size[0] for size in image_sizes])
    w = max([size[1] for size in image_sizes])
    S = 32
    h, w = ((h - 1) // S + 1) * S, ((w - 1) // S + 1) * S
    images = [F.pad(x, (0, w - x.shape[2], 0, h - x.shape[1], 0, 0)) \
        for x in images]
    images = torch.stack(images)
    return images


def _ind2il(ind, shapes_per_level, N):
    r = ind
    l = 0
    S = 0
    while r - S >= N * shapes_per_level[l][0] * shapes_per_level[l][1]:
        S += N * shapes_per_level[l][0] * shapes_per_level[l][1]
        l += 1
    i = (r - S) // (shapes_per_level[l][0] * shapes_per_level[l][1])
    return i, l

def debug_train(
    images, gt_instances, flattened_hms, reg_targets, labels, pos_inds,
    shapes_per_level, locations, strides):
    '''
    images: N x 3 x H x W
    flattened_hms: LNHiWi x C
    shapes_per_level: L x 2 [(H_i, W_i)]
    locations: LNHiWi x 2
    '''
    reg_inds = torch.nonzero(
        reg_targets.max(dim=1)[0] > 0).squeeze(1)
    N = len(images)
    images = _imagelist_to_tensor(images)
    repeated_locations = [torch.cat([loc] * N, dim=0) \
        for loc in locations]
    locations = torch.cat(repeated_locations, dim=0)
    gt_hms = _decompose_level(flattened_hms, shapes_per_level, N)
    masks = flattened_hms.new_zeros((flattened_hms.shape[0], 1))
    masks[pos_inds] = 1
    masks = _decompose_level(masks, shapes_per_level, N)
    for i in range(len(images)):
        image = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        color_maps = []
        for l in range(len(gt_hms)):
            color_map = _get_color_image(
                gt_hms[l][i].detach().cpu().numpy())
            color_maps.append(color_map