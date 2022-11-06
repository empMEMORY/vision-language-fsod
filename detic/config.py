# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_detic_config(cfg):
    _C = cfg

    _C.WITH_IMAGE_LABELS = False # Turn on co-training with classification data

    # Open-vocabulary classifier
    _C.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False # Use fixed classifier for open-vocabulary detection
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    _C.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0 # >= 0: not use
    
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False # CenterNet2
    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False # Federated Loss
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/metadata/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'max_size' # max, softmax, sum
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 0.1
    _C.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    _C.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = False # Used for image-box loss and caption loss
    _C.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128 # num proposals for image-labeled data
    _C.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False # Used for WSDDN
    _C.MODEL.ROI