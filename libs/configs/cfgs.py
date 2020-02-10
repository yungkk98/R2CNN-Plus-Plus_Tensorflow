# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

"""
Attention + BUS + pyramid
"""

# ------------------------------------------------
VERSION = 'R2CNN_custom'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True
# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 100
SMRY_ITER = 1000
SAVE_WEIGHTS_INTE = 20000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith('resnet'):
    weights_name = NET_NAME
elif NET_NAME.startswith('MobilenetV2'):
    weights_name = 'mobilenet/mobilenet_v2_1.0_224'
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

EVALUATE_H_DIR = ROOT_PATH + '/output' + '/evaluate_h_result_pickle/' + VERSION
EVALUATE_R_DIR = ROOT_PATH + '/output' + '/evaluate_r_result_pickle/' + VERSION
TEST_ANNOTATION_PATH = '/mnt/USBB/gx/DOTA/DOTA_clip/val/labeltxt'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
ROTATE_NMS_USE_GPU = True
FIXED_BLOCKS = 2  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 1 / 7.0
RPN_CLASSIFICATION_LOSS_WEIGHT = 2.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 4.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 2.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0


MUTILPY_BIAS_GRADIENT = None  # 2.0  # if None, will not multiply
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.0003  # 0.0003
DECAY_STEP = [150000, 250000]  # [100000, 200000] Not pyramid images training
MAX_ITERATION = 600000   # 300000 Not pyramid images training

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'ship'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = [800, 600, 700, 900, 1000, 1100, 1200]
IMG_MAX_LENGTH = 1000
CLASS_NUM = 4

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.0001


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [8]  # can not be modified in most situations
ANCHOR_SCALES = [0.0625, 0.125, 0.25, 0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 4., 4., 1 / 5., 6., 1 / 6., 7., 1 / 7., 9., 1 / 9.]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 10.0]
ANCHOR_SCALE_FACTORS = None


# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 512
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 10000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 300  # 300

ADD_ATTENTION = True      # work
ADD_CONTEXT = False       # not work
ADD_ANCHOR_SHIFT = False  # not work
ADD_FUSION = True         # work

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard

FAST_RCNN_H_NMS_IOU_THRESHOLD = 0.4
FAST_RCNN_R_NMS_IOU_THRESHOLD = 0.1
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 150
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.4
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 512  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.4

ADD_GTBOXES_TO_TRAIN = False



