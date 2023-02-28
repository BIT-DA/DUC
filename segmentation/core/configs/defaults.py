import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplab_resnet101"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False
_C.MODEL.USE_SE = True

_C.INPUT = CN()
_C.INPUT.SOURCE_INPUT_SIZE_TRAIN = (1280, 720)
_C.INPUT.TARGET_INPUT_SIZE_TRAIN = (1024, 512)
_C.INPUT.INPUT_SIZE_TEST = (1024, 512)
_C.INPUT.INPUT_SCALES_TRAIN = (1.0, 1.0)
_C.INPUT.IGNORE_LABEL = 255
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False

# GaussianBlur
_C.INPUT.GAUSSIANBLUR = False

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# RandomApply Transforms
_C.INPUT.RANDOMAPPLY = 0.0

# RandomGrayscale
_C.INPUT.GRAYSCALE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.0

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.SOURCE_TRAIN = ""
_C.DATASETS.TARGET_TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 16000
_C.SOLVER.STOP_ITER = 10000
_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.LR_METHOD = 'poly'
_C.SOLVER.BASE_LR = 0.02
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.BETA = 1.0
_C.SOLVER.LAMBDA = 0.01

# 4 images per batch, two for source and two for target
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.BATCH_SIZE_VAL = 1


_C.ACTIVE = CN()

# active strategy
_C.ACTIVE.NAME = 'DUC'
_C.ACTIVE.SELECT_ITER = [10000, 12000, 14000, 16000, 18000] # for 5%
# ratio selection for each round
_C.ACTIVE.RATIO = 0.01    # 0.01 for 5%, 0.0044 for 2.2%
_C.ACTIVE.KAPPA = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.OUTPUT_DIR = './output'
_C.LOG_NAME = 'log.txt'
_C.resume = ""
_C.PREPARE_DIR = ""
_C.CV_DIR = ""
_C.SEED = 2631
_C.DEBUG = 0



