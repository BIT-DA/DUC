#!/usr/bin/env bash

# train for deeplabv2, GTAV->cityscapes
CUDA_VISIBLE_DEVICES=0 python3 train_seg.py --cfg configs/deeplabv2_r101_gtav.yaml SOLVER.BATCH_SIZE 2 SOLVER.BASE_LR 3e-4 SOLVER.LAMBDA 0.01 ACTIVE.RATIO 0.01 ACTIVE.KAPPA 10 OUTPUT_DIR './output/deeplabv2/gtav' LOG_NAME 'log_lambda001_kappa10_3e-4.txt' SOLVER.STOP_ITER 50000

# test for deeplabv2, GTAV->cityscapes
CUDA_VISIBLE_DEVICES=0 python3 test_seg.py --cfg configs/deeplabv2_r101_gtav.yaml resume './output/deeplabv2/gtav' OUTPUT_DIR './output/deeplabv2/gtav' SOLVER.BATCH_SIZE_VAL 1





# train for deeplabv3+, GTAV->cityscapes
CUDA_VISIBLE_DEVICES=0 python3 train_seg.py --cfg configs/deeplabv3_plus_r101_gtav.yaml SOLVER.BATCH_SIZE 2 SOLVER.BASE_LR 3e-4 SOLVER.LAMBDA 0.01 ACTIVE.RATIO 0.01 ACTIVE.KAPPA 10 OUTPUT_DIR './output/deeplabv3+/gtav' LOG_NAME 'log_lambda001_kappa10_3e-4.txt' SOLVER.STOP_ITER 50000

# test for deeplabv3+, GTAV->cityscapes
CUDA_VISIBLE_DEVICES=0 python3 test_seg.py --cfg configs/deeplabv3_plus_r101_gtav.yaml resume './output/deeplabv3+/gtav' OUTPUT_DIR './output/deeplabv3+/gtav' SOLVER.BATCH_SIZE_VAL 1







