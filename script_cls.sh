#!/usr/bin/env bash

# Office-Home
python3 main.py --cfg configs/home.yaml GPU_ID 8 OUTPUT_DIR log TRAINER.BETA 1.0 OPTIM.LR 0.004  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 50 TRAINER.KAPPA 10 SAVE False LOG_NAME 'log_lr0004_beta1_lambda005_epoch50_kappa10.txt'


# VisDA-2017
python3 main.py --cfg configs/visda2017.yaml GPU_ID 8 OUTPUT_DIR log TRAINER.BETA 1.0 OPTIM.LR 0.001  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 40 TRAINER.KAPPA 10 LOG_NAME 'log_lr0001_beta1_lambda005_epoch40_kappa10.txt'


# miniDomainNet
python3 main.py --cfg configs/minidomainnet_clp.yaml GPU_ID 8 OUTPUT_DIR log_miniDomainNet_clp TRAINER.BETA 1.0 OPTIM.LR 0.002  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 50 TRAINER.KAPPA 10 LOG_NAME 'log_lr0002_beta1_lambda005_epoch50_kappa10_clp.txt'
python3 main.py --cfg configs/minidomainnet_pnt.yaml GPU_ID 8 OUTPUT_DIR log_miniDomainNet_pnt TRAINER.BETA 1.0 OPTIM.LR 0.002  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 50 TRAINER.KAPPA 10 LOG_NAME 'log_lr0002_beta1_lambda005_epoch50_kappa10_pnt.txt'
python3 main.py --cfg configs/minidomainnet_rel.yaml GPU_ID 8 OUTPUT_DIR log_miniDomainNet_rel TRAINER.BETA 1.0 OPTIM.LR 0.002  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 50 TRAINER.KAPPA 10 LOG_NAME 'log_lr0002_beta1_lambda005_epoch50_kappa10_rel.txt'
python3 main.py --cfg configs/minidomainnet_skt.yaml GPU_ID 8 OUTPUT_DIR log_miniDomainNet_skt TRAINER.BETA 1.0 OPTIM.LR 0.002  TRAINER.LAMBDA 0.05 SEED 2 TRAINER.MAX_EPOCHS 50 TRAINER.KAPPA 10 LOG_NAME 'log_lr0002_beta1_lambda005_epoch50_kappa10_skt.txt'
