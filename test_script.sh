#!/usr/bin/env bash

# VisDA-2017
#python3 test.py --cfg configs/visda2017_centerCrop.yaml --source synthetic --target real --weight_path /home/lishuang/xmx/2022NeurIPS/DUC_github_code/log/visda2017_1/synthetic2real/best_model_synthetic2real.pth GPU_ID 6
python3 test.py --cfg configs/home.yaml --source RealWorld --target Art --weight_path /home/lishuang/xmx/2022NeurIPS/DUC_github_code/log_delete/home/RealWorld2Art/best_model_RealWorld2Art.pth GPU_ID 6
